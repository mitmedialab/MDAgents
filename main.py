import os
import json
import time
import csv
import random
import openai
from openai import OpenAI
import google.generativeai as genai
import argparse
from tqdm import tqdm
from pathlib import Path
import base64
import requests
from collections import Counter
from prettytable import PrettyTable 
from termcolor import colored, cprint
from pptree import *
import re
import climage 


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='medqa')
parser.add_argument('--model', type=str, default='gemini-pro')
parser.add_argument('--difficulty', type=str, default='mixed')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--num_teams', type=int, default=3)
parser.add_argument('--num_agents', type=int, default=5)
parser.add_argument('--rag', type=bool, default=False)
parser.add_argument('--review', type=bool, default=False)
args = parser.parse_args()

seed = args.seed
model = args.model
dataset = args.dataset
new_difficulty = args.difficulty
num_teams = args.num_teams
num_agents = args.num_agents
rag = args.rag
review = args.review

if rag == True:
    from src.medrag import MedRAG
    if model == 'gpt-3.5':
        rag_model_info = 'OpenAI/gpt-3.5-turbo-16k'
    elif model == 'gpt-4':
        rag_model_info = 'OpenAI/gpt-4'
    elif model == 'gpt-4v':
        rag_model_info = 'OpenAI/gpt-4-vision-preview'
    elif model == 'gemini-pro':
        rag_model_info = 'gemini-pro'
    elif model == 'gemini-pro-vision':
        rag_model_info = 'gemini-pro-vision'

genai.configure(api_key='your_api_key')
openai.api_key = 'your_api_key' 
client = OpenAI(
    api_key='your_api_key', 
)

agent_emoji = ['\U0001F468\u200D\u2695\uFE0F', '\U0001F468\U0001F3FB\u200D\u2695\uFE0F', '\U0001F469\U0001F3FC\u200D\u2695\uFE0F', '\U0001F469\U0001F3FB\u200D\u2695\uFE0F', '\U0001f9d1\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3fd\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3fd\u200D\u2695\uFE0F', '\U0001F468\U0001F3FD\u200D\u2695\uFE0F']
random.shuffle(agent_emoji)

def extract_profession_data(input_string):
    input_string = re.sub(r'\s+', ' ', input_string)
    entries = re.split(r'(?<=Independent)\s*(?=\d+\.)', input_string)
    pattern = r'^(\d+)\.\s+(.*?)\s+-\s+Specializes in (.*?)(?:\s+-\s+Hierarchy:\s+(.*?))(?=\d+\.|$)'
    extracted_data = []

    for entry in entries:
        match = re.match(pattern, entry.strip())
        if match:
            name_description = f"{match.group(2)} - Specializes in {match.group(3)}"
            hierarchy = match.group(4).strip()
            extracted_data.append([name_description, hierarchy])

    return extracted_data

def parse_comments(text):
    expert_comments = {}
    pattern = r'(\d+\.\s+([A-Za-z]+)):\s+(.*?)(?=\d+\.\s+[A-Za-z]+:|$)'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        expert_comments[match[1].lower()] = match[2].strip()
    
    return expert_comments

def shuffle_dict(original_dict):
    items = list(original_dict.items())
    random.shuffle(items)
    shuffled_dict = dict(items)
    return shuffled_dict

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def parse_options(input_str):
    options_list = ['(' + option for option in input_str.split(" (") if option]
    return options_list

def get_weighted(question, candidates):
    global total_api_calls

    decision_prompt = f"""You are an experienced medical expert. Now, given the medical query and opinions from medical experts, please review the inputs very carefully and return your final decision by using weighted voting."""
    tmp_agent = Agent(instruction=decision_prompt, role='decision maker', model_info=model)
    tmp_agent.chat(decision_prompt)
    final_report = ""
    num_agent = 0
    for k,v in candidates.items():
        final_report += "Agent {}: {}\n".format(num_agent+1, v)
        num_agent += 1

    decision = tmp_agent.chat("Opinions:\n{}\n\nQuestion: {}\n\nNow, please return your answer in a format among the options.\nAnswer: ".format(final_report, question))
    total_api_calls += 1 
    return decision

def get_majority(y_hat):
    counts = Counter(y_hat.values())
    max_freq = max(counts.values())
    max_items = [item for item, freq in counts.items() if freq == max_freq]
    return random.choice(max_items)

def parse_hierarchy(info, emojis):
    moderator = Node('moderator (\U0001F468\u200D\u2696\uFE0F)')
    agents = [moderator]

    count = 0
    for expert, hierarchy in info:
        try:
            expert = expert.split('-')[0].split('.')[1].strip()
        except:
            expert = expert.split('-')[0].strip()
        
        if hierarchy == None:
            hierarchy = 'Independent'
        
        if 'independent' not in hierarchy.lower():
            parent = hierarchy.split(">")[0].strip()
            child = hierarchy.split(">")[1].strip()

            for agent in agents:
                if agent.name.split("(")[0].strip().lower() == parent.strip().lower():
                    child_agent = Node("{} ({})".format(child, emojis[count]), agent)
                    agents.append(child_agent)

        else:
            agent = Node("{} ({})".format(expert, emojis[count]), moderator)
            agents.append(agent)

        count += 1

    return agents

def parse_group_info(group_info):
    lines = group_info.split('\n')
    
    parsed_info = {
        'group_goal': '',
        'members': []
    }

    parsed_info['group_goal'] = "".join(lines[0].split('-')[1:])
    
    for line in lines[1:]:
        if line.startswith('Member'):
            member_info = line.split(':')
            member_role_description = member_info[1].split('-')

            member_role = member_role_description[0].strip()
            member_expertise = member_role_description[1].strip() if len(member_role_description) > 1 else ''
            
            parsed_info['members'].append({
                'role': member_role,
                'expertise_description': member_expertise
            })
    
    return parsed_info

class Agent:
    def __init__(self, instruction, role, examplers=None, model_info='gemini-pro', img_path=None):
        self.instruction = instruction
        self.role = role
        self.model_info = model_info
        self.img_path = img_path

        if self.model_info == 'gemini-pro':
            self.model = genai.GenerativeModel('gemini-pro')
            self._chat = self.model.start_chat(history=[])

        elif self.model_info in ['gpt-3.5', 'gpt-4']:
            self.messages = [
                {"role": "system", "content": instruction},
            ]
            if examplers != None:
                for exampler in examplers:
                    self.messages.append({"role": "user", "content": exampler['question']})
                    self.messages.append({"role": "assistant", "content": exampler['answer'] + "\n\n" + exampler['reason']})

        elif self.model_info == 'gpt-4v':
            self.messages = [{"role": "system", "content": instruction}]

            if examplers is not None:
                for exampler in examplers:
                    image_url = encode_image(self.img_path)
                    self.messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": exampler['question']},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_url}"}}
                            ]
                        }
                    )

            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=self.messages,
                max_tokens=512
            )
            self.messages.append({"role": "assistant", "content": response.choices[0].message.content})

        elif self.model_info == 'gemini-pro-vision':
            self.model = genai.GenerativeModel('gemini-pro-vision')
            self._chat = self.model.start_chat(history=[])
                
    def chat(self, message, img_path=None, chat_mode=True):
        if self.model_info == 'gemini-pro':
            for _ in range(3):
                try:
                    response = self._chat.send_message(message, stream=True)
                    responses = ""
                    for chunk in response:
                        responses += chunk.text + "\n"
                    break
                except:
                    time.sleep(0.5)
                    continue

            return responses

        elif self.model_info == 'gpt-3.5':
          self.messages.append({"role": "user", "content": message})
          response = client.chat.completions.create(
              model="gpt-3.5-turbo",
              messages=self.messages
          )
          self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
          return response.choices[0].message.content

        elif 'gpt-4' in self.model_info and img_path == None:
            self.messages.append({"role": "user", "content": message})

            tmp_messages = []
            for tm in self.messages:
                if isinstance(tm['content'], list):
                    tm_content = []
                    for _tm in tm['content']:
                        if _tm['type'] != 'text':
                            continue
                        else:
                            tm_content.append(_tm)
                    tmp_messages.append({'role': tm['role'], 'content': tm_content})

                else:
                    tmp_messages.append(tm)
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=tmp_messages
            )

            self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
            return response.choices[0].message.content

        elif self.model_info == 'gpt-4v':
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer {}".format('your_api_key')
            }
            tmp = [{"type": "text", "text": message}]

            if isinstance(img_path, list):
                for ip in img_path:
                    base64_image = encode_image(ip)
                    tmp1 = {
                        "type": "image_url",
                        "image_url": {
                        "url": f"data:image/jpeg;base64, {base64_image}"
                        }
                    }
                    tmp.append(tmp1)
            else:
                base64_image = encode_image(img_path)
                tmp1 = {
                    "type": "image_url",
                    "image_url": {
                    "url": f"data:image/jpeg;base64, {base64_image}"
                    }
                }
                tmp.append(tmp1)

                
            self.messages.append({"role": "user", "content": tmp})

            response = client.chat.completions.create(
                model="gpt-4-vision-preview", 
                messages=self.messages,
                max_tokens=512
            )

            return response.choices[0].message.content.strip()

        elif self.model_info == 'gemini-pro-vision':
            prompt = message
            content_list = [prompt]

            picture = {
                'mime_type': 'image/png',
                'data': Path(img_path).read_bytes()
            }
            content_list.append(picture)
            response = self._chat.send_message(content_list)

            responses = ""
            for chunk in response:
                responses += chunk.text + "\n"
            
            return responses


    def temp_responses(self, message, num_responses=5, img_path=None):
        if self.model_info in ['gpt-3.5', 'gpt-4']:
            if self.model_info == 'gpt-3.5':
                model_info = 'gpt-3.5-turbo'
                engine = ENGINE_GPT35
                openai.api_base = BASE_GPT35
                openai.api_key = API_KEY_GPT35
                
            
            self.messages.append({"role": "user", "content": message})

            temperatures = [0.3, 1.2]
            responses = {}
            for temperature in temperatures:
                if self.model_info == 'gpt-3.5':
                    model_info = 'gpt-3.5-turbo'
                else:
                    model_info = 'gpt-4'
                response = client.chat.completions.create(
                    model=model_info,
                    messages=self.messages,
                    temperature=temperature,
                )
                responses[temperature] = response.choices[0].message.content 
                
            return responses
        
        elif self.model_info == 'gemini-pro':
            response = self._chat.send_message(message, stream=True)
            responses = ""
            for chunk in response:
                responses += chunk.text + "\n"
            return responses

        elif self.model_info == 'gemini-pro-vision':
            prompt = message
            content_list = [prompt]

            picture = {
                'mime_type': 'image/png',
                'data': Path(img_path).read_bytes()
            }
            content_list.append(picture)

            response = self._chat.send_message(content_list)
            responses = ""
            for chunk in response:
                responses += chunk.text + "\n"
            
            return responses


        elif self.model_info == 'gpt-4v':
            n = num_responses
            temperatures = [0.3, 1.2]
            
            responses = {}
            for temperature in temperatures:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer {}".format('your_api_key')
                }
                tmp = [{"type": "text", "text": message}]

                base64_image = encode_image(img_path)
                tmp1 = {
                    "type": "image_url",
                    "image_url": {
                    "url": f"data:image/jpeg;base64, {base64_image}"
                    }
                }
                tmp.append(tmp1)
                response = client.chat.completions.create(
                    model="gpt-4-vision-preview",
                    messages=[{"role": "user", "content": tmp}]
                )
                responses[temperature] = response.choices[0].message.content.strip()

            return responses


class Group:
    def __init__(self, goal, members, question, examplers=None):
        self.goal = goal
        self.members = [] 
        for member_info in members:
            _agent = Agent('You are a {} who {}.'.format(member_info['role'], member_info['expertise_description'].lower()), role=member_info['role'], model_info=model)
            if model != 'gemini-pro-vision':
                _agent.chat('You are a {} who {}.'.format(member_info['role'], member_info['expertise_description'].lower()))
            self.members.append(_agent)
        self.question = question
        self.examplers = examplers

    def interact(self, comm_type, message=None, img_path=None):
        global total_api_calls

        if comm_type == 'internal':
            lead_member = None
            assist_members = []
            for member in self.members:
                member_role = member.role

                if 'lead' in member_role.lower():
                    lead_member = member
                else:
                    assist_members.append(member)

            if lead_member == None:
                try:
                    lead_member = assist_members[0]
                except:
                    lead_member = Agent('You are the lead of the medical group which aims to {self.goal}.', role='leader', model_info='gemini-pro-vision')
                    # delivery = lead_member.chat(f'''You are the lead of the medical group which aims to {self.goal}. ''' + delivery_prompt, img_path=img_path)
            

            delivery_prompt = f'''You are the lead of the medical group which aims to {self.goal}. You have the following assistant clinicians who work for you:'''
            for a_mem in assist_members:
                delivery_prompt += "\n{}".format(a_mem.role)
            
            delivery_prompt += "\n\nNow, given the medical query, provide a short answer to what kind investigations are needed from each assistant clinicians.\nQuestion: {}".format(self.question)
            try:
                if img_path != None:
                    try:
                        delivery = lead_member.chat(delivery_prompt, img_path)
                    except:    
                        lead_member = Agent('You are the lead of the medical group which aims to {self.goal}.', role='leader', model_info='gemini-pro-vision')
                        delivery = lead_member.chat(f'''You are the lead of the medical group which aims to {self.goal}. ''' + delivery_prompt, img_path=img_path)
            
                else:
                    delivery = lead_member.chat(delivery_prompt)
            except:
                if img_path != None:
                    try:
                        delivery = assist_members[0].chat(delivery_prompt, img_path)
                    except:    
                        lead_member = Agent('You are the lead of the medical group which aims to {self.goal}.', role='leader', model_info='gemini-pro-vision')
                        delivery = lead_member.chat(f'''You are the lead of the medical group which aims to {self.goal}. ''' + delivery_prompt, img_path=img_path)
            
                else:
                    delivery = lead_member.chat(delivery_prompt)
            
            total_api_calls +=1 
            investigations = []
            for a_mem in assist_members:
                if img_path != None:
                    if model == 'gemini-pro-vision':
                        try:
                            investigation = a_mem.chat("You are in a medical group where the goal is to {}. Your group lead is asking for the following investigations:\n{}\n\nGiven that your expertise is {}, return your investigation summary that contains the core information.".format(self.goal, delivery, a_mem.role), img_path)
                        except:
                            a_mem = Agent(f'You are a team member in a medical group which aims to {self.goal}. Your role is {a_mem.role}', role=a_mem.role, model_info='gemini-pro-vision')
                            investigation = a_mem.chat("You are in a medical group where the goal is to {}. Your group lead is asking for the following investigations:\n{}\n\nGiven that your expertise is {}, return your investigation summary that contains the core information.".format(self.goal, delivery, a_mem.role), img_path)
                    else:
                        investigation = a_mem.chat("You are in a medical group where the goal is to {}. Your group lead is asking for the following investigations:\n{}\n\nPlease remind your expertise and return your investigation summary that contains the core information.".format(self.goal, delivery), img_path)
                else:
                    investigation = a_mem.chat("You are in a medical group where the goal is to {}. Your group lead is asking for the following investigations:\n{}\n\nPlease remind your expertise and return your investigation summary that contains the core information.".format(self.goal, delivery))
                
                total_api_calls += 1 
                investigations.append([a_mem.role, investigation])
            
            gathered_investigation = ""
            for investigation in investigations:
                gathered_investigation += "[{}]\n{}\n".format(investigation[0], investigation[1])

            if self.examplers != None:
                investigation_prompt = f"""The gathered investigation from your asssitant clinicians is as follows:\n{gathered_investigation}.\n\nNow, after reviewing the following example cases, return your answer to the medical query among the option provided:\n\n{self.examplers}\nQuestion: {self.question}"""
            else:
                investigation_prompt = f"""The gathered investigation from your asssitant clinicians is as follows:\n{gathered_investigation}.\n\nNow, return your answer to the medical query among the option provided.\n\nQuestion: {self.question}"""
          

            try:
                response = lead_member.chat(investigation_prompt, img_path=img_path)
            except:
                lead_member = Agent('You are the lead of the medical group which aims to {self.goal}.', role='leader', model_info='gemini-pro-vision')
                response = lead_member.chat(f'''You are the lead of the medical group which aims to {self.goal}. ''' + investigation_prompt, img_path=img_path)
            total_api_calls +=1 

            return response

        elif comm_type == 'external':
            return


test_qa = []
examplers = []
if 'mmlu' in dataset:
    try:
        test_path = 'data/{}/{}_test_100.jsonl'.format(dataset.split('_')[0], dataset.split('_')[1])
        with open(test_path, 'r') as file2:
            for line in file2:
                data = json.loads(line)
                test_qa.append(data)
                examplers.append(data)
    except:
        test_path = 'data/{}/{}_test_100.jsonl'.format(dataset.split('_')[0], "_".join(dataset.split('_')[1:3]))
        with open(test_path, 'r') as file2:
            for line in file2:
                data = json.loads(line)
                test_qa.append(data)
                examplers.append(data)

elif 'mmmu' in dataset:
    train_path = 'data/mmmu/{}/validation-00000-of-00001.json'.format("_".join(dataset.split('_')[1:]))
    with open(train_path, 'r') as file1:
        data1 = json.load(file1)
        for d in data1:
            examplers.append(d)
    
    test_path = 'data/mmmu/{}/test_100.jsonl'.format("_".join(dataset.split('_')[1:]))
    with open(test_path, 'r') as file2:
        for line in file2:
            data = json.loads(line)
            test_qa.append(data)
            examplers.append(data)

elif dataset == 'medmcqa':
    train_path = 'data/{}/original_train.json'.format(dataset)
    with open(train_path, 'r') as file2:
        for line in file2:
            data = json.loads(line)
            question = data['question']
            opa = data['opa']
            opb = data['opb']
            opc = data['opc']
            opd = data['opd']
            label = data['cop']

            if label == 1:
                label = '(A)'
            elif label == 2:
                label = '(B)'
            elif label == 3:
                label = '(C)'
            elif label == 4:
                label = '(D)'

            examplers.append({'question': question, 'answer': label, 'opa': opa, 'opb': opb, 'opc': opc, 'opd': opd})


    test_path = 'data/{}/test_100.jsonl'.format(dataset)
    with open(test_path, 'r') as file2:
        for line in file2:
            data = json.loads(line)
            test_qa.append(data)


else:
    test_path = 'data/{}/test_100.jsonl'.format(dataset)
    with open(test_path, 'r') as file2:
        for line in file2:
            data = json.loads(line)
            test_qa.append(data)
            examplers.append(data)

no = 0
results = []
random.shuffle(test_qa)
for sample in tqdm(test_qa):
    if no == 50:
        break

    print("\n[INFO] no:", no)    
    total_api_calls = 0

    if dataset == 'medqa':
        question = sample['question'] + " Options: "
        options = []
        for k,v in sample['options'].items():
            options.append("({}) {}".format(k,v))
        random.shuffle(options)
        options = " ".join(options)
        question += options
        img_path = None

        rag_options = {
            "A": sample['options']['A'],
            "B": sample['options']['B'],
            "C": sample['options']['C'],
            "D": sample['options']['D'],
            "E": sample['options']['E']
        }

    elif dataset == 'medmcqa':
        options = ["(A) {}".format(sample['opa']), "(B) {}".format(sample['opb']), "(C) {}".format(sample['opc']), "(D) {}".format(sample['opd'])]
        random.shuffle(options)
        options = " ".join(options)
        question = sample['question'] + " " + options
        img_path = None

        rag_options = {
            "A": "{}".format(sample['opa']), 
            "B": "{}".format(sample['opb']), 
            "C": "{}".format(sample['opc']), 
            "D": "{}".format(sample['opd'])
        }

    elif dataset == 'pubmedqa':
        options = ['(A) Yes', '(B) No', '(C) Maybe']
        random.shuffle(options)
        options = " ".join(options)
        question = "Context: " + sample['context'] + "\nQuestion: " + sample['question'] + " Options: " + options

        rag_options = {
            'A': 'Yes', 
            'B': 'No', 
            'C': 'Maybe'
        }
        img_path = None


    elif dataset == 'ddxplus':
        initial_evidence = sample['initial_evidence']
        clinical_evidence = sample['evidences']
        age = sample['age']
        sex = sample['sex']
        options = sample['options']

        question = f"Patient Information:\n-Age: {age}\n-Sex: {sex}\n-Initial Evidence: {initial_evidence}\n\nClinical Evidence:\n{clinical_evidence}\n\nBased on the above information, please choose the most likely diagnosis:\n"
        question += "\n{}".format(options)

        rag_options = {} 
        tmp = sample['options'] 
        option_list = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)', '(I)']

        for i in range(len(option_list)):
            try:
                opx = tmp.split(option_list[i+1])[0].split(option_list[i])[1].strip()
                rag_options[option_list[i].replace(')','').replace('(','')] = opx
            except:
                continue
        
        img_path = None

    elif 'mmlu' in dataset:
        question = sample['question'] + " Options: (A) {} (B) {} (C) {} (D) {}".format(sample['opa'], sample['opb'], sample['opc'], sample['opd'])
        img_path = None

        rag_options = {
            "A": "{}".format(sample['opa']), 
            "B": "{}".format(sample['opb']), 
            "C": "{}".format(sample['opc']), 
            "D": "{}".format(sample['opd'])
        }
    
    elif dataset == 'pmc-vqa':
        question = sample['question'] + " Options: (A) {} (B) {} (C) {} (D) {}".format(sample['opa'].split(":")[-1].strip(), sample['opb'].split(":")[-1].strip(), sample['opc'].split(":")[-1].strip(), sample['opd'].split(":")[-1].strip())
        img_path = 'data/pmc-vqa/images/' + sample['img_path']

        rag_options = {
            "A": "{}".format(sample['opa']), 
            "B": "{}".format(sample['opb']), 
            "C": "{}".format(sample['opc']), 
            "D": "{}".format(sample['opd'])
        }

    elif dataset == 'path-vqa':
        options = ['(A) yes', '(B) no']
        random.shuffle(options)
        options = " ".join(options)
        question = sample['question'] + " " + options
        img_path = 'data/path-vqa/images/test/' + sample['img'] + ".jpg"

        rag_options = {
            'A': 'yes',
            'B': 'no'
        }

    elif 'mmmu' in dataset:
        _options = {}
        option_list = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)', '(I)', '(J)', '(K)']
        for oi,option in enumerate(sample['options']):
            _options["{}".format(option_list[oi])] = "{}".format(option)
        shuffled_dict = shuffle_dict(_options)
        options = ""
        for k,v in _options.items():
            options += "{} {} ".format(k,v)
        question = sample['question'] + " Options: {}".format(options)
        img_path = sample['image']

        rag_options = {}
        option_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
        for oi,option in enumerate(sample['options']):
            rag_options["{}".format(option_list[oi])] = "{}".format(option)

    try:
        output = climage.convert(img_path, is_unicode=True, width=50) 
    except:
        pass
    
    try:
        if new_difficulty == 'mixed':   
            difficulty_prompt = f"""Now, given the medical query as below, you need to decide the difficulty/complexity of it and provide short reasoning:\n{question}.\n\nPlease indicate the difficulty/complexity of the medical query among below options:\n1) basic: a single medical agent can output an answer.\n2) intermediate: number of medical experts with different expertise should dicuss and make final decision.\n3) advanced: multiple teams of clinicians from different departments need to collaborate with each other to make final decision."""
            
            if model == 'gpt-4v':
                medical_agent = Agent(instruction='You are a medical expert who conducts initial assessment and your job is to decide the difficulty/complexity of the medical query.', role='medical expert', model_info=model)
                medical_agent.chat('You are a medical expert who conducts initial assessment and your job is to decide the difficulty/complexity of the medical query.', img_path=img_path)
                response = medical_agent.chat(difficulty_prompt, img_path=img_path)
            
            elif model == 'gemini-pro-vision':
                medical_agent = Agent(instruction='You are a medical expert who conducts initial assessment and your job is to decide the difficulty/complexity of the medical query.', role='medical expert', model_info=model)
                response = medical_agent.chat('You are a medical expert who conducts initial assessment and your job is to decide the difficulty/complexity of the medical query. ' + difficulty_prompt, img_path=img_path)

            else:
                medical_agent = Agent(instruction='You are a medical expert who conducts initial assessment and your job is to decide the difficulty/complexity of the medical query.', role='medical expert', model_info='gemini-pro')
                medical_agent.chat('You are a medical expert who conducts initial assessment and your job is to decide the difficulty/complexity of the medical query.')
                response = medical_agent.chat(difficulty_prompt)

            if 'basic' in response.lower() or '1)' in response.lower():
                difficulty = 'basic'
            elif 'intermediate' in response.lower() or '2)' in response.lower():
                difficulty = 'intermediate'
            elif 'advanced' in response.lower() or '3)' in response.lower():
                difficulty = 'advanced'
            else:
                medical_agent = Agent(instruction='You are a medical expert who conducts initial assessment and your job is to decide the difficulty/complexity of the medical query.', role='medical expert', model_info='gemini-pro')
                medical_agent.chat('You are a medical expert who conducts initial assessment and your job is to decide the difficulty/complexity of the medical query.')
                response = medical_agent.chat(difficulty_prompt)

                if 'basic' in response.lower() or '1)' in response.lower():
                    difficulty = 'basic'
                elif 'intermediate' in response.lower() or '2)' in response.lower():
                    difficulty = 'intermediate'
                elif 'advanced' in response.lower() or '3)' in response.lower():
                    difficulty = 'advanced'

        else:
            difficulty = new_difficulty

        print("difficulty:", difficulty)
        print()

        if difficulty == 'basic':        
            if rag == True:
                medrag = MedRAG(llm_name="{}".format(rag_model_info), rag=True, retriever_name="MedCPT", corpus_name="Textbooks", img_path=img_path)
                final_decision, snippets, scores = medrag.answer(question=question, options=rag_options, k=32)

                total_api_calls += 1
            
            else:
                # 3-shot
                medical_agent = Agent(instruction='You are a helpful medical agent.', role='medical expert', model_info=model)
                fewshot_examplers = ""
                if dataset == 'medqa':
                    random.shuffle(examplers)
                    new_examplers = []
                    for ie,exampler in enumerate(examplers):
                        if ie > 2:
                            break
                        tmp_exampler = {}
                        exampler_question = exampler['question']
                        choices = []
                        for k,v in exampler['options'].items():
                            choices.append("({}) {}".format(k,v))
                        
                        random.shuffle(choices)
                        exampler_question += " " + ' '.join(choices)
                        exampler_answer = "Answer: ({}) {}\n\n".format(exampler['answer_idx'], exampler['answer'])

                        # chain-of-thought (cot)
                        exampler_reason = medical_agent.chat("You are a helpful medical agent. Below is an example of medical knowledge question and answer. After reviewing the below medical question and answering, can you provide 1-2 sentences of reason that support the answer as you didn't know the answer ahead?\n\nQuestion: {}\n\nAnswer: {}".format(exampler_question, exampler_answer))

                        tmp_exampler['question'] = exampler_question
                        tmp_exampler['reason'] = exampler_reason
                        tmp_exampler['answer'] = exampler_answer
                        new_examplers.append(tmp_exampler)
                
                elif dataset == 'pubmedqa':
                    random.shuffle(examplers)
                    new_examplers = []
                    for ie,exampler in enumerate(examplers):
                        if ie > 2:
                            break
                        tmp_exampler = {}
                        exampler_question = exampler['question']
                        choices = ['(A) Yes', '(B) No', '(C) Maybe']
                        random.shuffle(choices)
                        exampler_question += ' ' + ' '.join(choices)
                        
                        if 'yes' in exampler['label'].lower():
                            exampler_answer = "Answer: (A) Yes"
                        if 'no' in exampler['label'].lower():
                            exampler_answer = "Answer: (B) No"
                        if 'maybe' in exampler['label'].lower():
                            exampler_answer = "Answer: (C) Maybe"

                        # chain-of-thought (cot)
                        exampler_reason = medical_agent.chat("Below is an example of medical knowledge question and answer. After reviewing the below medical question and answering, can you provide 1-2 sentences of reason that support the answer as you didn't know the answer ahead?\n\nQuestion: {}\n\nAnswer: {}".format(exampler_question, exampler_answer))

                        tmp_exampler['question'] = exampler_question
                        tmp_exampler['reason'] = exampler_reason
                        tmp_exampler['answer'] = exampler_answer
                        new_examplers.append(tmp_exampler)
                
                elif dataset == 'ddxplus':
                    random.shuffle(examplers)
                    new_examplers = []
                    for ie,exampler in enumerate(examplers):
                        if ie > 2:
                            break
                        tmp_exampler = {}

                        exampler_question = ""
                        exampler_initial_evidence = exampler['initial_evidence']
                        exampler_clinical_evidence = exampler['evidences']
                        exampler_age = exampler['age']
                        exampler_sex = exampler['sex']
                        exampler_options = exampler['options']
                        exampler_answer = exampler['pathology']
                        exampler_question = f"Patient Information:\n-Age: {exampler_age}\n-Sex: {exampler_sex}\n-Initial Evidence: {exampler_initial_evidence}\n\nClinical Evidence:\n{exampler_clinical_evidence}\n\nBased on the above information, please choose the most likely diagnosis:\n{exampler_options}"
                    
                        choices = []
                        exampler_options = parse_options(exampler_options)
                        for e_option in exampler_options:
                            choices.append(e_option)
                        
                        random.shuffle(choices)
                        exampler_question += " " + ' '.join(choices)
                        exampler_answer = "Answer: {}\n\n".format(exampler_answer)

                        # chain-of-thought (cot)
                        exampler_reason = medical_agent.chat("Below is an example of medical knowledge question and answer. After reviewing the below medical question and answering, can you provide 1-2 sentences of reason that support the answer as you didn't know the answer ahead?\n\nQuestion: {}\n\nAnswer: {}".format(exampler_question, exampler_answer))

                        tmp_exampler['question'] = exampler_question
                        tmp_exampler['reason'] = exampler_reason
                        tmp_exampler['answer'] = exampler_answer
                        new_examplers.append(tmp_exampler)

                elif 'mmlu' in dataset:
                    random.shuffle(examplers)
                    new_examplers = []
                    for ie,exampler in enumerate(examplers):
                        if ie > 2:
                            break
                        tmp_exampler = {}
                        exampler_question = exampler['question']
                        choices = ["(A) {}".format(exampler['opa']), "(B) {}".format(exampler['opb']), "(C) {}".format(exampler['opc']), "(D) {}".format(exampler['opd'])] 
                        random.shuffle(choices)
                        exampler_question += ' ' + ' '.join(choices)
                        
                        exampler_answer = "Answer: ({})\n\n".format(exampler['answer'])

                        # chain-of-thought (cot)
                        exampler_reason = medical_agent.chat("Below is an example of medical knowledge question and answer. After reviewing the below medical question and answering, can you provide 1-2 sentences of reason that support the answer as you didn't know the answer ahead?\n\nQuestion: {}\n\nAnswer: {}".format(exampler_question, exampler_answer))

                        tmp_exampler['question'] = exampler_question
                        tmp_exampler['reason'] = exampler_reason
                        tmp_exampler['answer'] = exampler_answer
                        new_examplers.append(tmp_exampler)

                elif dataset == 'path-vqa':
                    random.shuffle(examplers)
                    new_examplers = []
                    for ie,exampler in enumerate(examplers):
                        if model == 'gemini-pro-vision':
                            medical_agent = Agent(instruction='You are a helpful medical agent.', role='medical expert', model_info=model)
                        if ie > 2:
                            break
                        tmp_exampler = {}
                        exampler_question = exampler['question']
                        exampler_img_path = 'data/path-vqa/images/test/' + exampler['img'] + ".jpg"
                        choices = ["(A) yes", "(B) no"] 
                        random.shuffle(choices)
                        exampler_question += ' ' + ' '.join(choices)
                        
                        if exampler['answer'] == 'yes':
                            exampler_answer = "Answer: (A) yes"
                        else:
                            exampler_answer = "Answer: (B) no"

                        # chain-of-thought (cot)
                        exampler_reason = medical_agent.chat("Below is an example of medical knowledge question and answer. After reviewing the below medical question and answering, can you provide 1-2 sentences of reason that support the answer as you didn't know the answer ahead?\n\nQuestion: {}\n\nAnswer: {}".format(exampler_question, exampler_answer), img_path=exampler_img_path)

                        tmp_exampler['question'] = exampler_question
                        tmp_exampler['reason'] = exampler_reason
                        tmp_exampler['answer'] = exampler_answer
                        tmp_exampler['img_path'] = exampler_img_path
                        new_examplers.append(tmp_exampler)

                elif 'mmmu' in dataset:
                    random.shuffle(examplers)
                    new_examplers = []
                    for ie,exampler in enumerate(examplers):
                        if model == 'gemini-pro-vision':
                            medical_agent = Agent(instruction='You are a helpful medical agent.', role='medical expert', model_info=model)
                        if ie > 2:
                            break
                        tmp_exampler = {}
                        exampler_question = exampler['question']
                        if 'val' in exampler['image']:
                            exampler_img_path = exampler['image'].replace('images', 'val_images')
                        else:
                            exampler_img_path = exampler['image']
                        
                        choices = []
                        option_list = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)', '(I)', '(J)', '(K)']
                        for oi,option in enumerate(exampler['options']):
                            choices.append("{} {}".format(option_list[oi], option))
                            
                        random.shuffle(choices)
                        exampler_question += ' ' + ' '.join(choices)
                        exampler_answer = "Answer: {}".format(exampler['answer'])

                        # chain-of-thought (cot)
                        if model == 'gemini-pro-vision':
                            cot_agent = Agent(instruction='You are a medical expert who conducts initial assessment and your job is to decide the difficulty/complexity of the medical query.', role='medical expert', model_info=model)
                            exampler_reason = cot_agent.chat("You are a medical expert who conducts initial assessment and your job is to decide the difficulty/complexity of the medical query. Below is an example of medical knowledge question and answer. After reviewing the below medical question and answering, can you provide 1-2 sentences of reason that support the answer as you didn't know the answer ahead?\n\nQuestion: {}\n\nAnswer: {}".format(exampler_question, exampler_answer), img_path=exampler_img_path)
                        
                        else:
                            exampler_reason = medical_agent.chat("Below is an example of medical knowledge question and answer. After reviewing the below medical question and answering, can you provide 1-2 sentences of reason that support the answer as you didn't know the answer ahead?\n\nQuestion: {}\n\nAnswer: {}".format(exampler_question, exampler_answer), img_path=exampler_img_path)

                        tmp_exampler['question'] = exampler_question
                        tmp_exampler['reason'] = exampler_reason
                        tmp_exampler['answer'] = exampler_answer
                        tmp_exampler['img_path'] = exampler_img_path
                        new_examplers.append(tmp_exampler)

                elif dataset == 'pmc-vqa':
                    random.shuffle(examplers)
                    new_examplers = []
                    for ie,exampler in enumerate(examplers):
                        if model == 'gemini-pro-vision':
                            medical_agent = Agent(instruction='You are a helpful medical agent.', role='medical expert', model_info=model)
                        if ie > 2:
                            break
                        tmp_exampler = {}
                        exampler_question = exampler['question']
                        exampler_img_path = 'data/pmc-vqa/images/' +exampler['img_path']
                        choices = ["(A) {}".format(exampler['opa']), "(B) {}".format(exampler['opb']), "(C) {}".format(exampler['opc']), "(D) {}".format(exampler['opd'])] 
                        random.shuffle(choices)
                        exampler_question += ' ' + ' '.join(choices)

                        exampler_answer = exampler['answer']
                        
                        # chain-of-thought (cot)
                        exampler_reason = medical_agent.chat("Below is an example of medical knowledge question and answer. After reviewing the below medical question and answering, can you provide 1-2 sentences of reason that support the answer as you didn't know the answer ahead?\n\nQuestion: {}\n\nAnswer: {}".format(exampler_question, exampler_answer), img_path=exampler_img_path)

                        tmp_exampler['question'] = exampler_question
                        tmp_exampler['reason'] = exampler_reason
                        tmp_exampler['answer'] = exampler_answer
                        tmp_exampler['img_path'] = exampler_img_path
                        new_examplers.append(tmp_exampler)

                elif dataset == 'medmcqa':
                    random.shuffle(examplers)
                    new_examplers = []
                    for ie,exampler in enumerate(examplers):
                        if ie > 2:
                            break
                        tmp_exampler = {}
                        exampler_question = exampler['question']
                        choices = ["(A) {}".format(exampler['opa']), "(B) {}".format(exampler['opb']), "(C) {}".format(exampler['opc']), "(D) {}".format(exampler['opd'])] 
                        random.shuffle(choices)
                        exampler_question += ' ' + ' '.join(choices)

                        exampler_answer = exampler['answer']
                        
                        # chain-of-thought (cot)
                        exampler_reason = medical_agent.chat("Below is an example of medical knowledge question and answer. After reviewing the below medical question and answering, can you provide 1-2 sentences of reason that support the answer as you didn't know the answer ahead?\n\nQuestion: {}\n\nAnswer: {}".format(exampler_question, exampler_answer))

                        tmp_exampler['question'] = exampler_question
                        tmp_exampler['reason'] = exampler_reason
                        tmp_exampler['answer'] = exampler_answer
                        new_examplers.append(tmp_exampler)
                
                if dataset in ['path-vqa', 'pmc-vqa'] or 'mmmu' in dataset:
                    single_agent = Agent(instruction='You are a helpful assistant that answers multiple choice questions about medical knowledge.', role='medical expert', examplers=new_examplers, model_info=model, img_path=img_path)
                    
                    if model == 'gemini-pro-vision':
                        final_decision = single_agent.temp_responses(f'''You are a helpful assistant that answers multiple choice questions about medical knowledge. The following are multiple choice questions (with answers) about medical knowledge. Let's think step by step.\n\n**Question:** {question}\nAnswer: ''', img_path=img_path)
                
                    else:
                        single_agent.chat('You are a helpful assistant that answers multiple choice questions about medical knowledge.')
                        final_decision = single_agent.temp_responses(f'''The following are multiple choice questions (with answers) about medical knowledge. Let's think step by step.\n\n**Question:** {question}\nAnswer: ''', img_path=img_path)

                else:
                    single_agent = Agent(instruction='You are a helpful assistant that answers multiple choice questions about medical knowledge.', role='medical expert', examplers=new_examplers, model_info=model)
                    single_agent.chat('You are a helpful assistant that answers multiple choice questions about medical knowledge.')
                    final_decision = single_agent.temp_responses(f'''The following are multiple choice questions (with answers) about medical knowledge. Let's think step by step.\n\n**Question:** {question}\nAnswer: ''', img_path=img_path)

                total_api_calls += 3

            print("y_hat:", final_decision)
            
            if dataset == 'medqa':
                print("y: ({}) {}".format(sample['answer_idx'], sample['answer']))
            
            elif dataset == 'pubmedqa':
                if 'yes' in sample['label'].lower():
                    print("y: (A) Yes")
                if 'no' in sample['label'].lower():
                    print("y: (B) No")
                if 'maybe' in sample['label'].lower():
                    print("y: (C) Maybe")
            
            elif dataset == 'ddxplus':
                print("y: {}".format(sample['pathology']))

            elif 'mmlu' in dataset:
                print("y: ({})".format(sample['answer']))

            elif 'mmmu' in dataset:
                print("y:", sample['answer'])
            
            elif dataset == 'path-vqa':
                print("y:", sample['answer'])

            print()

        elif difficulty == 'intermediate':
            cprint("[INFO] Step 1. Expert Recruitment", 'yellow', attrs=['blink'])
            recruit_prompt = f"""You are an experienced medical expert who recruits a group of experts with diverse identity and ask them to discuss and solve the given medical query."""
            
            if dataset in ['path-vqa', 'pmc-vqa'] or 'mmmu' in dataset:
                img_prompt = "You are an experienced medical expert. Given the image, your job is to point out the important features in the image."
                img_analyzer = Agent(instruction=img_prompt, role='medical image analyzer', model_info='gpt-3.5')
                # img_analyzer.chat(img_prompt, img_path=img_path)
                img_description = img_analyzer.chat("You are an experienced medical expert. Given the image, your job is to point out the important features in the image.. Please provide what you see important in the image as an experienced medical expert. Your answer should be in one sentence.", img_path=img_path)

            tmp_agent = Agent(instruction=recruit_prompt, role='recruiter', model_info='gpt-3.5')
            tmp_agent.chat(recruit_prompt)
            
            if dataset in ['path-vqa', 'pmc-vqa'] or 'mmmu' in dataset:
                recruited = tmp_agent.chat("Image: {}\n\nQuestion: {}\nYou can recruit {} experts with different medical expertise. Considering the medical question and the options for the answer, what kind of experts will you recruit to better make an accurate answer?\nAlso, you need to specify the communication structure between experts (e.g., Pulmonologist == Neonatologist == Medical Geneticist == Pediatrician > Cardiologist), or indicate if they are independent.\n\nFor example, if you want to recruit five experts, you answer can be like:\n1. Pediatrician - Specializes in the medical care of infants, children, and adolescents. - Hierarchy: Independent\n2. Cardiologist - Focuses on the diagnosis and treatment of heart and blood vessel-related conditions. - Hierarchy: Pediatrician > Cardiologist\n3. Pulmonologist - Specializes in the diagnosis and treatment of respiratory system disorders. - Hierarchy: Independent\n4. Neonatologist - Focuses on the care of newborn infants, especially those who are born prematurely or have medical issues at birth. - Hierarchy: Independent\n5. Medical Geneticist - Specializes in the study of genes and heredity. - Hierarchy: Independent\n\nPlease answer in above format, and do not include your reason.".format(img_description, question, num_agents))
            else:
                recruited = tmp_agent.chat("Question: {}\nYou can recruit {} experts in different medical expertise. Considering the medical question and the options for the answer, what kind of experts will you recruit to better make an accurate answer?\nAlso, you need to specify the communication structure between experts (e.g., Pulmonologist == Neonatologist == Medical Geneticist == Pediatrician > Cardiologist), or indicate if they are independent.\n\nFor example, if you want to recruit five experts, you answer can be like:\n1. Pediatrician - Specializes in the medical care of infants, children, and adolescents. - Hierarchy: Independent\n2. Cardiologist - Focuses on the diagnosis and treatment of heart and blood vessel-related conditions. - Hierarchy: Pediatrician > Cardiologist\n3. Pulmonologist - Specializes in the diagnosis and treatment of respiratory system disorders. - Hierarchy: Independent\n4. Neonatologist - Focuses on the care of newborn infants, especially those who are born prematurely or have medical issues at birth. - Hierarchy: Independent\n5. Medical Geneticist - Specializes in the study of genes and heredity. - Hierarchy: Independent\n\nPlease answer in above format, and do not include your reason.".format(question, num_agents))
            total_api_calls += 1 

            agents_info = [agent_info.split(" - Hierarchy: ") for agent_info in recruited.split('\n') if agent_info]
            agents_data = [(info[0], info[1]) if len(info) > 1 else (info[0], None) for info in agents_info]

            hierarchy_agents = parse_hierarchy(agents_data, agent_emoji)

            agent_list = ""
            for i,agent in enumerate(agents_data):
                agent_role = agent[0].split('-')[0].split('.')[1].strip().lower()
                description = agent[0].split('-')[1].strip().lower()
                agent_list += "Agent {}: {} - {}\n".format(i+1, agent_role, description)

            agent_dict = {}
            medical_agents = []
            for agent in agents_data:
                try:
                    agent_role = agent[0].split('-')[0].split('.')[1].strip().lower()
                    description = agent[0].split('-')[1].strip().lower()
                except:
                    continue
                
                inst_prompt = f"""You are a {agent_role} who {description}. Your job is to collaborate with other medical experts in a team."""
                if model == 'gemini-pro-vision':
                    _agent = Agent(instruction=inst_prompt, role=agent_role, model_info='gemini-pro')
                else:
                    _agent = Agent(instruction=inst_prompt, role=agent_role, model_info=model)
                
                _agent.chat(inst_prompt)
                agent_dict[agent_role] = _agent
                medical_agents.append(_agent)

            for idx, agent in enumerate(agents_data):
                try:
                    print("Agent {} ({} {}): {}".format(idx+1, agent_emoji[idx], agent[0].split("-")[0].strip(), agent[0].split("-")[1].strip()))
                except:
                    print("Agent {} ({}): {}".format(idx+1, agent_emoji[idx], agent[0]))

            
            fewshot_examplers = ""
            medical_agent = Agent(instruction='You are a helpful medical agent.', role='medical expert', model_info=model)
            if dataset == 'medqa':
                random.shuffle(examplers)
                for ie,exampler in enumerate(examplers):
                    if ie > 2:
                        break
                    exampler_question = "[Example {}]\n".format(ie+1) + exampler['question']
                    options = []
                    for k,v in exampler['options'].items():
                        options.append("({}) {}".format(k,v))
                    random.shuffle(options)
                    options = " ".join(options)
                    exampler_question += " " + options
                    # chain-of-thought (cot)
                    exampler_answer = "Answer: ({}) {}".format(exampler['answer_idx'], exampler['answer'])
                    exampler_reason = tmp_agent.chat("Below is an example of medical knowledge question and answer. After reviewing the below medical question and answering, can you provide 1-2 sentences of reason that support the answer as you didn't know the answer ahead?\n\nQuestion: {}\n\nAnswer: {}".format(exampler_question, exampler_answer))
                    
                    exampler_question += "\n{}\n{}\n\n".format(exampler_answer, exampler_reason)
                    fewshot_examplers += exampler_question

            elif dataset == 'pubmedqa':
                random.shuffle(examplers)
                for ie,exampler in enumerate(examplers):
                    if ie > 2:
                        break
                    
                    exampler_question = "[Example {}]\n".format(ie+1) + exampler['question']
                    options = ['(A) Yes', '(B) No', '(C) Maybe']
                    random.shuffle(options)
                    options = " ".join(options)
                    exampler_question += " " + options

                    if 'yes' in exampler['label'].lower():
                        exampler_answer = "Answer: (A) Yes"
                    if 'no' in exampler['label'].lower():
                        exampler_answer = "Answer: (B) No"
                    if 'maybe' in exampler['label'].lower():
                        exampler_answer = "Answer: (C) Maybe"

                    # chain-of-thought (cot)
                    exampler_reason = tmp_agent.chat("Below is an example of medical knowledge question and answer. After reviewing the below medical question and answering, can you provide 1-2 sentences of reason that support the answer as you didn't know the answer ahead?\n\nQuestion: {}\n\nAnswer: {}".format(exampler_question, exampler_answer))
                    
                    exampler_question += "\n{}\n{}\n\n".format(exampler_answer, exampler_reason)
                    fewshot_examplers += exampler_question
            
            elif dataset == 'ddxplus':
                random.shuffle(examplers)
                new_examplers = []
                for ie,exampler in enumerate(examplers):
                    if ie > 2:
                        break

                    exampler_question = ""
                    exampler_initial_evidence = exampler['initial_evidence']
                    exampler_clinical_evidence = exampler['evidences']
                    exampler_age = exampler['age']
                    exampler_sex = exampler['sex']
                    exampler_options = exampler['options']
                    exampler_answer = exampler['pathology']
                    exampler_question = f"Patient Information:\n-Age: {exampler_age}\n-Sex: {exampler_sex}\n-Initial Evidence: {exampler_initial_evidence}\n\nClinical Evidence:\n{exampler_clinical_evidence}\n\nBased on the above information, please choose the most likely diagnosis:\n{exampler_options}"
                
                    choices = []
                    exampler_options = parse_options(exampler_options)
                    for e_option in exampler_options:
                        choices.append(e_option)
                    
                    random.shuffle(choices)
                    exampler_question += " " + ' '.join(choices)
                    exampler_answer = "Answer: {}\n\n".format(exampler_answer)

                    # chain-of-thought (cot)
                    exampler_reason = medical_agent.chat("Below is an example of medical knowledge question and answer. After reviewing the below medical question and answering, can you provide 1-2 sentences of reason that support the answer as you didn't know the answer ahead?\n\nQuestion: {}\n\nAnswer: {}".format(exampler_question, exampler_answer))

                    exampler_question += "\n{}\n{}\n\n".format(exampler_answer, exampler_reason)
                    fewshot_examplers += exampler_question

            elif 'mmlu' in dataset:
                random.shuffle(examplers)
                new_examplers = []
                for ie,exampler in enumerate(examplers):
                    if ie > 2:
                        break
                    tmp_exampler = {}
                    exampler_question = exampler['question']
                    choices = ["(A) {}".format(exampler['opa']), "(B) {}".format(exampler['opb']), "(C) {}".format(exampler['opc']), "(D) {}".format(exampler['opd'])] 
                    random.shuffle(choices)
                    exampler_question += ' ' + ' '.join(choices)
                    
                    exampler_answer = "Answer: ({})\n\n".format(exampler['answer'])

                    # chain-of-thought (cot)
                    exampler_reason = medical_agent.chat("Below is an example of medical knowledge question and answer. After reviewing the below medical question and answering, can you provide 1-2 sentences of reason that support the answer as you didn't know the answer ahead?\n\nQuestion: {}\n\nAnswer: {}".format(exampler_question, exampler_answer))

                    tmp_exampler['question'] = exampler_question
                    tmp_exampler['reason'] = exampler_reason
                    tmp_exampler['answer'] = exampler_answer

                    exampler_question += "\n{}\n{}\n\n".format(exampler_answer, exampler_reason)
                    fewshot_examplers += exampler_question
            
            elif 'mmmu' in dataset:
                random.shuffle(examplers)
                new_examplers = []
                for ie,exampler in enumerate(examplers):
                    if ie > 2:
                        break
                    tmp_exampler = {}
                    exampler_question = exampler['question']
                    exampler_img_path = exampler['image']
                    
                    if 'val' in exampler['image']:
                        exampler_img_path = exampler_img_path.replace('images', 'val_images')
                    
                    choices = []
                    option_list = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)', '(I)', '(J)', '(K)']
                    for oi,option in enumerate(exampler['options']):
                        choices.append("{} {}".format(option_list[oi], option))
                        
                    random.shuffle(choices)
                    exampler_question += ' ' + ' '.join(choices)
                    exampler_answer = "Answer: ({})\n\n".format(exampler['answer'])

                    # chain-of-thought (cot)
                    if model == 'gemini-pro-vision':
                        cot_agent = Agent(instruction='You are a medical expert who conducts initial assessment and your job is to decide the difficulty/complexity of the medical query.', role='medical expert', model_info=model)
                        exampler_reason = cot_agent.chat("You are a experienced medical expert. Below is an example of medical knowledge question and answer. After reviewing the below medical question and answering, can you provide 1-2 sentences of reason that support the answer as you didn't know the answer ahead?\n\nQuestion: {}\n\nAnswer: {}".format(exampler_question, exampler_answer), img_path=exampler_img_path)
                    else:
                        exampler_reason = tmp_agent.chat("Below is an example of medical knowledge question and answer. After reviewing the below medical question and answering, can you provide 1-2 sentences of reason that support the answer as you didn't know the answer ahead?\n\nQuestion: {}\n\nAnswer: {}".format(exampler_question, exampler_answer), img_path=exampler_img_path)

                    tmp_exampler['question'] = exampler_question
                    tmp_exampler['reason'] = exampler_reason
                    tmp_exampler['answer'] = exampler_answer
                    tmp_exampler['img_path'] = exampler_img_path

                    exampler_question += "\n{}\n{}\n\n".format(exampler_answer, exampler_reason)
                    fewshot_examplers += exampler_question            

            print()
            cprint("[INFO] Step 2. Collaborative Decision Making", 'yellow', attrs=['blink'])
            cprint("[INFO] Step 2.1. Hierarchy Selection", 'yellow', attrs=['blink'])
            print_tree(hierarchy_agents[0], horizontal=False)
            print()

            # STEP 0. Initial Assessment
            # Number of interaction rounds
            num_rounds = 3
            num_turns = 3
            num_agents = len(medical_agents)

            # interaction_log init
            interaction_log = {} 
            for round_num in range(1, num_rounds + 1):
                interaction_log[f'Round {round_num}'] = {}
                for turn_num in range(1, num_turns + 1):
                    interaction_log[f'Round {round_num}'][f'Turn {turn_num}'] = {}
                    for source_agent_num in range(1, num_agents + 1):
                        source_agent = f'Agent {source_agent_num}'
                        interaction_log[f'Round {round_num}'][f'Turn {turn_num}'][f'{source_agent}'] = {}
                        for target_agent_num in range(1, num_agents + 1):
                            target_agent = f'Agent {target_agent_num}'
                            interaction_log[f'Round {round_num}'][f'Turn {turn_num}'][f'{source_agent}'][f'{target_agent}'] = None

            ##### Step 2.2. Participatory Debate #####
            anonymized_opinions = {}
            terminating_round = None
            cprint("[INFO] Step 2.2. Participatory Debate", 'yellow', attrs=['blink'])

            round_opinions = {n: {} for n in range(1, num_rounds+1)}
            round_answers = {n: None for n in range(1, num_rounds+1)}
            initial_report = ""
            for k,v in agent_dict.items():
                opinion = v.chat(f'''Given the examplers, please return your answer and a short reasons to the medical query among the option provided. Your responses will be used for research purposes only, so please have a definite answer.\n\n{fewshot_examplers}\n\nQuestion: {question}\n\nYour answer should be like below format.\n\nAnswer: ''', img_path=img_path)
                total_api_calls += 1 
                initial_report += "({}): {}\n".format(k.lower(), opinion)
                round_opinions[1][k.lower()] = opinion

            print("[Initial Report]")
            print(initial_report)
            print()
            
            final_answer = None
            # round-loop
            for n in range(1, num_rounds+1):
                print("== Round {} ==".format(n))
                round_name = f"Round {n}"
                # previous round's assessment
                agent_rs = Agent(instruction="You are a medical assistant who excels at summarizing and synthesizing based on multiple experts from various domain experts.", role="medical assistant", model_info=model)
                if model != 'gemini-pro-vision':
                    agent_rs.chat("You are a medical assistant who excels at summarizing and synthesizing based on multiple experts from various domain experts.")
                assessment = ""
                for k,v in round_opinions[n].items():
                    assessment += "({}): {}\n".format(k.lower(), v)

                if model == 'gemini-pro-vision':
                    report = agent_rs.chat(f'''You are a medical assistant who excels at summarizing and synthesizing based on multiple experts from various domain experts. Here are some reports from different medical domain experts.\n\n{assessment}\n\nYou need to complete the following steps\n1. Take careful and comprehensive consideration of the following reports.\n2. Extract key knowledge from the following reports.\n3. Derive the comprehensive and summarized analysis based on the knowledge\n4. Your ultimate goal is to derive a refined and synthesized report based on the following reports.\n\nYou should output in exactly the same format as: Key Knowledge:; Total Analysis:''', img_path=img_path)
                else:
                    report = agent_rs.chat(f'''Here are some reports from different medical domain experts.\n\n{assessment}\n\nYou need to complete the following steps\n1. Take careful and comprehensive consideration of the following reports.\n2. Extract key knowledge from the following reports.\n3. Derive the comprehensive and summarized analysis based on the knowledge\n4. Your ultimate goal is to derive a refined and synthesized report based on the following reports.\n\nYou should output in exactly the same format as: Key Knowledge:; Total Analysis:''')
                
                turn_opinions = {}
                # turn-loop
                for turn_num in range(num_turns):
                    num_yes = 0
                    turn_name = f"Turn {turn_num + 1}"
                    print(f"|_{turn_name}")

                    consensus = {}
                    for idx,v in enumerate(medical_agents):
                        all_comments = ""
                        for _k,_v in interaction_log[round_name][turn_name].items():
                            all_comments += "{} -> {}: {}\n".format(_k, 'Agent {}'.format(idx+1), _v['Agent {}'.format(idx+1)])

                        if rag == True:                        
                            medrag = MedRAG(llm_name="{}".format(rag_model_info), rag=True, retriever_name="MedCPT", corpus_name="Textbooks", img_path=img_path)                        
                            rag_answer, snippets, scores = medrag.answer(question=question, options=rag_options, k=32, role=v.role)
                        
                        if turn_num == 0:
                            if n == 1:
                                participate = v.chat("Given the opinions from other medical experts in your team, please indicate whether you want to talk to any expert (yes/no). If no, just provide your opinion.\n\nOpinions:\n{}".format(assessment))
                            else:
                                participate = v.chat("Given the conversations between medical experts in the previous round, please indicate whether you want to continue to talk with any expert (yes/no). If no, just provide your opinion.\n\nInteraction History:\n{}".format(interaction_log[f"Round {n-1}"].values()))
                            
                        else:
                            participate = v.chat("Given the interaction between medical experts in the previous turn, please indicate whether you want to talk to any expert (yes/no). If no, just provide your opinion.\n\nInteraction:\n{}".format(interaction_log[f"Round {n}"][f"Turn {turn_num}"]))
                        
                        total_api_calls +=1 

                        if 'yes' in participate.lower().strip():                
                            chosen_expert = v.chat(f"Enter the number of the expert you want to talk to:\n{agent_list}\nFor example, if you want to talk with Agent 1. Pediatrician, return just 1. If you want to talk with more than one expert, please return 1,2 and don't return the reasons.")
                            
                            chosen_experts = []
                            if "1." in chosen_expert or "1" in chosen_expert:
                                chosen_experts.append(1)
                            if "2." in chosen_expert or "2" in chosen_expert:
                                chosen_experts.append(2)
                            if "3." in chosen_expert or "3" in chosen_expert:
                                chosen_experts.append(3)
                            if "4." in chosen_expert or "4" in chosen_expert:
                                chosen_experts.append(4)
                            if "5." in chosen_expert or "5" in chosen_expert:
                                chosen_experts.append(5)

                            for ce in chosen_experts:
                                specific_question = v.chat("Please remind your medical expertise and then leave your opinion to an expert you chose (Agent {}. {}). You should deliver your opinion once you are confident enough and in a way to convince other expert with a short reason.".format(ce, medical_agents[ce-1].role))
                                
                                print(" Agent {} ({} {}) -> Agent {} ({} {}) : {}".format(idx+1, agent_emoji[idx], medical_agents[idx].role, ce, agent_emoji[ce-1], medical_agents[ce-1].role, specific_question))
                                interaction_log[round_name][turn_name]['Agent {}'.format(idx+1)]['Agent {}'.format(ce)] = specific_question
                        
                            num_yes += 1

                        else:
                            print(" Agent {} ({} {}): {}".format(idx+1, agent_emoji[idx], v.role, participate))

                    if num_yes == 0:
                        terminating_round = round_num
                        break
                
                if num_yes == 0:
                    terminating_round = round_num
                    break

                print()

                tmp_final_answer = {}
                for i, agent in enumerate(medical_agents):
                    response = agent.chat(f"Now that you've interacted with other medical experts, remind your expertise and the comments from other experts and make your final answer to the given question:\n{question}\nAnswer: ")
                    tmp_final_answer[agent.role] = response

                if review == True:
                    review_agent = Agent(instruction='You are an experienced medical expert who reviews medical assessment and gives feedback.', role='medical expert', model_info=model)
                    feedbacks = review_agent.chat(f"You are an experienced medical expert who reviews medical assessment and gives feedback. Given the opinions from each agent for the medical query, please return your comments to each agent. You should not be biased and be objective to correct or encourage each agent.\n\n[Question]\n{question}\n\n[Opinions]\n{tmp_final_answer}\n\nThe example answer format should be as follows and do not include any reasons:\n1. Neurilogist: (some feedback here ...)\n2. Physiatrist: (some feedback here)\n3. Surgeon: (some feedback here ...).")
   
                    for i, agent in enumerate(medical_agents):
                        _ = agent.chat(f"Here's some comments from the moderator. Before proceeding to the next round, please keep this in mind.\n[Comments]\n{feedbacks}.")
  
                if rag == True:
                    medrag = MedRAG(llm_name="{}".format(rag_model_info), rag=True, retriever_name="MedCPT", corpus_name="Textbooks", img_path=img_path)
                    rag_answer, snippets, scores = medrag.answer(question=question, options=rag_options, k=32)
                    
                    for i, agent in enumerate(medical_agents):
                        _ = agent.chat(f"Here's the answer based on the information from the recent medical literatures. Before proceeding to the next round, please keep this in mind.\n\n{rag_answer}.")

                round_answers[round_name] = tmp_final_answer
                final_answer = tmp_final_answer
   
            tmp = ['']
            for i in range(1, len(medical_agents)+1):
                tmp.append("Agent {} ({})".format(i, agent_emoji[i-1]))
            myTable = PrettyTable(tmp) 

            for i in range(1, len(medical_agents)+1):
                tmp = ["Agent {} ({})".format(i, agent_emoji[i-1])]
                for j in range(1, len(medical_agents)+1):
                    if i == j:
                        tmp.append('')
                    else:
                        i2j = False
                        j2i = False
                        # num_round
                        for k in range(1, len(list(interaction_log.keys()))+1):
                            # num_turn
                            for l in range(1, len(list(interaction_log['Round 1'].items()))+1):
                                if interaction_log['Round {}'.format(k)]['Turn {}'.format(l)]['Agent {}'.format(i)]['Agent {}'.format(j)] != None:
                                    i2j = True
                                elif interaction_log['Round {}'.format(k)]['Turn {}'.format(l)]['Agent {}'.format(j)]['Agent {}'.format(i)] != None:
                                    j2i = True
                        
                        if i2j == False and j2i == False:
                            tmp.append(' ')
                        elif i2j == True and j2i == False:
                            tmp.append('\u270B ({}->{})'.format(i,j))
                        elif j2i == True and i2j == False:
                            tmp.append('\u270B ({}<-{})'.format(i,j))
                        elif i2j == True and j2i == True:
                            tmp.append('\u270B ({}<->{})'.format(i,j))

                myTable.add_row(tmp) 
                if i != len(medical_agents):
                    myTable.add_row(['' for _ in range(len(medical_agents)+1)]) 
            
            print(myTable)
            
            ##################################
            ##### Step 3. Final Decision #####
            ##################################
            cprint("\n[INFO] Step 3. Final Decision", 'yellow', attrs=['blink'])
            if model == 'gemini-pro-vision':
                model = 'gemini-pro'
            
            moderator1 = Agent("You are a final medical decision maker who reviews all opinions from different medical experts and make final decision.", "Moderator", model_info=model)
            moderator1.chat('You are a final medical decision maker who reviews all opinions from different medical experts and make final decision.')
            
            moderator3 = Agent("You are a final medical decision maker who reviews all opinions from different medical experts and make final decision.", "Moderator", model_info=model)
            moderator3.chat('You are a final medical decision maker who reviews all opinions from different medical experts and make final decision.')
            
            moderator4 = Agent("You are a final medical decision maker who reviews all opinions from different medical experts and make final decision.", "Moderator", model_info=model)
            moderator4.chat('You are a final medical decision maker who reviews all opinions from different medical experts and make final decision.')
   
            new_interaction_log = ""
            for k,v in interaction_log.items():
                if k == f'Round {terminating_round}':
                    break
                new_interaction_log += "[{}]\n".format(k)
                
                for k1,v1 in v.items():
                    new_interaction_log += "[{}]\n".format(k1)

                    for k2,v2 in v1.items():
                        for k3,v3 in v2.items():
                            if k2 == k3:
                                continue

                            if v3 != None:
                                new_interaction_log += "{} -> {}: {}\n".format(k2, k3, v3)
                    
                    new_interaction_log += "\n"
                new_interaction_log += "\n\n"
            
            final_decision = {}
            final_decision1 = moderator1.temp_responses(f"Given the inital assessment and the interaction between agents over the rounds, please review each agent's opinion and make the final answer to the question with short reasoning. Your answer should be like below format:\nAnswer: A) 6th pharyngeal arch\n[Initial Assessment]\n{initial_report}\n\n[Interaction Log]\n{new_interaction_log}\n\nQuestion: {question}", img_path=img_path)
            final_decision3 = moderator3.temp_responses(f"Given the inital assessment and the interaction between agents over the rounds, please review each agent's opinion and make the final answer to the question by majority voting with short reasoning. Your answer should be like below format:\nAnswer: C) 2th pharyngeal arch\n[Initial Assessment]\n{initial_report}\n\n[Interaction Log]\n{new_interaction_log}\n\nQuestion: {question}", img_path=img_path)
            final_decision4 = moderator4.temp_responses(f"Given the inital assessment and the interaction between agents over the rounds, please review each agent's opinion and make the final answer to the question by first ranking the agent's opinion and taking weighted voting with short reasoning. Your answer should be like below format:\nAnswer: A) 6th pharyngeal arch\n[Initial Assessment]\n{initial_report}\n\n[Interaction Log]\n{new_interaction_log}\n\nQuestion: {question}", img_path=img_path)
            
            final_decision = {'overall': final_decision1, 'majority': final_decision3, 'weighted': final_decision4}

            print("{} moderator's final decision:".format("\U0001F468\u200D\u2696\uFE0F"), final_decision3)
            print()
            
            if dataset == 'medqa':
                print('y: ({}) {}'.format(sample['answer_idx'], sample['answer']))
                print()

            elif dataset == 'pubmedqa':
                if 'yes' in sample['label'].lower():
                    print("y: (A) Yes")
                if 'no' in sample['label'].lower():
                    print("y: (B) No")
                if 'maybe' in sample['label'].lower():
                    print("y: (C) Maybe")
            
            elif dataset == 'ddxplus':
                print("y: {}".format(sample['pathology']))

            elif 'mmlu' in dataset:
                print("y: ({})".format(sample['answer']))

            elif dataset == 'path-vqa':
                print("y:", sample['answer'])
            
            elif dataset == 'pmc-vqa':
                if sample['answer'] in sample['opa']:
                    print("y: (A)", sample['answer'])
                elif sample['answer'] in sample['opb']:
                    print("y: (B)", sample['answer'])
                elif sample['answer'] in sample['opc']:
                    print("y: (C)", sample['answer'])
                elif sample['answer'] in sample['opd']:
                    print("y: (D)", sample['answer'])
                else:
                    print("y:", sample['answer'])
            
            else:
                print()
                    
        elif difficulty == 'advanced':
            # STEP 1. Multidisciplinary Teams (MDTs) recruitment
            print("[STEP 1] Recruitment")
            group_instances = []

            recruit_prompt = f"""You are an experienced medical expert. Given the complex medical query, you need to organize Multidisciplinary Teams (MDTs) and the members in MDT to make accurate and robust answer."""
            if dataset in ['path-vqa', 'pmc-vqa'] or 'mmmu' in dataset:
                img_prompt = "You are an experienced medical expert. Given the image, your job is to point out the important features in the image."
                img_analyzer = Agent(instruction=img_prompt, role='medical image analyzer', model_info='gemini-pro-vision')
                # img_analyzer.chat(img_prompt)
                img_description = img_analyzer.chat("You are an experienced medical expert. Given the complex medical query, you need to organize Multidisciplinary Teams (MDTs) and the members in MDT to make accurate and robust answer. Please provide what you see important in the image as an experienced medical expert. Your answer should be in one sentence.", img_path=img_path)

            tmp_agent = Agent(instruction=recruit_prompt, role='recruiter', model_info='gpt-4v')
            if dataset in ['path-vqa', 'pmc-vqa'] or 'mmmu' in dataset:
                recruited = tmp_agent.chat("You are an experienced medical expert. Given the complex medical query, you need to organize Multidisciplinary Teams (MDTs) and the members in MDT to make accurate and robust answer.\n\nImage: {}\n\nQuestion: {}\n\nYou can organize {} MDTs with different specialties or purposes and each MDT should have {} clinicians. Considering the medical question and the options, please return your recruitment plan to better make an accurate answer.\n\nFor example, the following can an example answer:\nGroup 1 - Initial Assessment Team (IAT)\nMember 1: Otolaryngologist (ENT Surgeon) (Lead) - Specializes in ear, nose, and throat surgery, including thyroidectomy. This member leads the group due to their critical role in the surgical intervention and managing any surgical complications, such as nerve damage.\nMember 2: General Surgeon - Provides additional surgical expertise and supports in the overall management of thyroid surgery complications.\nMember 3: Anesthesiologist - Focuses on perioperative care, pain management, and assessing any complications from anesthesia that may impact voice and airway function.\n\nGroup 2 - Diagnostic Evidence Team (DET)\nMember 1: Endocrinologist (Lead) - Oversees the long-term management of Graves' disease, including hormonal therapy and monitoring for any related complications post-surgery.\nMember 2: Speech-Language Pathologist - Specializes in voice and swallowing disorders, providing rehabilitation services to improve the patient's speech and voice quality following nerve damage.\nMember 3: Neurologist - Assesses and advises on nerve damage and potential recovery strategies, contributing neurological expertise to the patient's care.\n\nGroup 3 - Patient History Team (PHT)\nMember 1: Psychiatrist or Psychologist (Lead) - Addresses any psychological impacts of the chronic disease and its treatments, including issues related to voice changes, self-esteem, and coping strategies.\nMember 2: Physical Therapist - Offers exercises and strategies to maintain physical health and potentially support vocal function recovery indirectly through overall well-being.\nMember 3: Vocational Therapist - Assists the patient in adapting to changes in voice, especially if their profession relies heavily on vocal communication, helping them find strategies to maintain their occupational roles.\n\nGroup 4 - Final Review and Decision Team (FRDT)\nMember 1: Senior Consultant from each specialty (Lead) - Provides overarching expertise and guidance in decision\nMember 2: Clinical Decision Specialist - Coordinates the different recommendations from the various teams and formulates a comprehensive treatment plan.\nMember 3: Advanced Diagnostic Support - Utilizes advanced diagnostic tools and techniques to confirm the exact extent and cause of nerve damage, aiding in the final decision.\n\nAbove is just an example, thus, you should organize your own unique MDTs but you should include Initial Assessment Team (IAT) and Final Review and Decision Team (FRDT) in your recruitment plan. When you retrun your answer, please strictly refer to the above format.".format(img_description, question, num_teams, num_agents), img_path=img_path)
            else:
                recruited = tmp_agent.chat("Question: {}\n\nYou should organized {} MDTs with different specialties or purposes and each MDT should have {} clinicians. Considering the medical question and the options, please return your recruitment plan to better make an accurate answer.\n\nFor example, the following can an example answer:\nGroup 1 - Initial Assessment Team (IAT)\nMember 1: Otolaryngologist (ENT Surgeon) (Lead) - Specializes in ear, nose, and throat surgery, including thyroidectomy. This member leads the group due to their critical role in the surgical intervention and managing any surgical complications, such as nerve damage.\nMember 2: General Surgeon - Provides additional surgical expertise and supports in the overall management of thyroid surgery complications.\nMember 3: Anesthesiologist - Focuses on perioperative care, pain management, and assessing any complications from anesthesia that may impact voice and airway function.\n\nGroup 2 - Diagnostic Evidence Team (DET)\nMember 1: Endocrinologist (Lead) - Oversees the long-term management of Graves' disease, including hormonal therapy and monitoring for any related complications post-surgery.\nMember 2: Speech-Language Pathologist - Specializes in voice and swallowing disorders, providing rehabilitation services to improve the patient's speech and voice quality following nerve damage.\nMember 3: Neurologist - Assesses and advises on nerve damage and potential recovery strategies, contributing neurological expertise to the patient's care.\n\nGroup 3 - Patient History Team (PHT)\nMember 1: Psychiatrist or Psychologist (Lead) - Addresses any psychological impacts of the chronic disease and its treatments, including issues related to voice changes, self-esteem, and coping strategies.\nMember 2: Physical Therapist - Offers exercises and strategies to maintain physical health and potentially support vocal function recovery indirectly through overall well-being.\nMember 3: Vocational Therapist - Assists the patient in adapting to changes in voice, especially if their profession relies heavily on vocal communication, helping them find strategies to maintain their occupational roles.\n\nGroup 4 - Final Review and Decision Team (FRDT)\nMember 1: Senior Consultant from each specialty (Lead) - Provides overarching expertise and guidance in decision\nMember 2: Clinical Decision Specialist - Coordinates the different recommendations from the various teams and formulates a comprehensive treatment plan.\nMember 3: Advanced Diagnostic Support - Utilizes advanced diagnostic tools and techniques to confirm the exact extent and cause of nerve damage, aiding in the final decision.\n\nAbove is just an example, thus, you should organize your own unique MDTs but you should include Initial Assessment Team (IAT) and Final Review and Decision Team (FRDT) in your recruitment plan. When you retrun your answer, please strictly refer to the above format.".format(question, num_teams, num_agents))
        
            total_api_calls +=1 

            groups = [group.strip() for group in recruited.split("Group") if group.strip()]
            group_strings = ["Group " + group for group in groups]
            i1 = 0
            for gs in group_strings:
                res_gs = parse_group_info(gs)
                if res_gs['group_goal'] == '':
                    continue
                print("Group {} - {}".format(i1+1, res_gs['group_goal']))
                for i2,member in enumerate(res_gs['members']):
                    print(" Member {} ({}): {}".format(i2+1, member['role'], member['expertise_description']))
                print()

                group_instance = Group(res_gs['group_goal'], res_gs['members'], question)#, fewshot_examplers)
                group_instances.append(group_instance)

                i1 += 1

            # STEP 2. initial assessment from each group
            # STEP 2.1. IAP Process
            # - IAT reviews initial evidence and patient profile.
            # - IAT LLM Agent analyzes initial symptoms and suggests preliminary assessment.
            # - Output: Initial Assessment Report 
            initial_assessments = []
            for group_instance in group_instances:
                if 'initial' in group_instance.goal.lower() or 'iap' in group_instance.goal.lower():
                    init_assessment = group_instance.interact(comm_type='internal', img_path=img_path)
                    initial_assessments.append([group_instance.goal, init_assessment])

            initial_assessment_report = ""
            for idx,init_assess in enumerate(initial_assessments):
                initial_assessment_report += "Group {} - {}\n{}\n\n".format(idx+1, init_assess[0], init_assess[1])

            # STEP 2.2. other MDTs Process
            asessments = []
            for group_instance in group_instances:
                if 'initial' not in group_instance.goal.lower() and 'iap' not in group_instance.goal.lower():
                    if model == 'gemini-pro-vision':
                        assessment = group_instance.interact(comm_type='internal', img_path=img_path)
                    else:
                        assessment = group_instance.interact(comm_type='internal')
                    asessments.append([group_instance.goal, assessment])
            
            assessment_report = ""
            for idx,assess in enumerate(asessments):
                assessment_report += "Group {} - {}\n{}\n\n".format(idx+1, assess[0], assess[1])
            
            # STEP 2.3. FRDT Process
            # - FRDT compiles reports from IAT, DET, and PHT.
            # - FRDT reviews all LLM Agent outputs and conducts a comprehensive analysis.
            # - FRDT LLM Agent synthesizes information and suggests the most probable diagnosis.
            # - If consensus is not reached, FRDT discusses discrepancies and reviews additional data if necessary.
            # - Final Decision
            final_decisions = []
            for group_instance in group_instances:
                if 'review' in group_instance.goal.lower() or 'decision' in group_instance.goal.lower() or 'frdt' in group_instance.goal.lower():
                    decision = group_instance.interact(comm_type='internal', img_path=img_path)
                    final_decisions.append([group_instance.goal, decision])
            
            compiled_report = ""
            for idx,decision in enumerate(final_decisions):
                compiled_report += "Group {} - {}\n{}\n\n".format(idx+1, decision[0], decision[1])

            # STEP 3. Final Decision
            decision_prompt = f"""You are an experienced medical expert. Now, given the investigations from multidisciplinary teams (MDT), please review them very carefully and return your final decision to the medical query."""
            tmp_agent = Agent(instruction=decision_prompt, role='decision maker', model_info=model)

            if rag == True:
                medrag = MedRAG(llm_name="{}".format(rag_model_info), rag=True, retriever_name="MedCPT", corpus_name="Textbooks", img_path=img_path)
                rag_answer, snippets, scores = medrag.answer(question=question, options=rag_options, k=32)
                final_decision = tmp_agent.temp_responses(f"""You are an experienced medical expert. Now, given the investigations from multidisciplinary teams (MDT), please review them very carefully and return your final decision to the medical query. Retrieved Information:\n{rag_answer}\n\nInvestigation:\n{initial_assessment_report}\n\nQuestion: {question}""", num_responses=5, img_path=img_path)
                        
            else:
                final_decision = tmp_agent.temp_responses(f"""You are an experienced medical expert. Now, given the investigations from multidisciplinary teams (MDT), please review them very carefully and return your final decision to the medical query. Investigation:\n{initial_assessment_report}\n\nQuestion: {question}""", num_responses=5, img_path=img_path)
            print("y_hat:", final_decision)

            if dataset == 'medqa':
                print("y: ({}) {}".format(sample['answer_idx'], sample['answer']))
            
            elif dataset == 'pubmedqa':
                if 'yes' in sample['label'].lower():
                    print("y: (A) Yes")
                if 'no' in sample['label'].lower():
                    print("y: (B) No")
                if 'maybe' in sample['label'].lower():
                    print("y: (C) Maybe")

            elif dataset == 'ddxplus':
                print("y: {}".format(sample['pathology']))

            elif 'mmlu' in dataset:
                print("y: ({})".format(sample['answer']))

            elif 'mmmu' in dataset:
                print("y:", sample['answer'])

            elif dataset == 'path-vqa':
                print("y:", sample['answer'])
            
            elif dataset == 'pmc-vqa':
                print("y:", sample['answer'])

            print()
            
        print("# api calls:", total_api_calls)
        print()


        if dataset == 'medqa':
            results.append({'question': question, 'label': sample['answer_idx'], 'answer': sample['answer'], 'options': sample['options'], 'response': final_decision, 'difficulty': difficulty, 'total_api_calls': total_api_calls})
        elif dataset == 'medmcqa':
            results.append({'question': question, 'response': final_decision, 'opa': sample['opa'], 'opb': sample['opb'], 'opc': sample['opc'], 'opd': sample['opd']})
        elif dataset == 'pubmedqa':
            results.append({'question': question, 'label': sample['label'], 'options': '(A) Yes (B) No (C) Maybe', 'response': final_decision, 'difficulty': difficulty, 'total_api_calls': total_api_calls})
        elif dataset == 'ddxplus':
            results.append({'question': question, 'response': final_decision, 'difficulty': difficulty, 'label': sample['pathology'], 'options': sample['options'], 'total_api_calls': total_api_calls})
        elif 'mmlu' in dataset:
            results.append({'question': question, 'response': final_decision, 'difficulty': difficulty, 'label': sample['answer'], 'options': "(A) {} (B) {} (C) {} (D) {}".format(sample['opa'], sample['opb'], sample['opc'], sample['opd']), 'total_api_calls': total_api_calls})
        elif 'mmmu' in dataset:
            results.append({'id': sample['id'], 'question': question, 'response': final_decision, 'difficulty': difficulty, 'label': sample['answer'], 'options': sample['options'], 'total_api_calls': total_api_calls})
        elif dataset == 'path-vqa':
            results.append({'no': no, 'question': question, 'difficulty': difficulty, 'response': final_decision,  'label': sample['answer'], 'img_path': sample['img'], 'options': "(A) yes (B) no", 'total_api_calls': total_api_calls})
        elif dataset == 'pmc-vqa':
            results.append({'no': no, 'question': question, 'difficulty': difficulty, 'response': final_decision,  'label': sample['answer'], 'img_path': sample['img_path'], 'opa': sample['opa'], 'opb': sample['opb'], 'opc': sample['opc'], 'opd': sample['opd'], 'total_api_calls': total_api_calls})
                
        no += 1

    except Exception as e:
        print(e)
        continue


if rag == True:
    model = 'rag_' + model

if review == True:
    model = 'review_' + model

with open('output/{}_{}_sd{}.json'.format(model, dataset, seed), 'w') as file:
    json.dump(results, file, indent=4)
