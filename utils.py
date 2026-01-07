import os
import json
import random
from prettytable import PrettyTable 
from termcolor import cprint
from pptree import Node
import google.generativeai as genai
from openai import OpenAI
from pptree import *
import re 

# --- Robust regex helpers (avoid brittle split/indexing on LLM outputs) ---
_EXPERT_HIER_LINE_RE = re.compile(r'^\s*(?P<expert>.+?)(?:\s*-\s*Hierarchy:\s*(?P<hierarchy>.+))?\s*$', re.IGNORECASE)
_EXPERT_ROLE_DESC_RE = re.compile(r'^\s*(?:\d+\.\s*)?(?P<role>.+?)(?:\s*-\s*(?P<desc>.+))?\s*$')

# -----------------------
# Robustness helpers
# -----------------------
DEFAULT_LLM_RETRIES = int(os.environ.get("MDAGENTS_LLM_RETRIES", "5"))

def _retry_call(name, fn, max_tries=None, retry_exceptions=(IndexError,), sleep_s=None):
    """
    Retry a callable as the last resort when the LLM output (or downstream parsing) is brittle.
    This is intentionally lightweight so we don't change any parsing logic; we just re-ask.
    """
    if max_tries is None:
        max_tries = DEFAULT_LLM_RETRIES
    last_err = None
    for attempt in range(1, max_tries + 1):
        try:
            return fn()
        except retry_exceptions as e:
            last_err = e
            print(f"[WARN] {name} failed (attempt {attempt}/{max_tries}): {type(e).__name__}: {e}. Retrying...")
            if sleep_s:
                try:
                    import time
                    time.sleep(sleep_s)
                except Exception:
                    pass
        except Exception as e:
            # Also retry on generic transient failures (API hiccups, etc.)
            last_err = e
            print(f"[WARN] {name} failed (attempt {attempt}/{max_tries}): {type(e).__name__}: {e}. Retrying...")
            if sleep_s:
                try:
                    import time
                    time.sleep(sleep_s)
                except Exception:
                    pass
    # If still failing, raise the last error so caller can persist progress.
    raise last_err

class Agent:
    # Class-level counter for total API calls across all agents
    total_api_calls = 0
    
    def __init__(self, instruction, role, examplers=None, model_info='gpt-4o-mini', img_path=None):
        self.instruction = instruction
        self.role = role
        self.model_info = model_info
        self.img_path = img_path
        self.api_calls = 0  # Instance-level counter

        if self.model_info == 'gemini-pro':
            self.model = genai.GenerativeModel('gemini-pro')
            self._chat = self.model.start_chat(history=[])
        elif self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini']:
            self.client = OpenAI(api_key=os.environ['openai_api_key'])
            self.messages = [
                {"role": "system", "content": instruction},
            ]
            if examplers is not None:
                for exampler in examplers:
                    self.messages.append({"role": "user", "content": exampler['question']})
                    self.messages.append({"role": "assistant", "content":  ("Let's think step by step. " + exampler['reason'] + " "  if 'reason' in exampler else '') + exampler['answer']})

        print(f"[DEBUG] Print out the messages for Agent {self.messages}")

    def chat(self, message, img_path=None, chat_mode=True):
        if self.model_info == 'gemini-pro':
            for _ in range(10):
                try:
                    response = self._chat.send_message(message, stream=True)
                    
                    # Track API call
                    self.api_calls += 1
                    Agent.total_api_calls += 1
                    
                    responses = ""
                    for chunk in response:
                        responses += chunk.text + "\n"
                    return responses
                except:
                    continue
            return "Error: Failed to get response from Gemini."

        elif self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini']:
            self.messages.append({"role": "user", "content": message})
            
            if self.model_info == 'gpt-3.5':
                model_name = "gpt-3.5-turbo"
            else:
                model_name = "gpt-4o-mini"

            response = self.client.chat.completions.create(
                model=model_name,
                messages=self.messages
            )
            
            # Track API call
            self.api_calls += 1
            Agent.total_api_calls += 1

            self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
            return response.choices[0].message.content

    def temp_responses(self, message, temperatures=[0.0], img_path=None):
        if self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini']:      
            self.messages.append({"role": "user", "content": message})
            
            temperatures = list(set(temperatures))
            
            responses = {}
            for temperature in temperatures:
                if self.model_info == 'gpt-3.5':
                    model_info = 'gpt-3.5-turbo'
                else:
                    model_info = 'gpt-4o-mini'
                response = self.client.chat.completions.create(
                    model=model_info,
                    messages=self.messages,
                    temperature=temperature,
                )
                
                # Track API call
                self.api_calls += 1
                Agent.total_api_calls += 1
                
                responses[temperature] = response.choices[0].message.content
                
            return responses
        
        elif self.model_info == 'gemini-pro':
            response = self._chat.send_message(message, stream=True)
            
            # Track API call
            self.api_calls += 1
            Agent.total_api_calls += 1
            
            responses = ""
            for chunk in response:
                responses += chunk.text + "\n"
            return responses

    @classmethod
    def get_total_api_calls(cls):
        """Get total API calls across all agents."""
        return cls.total_api_calls
    
    @classmethod
    def reset_total_api_calls(cls):
        """Reset total API call counter."""
        cls.total_api_calls = 0
    
    def get_api_calls(self):
        """Get API calls for this agent instance."""
        return self.api_calls
    
    def reset_api_calls(self):
        """Reset API call counter for this agent instance."""
        self.api_calls = 0

class Group:
    def __init__(self, goal, members, question, examplers=None):
        self.goal = goal
        self.members = []
        for member_info in members:
            _agent = Agent('You are a {} who {}.'.format(member_info['role'], member_info['expertise_description'].lower()), role=member_info['role'], model_info='gpt-4o-mini')
            self.members.append(_agent)
        self.question = question
        self.examplers = examplers

    def interact(self, comm_type, message=None, img_path=None):
        if comm_type == 'internal':
            # Identify lead vs assistants (fallback to first member if no explicit lead)
            lead_member = None
            assist_members = []
            for member in self.members:
                if 'lead' in (member.role or '').lower():
                    lead_member = member
                else:
                    assist_members.append(member)

            if lead_member is None:
                lead_member = assist_members[0] if assist_members else self.members[0]
                assist_members = [m for m in self.members if m is not lead_member]

            extra_ctx = f"\n\nAdditional context / reports from other teams:\n{message}\n" if message else ""

            # Lead delegates investigations
            delivery_prompt = (
                f"You are the lead of a medical group which aims to {self.goal}. "
                "You have the following assistant clinicians who work for you:\n" +
                "\n".join([m.role for m in assist_members]) +
                f"\n\nMedical query:\n{self.question}"
                f"{extra_ctx}\n\n"
                "Provide a short assignment of what investigations / considerations are needed from each assistant clinician."
            )
            try:
                delivery = lead_member.chat(delivery_prompt)
            except Exception:
                delivery = assist_members[0].chat(delivery_prompt) if assist_members else lead_member.chat(delivery_prompt)

            # Assistants return investigation summaries
            investigations = []
            for a_mem in assist_members:
                investigation = a_mem.chat(
                    "You are in a medical group where the goal is to {}. Your group lead is asking for the following investigations:\n{}\n\n"
                    "Please remind your expertise and return your investigation summary that contains the core information.".format(self.goal, delivery)
                )
                investigations.append([a_mem.role, investigation])

            gathered_investigation = ""
            for role, inv in investigations:
                gathered_investigation += f"[{role}]\n{inv}\n"

            # 3) Lead synthesizes and answers the MCQ
            base = (
                f"The gathered investigations from assistant clinicians are as follows:\n{gathered_investigation}\n"
                f"{extra_ctx}\n"
                "Now, answer the medical multiple-choice query among the options provided. "
                f"Question: {self.question}"
            )
            if self.examplers is not None:
                base = (
                    f"{extra_ctx}\n"
                    f"After reviewing the following example cases, answer the query.\n\n{self.examplers}\n\n" + base
                )

            response = lead_member.chat(base)
            return response

        elif comm_type == 'external':
            # External communication between different teams in the ICT framework.
            # This is the sequential, report-driven process that is already implemented in `process_advanced_query`.
            return None

def parse_hierarchy(info, emojis):
    moderator = Node('moderator (\U0001F468\u200D\u2696\uFE0F)')
    agents = [moderator]

    count = 0
    for expert, hierarchy in info:
        m = _EXPERT_ROLE_DESC_RE.match(expert or '')
        expert_name = (m.group('role') if m else (expert or '')).strip()

        emoji = emojis[count % len(emojis)] if emojis else ''
        emoji_str = f" ({emoji})" if emoji else ""
        
        if hierarchy is None:
            hierarchy = 'Independent'

        if hierarchy and 'independent' not in hierarchy.lower():
            # Accept chains like: A > B > C (parent=A, child=B) to match original split('>')[1] behavior.
            parts = [p.strip() for p in re.findall(r'[^>]+', hierarchy) if p.strip()]
            if len(parts) >= 2:
                parent, child = parts[0], parts[1]

                # Attach child under the matched parent if found; otherwise fall back to moderator.
                attached = False
                for agent in agents:
                    agent_base = re.sub(r'\s*\(.*$', '', agent.name).strip().lower()
                    if agent_base == parent.strip().lower():
                        child_agent = Node(f"{child}{emoji_str}", agent)
                        agents.append(child_agent)
                        attached = True
                        break

                if not attached:
                    agent = Node(f"{expert_name}{emoji_str}", moderator)
                    agents.append(agent)
            else:
                agent = Node(f"{expert_name}{emoji_str}", moderator)
                agents.append(agent)
        else:
            agent = Node(f"{expert_name}{emoji_str}", moderator)
            agents.append(agent)

        count += 1

    return agents

def parse_group_info(group_info):
    parsed_info = {
        'group_goal': '',
        'members': []
    }

    group_goal_match = re.search(r'(Group\s+\d+\s*-?\s*|###\s*Group\s+\d+\s*-?\s*)([^\n]+)', group_info, re.IGNORECASE)
    if group_goal_match:
        goal_full = group_goal_match.group(2).strip()
        goal_full = re.sub(r'Group\s+\d+\s*-\s*', '', goal_full, flags=re.IGNORECASE).strip()
        parsed_info['group_goal'] = goal_full
    
    # Parse members line by line - more robust for various LLM output formats
    # Handles: "Member N: Role - Description", "Member N: **Role** - **Description**", etc.
    lines = group_info.split('\n')
    for line in lines:
        # Match "Member N:" at the start of line (with optional markdown bullets/dashes)
        member_match = re.match(r'^\s*[-*]*\s*\*?\*?Member\s+\d+:\s*(.+)', line, re.IGNORECASE)
        if member_match:
            content = member_match.group(1).strip()
            
            # Remove all markdown asterisks first
            content = re.sub(r'\*+', '', content)
            
            # Split by dash to separate role from description
            if ' - ' in content:
                parts = content.split(' - ', 1)
                role_part = parts[0].strip()
                desc_part = parts[1].strip() if len(parts) > 1 else ''
            else:
                role_part = content.strip()
                desc_part = ''
            
            # Remove parenthetical notes like "(Lead)" from role
            role_clean = re.sub(r'\s*\([^)]*\)\s*', ' ', role_part).strip()
            
            if role_clean:
                parsed_info['members'].append({
                    'role': role_clean,
                    'expertise_description': desc_part
                })
    
    return parsed_info

def setup_model(model_name):
    if 'gemini' in model_name:
        genai.configure(api_key=os.environ['genai_api_key'])
        return genai, None
    elif 'gpt' in model_name:
        client = OpenAI(api_key=os.environ['openai_api_key'])
        return None, client
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def load_data(dataset):
    test_qa = []
    examplers = []

    test_path = f'data/{dataset}/test.jsonl'
    with open(test_path, 'r') as file:
        for line in file:
            test_qa.append(json.loads(line))

    train_path = f'data/{dataset}/train.jsonl'
    with open(train_path, 'r') as file:
        for line in file:
            examplers.append(json.loads(line))

    return test_qa, examplers

def create_question(sample, dataset):
    if dataset == 'medqa':
        question = sample['question'] + " Options: "
        options = []
        for k, v in sample['options'].items():
            options.append("({}) {}".format(k, v))
        random.shuffle(options)
        question += " ".join(options)
        return question, None
    return sample['question'], None

def determine_difficulty(question, difficulty):
    if difficulty != 'adaptive':
        return difficulty, None
    
    difficulty_prompt = f"""Now, given the medical query as below, you need to decide the difficulty/complexity of it:\n{question}.\n\nPlease indicate the difficulty/complexity of the medical query among below options:\n1) basic: a single medical agent can output an answer.\n2) intermediate: number of medical experts with different expertise should dicuss and make final decision.\n3) advanced: multiple teams of clinicians from different departments need to collaborate with each other to make final decision."""
    
    moderator = Agent(instruction='You are a medical expert who conducts initial assessment and your job is to decide the difficulty/complexity of the medical query.', role='medical expert', model_info='gpt-4o-mini')
    response = moderator.chat(difficulty_prompt)

    if 'basic' in response.lower() or '1)' in response.lower():
        return 'basic', None
    elif 'intermediate' in response.lower() or '2)' in response.lower():
        return 'intermediate', moderator
    elif 'advanced' in response.lower() or '3)' in response.lower():
        return 'advanced', None
    
    return 'intermediate', None # Default fallback

def process_basic_query(question, examplers, args, fewshot=3):
    cprint("[INFO] Step 1. Single-Agent Few-shot Preparation", 'yellow', attrs=['blink'])    
    medical_agent = Agent(instruction='You are a helpful medical agent.', role='medical expert', model_info=args.model)
    fewshot_examplers = []
    if args.dataset == 'medqa':
        random.shuffle(examplers)
        for _, exampler in enumerate(examplers[:fewshot]):
            tmp_exampler = {}
            exampler_question = f"Question: {exampler['question']}"
            options = [f"({k}) {v}" for k, v in exampler['options'].items()]
            random.shuffle(options)

            exampler_question += " " + ' '.join(options)
            exampler_answer = f"Answer: ({exampler['answer_idx']}) {exampler['answer']}"
            exampler_reason = "Reason: " + medical_agent.chat(f"You are a helpful medical agent. Below is an example of medical knowledge question and answer. After reviewing the below medical question and answering, provide 2-3 sentences of reason that support the answer suppose you didn't know the answer ahead?\n\nQuestion: {exampler_question}\n\nAnswer: {exampler_answer}")

            tmp_exampler['question'] = exampler_question
            tmp_exampler['reason'] = exampler_reason
            tmp_exampler['answer'] = exampler_answer
            fewshot_examplers.append(tmp_exampler)
    
    cprint("[INFO] Step 2. Single-Agent Final Decision", 'yellow', attrs=['blink'])
    single_agent = Agent(instruction="You are a helpful assistant that answers multiple choice questions about medical knowledge.", role='medical expert', examplers=fewshot_examplers, model_info=args.model)
    final_decision = single_agent.temp_responses(
        f"The following are multiple choice questions (with answers) about medical knowledge. Let's think step by step.\n\nQuestion: {question}\nAnswer: ",
        temperatures=[args.temperature] if hasattr(args, 'temperature') else [0.0],
        img_path=None
    )
    
    return final_decision

def process_intermediate_query(question, examplers, moderator, args, fewshot=None):
    """
    Intermediate (MDT) setting with a moderator feedback loop:
      - Recruit N experts + hierarchy
      - Collect initial opinions
      - For each round:
          * participatory debate (agents optionally message each other)
          * agents update final answers for the round
          * consensus check; if not reached, moderator reviews and provides per-agent feedback
      - Final decision maker reviews all agent answers and produces the final answer
    """
    # Create moderator if not provided (when difficulty is not 'adaptive')
    if moderator is None:
        moderator = Agent(
            instruction='You are a medical expert who conducts initial assessment and moderates the discussion.',
            role='moderator',
            model_info='gpt-4o-mini'
        )
    
    print()
    cprint("[INFO] Step 1. Expert Recruitment", 'yellow', attrs=['blink'])

    num_agents = 3 # You can adjust this number as needed

    def _recruit_and_parse_intermediate():
        recruiter = Agent(instruction="You are an experienced medical expert who recruits a group of experts with diverse identity and ask them to discuss and solve the given medical query.", 
                          role='recruiter', 
                          model_info='gpt-4o-mini')
        recruited = recruiter.chat(
            f"Question: {question}\n"
            f"You can recruit {num_agents} experts in different medical expertise. Considering the medical question and the options for the answer, "
            "what kind of experts will you recruit to better make an accurate answer?\n"
            "Also, you need to specify the communication structure between experts "
            "(e.g., Pulmonologist == Neonatologist == Medical Geneticist == Pediatrician > Cardiologist), or indicate if they are independent.\n\n"
            "For example, if you want to recruit five experts, you answer can be like:\n"
            "1. Pediatrician - Specializes in the medical care of infants, children, and adolescents. - Hierarchy: Independent\n"
            "2. Cardiologist - Focuses on the diagnosis and treatment of heart and blood vessel-related conditions. - Hierarchy: Pediatrician > Cardiologist\n"
            "3. Pulmonologist - Specializes in the diagnosis and treatment of respiratory system disorders. - Hierarchy: Independent\n"
            "4. Neonatologist - Focuses on the care of newborn infants, especially those who are born prematurely or have medical issues at birth. - Hierarchy: Independent\n"
            "5. Medical Geneticist - Specializes in the study of genes and heredity. - Hierarchy: Independent\n\n"
            "Please answer in above format, and do not include your reason."
        )
        # print("[DEBUG] Recruited Experts and Hierarchy:\n", recruited)

        agents_data = []
        for _line in re.findall(r'[^\n]+', recruited or ''):
            _line = _line.strip()
            if not _line:
                continue
            m = _EXPERT_HIER_LINE_RE.match(_line)
            expert_txt = (m.group('expert') if m else _line).strip()
            hierarchy_txt = (m.group('hierarchy').strip() if (m and m.group('hierarchy')) else None)
            agents_data.append((expert_txt, hierarchy_txt))

        if not agents_data:
            raise IndexError('No experts parsed from recruitment output')

        # Keep only the requested number of experts (LLM may output extra lines)
        agents_data = agents_data[:num_agents]
        return agents_data

    agents_data = _retry_call('intermediate_recruitment', _recruit_and_parse_intermediate)

    agent_emoji = ['\U0001F468\u200D\u2695\uFE0F', '\U0001F468\U0001F3FB\u200D\u2695\uFE0F', '\U0001F469\U0001F3FC\u200D\u2695\uFE0F', '\U0001F469\U0001F3FB\u200D\u2695\uFE0F', '\U0001f9d1\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3fd\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3fd\u200D\u2695\uFE0F', '\U0001F468\U0001F3FD\u200D\u2695\uFE0F']
    random.shuffle(agent_emoji)

    hierarchy_agents = parse_hierarchy(agents_data, agent_emoji)

    # Extract hierarchy map from the already-parsed tree structure
    # Returns: {agent_idx: {"parent": parent_idx or None, "children": [child_idxs]}}
    def _extract_hierarchy_from_tree(hierarchy_agents, agents_data):
        """
        Extract parent-child relationships from the tree built by parse_hierarchy().
        Build a mapping of agent indices to their parent and children in the hierarchy.
        """
        hierarchy_map = {}
        
        # Initialize all agents
        for idx in range(len(agents_data)):
            hierarchy_map[idx] = {"parent": None, "children": []}
        
        # Map agent roles to indices for quick lookup
        agent_role_to_idx = {}
        for idx, (expert_str, _) in enumerate(agents_data):
            m = _EXPERT_ROLE_DESC_RE.match(expert_str or '')
            role = (m.group('role') if m else (expert_str or '')).strip().lower()
            agent_role_to_idx[role] = idx
        
        # Traverse the tree to extract parent-child relationships
        # Skip the moderator node (index 0 in hierarchy_agents)
        for node in hierarchy_agents[1:]:  # Skip moderator
            # Extract role name from node (format: "role (emoji)")
            node_name = re.sub(r'\s*\(.*$', '', node.name).strip().lower()
            
            # Find corresponding agent index
            child_idx = None
            for role, idx in agent_role_to_idx.items():
                if node_name == role or node_name in role:
                    child_idx = idx
                    break
            
            if child_idx is not None and node.parent:
                # Extract parent name from parent node
                parent_name = re.sub(r'\s*\(.*$', '', node.parent.name).strip().lower()
                
                # Find parent agent index (or skip if it's the moderator)
                if parent_name == 'moderator':
                    parent_idx = None  # Independent (directly under moderator)
                else:
                    parent_idx = None
                    for role, idx in agent_role_to_idx.items():
                        if parent_name == role or parent_name in role:
                            parent_idx = idx
                            break
                
                # Set parent relationship
                if parent_idx is not None:
                    hierarchy_map[child_idx]["parent"] = parent_idx
                    if child_idx not in hierarchy_map[parent_idx]["children"]:
                        hierarchy_map[parent_idx]["children"].append(child_idx)
        
        return hierarchy_map
    
    hierarchy_map = _extract_hierarchy_from_tree(hierarchy_agents, agents_data)

    agent_list = ""
    for i, agent in enumerate(agents_data):
        m = _EXPERT_ROLE_DESC_RE.match(agent[0] or '')
        agent_role = (m.group('role') if m else (agent[0] or '')).strip().lower()
        description = ((m.group('desc') or '') if m else '').strip().lower()
        agent_list += f"Agent {i+1}: {agent_role} - {description}\n"

    agent_dict = {}
    medical_agents = []
    for agent in agents_data:
        m = _EXPERT_ROLE_DESC_RE.match(agent[0] or '')
        agent_role = (m.group('role') if m else (agent[0] or '')).strip().lower()
        description = ((m.group('desc') or '') if m else '').strip().lower()

        inst_prompt = f"You are a {agent_role} who {description}. Your job is to collaborate with other medical experts in a team."
        _agent = Agent(instruction=inst_prompt, role=agent_role, model_info=args.model)
        agent_dict[agent_role] = _agent
        medical_agents.append(_agent)

    for idx, agent in enumerate(agents_data):
        m = _EXPERT_ROLE_DESC_RE.match(agent[0] or '')
        role_txt = (m.group('role') if m else (agent[0] or '')).strip()
        desc_txt = ((m.group('desc') or '') if m else '').strip()
        if desc_txt:
            print(f"Agent {idx+1} ({agent_emoji[idx]} {role_txt}): {desc_txt}")
        else:
            print(f"Agent {idx+1} ({agent_emoji[idx]}): {role_txt}")

    # Few-shot prompting is only used in the low-complexity (basic) setting in the paper.
    # Keep intermediate as zero-shot by default.
    fewshot_examplers = ""
    if fewshot is not None and fewshot > 0:
        medical_agent = Agent(instruction='You are a helpful medical agent.', role='medical expert', model_info=args.model)
        random.shuffle(examplers)
        for ie, exampler in enumerate(examplers[:fewshot]):
            exampler_question = f"[Example {ie+1}]\n" + exampler['question']
            options = [f"({k}) {v}" for k, v in exampler['options'].items()]
            random.shuffle(options)

            exampler_question += " " + ' '.join(options)
            exampler_answer = f"Answer: ({exampler['answer_idx']}) {exampler['answer']}"
            exampler_reason = medical_agent.chat(
                "Below is an example of medical knowledge question and answer. After reviewing the below medical question and answering, "
                "can you provide 1-2 sentences of reason that support the answer as you didn't know the answer ahead?\n\n"
                f"Question: {exampler_question}\n\nAnswer: {exampler_answer}"
            )

            exampler_question += f"\n{exampler_answer}\n{exampler_reason}\n\n"
            fewshot_examplers += exampler_question

    print()
    cprint("[INFO] Hierarchy Selection", 'yellow', attrs=['blink'])
    print_tree(hierarchy_agents[0], horizontal=False)
    print()

    # Moderator (feedback loop + consensus checking)
    moderator_prompt = (
        "You are now a moderator in a multidisciplinary medical team discussion. "
        "Your job is to moderate the discussion, check whether the team has reached consensus, "
        "and when there is disagreement, provide targeted feedback to each expert to help the team converge "
        "toward a correct and consistent final answer."
    )
    moderator.chat(moderator_prompt)

    num_rounds = 5
    num_turns = 5

    # Logs
    # interaction_log[round_name][turn_name]['Agent i']['Agent j'] = message
    interaction_log = {}
    feedback_log = {}  # feedback_log[round_name][role] = feedback

    # Helper function to print summary table
    def _print_summary_table(turn_data, n_agents):
        """Print the interaction summary table showing which agents communicated.
        Args:
            turn_data: Dictionary with structure {'Agent i': {'Agent j': msg, ...}, ...}
            n_agents: Number of agents
        """
        header = [""] + [f"Agent {i+1}" for i in range(n_agents)]
        myTable = PrettyTable(header)

        matrix = [[False] * n_agents for _ in range(n_agents)]
        # Process the turn data directly
        for src, dsts in turn_data.items():
            msrc = re.search(r'Agent\s+(\d+)', src)
            if not msrc:
                continue
            i = int(msrc.group(1)) - 1
            for dst in dsts.keys():
                mdst = re.search(r'Agent\s+(\d+)', dst)
                if not mdst:
                    continue
                j = int(mdst.group(1)) - 1
                if 0 <= i < n_agents and 0 <= j < n_agents and i != j:
                    matrix[i][j] = True

        for i in range(1, n_agents + 1):
            row = [f"Agent {i}"]
            for j in range(1, n_agents + 1):
                if i == j:
                    row.append(" ")
                    continue
                i2j = matrix[i-1][j-1]
                j2i = matrix[j-1][i-1]
                if not i2j and not j2i:
                    row.append(" ")
                elif i2j and not j2i:
                    row.append(f'\u270B ({i}->{j})')
                elif j2i and not i2j:
                    row.append(f'\u270B ({i}<-{j})')
                else:
                    row.append(f'\u270B ({i}<->{j})')
            myTable.add_row(row)
            if i != n_agents:
                myTable.add_row(['' for _ in range(n_agents + 1)])

        print(myTable)
        print()

    # Initial opinions
    cprint("[INFO] Step 2. Initial Opinions", 'yellow', attrs=['blink'])
    opinions = {}
    for idx, agent in enumerate(medical_agents):
        prompt = (
            (f"Given the examplers, please return your answer to the medical query among the option provided.\n\n"
             f"{fewshot_examplers}\n\n" if fewshot_examplers else "") +
            f"Question: {question}\n\n"
        )
        resp = agent.chat(prompt, img_path=None)
        opinions[agent.role] = resp
        # Print like the original: include agent number + emoji + role
        print(f" Agent {idx+1} ({agent_emoji[idx]} {agent.role}) : {resp}")

    # Collaborative decision making rounds
    print()
    cprint("[INFO] Step 3. Collaborative Decision Making", 'yellow', attrs=['blink'])  
    final_answers = dict(opinions)
    round_feedback = {}  # role -> feedback for next round

    for r in range(1, num_rounds + 1):
        round_name = f"Round {r}"
        interaction_log.setdefault(round_name, {})
        feedback_log.setdefault(round_name, {})

        print(f"== {round_name} ==")

        # Apply moderator feedback from the previous round (if any)
        if round_feedback:
            cprint("[INFO] Moderator Feedback", 'yellow', attrs=['blink'])
            for idx, agent in enumerate(medical_agents):
                fb = (round_feedback.get(agent.role) or "").strip()
                if fb:
                    print(f" \U0001F468\u200D\u2696\uFE0F moderator -> Agent {idx+1} ({agent_emoji[idx]} {agent.role}) : {fb}")
                    agent.chat(
                        f"Moderator feedback for you:\n{fb}\n\n"
                        "Acknowledge the feedback and adjust your thinking. Then be ready to discuss.",
                        img_path=None
                    )

        assessment = "".join(f"({k}): {v}\n" for k, v in opinions.items())

        # Participatory debate (T turns)
        cprint("[INFO] Participatory Debate", 'yellow', attrs=['blink'])
        
        # Build list of valid communication partners based on hierarchy
        def _get_valid_partners(agent_idx, hierarchy_map):
            """
            Get list of agents that this agent can communicate with based on hierarchy.
            An agent can communicate with:
            - Its parent (if it has one)
            - Its children (if it has any)
            - Its siblings (agents with the same parent)
            
            Special case: Agents with parent=None are direct children of Moderator.
            They should be able to talk to each other (as siblings under Moderator).
            """
            valid = []
            hm = hierarchy_map.get(agent_idx, {})
            
            # Can talk to parent
            if hm.get("parent") is not None:
                valid.append(hm["parent"])
            
            # Can talk to children
            if hm.get("children"):
                valid.extend(hm["children"])
            
            # Can talk to siblings (other children of the same parent)
            if hm.get("parent") is not None:
                # Normal case: agent has a parent agent
                parent_idx = hm["parent"]
                parent_hm = hierarchy_map.get(parent_idx, {})
                siblings = [s for s in parent_hm.get("children", []) if s != agent_idx]
                valid.extend(siblings)
            else:
                # Special case: agent is directly under Moderator (parent=None)
                # Find all other agents that are also directly under Moderator (siblings)
                moderator_level_siblings = [idx for idx in range(len(hierarchy_map)) 
                                           if idx != agent_idx and hierarchy_map[idx].get("parent") is None]
                valid.extend(moderator_level_siblings)
            
            return valid
        
        num_yes_total = 0
        for t in range(1, num_turns + 1):
            turn_name = f"Turn {t}"
            interaction_log[round_name].setdefault(turn_name, {})
            print(f"|_{turn_name}")

            num_yes = 0
            for idx, agent in enumerate(medical_agents):
                participate = agent.chat(
                    "Given the opinions from other medical agents, indicate whether you want to talk to any expert (yes/no). "
                    "If not, provide your opinion.\n\n"
                    f"Opinions:\n{assessment}",
                    img_path=None
                )

                if re.search(r'(?i)\byes\b', (participate or "").strip()):
                    # Get valid communication partners based on hierarchy
                    valid_partners = _get_valid_partners(idx, hierarchy_map)
                    
                    if not valid_partners:
                        # No valid partners in hierarchy, skip communication
                        print(f" Agent {idx+1} ({agent_emoji[idx]} {agent.role}): No valid partners in hierarchy")
                        continue
                    
                    # Build filtered agent list showing only valid partners
                    filtered_agent_list = ""
                    for i, agent_data in enumerate(agents_data):
                        if i in valid_partners:
                            m = _EXPERT_ROLE_DESC_RE.match(agent_data[0] or '')
                            agent_role = (m.group('role') if m else (agent_data[0] or '')).strip().lower()
                            description = ((m.group('desc') or '') if m else '').strip().lower()
                            filtered_agent_list += f"Agent {i+1}: {agent_role} - {description}\n"
                    
                    chosen_expert = agent.chat(
                        "Next, indicate the agent(s) you want to talk to (only from your team hierarchy).\n"
                        f"{filtered_agent_list}\n"
                        "Return ONLY the number(s), e.g., 1 or 1,2. Do not include reasons.",
                        img_path=None
                    )
                    chosen_experts = [int(ce) for ce in re.split(r'[^0-9]+', chosen_expert or '') if ce.strip().isdigit()]
                    chosen_experts = [ce for ce in chosen_experts if 1 <= ce <= len(medical_agents) and (ce - 1) in valid_partners]
                    chosen_experts = list(dict.fromkeys(chosen_experts))  # unique, preserve order

                    for ce in chosen_experts:
                        msg = agent.chat(
                            "Remind your medical expertise and leave your opinion to the expert you chose. "
                            "Deliver your opinion once you are confident and in a way to convince the other expert with a short reason.\n\n"
                            f"Question:\n{question}",
                            img_path=None
                        )

                        # Print + log 
                        print(f" Agent {idx+1} ({agent_emoji[idx]} {agent.role}) -> Agent {ce} ({agent_emoji[ce-1]} {medical_agents[ce-1].role}) : {msg}")
                        interaction_log[round_name][turn_name].setdefault(f"Agent {idx+1}", {})
                        interaction_log[round_name][turn_name][f"Agent {idx+1}"][f"Agent {ce}"] = msg

                    num_yes += 1
                    num_yes_total += 1
                else:
                    # "no" path: store updated opinion for the next turn/round
                    if participate and participate.strip():
                        opinions[agent.role] = participate
                        assessment = "".join(f"({k}): {v}\n" for k, v in opinions.items())
                    print(f" Agent {idx+1} ({agent_emoji[idx]} {agent.role}): \U0001F910")

            if num_yes == 0:
                print(" No agents chose to participate in this turn. End this turn.")
                break

            # Print summary table for this turn only
            print()
            cprint("[INFO] Summary Table", 'cyan', attrs=['blink'])
            _print_summary_table(interaction_log[round_name][turn_name], len(medical_agents))

        if num_yes_total == 0:
            print(" No agents chose to participate in this round. End this round.")
            break
        
        # Agents update final answers for this round
        tmp_final_answer = {}
        for agent in medical_agents:
            response = agent.chat(
                "Now that you've interacted with other medical experts (and received moderator feedback if any), "
                "remind your expertise and the comments from other experts and make your final answer to the given question.\n"
                f"Question: {question}\n\n",
                img_path=None
            )
            tmp_final_answer[agent.role] = response

        final_answers = tmp_final_answer

        # Moderator consensus check (moderator decides if another round is needed)
        cprint("\n[INFO] Moderator Consensus Check", 'yellow', attrs=['blink'])
        answers_text = "".join(f"[{role}] {ans}\n" for role, ans in final_answers.items())
        moderator_consensus = moderator.chat(
            "You are moderating the team. Decide whether the team has reached consensus on the final option.\n"
            "Consensus means the experts' final answers point to the same option letter, or are clearly aligned.\n\n"
            "Return EXACTLY one of the following (and nothing else):\n"
            "Consensus: YES\n"
            "Consensus: NO\n\n"
            f"Question:\n{question}\n\n"
            f"Experts' current answers:\n{answers_text}\n",
            img_path=None
        )
                
        consensus_yes = bool(re.search(r'(?im)^\s*Consensus\s*:\s*YES\s*$', moderator_consensus or ''))
        consensus_no = bool(re.search(r'(?im)^\s*Consensus\s*:\s*NO\s*$', moderator_consensus or ''))

        # If the moderator response is malformed, default to NO (continue refining)
        if not (consensus_yes or consensus_no):
            consensus_yes = False
            consensus_no = True

        print(f" \U0001F468\u200D\u2696\uFE0F Moderator consensus check: {'YES' if consensus_yes else 'NO'}")

        if consensus_yes:
            cprint("\n[INFO] Consensus reached! Ending discussion.", 'green', attrs=['blink'])
            round_feedback = {}
            break

        # Moderator provides feedback for next round if not converged
        cprint("\n[INFO] Disagreement detected", 'yellow', attrs=['blink'])
        
        # Build compact discussion log so far (across rounds/turns)
        log_text = ""
        for _rnd, _rnd_payload in interaction_log.items():
            for _turn, payload in _rnd_payload.items():
                for src, dsts in payload.items():
                    for dst, msg in dsts.items():
                        log_text += f"[{_rnd} / {_turn}] {src} -> {dst}: {msg}\n"

        moderator_feedback = moderator.chat(
            "You are moderating the team. Provide targeted feedback to help the experts converge in the next round.\n"
            "Return ONLY in the following repeated format (no other text):\n"
            "Agent: <role>\n"
            "Feedback: <targeted feedback>\n\n"
            "(repeat Agent/Feedback for each expert)\n\n"
            f"Question:\n{question}\n\n"
            f"Experts' current answers:\n{answers_text}\n"
            f"Discussion log so far (if any):\n{log_text if log_text else '(no direct messages)'}\n",
            img_path=None
        )

        # Parse and store feedback for the next round
        parsed_fb = {}
        for fb_block in re.split(r'(?im)^\s*Agent\s*:\s*', moderator_feedback or ''):
            fb_block = fb_block.strip()
            if not fb_block:
                continue
            fb_lines = fb_block.splitlines()
            role = (fb_lines[0] or '').strip().lower()
            body = "\n".join(fb_lines[1:]).strip()
            body = re.sub(r'(?im)^\s*Feedback\s*:\s*', '', body).strip()
            if role and body:
                parsed_fb[role] = body

        # Fallback: broadcast the whole moderator response if parsing fails
        if not parsed_fb:
            parsed_fb = {agent.role: (moderator_feedback or '').strip() for agent in medical_agents}

        round_feedback = parsed_fb

        cprint("\n[INFO] Moderator Review & Feedback", 'yellow', attrs=['blink'])
        for idx, agent in enumerate(medical_agents):
            fb = (round_feedback.get(agent.role) or "").strip()
            if fb:
                feedback_log[round_name][agent.role] = fb
                print(f" \U0001F468\u200D\u2696\uFE0F Moderator -> Agent {idx+1} ({agent_emoji[idx]} {agent.role}) : {fb}")

        # Next round starts from the agents' last answers
        opinions = dict(final_answers)

    # Final decision maker (review all opinions)
    cprint("\n[INFO] Step 4. Final Decision", 'yellow', attrs=['blink'])

    decision_maker = Agent(
        "You are a final medical decision maker who reviews all opinions from different medical experts and their conversation history to make the final decision.",
        role='decision maker',
        model_info=args.model
    )
    
    answers_text = "".join(f"[{role}] {ans}\n" for role, ans in final_answers.items())
    
    # Build full conversation history for decision maker
    conversation_history = ""
    for round_name, round_data in interaction_log.items():
        conversation_history += f"\n=== {round_name} ===\n"
        for turn_name, turn_data in round_data.items():
            conversation_history += f"\n{turn_name}:\n"
            for src, dsts in turn_data.items():
                for dst, msg in dsts.items():
                    conversation_history += f"  {src} → {dst}:\n    {msg}\n"
    
    # print("\n[DEBUG] Full Conversation History for Decision Maker:\n", conversation_history)
    
    _decision = decision_maker.temp_responses(
        "You are reviewing the final decision from a multidisciplinary team discussion. "
        "Consider the experts' reasoning, the conversation history showing how they interacted and converged (or disagreed), "
        "and their final answers to make an informed final decision.\n\n"
        f"Question:\n{question}\n\n"
        f"Conversation History:\n{conversation_history if conversation_history.strip() else '(No direct interactions occurred)'}\n\n"
        f"Experts' Final Answers:\n{answers_text}\n"
        "Based on the conversation history and final answers, please make the final answer to the question by considering consensus and reasoning quality:\n"
        "Answer: ",
        temperatures=[args.temperature] if hasattr(args, 'temperature') else [0.0],
        img_path=None 
    )
    final_decision = {'majority': _decision,}
    
    print(f"\U0001F468\u200D\u2696\uFE0F  Moderator's final decision (by majority vote):", _decision)

    return final_decision

def process_advanced_query(question, args):
    """
    Advanced (ICT) setting:
      - Recruit multiple MDTs including an Initial Assessment Team (IAT) and a Final Review/Decision Team (FRDT)
      - Each MDT internally discusses and produces a report
      - FRDT reviews other MDT reports to produce a final review report
      - Final decision maker reviews ALL reports (IAT + other MDTs + FRDT) and outputs final answer
    """
    print()
    cprint("[INFO] Step 1. MDTs Recruitment", 'yellow', attrs=['blink'])
    group_instances = []

    recruit_prompt = (
        "You are an experienced medical expert. Given the complex medical query, you need to organize "
        "Multidisciplinary Teams (MDTs) and the members in MDT to make accurate and robust answer."
    )

    num_teams = 3  # Number of MDTs
    num_agents = 3

    def _recruit_and_parse_advanced():
        recruit_prompt = (
            "You are an experienced medical expert. Given the complex medical query, you need to organize "
            "Multidisciplinary Teams (MDTs) and the members in MDT to make accurate and robust answer."
        )
        recruiter = Agent(instruction=recruit_prompt, role='recruiter', model_info='gpt-4o-mini')

        recruited = recruiter.chat(
            f"Question: {question}\n\n"
            f"You should organize {num_teams + 2} MDTs with different specialties or purposes and each MDT should have {num_agents} clinicians. "
            "Considering the medical question and the options, please return your recruitment plan to better make an accurate answer.\n\n"
            "You MUST include:\n"
            "- Group 1 - Initial Assessment Team (IAT)\n"
            f"- {num_teams} MDTs\n"
            f"- Group {num_teams + 2} - Final Review and Decision Team (FRDT)\n\n"
            "Return in this format:\n"
            "Group X - <Team Name> (optional acronym)\n"
            "Member 1: <Role> (Lead) - <Expertise>\n"
            "Member 2: <Role> - <Expertise>\n"
            "Member 3: <Role> - <Expertise>\n"
        )

        # Parse groups robustly
        groups = [g.strip() for g in (recruited or '').split('Group') if g.strip()]
        # Drop any leading non-group chunk created by split (e.g., stray '###')
        groups = [g for g in groups if re.match(r'^\d+\s*[-–—:]', g) or re.match(r'^\d+\b', g)]
        group_strings = ['Group ' + g for g in groups]
        
        if not group_strings:
            raise IndexError("No groups parsed from recruitment output")

        parsed_groups = []
        for gs in group_strings:
            res_gs = parse_group_info(gs)
            if not res_gs.get('members'):
                raise IndexError('Parsed group has no members')
            parsed_groups.append(res_gs)

        return group_strings, parsed_groups

    _, parsed_groups = _retry_call('advanced_recruitment', _recruit_and_parse_advanced)

    # Define emojis for teams/members
    member_emoji = ['\U0001F468\u200D\u2695\uFE0F', '\U0001F468\U0001F3FB\u200D\u2695\uFE0F', '\U0001F469\U0001F3FC\u200D\u2695\uFE0F', '\U0001F469\U0001F3FB\u200D\u2695\uFE0F', '\U0001f9d1\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3fd\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3fd\u200D\u2695\uFE0F', '\U0001F468\U0001F3FD\u200D\u2695\uFE0F']
    
    for i1, res_gs in enumerate(parsed_groups):
        print(f"\nGroup {i1+1} - {res_gs['group_goal']}")
        for i2, member in enumerate(res_gs['members']):
            member_icon = member_emoji[i2 % len(member_emoji)]
            role = member['role']
            desc = member['expertise_description']
            if desc:
                print(f"  {member_icon} Member {i2+1} ({role}): {desc}")
            else:
                print(f"  {member_icon} Member {i2+1} ({role})")

        group_instance = Group(res_gs['group_goal'], res_gs['members'], question)
        group_instances.append(group_instance)
    print()

    # Identify teams
    def _is_iat(goal):
        g = (goal or "").lower()
        return ("initial" in g) or ("iat" in g) or ("iap" in g)

    def _is_frdt(goal):
        g = (goal or "").lower()
        return ("frdt" in g) or ("final review" in g) or ("final decision" in g) or ("review" in g and "final" in g) or ("decision" in g and "final" in g)

    iat_team = [gi for gi in group_instances if _is_iat(gi.goal)]
    frdt_team = [gi for gi in group_instances if _is_frdt(gi.goal)]
    other_teams = [gi for gi in group_instances if (gi not in iat_team and gi not in frdt_team)]
    
    def _generate_report(raw_team_output):
        reporter = Agent(instruction="You are a medical assistant who excels at summarizing and synthesizing based on multiple experts from various domain experts.", role="medical assistant", model_info=args.model)
        prompt = (
            "Given the MDT raw discussion output (agent answers / investigations) below, please complete the following steps:\n"
            "1. Take careful and comprehensive consideration of the provided reports.\n"
            "2. Extract key knowledge from the reports.\n"
            "3. Derive a comprehensive and summarized analysis based on the extracted knowledge.\n"
            "4. Generate a refined and synthesized report based on your analysis.\n\n"
            "MDT raw output:\n"
            f"{raw_team_output}\n"
        )
        report = reporter.temp_responses(
            prompt,
            temperatures=[args.temperature] if hasattr(args, 'temperature') else [0.0]
        )
        temp_key = args.temperature if hasattr(args, 'temperature') else 0.0
        return report[temp_key]

    # STEP 2. MDT Internal Discussions (reports)
    cprint("[INFO] Step 2. MDT Internal Discussions", 'yellow', attrs=['blink'])

    # 2.1 Initial Assessment
    initial_assessments = []
    for gi in iat_team:
        response = gi.interact(comm_type='internal')
        initial_assessments.append([gi.goal, response])

    initial_assessment_report = ""
    for idx, init_assess in enumerate(initial_assessments):
        initial_assessment_report += f"Group {idx+1} - {init_assess[0]}\n{init_assess[1]}\n\n"
    initial_assessment_report = _generate_report(initial_assessment_report) 
    print()  
    print("Initial Assessment Report:\n", initial_assessment_report)
    
    # 2.2 Other MDTs
    assessments = []
    for gi in other_teams:
        response = gi.interact(comm_type='internal')
        assessments.append([gi.goal, response])

    assessment_report = ""
    for idx, assess in enumerate(assessments):
        assessment_report += f"Group {idx+1} - {assess[0]}\n{assess[1]}\n\n"
    assessment_report = _generate_report(assessment_report)
    print()
    print("Other MDTs Report:\n", assessment_report)

    # 2.3 FRDT review (give it all prior reports)
    frdt_reviews = []
    frdt_context = (
        "You are the Final Review and Decision Team (FRDT). "
        "Below are reports from the Initial Assessment Team and other MDTs. "
        "Use them to produce a careful final review report.\n\n"
        f"[Initial Assessment]\n{initial_assessment_report}\n"
        f"[Other MDT Reports]\n{assessment_report}\n"
    )
    for gi in frdt_team:
        review = gi.interact(comm_type='internal', message=frdt_context)
        frdt_reviews.append([gi.goal, review])

    frdt_report = ""
    for idx, decision in enumerate(frdt_reviews):
        frdt_report += f"Group {idx+1} - {decision[0]}\n{decision[1]}\n\n"
    frdt_report = _generate_report(frdt_report)
    print()
    print("FRDT Report:\n", frdt_report)

    # STEP 3. Final Decision Maker uses ALL reports
    cprint("\n[INFO] Step 3. Final Decision", 'yellow', attrs=['blink'])
    decision_prompt = (
        "You are an experienced medical expert. Now, given the investigations from multidisciplinary teams (MDT), "
        "please review them very carefully and return your final decision to the medical query."
    )
    decision_maker = Agent(instruction=decision_prompt, role='decision maker', model_info=args.model)

    all_reports = (
        f"[Initial Assessment Team]\n{initial_assessment_report}\n"
        f"[Other MDTs]\n{assessment_report}\n"
        f"[Final Review & Decision Team]\n{frdt_report}\n"
    )

    final_decision = decision_maker.temp_responses(
        f"Reports:\n{all_reports}\n"
        f"Question: {question}\n\n"
        "Answer: ",
        temperatures=[args.temperature] if hasattr(args, 'temperature') else [0.0]
    )
    print(f"\U0001F468\u200D\u2696\uFE0F  Decision Maker's final decision:", final_decision)
    
    return final_decision

