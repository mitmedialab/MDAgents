import random
from termcolor import cprint
from utils import Agent

def cot_examplers(examplers, args, fewshot_num=8):
    # print()
    # cprint(f"[INFO] Generating CoT response with {fewshot_num} few-shot examplers.", 'cyan')
    medical_agent = Agent(instruction='You are a helpful medical agent.', role='medical expert', model_info=args.model)
    fewshot_examplers = []
    if args.dataset == 'medqa':
        random.shuffle(examplers)
        for _, exampler in enumerate(examplers[:fewshot_num]):
            tmp_exampler = {}
            exampler_question = f"Question: {exampler['question']}"
            options = [f"({k}) {v}" for k, v in exampler['options'].items()]
            random.shuffle(options)
            exampler_question += " " + ' '.join(options)
            exampler_answer = f"Answer: ({exampler['answer_idx']}) {exampler['answer']}"
            exampler_reason = "Reason: " + medical_agent.chat(f"You are a helpful medical agent. Below is an example of medical knowledge question and answer. After reviewing the below medical question and answering, provide 1-2 sentences of reason that support the answer suppose you didn't know the answer ahead?\n\nQuestion: {exampler_question}\n\nAnswer: {exampler_answer}")

            tmp_exampler['question'] = exampler_question
            tmp_exampler['reason'] = exampler_reason
            tmp_exampler['answer'] = exampler_answer
            fewshot_examplers.append(tmp_exampler)
            # print()
            # print(f"Fewshot exampler #{_ + 1}:\n{tmp_exampler['question']}\n{tmp_exampler['reason']}\n{tmp_exampler['answer']}\n")

    return fewshot_examplers


def cot_sc_query(question, examplers, args, fewshot_num=8, sampling_num=10):
    sampling_responses = []

    cprint(f"\n[INFO] Generating CoT-SC responses with {fewshot_num} few-shot examplers and {sampling_num} sampling paths.", 'cyan')
    for _ in range(sampling_num):
        single_agent = Agent(
            instruction="You are a helpful assistant that answers multiple choice questions about medical knowledge.", 
            role='medical expert', 
            examplers=cot_examplers(examplers, args, fewshot_num), 
            model_info=args.model
        )

        response_dict = single_agent.temp_responses(
            f"The following are multiple choice questions (with answers) about medical knowledge. Let's think step by step.\n\nQuestion: {question}\nAnswer: ", 
            temperatures=[0.7], 
            img_path=None
        )
        sampling_responses.append(list(response_dict.values())[0])

    # Majority voting
    decision_agent = Agent(
        instruction="You are a final medical decision maker who reviews all opinions and makes the final decision.",
        role="decision maker",
        model_info=args.model
    )
    
    formatted_opinions = "\n".join([f"Path {i+1}: {ans}" for i, ans in enumerate(sampling_responses)])
    final_prompt = (
        f"The following are {sampling_num} different reasoning paths and answers generated for the same question.\n"
        f"Opinions:\n{formatted_opinions}\n"
        f"Perform majority voting to select the most consistent answer."
    )

    cprint(f"\n[INFO] Opinion paths", 'cyan')
    print(formatted_opinions)
    
    cprint(f"\n[INFO] Final decision by majority voting", 'cyan')
    final_decision = decision_agent.temp_responses(final_prompt)
    print(final_decision[0.0])

    return final_decision