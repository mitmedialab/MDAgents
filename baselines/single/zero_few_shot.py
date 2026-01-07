import random
from termcolor import cprint
from utils import Agent

def zero_few_shot_query(question, examplers, args, fewshot_num=8):
    if args.method == 'zero_shot':
        cprint(f"\n[INFO] Generating Zero-Shot response.", 'cyan')
        single_agent = Agent(
            instruction="You are a helpful assistant that answers multiple choice questions about medical knowledge.", 
            role='medical expert',
            model_info=args.model
        )
        response_dict = single_agent.temp_responses(
            f"The following is a multiple choice question (with answer choices) about medical knowledge. Please provide the final answer directly.\n\nQuestion: {question}\nAnswer: ", 
            temperatures=[0.0], 
            img_path=None
        )
        final_decision = list(response_dict.values())[0]
        cprint(final_decision)

        return final_decision
    
    elif args.method == 'few_shot':
        cprint(f"\n[INFO] Generating Few-Shot response with {fewshot_num} few-shot examplers.", 'cyan')
        if args.dataset == 'medqa':
            random.shuffle(examplers)
            fewshot_examplers = []
            for _, exampler in enumerate(examplers[:fewshot_num]):
                tmp_exampler = {}
                exampler_question = f"Question: {exampler['question']}"
                options = [f"({k}) {v}" for k, v in exampler['options'].items()]
                random.shuffle(options)
                exampler_question += " " + ' '.join(options)
                exampler_answer = f"Answer: ({exampler['answer_idx']}) {exampler['answer']}"

                tmp_exampler['question'] = exampler_question
                tmp_exampler['answer'] = exampler_answer
                fewshot_examplers.append(tmp_exampler)
                print()
                print(f"Fewshot exampler #{_ + 1}:\n{tmp_exampler['question']}\n{tmp_exampler['answer']}")

        single_agent = Agent(
            instruction="You are a helpful assistant that answers multiple choice questions about medical knowledge.", 
            role='medical expert', 
            fewshot_examplers=fewshot_examplers,
            model_info=args.model
        )

        response_dict = single_agent.temp_responses(
            f"The following is a multiple choice question (with answer choices) about medical knowledge. Please provide the final answer directly.\n\nQuestion: {question}\nAnswer: ", 
            temperatures=[0.0], 
            img_path=None
        )
        final_decision = list(response_dict.values())[0]
        cprint(final_decision)
        
        return final_decision