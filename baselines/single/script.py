import os
import sys
import json
import random
import traceback
import argparse
from tqdm import tqdm

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from utils import setup_model, load_data, create_question
from main import Logger, _atomic_json_dump

from baselines.single.zero_few_shot import zero_few_shot_query
from baselines.single.cot import cot_query
from baselines.single.cot_sc import cot_sc_query

if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument('--dataset', type=str, default='medqa')
    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    parser.add_argument('--method', type=str, default='cot', choices=['zero_shot', 'few_shot', 'cot', 'cot_sc'], help='Method to use')
    parser.add_argument('--fewshot', type=int, default=8, help='Number of few-shot examples to use. If method is zero-shot, this is ignored.')
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    
    # Redirect stdout to log file
    file_name = f"{args.dataset}_{args.model}_{args.method}_{args.num_samples}{'_' + str(args.seed) if args.seed is not None else ''}"

    if not os.path.exists('logs'):
        os.makedirs('logs')
    sys.stdout = Logger(f"logs/{file_name}.log")

    # Output + resume paths
    output_dir = os.path.join(os.getcwd(), 'output')
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{file_name}.json")
    progress_path = os.path.join('logs', f"{file_name}.progress.json")

    # Load previous results if present (resume)
    results = []
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                results = json.load(f)
            if not isinstance(results, list):
                print(f"[WARN] Existing output is not a list. Starting fresh: {output_path}")
                results = []
        except Exception as e:
            print(f"[WARN] Failed to load existing output ({output_path}): {e}. Starting fresh.")
            results = []

    start_no = len(results)
    if start_no > 0:
        print(f"[INFO] Resuming from sample index {start_no} (already saved {start_no} results).")

    # Keep a tiny progress file too (helpful if output gets edited)
    _atomic_json_dump({"next_index": start_no}, progress_path)

    model, client = setup_model(args.model)
    test_qa, examplers = load_data(args.dataset)

    # Randomly select 50 test samples for quicker testing (remove this part for full eval)
    if args.seed is not None:
        random.seed(args.seed)
        
    if args.num_samples is not None and args.num_samples < len(test_qa):
        test_qa = random.sample(test_qa, args.num_samples)

    # Main loop (auto-save after each sample)
    for no, sample in enumerate(
        tqdm(test_qa[start_no:], total=len(test_qa), initial=start_no),
        start=start_no
    ):
        if no == args.num_samples:
            break

        if no == 0:
            print(f"[INFO] no: {no}")
        else:    
            print(f"\n\n[INFO] no: {no}")

        try:
            question, img_path = create_question(sample, args.dataset)
            
            if args.method == 'zero_shot' or args.method == 'few_shot':
                final_decision = zero_few_shot_query(question, examplers, args, fewshot_num=args.fewshot)
            elif args.method == 'cot':
                final_decision = cot_query(question, examplers, args, fewshot_num=args.fewshot)
            elif args.method == 'cot_sc':
                final_decision = cot_sc_query(question, examplers, args, fewshot_num=args.fewshot)

            if args.dataset == 'medqa':
                results.append({
                    'question': question,
                    'label': sample['answer_idx'],
                    'answer': sample['answer'],
                    'options': sample['options'],
                    'response': final_decision,
                })
            else:
                # Will update later for other datasets
                results.append({
                    'question': question,
                    'response': final_decision,
                })

            # Save after each successful sample
            _atomic_json_dump(results, output_path)
            _atomic_json_dump({"next_index": len(results)}, progress_path)

        except KeyboardInterrupt:
            print("\n[WARN] Interrupted by user (KeyboardInterrupt). Saving progress and exiting...")
            _atomic_json_dump(results, output_path)
            _atomic_json_dump({"next_index": len(results)}, progress_path)
            break

        except Exception as e:
            print(f"\n[ERROR] Exception at sample index {no}: {type(e).__name__}: {e}")
            traceback.print_exc()
            print("[INFO] Saving progress up to last completed sample and exiting...")
            _atomic_json_dump(results, output_path)
            _atomic_json_dump({"next_index": len(results)}, progress_path)
            break

    # Final save (in case loop ends normally)
    _atomic_json_dump(results, output_path)
    _atomic_json_dump({"next_index": len(results)}, progress_path)
    print(f"[INFO] Done. Saved {len(results)} samples to: {output_path}")
