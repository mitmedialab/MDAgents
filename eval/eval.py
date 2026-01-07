import os
import argparse
import json
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import random

try:
    from eval.call_llm import ask
except ImportError:
    from call_llm import ask


# Default valid choices for MedQA dataset
VALID_CHOICES = {"A", "B", "C", "D", "E"}
VALID_CHOICES_PLUS_X = VALID_CHOICES | {"X"}


def extract_response(sample: dict) -> str:
    """Extract response content for a given sample"""
    return sample["response"]["majority"]["0.0"] if sample["difficulty"] == "intermediate" else sample["response"]["0.0"]


def extract_answer(response: str) -> str:
    """Extract the answer from a LLM response by calling Ollama API"""
    system_prompt = (
        "You are a grading helper. Your only job is to extract the single final answer choice letter from a model response.\n"
        "Rules:\n"
        "- Output MUST be exactly one character: A, B, C, D, or E.\n"
        "- Output MUST contain nothing else (no words, no punctuation, no spaces, no newlines).\n"
        "- If the answer is not clear, output X (a single character).\n"
    )
    user_prompt = (
        "Extract the final answer letter (A/B/C/D/E) from the response below.\n\n"
        f"RESPONSE:\n<<<\n{response}\n>>>\n"
    )

    extracted_answer = ask(
        user_prompt=user_prompt,
        sys_prompt=system_prompt,
        model_name="gpt-oss:20b",
        thinking=False,
        max_tokens=3,
        temperature=0.0,
        infinite_retry=True,
    )

    # Normalize to a single char
    out = (extracted_answer or "").strip().upper()
    if not out:
        return "X"
    # Keep only first valid char among A-E/X
    for ch in out:
        if ch in {"A", "B", "C", "D", "E", "X"}:
            return ch
    return "X"


def _atomic_write_json(path: str, obj: Any) -> None:
    """Write JSON atomically to avoid corruption on interruption."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _load_existing_results(result_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(result_path):
        return []
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        # If file is corrupted/partial, back it up and start fresh
        bak = result_path + ".corrupted.bak"
        try:
            os.replace(result_path, bak)
            print(f"[WARN] Existing result file was corrupted. Backed up to: {bak}")
        except Exception:
            pass
        return []


def extract_and_save_predictions(
    samples: List[dict],
    result_path: str,
    flush_every: int = 1,
) -> List[Dict[str, Any]]:
    """
    Extract predictions and continuously save to result_path.
    - Resume: if result_path exists, skip already processed indices.
    - tqdm progress bar.
    - Writes atomically to avoid file corruption.
    """
    existing = _load_existing_results(result_path)

    # Resume strategy: store "idx" with each record; use it to skip
    done_idx = set()
    for rec in existing:
        if isinstance(rec, dict) and "idx" in rec:
            done_idx.add(rec["idx"])

    results: List[Dict[str, Any]] = existing[:]  # keep what we already have

    pbar = tqdm(total=len(samples), desc="Extracting labels", unit="sample")
    # If resuming, reflect progress
    if done_idx:
        pbar.update(len(done_idx))

    writes_since_flush = 0

    for i, sample in enumerate(samples):
        if i in done_idx:
            continue

        label = sample.get("label")
        response = extract_response(sample)
        pred = extract_answer(response)

        rec = {
            "idx": i,
            "label": label,
            "extracted_answer": pred,
        }
        results.append(rec)
        done_idx.add(i)

        writes_since_flush += 1
        if writes_since_flush >= flush_every:
            _atomic_write_json(result_path, results)
            writes_since_flush = 0

        pbar.update(1)

    # Final flush
    _atomic_write_json(result_path, results)
    pbar.close()

    return results

def _normalize_label(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().upper()
    if not s:
        return None
    # If label might be like "(B)" or "B)" etc, normalize
    for ch in s:
        if ch in VALID_CHOICES:
            return ch
    return None

def evaluate_predictions(
    predictions: List[Dict[str, Any]],
    total_samples: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate extracted predictions:
    - accuracy on parsed subset (excluding X / invalid)
    - coverage (parsed subset / total predictions)
    - overall accuracy if counting X as wrong
    - basic confusion-like stats (optional)
    """
    n_total = len(predictions)
    if total_samples is None:
        total_samples = n_total

    correct_parsed = 0
    parsed = 0
    invalid = 0
    x_count = 0

    # Optional per-label tracking
    per_label = {c: {"tp": 0, "count": 0} for c in VALID_CHOICES}

    for rec in predictions:
        gold = _normalize_label(rec.get("label"))
        pred = _normalize_label(rec.get("extracted_answer"))

        if gold is None:
            continue  # skip if no gold label
        
        per_label[gold]["count"] += 1

        raw_pred = str(rec.get("extracted_answer", "")).strip().upper()
        if raw_pred == "X":
            x_count += 1
            continue

        if pred is None:
            invalid += 1
            continue

        parsed += 1
        if pred == gold:
            correct_parsed += 1
            per_label[gold]["tp"] += 1

    accuracy_parsed = correct_parsed / parsed if parsed else 0.0
    coverage = parsed / n_total if n_total else 0.0

    # If you want "overall accuracy counting X/invalid as wrong"
    # (still excludes records missing gold label)
    overall_correct = correct_parsed  # only parsed-correct are correct
    overall_accuracy = overall_correct / n_total if n_total > 0 else 0.0

    summary = {
        "total_predictions": n_total,
        "total_samples_expected": total_samples,
        "parsed_predictions": parsed,
        "coverage_parsed_over_predictions": coverage,
        "x_count": x_count,
        "invalid_pred_count": invalid,
        "correct_parsed": correct_parsed,
        "accuracy_on_parsed": accuracy_parsed,
        "overall_accuracy_counting_X_invalid_wrong": overall_accuracy,
        "per_label": {
            c: {
                "support": per_label[c]["count"],
                "accuracy": (per_label[c]["tp"] / per_label[c]["count"]) if per_label[c]["count"] else None,
            }
            for c in sorted(VALID_CHOICES)
        },
    }

    if verbose:
        print("\n===== Evaluation =====")
        print(f"Total predictions loaded: {n_total}")
        if total_samples is not None:
            print(f"Expected samples (from dataset file): {total_samples}")
        print(f"Parsed predictions (A-E only): {parsed}")
        print(f"Coverage (parsed / predictions): {coverage:.3f}")
        print(f"X (unparseable) count: {x_count}")
        print(f"Invalid pred count (not A-E/X): {invalid}")
        print(f"Accuracy on parsed subset: {accuracy_parsed:.4f} ({correct_parsed}/{parsed})")
        print(f"Overall accuracy (X/invalid as wrong): {overall_accuracy:.4f} ({overall_correct}/{n_total})")

    return summary


if __name__ == "__main__":
    # CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str)
    parser.add_argument("--flush_every", type=int, default=1, help="Write to JSON every N new predictions")
    parser.add_argument("--mode", type=str, choices=["extract", "eval", "both"], default="both", help="Run extraction, evaluation, or both")
    parser.add_argument("--num_samples", type=int, default=None, help="(Optional) Select random N samples for evaluation")
    args = parser.parse_args()

    # Response file + result paths for extraction
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    files = args.file_path.split("/")
    file_name = files[-1].replace('.json', '')  # strip .json if present
    result_dir = os.path.join(project_root, "results", "preds")
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, f"{file_name}.predictions.json")
    
    # Evaluation summary path
    eval_dir = os.path.join(project_root, "results", "metrics")
    os.makedirs(eval_dir, exist_ok=True)
    eval_path = os.path.join(eval_dir, f"{file_name}.eval.json")
    
    # Load samples
    samples: List[dict] = []
    if os.path.exists(args.file_path):
        with open(args.file_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

    # ----- PART 1: Extract and save predictions ----- #
    if args.mode in {"extract", "both"}:
        if not samples:
            raise FileNotFoundError(f"Cannot extract: dataset response file not found: {args.file_path}")
        predictions = extract_and_save_predictions(samples, result_path=result_path, flush_every=args.flush_every)
        print(f"[INFO] Saved predictions to: {result_path} (total records: {len(predictions)})")

    # ----- PART 2: Evaluate predictions ----- #
    if args.mode in {"eval", "both"}:
        preds = _load_existing_results(result_path)
        if not preds:
            raise FileNotFoundError(f"Cannot evaluate: predictions file not found/empty: {result_path}")
        if args.num_samples is not None:
            preds = random.sample(preds, args.num_samples)
        summary = evaluate_predictions(preds, total_samples=len(samples) if samples else None, verbose=True)

        _atomic_write_json(eval_path, summary)
        print(f"[INFO] Saved eval summary to: {eval_path}")