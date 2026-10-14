"""
Refusal-behavior comparison across base model, a "normal" fine-tune (no
controlled IDK ratio), and an IDK-ratio-controlled fine-tune. Measures
whether each model refuses the RIGHT questions, not just whether it's
accurate overall -- a model can have decent calibration numbers while still
confidently answering questions it should have refused, or refusing ones
it actually knew.

Definitions:
  - IDK precision = of everything the model refused, how much SHOULD have been refused
  - IDK recall    = of everything that SHOULD have been refused, how much was
  - IDK F1        = harmonic mean of the two
  - False refusal (FP) = refused a question it should have answered (overcautious)
  - Missed IDK (FN)    = confidently answered something it should have refused (hallucination)
  - Accuracy on answerable = of non-IDK questions, word-overlap with reference >= 0.4

Uses the SAME IDK detection pattern as eval_compare.py and prepare_expc_data.py,
so all three scripts agree on what counts as an abstention example.

USAGE:
    python compare_idk_stats.py --test expc_idk15_test.jsonl
    python compare_idk_stats.py --test expc_idk15_test.jsonl --skip-normal
    python compare_idk_stats.py --test expc_idk15_test.jsonl --skip-idk
"""

import json
import re
import csv
import argparse
import gc
import os
from pathlib import Path
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

try:
    import matplotlib.pyplot as plt
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
MAX_NEW_TOKENS = 100
RESULTS_DIR = "results"

# Same pattern as eval_compare.py / prepare_expc_data.py -- keep these three in sync.
IDK_PATTERN = re.compile(
    r"don't know|I don't|cannot verify|cannot check|cannot submit|"
    r"cannot see|real-time|moment to moment|I cannot|not in my knowledge",
    re.IGNORECASE,
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


def load_test_set(path):
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            q, a = d["question"].strip(), d["answer"].strip()
            is_idk_gt = bool(IDK_PATTERN.search(a))
            entries.append({"question": q, "ground_truth": a, "is_idk_gt": is_idk_gt})
    return entries


def load_model(adapter_path=None):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map={"": 0}
    )
    model = PeftModel.from_pretrained(base, adapter_path) if adapter_path else base
    return model, tokenizer


def free_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


@torch.no_grad()
def run_eval(model, tokenizer, test_entries, label):
    correct = 0
    refused = 0
    tp_idk = fp_idk = fn_idk = 0
    per_example = []

    model.eval()
    for entry in tqdm(test_entries, desc=f"Evaluating {label}", unit="ex"):
        prompt = f"<|user|>\n{entry['question']}<|end|>\n<|assistant|>\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_tokens = out[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        is_idk_pred = bool(IDK_PATTERN.search(response))

        if is_idk_pred:
            refused += 1
            if entry["is_idk_gt"]:
                tp_idk += 1
            else:
                fp_idk += 1
        else:
            if entry["is_idk_gt"]:
                fn_idk += 1
            else:
                gt_words = set(entry["ground_truth"].lower().split())
                pred_words = set(response.lower().split())
                overlap = len(gt_words & pred_words) / max(len(gt_words), 1)
                if overlap > 0.4:
                    correct += 1

        per_example.append({
            "question": entry["question"], "ground_truth": entry["ground_truth"],
            "generated": response, "is_idk_gt": entry["is_idk_gt"], "is_idk_pred": is_idk_pred,
        })

    n_answerable = sum(1 for e in test_entries if not e["is_idk_gt"])
    n_total = len(test_entries)

    precision = tp_idk / (tp_idk + fp_idk) if (tp_idk + fp_idk) > 0 else 0
    recall = tp_idk / (tp_idk + fn_idk) if (tp_idk + fn_idk) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    acc_on_answerable = correct / n_answerable if n_answerable > 0 else 0
    refusal_rate = refused / n_total if n_total > 0 else 0

    print(f"\n  -- {label.upper()} --------------------------------")
    print(f"  Refusal rate          : {refusal_rate:.3f} ({refused}/{n_total})")
    print(f"  IDK precision         : {precision:.3f}  (of refusals, how many were correct)")
    print(f"  IDK recall            : {recall:.3f}  (of IDK questions, how many were refused)")
    print(f"  IDK F1                : {f1:.3f}")
    print(f"  False refusals (FP)   : {fp_idk}  (refused answerable questions)")
    print(f"  Missed IDK (FN)       : {fn_idk}  (answered IDK questions confidently)")
    print(f"  Acc on answerable     : {acc_on_answerable:.3f}  (word-overlap >= 0.4)")

    return {
        "label": label, "refusal_rate": refusal_rate, "precision": precision,
        "recall": recall, "f1": f1, "fp_idk": fp_idk, "fn_idk": fn_idk,
        "acc_on_answerable": acc_on_answerable, "per_example": per_example,
    }


def save_error_examples(result, path):
    fps = [e for e in result["per_example"] if e["is_idk_pred"] and not e["is_idk_gt"]]
    fns = [e for e in result["per_example"] if not e["is_idk_pred"] and e["is_idk_gt"]]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["error_type", "question", "ground_truth", "generated"])
        writer.writeheader()
        for e in fps:
            writer.writerow({"error_type": "false_refusal_FP", "question": e["question"],
                              "ground_truth": e["ground_truth"], "generated": e["generated"]})
        for e in fns:
            writer.writerow({"error_type": "missed_idk_FN", "question": e["question"],
                              "ground_truth": e["ground_truth"], "generated": e["generated"]})
    print(f"Saved {len(fps)} false-refusal and {len(fns)} missed-IDK examples to {path}")


def plot_refusal_metrics(results, out_path):
    if not PLOTTING_AVAILABLE:
        print("matplotlib not installed, skipping refusal metric chart (pip install matplotlib)")
        return
    metrics = [("refusal_rate", "Refusal rate"), ("precision", "IDK precision"),
               ("recall", "IDK recall"), ("f1", "IDK F1"), ("acc_on_answerable", "Acc on answerable")]
    x = np.arange(len(metrics))
    width = 0.8 / len(results)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#7f7f7f", "#1f77b4", "#2ca02c"]
    for i, r in enumerate(results):
        vals = [r[key] for key, _ in metrics]
        ax.bar(x + i * width, vals, width, label=r["label"], color=colors[i % len(colors)])
    ax.set_xticks(x + width * (len(results) - 1) / 2)
    ax.set_xticklabels([name for _, name in metrics], rotation=15)
    ax.set_ylim(0, 1)
    ax.set_title("Refusal behavior comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved refusal metric chart to {out_path}")

def plot_error_breakdown(results, out_path):
    if not PLOTTING_AVAILABLE:
        print("matplotlib not installed, skipping error breakdown chart")
        return
    x = np.arange(len(results))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fp_vals = [r["fp_idk"] for r in results]
    fn_vals = [r["fn_idk"] for r in results]
    ax.bar(x - width / 2, fp_vals, width, label="False refusals (FP)", color="#ff7f0e")
    ax.bar(x + width / 2, fn_vals, width, label="Missed IDK / hallucinated (FN)", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in results], rotation=15)
    ax.set_ylabel("Count")
    ax.set_title("Error breakdown: overcautious vs hallucinated")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved error breakdown chart to {out_path}")


def print_pairwise_comparisons(results):
    """Three explicit pairwise comparisons, since that's usually what a report
    actually needs: does fine-tuning help at all, does the IDK ratio help
    specifically, and normal-FT vs IDK-FT head to head."""
    by_label = {r["label"]: r for r in results}
    pairs = []
    if "base" in by_label and "normal-ft" in by_label:
        pairs.append(("Base vs Normal FT", by_label["base"], by_label["normal-ft"]))
    if "base" in by_label and "idk-ft" in by_label:
        pairs.append(("Base vs IDK FT", by_label["base"], by_label["idk-ft"]))
    if "normal-ft" in by_label and "idk-ft" in by_label:
        pairs.append(("Normal FT vs IDK FT", by_label["normal-ft"], by_label["idk-ft"]))

    metric_keys = [
        ("refusal_rate", None), ("precision", True), ("recall", True),
        ("f1", True), ("acc_on_answerable", True),
    ]

    for title, a, b in pairs:
        print("\n" + "=" * 70)
        print(f"  {title}")
        print("=" * 70)
        for key, higher_better in metric_keys:
            va, vb = a[key], b[key]
            delta = vb - va
            if higher_better is None:
                tag = ""
            else:
                improved = (delta > 0) if higher_better else (delta < 0)
                tag = "  (better)" if improved else "  (worse)"
            print(f"  {key:<20}: {va:.3f} -> {vb:.3f}  ({delta:+.3f}){tag}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal-adapter", type=str, default="phi3-vit-lora-expc",
                         help="Path to the 'normal' fine-tuned adapter (no controlled IDK ratio)")
    parser.add_argument("--idk-adapter", type=str, default="phi3-vit-lora-expc-idk15",
                         help="Path to the IDK-ratio-controlled fine-tuned adapter")
    parser.add_argument("--test", type=str, required=True, help="Path to test JSONL (question/answer)")
    parser.add_argument("--skip-normal", action="store_true", help="Only evaluate base + idk-adapter")
    parser.add_argument("--skip-idk", action="store_true", help="Only evaluate base + normal-adapter")
    args = parser.parse_args()

    test_entries = load_test_set(args.test)
    n_idk = sum(1 for e in test_entries if e["is_idk_gt"])
    print(f"Test entries: {len(test_entries)}  (IDK: {n_idk}, Answerable: {len(test_entries) - n_idk})")

    results = []

    # Base model results are cached, keyed to the test file -- so a crash
    # partway through (e.g. a bad adapter path) doesn't cost you the ~40+
    # minutes the base-model pass takes on a re-run.
    base_cache_path = f"base_idk_cache_{Path(args.test).stem}.json"
    if os.path.exists(base_cache_path):
        print(f"Found cached base model results at {base_cache_path}, reusing them "
              f"(delete this file to force a fresh base-model run).")
        with open(base_cache_path, encoding="utf-8") as f:
            cached = json.load(f)
            cached.pop("per_example", None)  # per_example isn't cached (too large); not needed for summary
            cached["per_example"] = []
        base_result = cached
    else:
        print("\nLoading BASE model...")
        model, tokenizer = load_model(adapter_path=None)
        base_result = run_eval(model, tokenizer, test_entries, "base")
        free_model(model)
        with open(base_cache_path, "w", encoding="utf-8") as f:
            cacheable = {k: v for k, v in base_result.items() if k != "per_example"}
            json.dump(cacheable, f, indent=2)
    results.append(base_result)

    if not args.skip_normal:
        print(f"\nLoading NORMAL fine-tuned model ({args.normal_adapter})...")
        model, tokenizer = load_model(adapter_path=args.normal_adapter)
        normal_result = run_eval(model, tokenizer, test_entries, "normal-ft")
        free_model(model)
        save_error_examples(normal_result, "error_examples_normal_ft.csv")
        results.append(normal_result)

    if not args.skip_idk:
        print(f"\nLoading IDK-trained fine-tuned model ({args.idk_adapter})...")
        model, tokenizer = load_model(adapter_path=args.idk_adapter)
        idk_result = run_eval(model, tokenizer, test_entries, "idk-ft")
        free_model(model)
        save_error_examples(idk_result, "error_examples_idk_ft.csv")
        results.append(idk_result)

    print("\n" + "=" * 70)
    print("  SUMMARY: " + " -> ".join(r["label"] for r in results))
    print("=" * 70)
    for key in ["refusal_rate", "precision", "recall", "f1", "acc_on_answerable"]:
        vals = " -> ".join(f"{r[key]:.3f}" for r in results)
        print(f"  {key:<20}: {vals}")

    print_pairwise_comparisons(results)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_refusal_metrics(results, f"{RESULTS_DIR}/refusal_metrics.png")
    plot_error_breakdown(results, f"{RESULTS_DIR}/error_breakdown.png")


if __name__ == "__main__":
    main()
