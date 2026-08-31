"""
Compares base model vs UP TO TWO fine-tuned models (e.g. a "normal" fine-tune
with no controlled IDK ratio, and an IDK-ratio-controlled one) on the same
held-out eval set: Accuracy, Exact Match, Token F1, ECE, Brier Score, NLL.
Also breaks each metric down separately for IDK-labeled vs answerable rows,
and plots everything.

Loads ONE model at a time (evaluate, free memory, then next) -- important on
a 4GB card, since loading two 4-bit models simultaneously will OOM.

Metric definitions:
  - Correctness (for ECE/Brier) = exact match after normalization (lowercase,
    strip punctuation/whitespace). Strict binary label.
  - Confidence = mean per-token probability of the model's own generated
    tokens (geometric mean via averaging log-probs then exponentiating).
  - NLL = teacher-forced negative log-likelihood of the REFERENCE answer
    given the prompt (not the generated answer).
  - IDK rows = reference answer matches IDK_PATTERN (the SAME pattern used
    in prepare_expc_data.py / compare_idk_stats.py, so all three scripts
    agree on what counts as an abstention example). Earlier versions of this
    script used a stale "starts with NOT FOUND" check left over from a much
    older dataset format -- that no longer matches how IDK rows are phrased
    in combined_qa_finetuning.csv ("I don't know...", "I cannot verify...",
    etc.), which silently meant nothing was ever being flagged as IDK here.
    Fixed below.

BEFORE RUNNING:
    pip install matplotlib
"""

import json
import re
import string
import gc
import os
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# ---------- CLI ----------

parser = argparse.ArgumentParser()
parser.add_argument("--normal-adapter", type=str, default="phi3-vit-lora-expc",
                     help="Path to the 'normal' fine-tuned adapter (no controlled IDK ratio)")
parser.add_argument("--idk-adapter", type=str, default="phi3-vit-lora-expc-idk15",
                     help="Path to the IDK-ratio-controlled fine-tuned adapter")
parser.add_argument("--test", type=str, default="expc_idk15_test.jsonl",
                     help="Path to eval JSONL (question/answer)")
parser.add_argument("--skip-normal", action="store_true", help="Only evaluate base + idk-adapter")
parser.add_argument("--skip-idk", action="store_true", help="Only evaluate base + normal-adapter")
cli_args = parser.parse_args()

# ---------- CONFIG ----------

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
EVAL_DATA_PATH = cli_args.test
MAX_NEW_TOKENS = 64
N_BINS = 10

# Cache filename is keyed to the specific eval file, so switching test sets
# never silently reuses base-model results computed on a different set of questions.
BASE_CACHE_PATH = f"base_eval_cache_{Path(EVAL_DATA_PATH).stem}.json"
RESULTS_DIR = "results"

# Per-example confidence/correctness arrays, keyed by label, written for
# run_calibration_utils.py -> calibration_eval_utils.py to consume (AUROC +
# reliability diagram). Only populated for models evaluated fresh in THIS
# run, since the base-model cache only stores the aggregate summary, not
# per-example arrays -- see the NOTE printed below when that happens.
CALIBRATION_EXPORT_PATH = f"{RESULTS_DIR}/calibration_export.json"

# Same IDK detection pattern used across prepare_expc_data.py and
# compare_idk_stats.py, so "what counts as an IDK row" agrees everywhere.
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

# ---------- TEXT NORMALIZATION / SCORING ----------

def normalize(s):
    s = s.lower()
    s = re.sub(f"[{re.escape(string.punctuation)}]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def exact_match(pred, ref):
    return int(normalize(pred) == normalize(ref))

def token_f1(pred, ref):
    pred_toks = normalize(pred).split()
    ref_toks = normalize(ref).split()
    if not pred_toks or not ref_toks:
        return float(pred_toks == ref_toks)
    common = {}
    for t in pred_toks:
        common[t] = min(pred_toks.count(t), ref_toks.count(t))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(ref_toks)
    return 2 * precision * recall / (precision + recall)

# ---------- LOAD DATA ----------

def load_eval_set(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            q, a = d.get("question", "").strip(), d.get("answer", "").strip()
            if not q or not a:
                continue
            records.append({
                "question": q,
                "answer": a,
                "is_idk": bool(IDK_PATTERN.search(a)),
            })
    return records

# ---------- CORE: RUN ONE MODEL OVER THE EVAL SET ----------

@torch.no_grad()
def evaluate_model(model, tokenizer, eval_set, label):
    results = []
    model.eval()
    for ex in tqdm(eval_set, desc=f"Evaluating {label}", unit="ex"):
        prompt = f"<|user|>\n{ex['question']}<|end|>\n<|assistant|>\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        gen_out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_tokens = gen_out.sequences[0][inputs["input_ids"].shape[1]:]
        pred_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        token_logprobs = []
        for step_scores, tok_id in zip(gen_out.scores, gen_tokens):
            logprob = F.log_softmax(step_scores[0], dim=-1)[tok_id]
            token_logprobs.append(logprob.item())
        confidence = float(torch.tensor(token_logprobs).mean().exp()) if token_logprobs else 0.0

        ref_ids = tokenizer(ex["answer"], return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        full_ids = torch.cat([inputs["input_ids"], ref_ids], dim=1)
        labels = full_ids.clone()
        labels[:, :inputs["input_ids"].shape[1]] = -100
        nll = model(input_ids=full_ids, labels=labels).loss.item()

        em = exact_match(pred_text, ex["answer"])
        f1 = token_f1(pred_text, ex["answer"])

        results.append({
            "question": ex["question"], "reference": ex["answer"], "prediction": pred_text,
            "confidence": confidence, "correct": em, "f1": f1, "nll": nll, "is_idk": ex["is_idk"],
        })
    return results

# ---------- AGGREGATE METRICS ----------

def compute_ece(results, n_bins=N_BINS):
    if not results:
        return 0.0
    bins = [[] for _ in range(n_bins)]
    for r in results:
        idx = min(int(r["confidence"] * n_bins), n_bins - 1)
        bins[idx].append(r)
    ece = 0.0
    n = len(results)
    for b in bins:
        if not b:
            continue
        avg_conf = sum(r["confidence"] for r in b) / len(b)
        avg_acc = sum(r["correct"] for r in b) / len(b)
        ece += (len(b) / n) * abs(avg_conf - avg_acc)
    return ece

def get_reliability_bins(results, n_bins=N_BINS):
    bins = [[] for _ in range(n_bins)]
    for r in results:
        idx = min(int(r["confidence"] * n_bins), n_bins - 1)
        bins[idx].append(r)
    out = []
    for b in bins:
        if not b:
            out.append((None, None, 0))
            continue
        avg_conf = sum(r["confidence"] for r in b) / len(b)
        avg_acc = sum(r["correct"] for r in b) / len(b)
        out.append((avg_conf, avg_acc, len(b)))
    return out

def compute_brier(results):
    if not results:
        return 0.0
    return sum((r["confidence"] - r["correct"]) ** 2 for r in results) / len(results)

def summarize(results, label):
    n = len(results)
    accuracy = sum(r["correct"] for r in results) / n
    f1 = sum(r["f1"] for r in results) / n
    ece = compute_ece(results)
    brier = compute_brier(results)
    nll = sum(r["nll"] for r in results) / n

    idk_rows = [r for r in results if r["is_idk"]]
    ans_rows = [r for r in results if not r["is_idk"]]

    return {
        "label": label, "accuracy": accuracy, "exact_match": accuracy,
        "token_f1": f1, "ece": ece, "brier": brier, "nll": nll,
        "n_idk": len(idk_rows), "n_answerable": len(ans_rows),
        "idk_accuracy": (sum(r["correct"] for r in idk_rows) / len(idk_rows)) if idk_rows else None,
        "answerable_accuracy": (sum(r["correct"] for r in ans_rows) / len(ans_rows)) if ans_rows else None,
        "idk_ece": compute_ece(idk_rows) if idk_rows else None,
        "answerable_ece": compute_ece(ans_rows) if ans_rows else None,
        "reliability_bins": get_reliability_bins(results),
    }

# ---------- MODEL LOADING ----------

def load_base_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map={"": 0}
    )
    return model, tokenizer

def load_finetuned_model(adapter_path):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map={"": 0}
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    return model, tokenizer

def free_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# ---------- PLOTTING ----------

def plot_reliability_diagram(summaries, out_path):
    if not PLOTTING_AVAILABLE:
        print("matplotlib not installed, skipping reliability diagram (pip install matplotlib)")
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    colors = ["#d62728", "#1f77b4", "#2ca02c"]
    for i, s in enumerate(summaries):
        xs, ys = [], []
        for avg_conf, avg_acc, count in s["reliability_bins"]:
            if avg_conf is not None:
                xs.append(avg_conf)
                ys.append(avg_acc)
        ax.plot(xs, ys, marker="o", label=s["label"], color=colors[i % len(colors)])
    ax.set_xlabel("Mean predicted confidence (per bin)")
    ax.set_ylabel("Mean accuracy (per bin)")
    ax.set_title("Reliability diagram: confidence vs actual accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved reliability diagram to {out_path}")

def plot_metric_bars(summaries, out_path):
    if not PLOTTING_AVAILABLE:
        print("matplotlib not installed, skipping metric bar chart")
        return
    metrics = [("accuracy", "Accuracy"), ("token_f1", "Token F1"),
               ("ece", "ECE (lower better)"), ("brier", "Brier (lower better)")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    for ax, (key, title) in zip(axes, metrics):
        vals = [s[key] for s in summaries]
        labels = [s["label"] for s in summaries]
        ax.bar(labels, vals, color=["#7f7f7f", "#1f77b4", "#2ca02c"][:len(summaries)])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved metric comparison bar chart to {out_path}")

def plot_idk_vs_answerable(summaries, out_path):
    if not PLOTTING_AVAILABLE:
        print("matplotlib not installed, skipping IDK-vs-answerable chart")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    import numpy as np
    x = np.arange(len(summaries))
    width = 0.35
    idk_vals = [s["idk_accuracy"] if s["idk_accuracy"] is not None else 0 for s in summaries]
    ans_vals = [s["answerable_accuracy"] if s["answerable_accuracy"] is not None else 0 for s in summaries]
    ax.bar(x - width / 2, idk_vals, width, label="Accuracy on IDK rows")
    ax.bar(x + width / 2, ans_vals, width, label="Accuracy on answerable rows")
    ax.set_xticks(x)
    ax.set_xticklabels([s["label"] for s in summaries], rotation=15)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy split: IDK rows vs answerable rows")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved IDK-vs-answerable chart to {out_path}")

# ---------- REPORTING ----------

def print_comparison(summaries):
    metric_defs = [
        ("Accuracy", "accuracy", True), ("Token F1", "token_f1", True),
        ("ECE", "ece", False), ("Brier Score", "brier", False), ("NLL", "nll", False),
    ]
    base = summaries[0]
    print("=" * 90)
    print("  EXPERIMENT COMPARISON: " + " vs ".join(s["label"] for s in summaries))
    print("=" * 90)
    header = f"{'Metric':<16}" + "".join(f"{s['label']:<20}" for s in summaries)
    print(header)
    print("-" * 90)
    for name, key, higher_better in metric_defs:
        row = f"{name:<16}"
        for s in summaries:
            row += f"{s[key]:<20.3f}"
        print(row)
    print("-" * 90)
    print("  IDK-row / answerable-row accuracy breakdown:")
    for s in summaries:
        idk_a = f"{s['idk_accuracy']:.3f}" if s["idk_accuracy"] is not None else "n/a"
        ans_a = f"{s['answerable_accuracy']:.3f}" if s["answerable_accuracy"] is not None else "n/a"
        print(f"    {s['label']:<20} IDK acc: {idk_a} (n={s['n_idk']})   "
              f"Answerable acc: {ans_a} (n={s['n_answerable']})")
    print("=" * 90)

def export_calibration_data(calibration_data, path=CALIBRATION_EXPORT_PATH):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not calibration_data:
        print(f"NOTE: no per-example calibration data available this run "
              f"(likely every model was loaded from cache) -- "
              f"{path} not written. Delete the cache file(s) and re-run to "
              f"populate it.")
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(calibration_data, f, indent=2)
    print(f"Saved calibration export ({', '.join(calibration_data.keys())}) to {path}")


def save_comparison_md(summaries):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    metric_defs = [
        ("Accuracy", "accuracy"), ("Token F1", "token_f1"),
        ("ECE", "ece"), ("Brier Score", "brier"), ("NLL", "nll"),
    ]
    lines = ["# Experiment Comparison\n"]
    header = "| Metric | " + " | ".join(s["label"] for s in summaries) + " |"
    sep = "|---|" + "|".join(["---"] * len(summaries)) + "|"
    lines += [header, sep]
    for name, key in metric_defs:
        lines.append("| " + name + " | " + " | ".join(f"{s[key]:.3f}" for s in summaries) + " |")
    lines.append("\n## IDK vs answerable accuracy\n")
    lines.append("| Model | IDK accuracy (n) | Answerable accuracy (n) |")
    lines.append("|---|---|---|")
    for s in summaries:
        idk_a = f"{s['idk_accuracy']:.3f}" if s["idk_accuracy"] is not None else "n/a"
        ans_a = f"{s['answerable_accuracy']:.3f}" if s["answerable_accuracy"] is not None else "n/a"
        lines.append(f"| {s['label']} | {idk_a} ({s['n_idk']}) | {ans_a} ({s['n_answerable']}) |")
    with open(f"{RESULTS_DIR}/comparison.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved to {RESULTS_DIR}/comparison.md")

# ---------- MAIN ----------

def print_pairwise_comparisons(summaries):
    by_label = {s["label"]: s for s in summaries}
    pairs = []
    if "Base" in by_label and "Normal FT" in by_label:
        pairs.append(("Base vs Normal FT", by_label["Base"], by_label["Normal FT"]))
    if "Base" in by_label and "IDK FT" in by_label:
        pairs.append(("Base vs IDK FT", by_label["Base"], by_label["IDK FT"]))
    if "Normal FT" in by_label and "IDK FT" in by_label:
        pairs.append(("Normal FT vs IDK FT", by_label["Normal FT"], by_label["IDK FT"]))

    metric_defs = [
        ("Accuracy", "accuracy", True), ("Token F1", "token_f1", True),
        ("ECE", "ece", False), ("Brier Score", "brier", False), ("NLL", "nll", False),
    ]

    for title, a, b in pairs:
        print("\n" + "=" * 70)
        print(f"  {title}")
        print("=" * 70)
        for name, key, higher_better in metric_defs:
            va, vb = a[key], b[key]
            delta = vb - va
            improved = (delta > 0) if higher_better else (delta < 0)
            tag = "  (better)" if improved else "  (worse)"
            print(f"  {name:<14}: {va:.3f} -> {vb:.3f}  ({delta:+.3f}){tag}")
        idk_a = f"{a['idk_accuracy']:.3f}" if a["idk_accuracy"] is not None else "n/a"
        idk_b = f"{b['idk_accuracy']:.3f}" if b["idk_accuracy"] is not None else "n/a"
        ans_a = f"{a['answerable_accuracy']:.3f}" if a["answerable_accuracy"] is not None else "n/a"
        ans_b = f"{b['answerable_accuracy']:.3f}" if b["answerable_accuracy"] is not None else "n/a"
        print(f"  {'IDK-row acc':<14}: {idk_a} -> {idk_b}")
        print(f"  {'Answerable acc':<14}: {ans_a} -> {ans_b}")


def main():
    eval_set = load_eval_set(EVAL_DATA_PATH)
    n_idk = sum(1 for e in eval_set if e["is_idk"])
    print(f"Evaluating on {len(eval_set)} held-out examples "
          f"({n_idk} IDK-labeled, {len(eval_set) - n_idk} answerable).\n")

    summaries = []
    calibration_data = {}

    # ---- Base model (cached across runs) ----
    if os.path.exists(BASE_CACHE_PATH):
        print(f"Found cached base model results at {BASE_CACHE_PATH}, reusing them "
              f"(delete this file to force a fresh base-model run).")
        with open(BASE_CACHE_PATH, encoding="utf-8") as f:
            base_summary = json.load(f)
        print("NOTE: base model loaded from cache -- per-example confidence/"
              "correctness wasn't stored, so 'Base' will be missing from "
              "calibration_export.json this run. Delete the cache file above "
              "and re-run if you need Base in the AUROC/reliability comparison.")
    else:
        print("Loading BASE model...")
        model, tokenizer = load_base_model()
        base_results = evaluate_model(model, tokenizer, eval_set, "base")
        base_summary = summarize(base_results, "Base")
        free_model(model)
        print("Base model done, memory freed.\n")
        with open(BASE_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(base_summary, f, indent=2)
        calibration_data["Base"] = {
            "confidences": [r["confidence"] for r in base_results],
            "corrects": [r["correct"] for r in base_results],
        }
    summaries.append(base_summary)

    # ---- Normal fine-tune (no controlled IDK ratio) ----
    if not cli_args.skip_normal:
        print(f"Loading NORMAL fine-tuned model ({cli_args.normal_adapter})...")
        model, tokenizer = load_finetuned_model(cli_args.normal_adapter)
        normal_results = evaluate_model(model, tokenizer, eval_set, "normal-ft")
        normal_summary = summarize(normal_results, "Normal FT")
        free_model(model)
        print("Normal fine-tuned model done, memory freed.\n")
        summaries.append(normal_summary)
        calibration_data["Normal FT"] = {
            "confidences": [r["confidence"] for r in normal_results],
            "corrects": [r["correct"] for r in normal_results],
        }

    # ---- IDK-ratio-controlled fine-tune ----
    if not cli_args.skip_idk:
        print(f"Loading IDK-trained fine-tuned model ({cli_args.idk_adapter})...")
        model, tokenizer = load_finetuned_model(cli_args.idk_adapter)
        idk_results = evaluate_model(model, tokenizer, eval_set, "idk-ft")
        idk_summary = summarize(idk_results, "IDK FT")
        free_model(model)
        print("IDK-trained model done, memory freed.\n")
        summaries.append(idk_summary)
        calibration_data["IDK FT"] = {
            "confidences": [r["confidence"] for r in idk_results],
            "corrects": [r["correct"] for r in idk_results],
        }

    print_comparison(summaries)
    print_pairwise_comparisons(summaries)
    save_comparison_md(summaries)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_reliability_diagram(summaries, f"{RESULTS_DIR}/reliability_diagram.png")
    plot_metric_bars(summaries, f"{RESULTS_DIR}/metric_comparison.png")
    plot_idk_vs_answerable(summaries, f"{RESULTS_DIR}/idk_vs_answerable.png")

    export_calibration_data(calibration_data)

if __name__ == "__main__":
    main()
