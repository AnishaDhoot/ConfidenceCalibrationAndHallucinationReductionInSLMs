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

# The old eval used exact string match for accuracy, which is basically always
# 0 for generative QA (the model paraphrases, gold answer is worded differently).
# That also broke ECE/Brier/AUROC since they all read the same `correct` flag.
# Here correctness is refusal-aware + F1-based instead. EM and F1 are still
# printed so you can see the gap.

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

try:
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


p = argparse.ArgumentParser()
p.add_argument("--normal-adapter", default="phi3-vit-lora-expc-idk0")
p.add_argument("--idk-adapter", default="phi3-vit-lora-expc-idk15")
p.add_argument("--test", default="expc_idk15_test.jsonl")
p.add_argument("--f1-threshold", type=float, default=0.5)
p.add_argument("--skip-normal", action="store_true")
p.add_argument("--skip-idk", action="store_true")
args = p.parse_args()

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
TEST_PATH = args.test
MAX_NEW_TOKENS = 64
N_BINS = 10
F1_THRESHOLD = args.f1_threshold
RESULTS_DIR = "results"

# new cache name so we never mix with the old exact-match cache
BASE_CACHE = f"base_eval_cache_soft_{Path(TEST_PATH).stem}.json"
CALIB_EXPORT = f"{RESULTS_DIR}/calibration_export.json"

# what counts as the model refusing / abstaining
IDK_RE = re.compile(
    r"don't know|I don't have|cannot verify|cannot check|cannot submit|"
    r"cannot see|I cannot|not in my knowledge|do not have access|"
    r"don't have enough information",
    re.IGNORECASE,
)

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


def normalize(s):
    s = s.lower()
    s = re.sub(f"[{re.escape(string.punctuation)}]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def exact_match(pred, ref):
    return int(normalize(pred) == normalize(ref))


def token_f1(pred, ref):
    p_toks, r_toks = normalize(pred).split(), normalize(ref).split()
    if not p_toks or not r_toks:
        return float(p_toks == r_toks)
    same = sum(min(p_toks.count(t), r_toks.count(t)) for t in set(p_toks))
    if same == 0:
        return 0.0
    prec, rec = same / len(p_toks), same / len(r_toks)
    return 2 * prec * rec / (prec + rec)


def refused(text):
    return bool(IDK_RE.search(text))


def is_correct(pred, ref, is_idk):
    # IDK question -> right only if the model also backs off
    if is_idk:
        return int(refused(pred))
    # answerable question -> refusing here is a false refusal, so wrong
    if refused(pred):
        return 0
    return int(token_f1(pred, ref) >= F1_THRESHOLD)


def load_eval_set(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            q, a = d.get("question", "").strip(), d.get("answer", "").strip()
            if q and a:
                rows.append({"question": q, "answer": a, "is_idk": refused(a)})
    return rows


@torch.no_grad()
def run_model(model, tokenizer, eval_set, label):
    out = []
    model.eval()
    for ex in tqdm(eval_set, desc=f"Evaluating {label}", unit="ex"):
        prompt = f"<|user|>\n{ex['question']}<|end|>\n<|assistant|>\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        gen = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            output_scores=True, return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        new_tokens = gen.sequences[0][inputs["input_ids"].shape[1]:]
        pred = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # length-normalized sequence probability as the confidence score
        logprobs = [F.log_softmax(s[0], dim=-1)[t].item()
                    for s, t in zip(gen.scores, new_tokens)]
        conf = float(torch.tensor(logprobs).mean().exp()) if logprobs else 0.0

        # teacher-forced NLL of the reference answer
        ref_ids = tokenizer(ex["answer"], return_tensors="pt",
                            add_special_tokens=False).input_ids.to(model.device)
        full = torch.cat([inputs["input_ids"], ref_ids], dim=1)
        labels = full.clone()
        labels[:, :inputs["input_ids"].shape[1]] = -100
        nll = model(input_ids=full, labels=labels).loss.item()

        out.append({
            "question": ex["question"], "reference": ex["answer"], "prediction": pred,
            "confidence": conf, "correct": is_correct(pred, ex["answer"], ex["is_idk"]),
            "exact_match": exact_match(pred, ex["answer"]),
            "f1": token_f1(pred, ex["answer"]), "nll": nll, "is_idk": ex["is_idk"],
        })
    return out


def ece(rows, n_bins=N_BINS):
    if not rows:
        return 0.0
    bins = [[] for _ in range(n_bins)]
    for r in rows:
        bins[min(int(r["confidence"] * n_bins), n_bins - 1)].append(r)
    total = len(rows)
    score = 0.0
    for b in bins:
        if b:
            avg_conf = sum(r["confidence"] for r in b) / len(b)
            avg_acc = sum(r["correct"] for r in b) / len(b)
            score += len(b) / total * abs(avg_conf - avg_acc)
    return score


def reliability_bins(rows, n_bins=N_BINS):
    bins = [[] for _ in range(n_bins)]
    for r in rows:
        bins[min(int(r["confidence"] * n_bins), n_bins - 1)].append(r)
    out = []
    for b in bins:
        if b:
            out.append((sum(r["confidence"] for r in b) / len(b),
                        sum(r["correct"] for r in b) / len(b), len(b)))
        else:
            out.append((None, None, 0))
    return out


def brier(rows):
    if not rows:
        return 0.0
    return sum((r["confidence"] - r["correct"]) ** 2 for r in rows) / len(rows)


def auroc(rows):
    if not (HAS_SKLEARN and rows):
        return None
    y = [r["correct"] for r in rows]
    if len(set(y)) < 2:
        return None
    return float(roc_auc_score(y, [r["confidence"] for r in rows]))


def summarize(rows, label):
    n = len(rows)
    idk = [r for r in rows if r["is_idk"]]
    ans = [r for r in rows if not r["is_idk"]]
    return {
        "label": label,
        "accuracy": sum(r["correct"] for r in rows) / n,
        "exact_match": sum(r["exact_match"] for r in rows) / n,
        "token_f1": sum(r["f1"] for r in rows) / n,
        "ece": ece(rows), "brier": brier(rows),
        "nll": sum(r["nll"] for r in rows) / n, "auroc": auroc(rows),
        "n_idk": len(idk), "n_answerable": len(ans),
        "idk_accuracy": (sum(r["correct"] for r in idk) / len(idk)) if idk else None,
        "answerable_accuracy": (sum(r["correct"] for r in ans) / len(ans)) if ans else None,
        "reliability_bins": reliability_bins(rows),
    }


def load_base():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb, device_map={"": 0})
    return m, tok


def load_adapter(path):
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb, device_map={"": 0})
    return PeftModel.from_pretrained(base, path), tok


def free(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def print_table(summaries):
    metrics = [("Accuracy (semantic)", "accuracy"), ("Exact Match", "exact_match"),
               ("Token F1", "token_f1"), ("ECE", "ece"), ("Brier", "brier"),
               ("NLL", "nll"), ("AUROC", "auroc")]
    print("=" * 92)
    print(f"  Comparison (F1>={F1_THRESHOLD}): " + " vs ".join(s["label"] for s in summaries))
    print("=" * 92)
    print(f"{'Metric':<22}" + "".join(f"{s['label']:<20}" for s in summaries))
    print("-" * 92)
    for name, key in metrics:
        row = f"{name:<22}"
        for s in summaries:
            v = s[key]
            row += f"{v:<20.3f}" if isinstance(v, float) else f"{'n/a':<20}"
        print(row)
    print("-" * 92)
    for s in summaries:
        ia = f"{s['idk_accuracy']:.3f}" if s["idk_accuracy"] is not None else "n/a"
        aa = f"{s['answerable_accuracy']:.3f}" if s["answerable_accuracy"] is not None else "n/a"
        print(f"  {s['label']:<20} IDK acc: {ia} (n={s['n_idk']})   "
              f"Answerable acc: {aa} (n={s['n_answerable']})")
    print("=" * 92)


def save_md(summaries):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    metrics = [("Accuracy (semantic)", "accuracy"), ("Exact Match", "exact_match"),
               ("Token F1", "token_f1"), ("ECE", "ece"), ("Brier", "brier"),
               ("NLL", "nll"), ("AUROC", "auroc")]
    lines = ["# Experiment comparison (semantic correctness)", "",
             f"IDK rows count as correct when the model refuses; answerable rows "
             f"when it answers with token-F1 >= {F1_THRESHOLD}.", "",
             "| Metric | " + " | ".join(s["label"] for s in summaries) + " |",
             "|---|" + "|".join(["---"] * len(summaries)) + "|"]
    for name, key in metrics:
        cells = [f"{s[key]:.3f}" if isinstance(s[key], float) else "n/a" for s in summaries]
        lines.append("| " + name + " | " + " | ".join(cells) + " |")
    with open(f"{RESULTS_DIR}/comparison_fixed.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {RESULTS_DIR}/comparison_fixed.md")


def main():
    eval_set = load_eval_set(TEST_PATH)
    n_idk = sum(e["is_idk"] for e in eval_set)
    print(f"{len(eval_set)} examples ({n_idk} IDK, {len(eval_set) - n_idk} answerable)\n")

    summaries, calib = [], {}

    # base model is slow, so cache it per test file
    if os.path.exists(BASE_CACHE):
        print(f"Using cached base results ({BASE_CACHE})")
        with open(BASE_CACHE, encoding="utf-8") as f:
            cached = json.load(f)
        summaries.append(cached["summary"])
        calib["Base"] = cached["calib"]
    else:
        print("Loading base model...")
        m, tok = load_base()
        res = run_model(m, tok, eval_set, "base")
        free(m)
        base_summary = summarize(res, "Base")
        calib["Base"] = {"confidences": [r["confidence"] for r in res],
                         "corrects": [r["correct"] for r in res]}
        with open(BASE_CACHE, "w", encoding="utf-8") as f:
            json.dump({"summary": base_summary, "calib": calib["Base"]}, f, indent=2)
        summaries.append(base_summary)

    if not args.skip_normal:
        print(f"Loading normal fine-tune ({args.normal_adapter})...")
        m, tok = load_adapter(args.normal_adapter)
        res = run_model(m, tok, eval_set, "normal-ft")
        free(m)
        summaries.append(summarize(res, "Normal FT"))
        calib["Normal FT"] = {"confidences": [r["confidence"] for r in res],
                              "corrects": [r["correct"] for r in res]}

    if not args.skip_idk:
        print(f"Loading IDK fine-tune ({args.idk_adapter})...")
        m, tok = load_adapter(args.idk_adapter)
        res = run_model(m, tok, eval_set, "idk-ft")
        free(m)
        summaries.append(summarize(res, "IDK FT"))
        calib["IDK FT"] = {"confidences": [r["confidence"] for r in res],
                           "corrects": [r["correct"] for r in res]}

    print_table(summaries)
    save_md(summaries)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(CALIB_EXPORT, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2)
    print(f"Wrote {CALIB_EXPORT} ({', '.join(calib)})")


if __name__ == "__main__":
    main()