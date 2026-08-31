"""
Calibration/hallucination-detection utilities for the VIT Q&A LoRA project.

This is the toolbox version of metrics that eval_compare.py and
compare_idk_stats.py don't already compute:
  1. AUROC for hallucination/error detection (plain, and an IDK-specific
     variant tailored to this project's refusal behavior)
  2. Semantic consistency (SelfCheckGPT-style, sampling-based)
  3. Reliability diagram + ECE -- both a standalone version AND a version
     that plugs directly into eval_compare.py's summarize() output, so you
     don't have to re-extract confidence/correctness arrays by hand.

Kept in sync with the rest of the pipeline:
  - MODEL_NAME, bnb_config, IDK_PATTERN are copy-pasted verbatim from
    eval_compare.py / compare_idk_stats.py / prepare_expc_data.py, so "what
    counts as an IDK row" agrees everywhere. If you change the pattern in
    one file, change it in all four.
  - load_base_model() / load_finetuned_model() / free_model() match
    eval_compare.py's model-loading functions exactly (one model on GPU at
    a time -- loading two 4-bit models simultaneously will OOM on a 4GB
    card). Adapter path defaults match: "phi3-vit-lora-expc" (normal FT),
    "phi3-vit-lora-expc-idk15" (IDK-ratio-controlled FT).
  - All plots save under RESULTS_DIR = "results", same as eval_compare.py
    and compare_idk_stats.py, instead of dumping PNGs in the working dir.

NOTE ON MAX_NEW_TOKENS: eval_compare.py uses 64, compare_idk_stats.py uses
100. This file defaults to 64 (matching eval_compare.py, since that's the
calibration-focused script this module is meant to extend) -- but the two
numbers being different means generations from the two scripts aren't
directly comparable length-wise. Worth reconciling if you start feeding
compare_idk_stats.py output into these functions.

Install:
    pip install scikit-learn sentence-transformers matplotlib numpy --break-system-packages

USAGE (wiring into eval_compare.py):
    Inside eval_compare.py's per-model loop, right after
    `results = evaluate_model(model, tokenizer, eval_set, label)`, you can
    now also call:
        from calibration_eval_utils import auroc_from_results, auroc_hallucination_detection
        auroc = auroc_from_results(results, label=label)
        halluc_auroc = auroc_hallucination_detection(results, label=label)
    `results` there is already a list of dicts with "confidence", "correct",
    "is_idk", "prediction" -- exactly the shape these helpers expect.
"""

import os
import re
import gc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ---------------------------------------------------------------------------
# Shared config -- copied verbatim from eval_compare.py / compare_idk_stats.py
# ---------------------------------------------------------------------------

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
MAX_NEW_TOKENS = 64
RESULTS_DIR = "results"

# Same pattern as eval_compare.py / compare_idk_stats.py / prepare_expc_data.py
# -- keep all four in sync.
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


# ---------------------------------------------------------------------------
# Model loading -- identical to eval_compare.py's load_base_model /
# load_finetuned_model / free_model, duplicated here so this module is
# usable standalone (eval_compare.py runs argparse at import time, so it
# can't be imported cleanly from another script).
# ---------------------------------------------------------------------------

def load_base_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map={"": 0}
    )
    return model, tokenizer


def load_finetuned_model(adapter_path="phi3-vit-lora-expc-idk15"):
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


def sample_from_model(model, tokenizer, question, temperature=0.8, max_new_tokens=MAX_NEW_TOKENS):
    """
    Draws ONE sample from `model` for `question`, using the same Phi-3 chat
    template as eval_compare.py / compare_idk_stats.py. This is the concrete
    sample_fn to hand to semantic_consistency_score() below -- no more
    placeholder call_my_model().

    Example:
        model, tokenizer = load_finetuned_model("phi3-vit-lora-expc-idk15")
        score, samples = semantic_consistency_score(
            "What is the last date to pay hostel fees?",
            sample_fn=lambda q: sample_from_model(model, tokenizer, q),
            n_samples=5,
        )
        free_model(model)
    """
    prompt = f"<|user|>\n{question}<|end|>\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# 1. AUROC for hallucination / error detection
# ---------------------------------------------------------------------------

def compute_auroc(confidence_scores, correctness_labels, plot=False,
                   title="Hallucination Detection ROC", save_path=None, label=None):
    """
    confidence_scores: list/array of floats in [0,1] (or any monotonic score),
        one per QA example. Higher = model is more confident it's correct.
    correctness_labels: list/array of 0/1, same length.
        1 = model's answer was correct, 0 = hallucinated/wrong.

    Returns: auroc (float), and optionally plots + saves the ROC curve into
    RESULTS_DIR (matching where eval_compare.py / compare_idk_stats.py save
    their charts).
    """
    confidence_scores = np.asarray(confidence_scores)
    correctness_labels = np.asarray(correctness_labels)

    if len(set(correctness_labels.tolist())) < 2:
        print(f"  [compute_auroc] Only one class present in correctness_labels"
              f"{f' for {label}' if label else ''} -- AUROC is undefined, skipping.")
        return None

    auroc = roc_auc_score(correctness_labels, confidence_scores)

    if plot:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        save_path = save_path or f"{RESULTS_DIR}/roc_curve{f'_{label}' if label else ''}.png"
        fpr, tpr, _ = roc_curve(correctness_labels, confidence_scores)
        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, label=f"AUROC = {auroc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{title}{f' ({label})' if label else ''}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Saved ROC curve to {save_path}")

    return float(auroc)


def auroc_from_results(results, plot=False, label=None):
    """
    Convenience wrapper for eval_compare.py's `results` list -- the list of
    dicts returned by evaluate_model(), each with "confidence" and "correct"
    keys already computed. Saves you from unpacking two parallel arrays by
    hand every time you want this metric.

    Example (inside eval_compare.py, right after evaluate_model() returns):
        auroc = auroc_from_results(results, plot=True, label="idk-ft")
    """
    confidences = [r["confidence"] for r in results]
    corrects = [r["correct"] for r in results]
    return compute_auroc(confidences, corrects, plot=plot,
                          title="Hallucination Detection ROC", label=label)


def auroc_hallucination_detection(results, plot=False, label=None):
    """
    Project-specific variant: restricted to IDK-labeled rows only (is_idk ==
    True), asking a more pointed question than plain AUROC does -- when the
    model DOESN'T refuse an IDK question (a "missed IDK" / hallucination, in
    compare_idk_stats.py's terms), was its confidence on that hallucinated
    answer at least lower than its confidence on the IDK rows it correctly
    refused? We want confidence to go DOWN on hallucinations, so this scores
    AUROC of (1 - confidence) predicting "hallucinated".

    Recomputes is_idk_pred from each row's "prediction" text via IDK_PATTERN,
    since eval_compare.py's results don't carry that field directly (only
    compare_idk_stats.py does).

    Returns None (with a message) if there are fewer than 2 IDK rows, or if
    the IDK rows are all-hallucinated / all-refused (AUROC undefined).
    """
    idk_rows = [r for r in results if r.get("is_idk")]
    if len(idk_rows) < 2:
        print(f"  [auroc_hallucination_detection] Fewer than 2 IDK rows"
              f"{f' for {label}' if label else ''} -- skipping.")
        return None

    hallucinated = []
    inv_confidence = []
    for r in idk_rows:
        is_idk_pred = bool(IDK_PATTERN.search(r["prediction"]))
        hallucinated.append(0 if is_idk_pred else 1)  # 1 = hallucinated (bad)
        inv_confidence.append(1.0 - r["confidence"])   # higher = "looks like a hallucination"

    return compute_auroc(inv_confidence, hallucinated, plot=plot,
                          title="Hallucination Detection AUROC (IDK rows only)", label=label)


def sequence_logprob_confidence(token_logprobs):
    """
    Convert a list of per-token log-probabilities for a generated answer
    into a single confidence score in [0,1]. eval_compare.py already does
    this inline (mean log-prob -> exp()); kept here as a standalone helper
    for anyone building a similar loop from scratch.

    token_logprobs: list of floats (natural log-probs of each generated token)
    """
    token_logprobs = np.asarray(token_logprobs)
    avg_logprob = token_logprobs.mean()          # length-normalized
    confidence = np.exp(avg_logprob)              # back to probability space
    return float(confidence)


def self_consistency_confidence(sampled_answers, similarity_fn):
    """
    Confidence via agreement across N sampled generations for the same question.

    sampled_answers: list[str], N samples generated at temperature > 0
    similarity_fn: function(str, str) -> float in [0,1], e.g. embedding cosine sim

    Returns: confidence score = mean pairwise similarity across all samples
    """
    n = len(sampled_answers)
    if n < 2:
        return 1.0
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(similarity_fn(sampled_answers[i], sampled_answers[j]))
    return float(np.mean(sims))


# ---------------------------------------------------------------------------
# 2. Semantic consistency metric (SelfCheckGPT-style)
# ---------------------------------------------------------------------------

def embedding_similarity_fn():
    """
    Returns a similarity_fn(str, str) -> float using sentence-transformers
    cosine similarity. Load the model once, reuse across calls.
    """
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer("all-MiniLM-L6-v2")

    def sim(a, b):
        emb = model.encode([a, b], convert_to_tensor=True)
        return float(util.cos_sim(emb[0], emb[1]))

    return sim


def semantic_consistency_score(question, sample_fn, n_samples=5, similarity_fn=None):
    """
    Full pipeline: sample the model n_samples times for `question`,
    then compute the mean pairwise semantic similarity across samples.

    question: str
    sample_fn: function(question:str) -> str. Use sample_from_model() above
        (bound to a loaded model/tokenizer via a lambda) rather than writing
        your own -- it already matches the project's Phi-3 chat template.
    n_samples: how many independent generations to draw
    similarity_fn: optional; defaults to embedding cosine similarity

    Returns: (consistency_score: float, sampled_answers: list[str])

    Example:
        model, tokenizer = load_finetuned_model("phi3-vit-lora-expc-idk15")
        score, samples = semantic_consistency_score(
            "What is the last date to pay hostel fees?",
            sample_fn=lambda q: sample_from_model(model, tokenizer, q, temperature=0.8),
            n_samples=5,
        )
        free_model(model)
    """
    if similarity_fn is None:
        similarity_fn = embedding_similarity_fn()

    samples = [sample_fn(question) for _ in range(n_samples)]
    score = self_consistency_confidence(samples, similarity_fn)
    return score, samples


def nli_consistency_fn():
    """
    Alternative, more rigorous similarity_fn using an NLI model to check
    contradiction between two generated answers instead of raw embedding
    similarity. Returns 1.0 if not contradicting, 0.0 if contradicting,
    0.5 if neutral. Slower but catches cases embeddings miss (paraphrases
    that are semantically opposite but lexically close) -- worth it for the
    IDK rows specifically, where "I don't know" vs a confident wrong answer
    can end up embedding-similar despite being semantically opposite.
    """
    from transformers import AutoTokenizer as _AutoTok, AutoModelForSequenceClassification

    tok = _AutoTok.from_pretrained("microsoft/deberta-large-mnli")
    model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli")
    model.eval()

    def sim(a, b):
        inputs = tok(a, b, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        # deberta-mnli label order: [contradiction, neutral, entailment]
        contradiction_prob = probs[0].item()
        return 1.0 - contradiction_prob

    return sim


# ---------------------------------------------------------------------------
# 3. Reliability diagram + ECE
# ---------------------------------------------------------------------------

def reliability_diagram(confidence_scores, correctness_labels, n_bins=10,
                         title="Reliability Diagram", save_path=None, label=None):
    """
    Standalone version: builds bins itself from raw confidence/correctness
    arrays. Use this if you're not already holding an eval_compare.py-style
    `results` list or `summary` dict -- otherwise prefer
    reliability_diagram_from_results() / plot_reliability_from_summary()
    below, which reuse work eval_compare.py already did.

    confidence_scores: array of floats in [0,1], the model's confidence per example
    correctness_labels: array of 0/1, whether the answer was correct
    n_bins: number of confidence bins (10, matching eval_compare.py's N_BINS)

    Plots predicted confidence vs actual accuracy per bin, saves under
    RESULTS_DIR, and returns the computed ECE.
    """
    confidence_scores = np.asarray(confidence_scores)
    correctness_labels = np.asarray(correctness_labels)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    bin_ids = np.digitize(confidence_scores, bin_edges[1:-1])

    for b in range(n_bins):
        mask = bin_ids == b
        bin_counts[b] = mask.sum()
        if mask.sum() > 0:
            bin_accs[b] = correctness_labels[mask].mean()
            bin_confs[b] = confidence_scores[mask].mean()

    total = bin_counts.sum()
    ece = np.sum((bin_counts / total) * np.abs(bin_accs - bin_confs)) if total else 0.0

    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = save_path or f"{RESULTS_DIR}/reliability_diagram_utils{f'_{label}' if label else ''}.png"

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(bin_centers, bin_accs, width=1 / n_bins, edgecolor="black",
           color="steelblue", alpha=0.8, label="Accuracy")
    ax.plot([0, 1], [0, 1], linestyle="--", color="red", label="Perfect calibration")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title}{f' ({label})' if label else ''}\nECE = {ece:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved reliability diagram to {save_path}")

    return float(ece)


def reliability_diagram_from_results(results, n_bins=10, label=None):
    """
    Same as reliability_diagram() above, but takes an eval_compare.py-style
    `results` list directly (list of dicts with "confidence" / "correct"
    keys) instead of two parallel arrays.
    """
    confidences = [r["confidence"] for r in results]
    corrects = [r["correct"] for r in results]
    return reliability_diagram(confidences, corrects, n_bins=n_bins, label=label)


def plot_reliability_from_summary(summary, save_path=None):
    """
    Cheapest option of the three: eval_compare.py's summarize() already
    computes "reliability_bins" (a list of (avg_conf, avg_acc, count) per
    bin) and "ece" -- this just plots what's already there instead of
    recomputing bins from scratch. Pass it the summary dict straight from
    summarize()/base_summary/normal_summary/idk_summary.

    Example:
        from eval_compare import summarize  # if refactored into an importable module
        summary = summarize(results, "IDK FT")
        plot_reliability_from_summary(summary)
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = save_path or f"{RESULTS_DIR}/reliability_diagram_{summary['label'].replace(' ', '_')}.png"

    xs, ys, sizes = [], [], []
    for avg_conf, avg_acc, count in summary["reliability_bins"]:
        if avg_conf is not None:
            xs.append(avg_conf)
            ys.append(avg_acc)
            sizes.append(count)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.scatter(xs, ys, s=[20 + 4 * c for c in sizes], color="#1f77b4", zorder=3)
    ax.plot(xs, ys, color="#1f77b4", alpha=0.6)
    ax.set_xlabel("Mean predicted confidence (per bin)")
    ax.set_ylabel("Mean accuracy (per bin)")
    ax.set_title(f"Reliability diagram: {summary['label']}\nECE = {summary['ece']:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved reliability diagram to {save_path}")


if __name__ == "__main__":
    # Synthetic self-test ONLY -- no model loading, no real VIT questions.
    # This just checks the AUROC / ECE plumbing works before you wire it
    # into eval_compare.py or compare_idk_stats.py for real.
    print("Running synthetic smoke test (not a real model eval)...")
    rng = np.random.default_rng(0)
    n = 200
    confs = rng.uniform(0, 1, n)
    labels = (rng.uniform(0, 1, n) < confs * 0.7).astype(int)  # somewhat calibrated

    auroc = compute_auroc(confs, labels, plot=True, label="synthetic")
    print("AUROC:", auroc)
    ece = reliability_diagram(confs, labels, n_bins=10, label="synthetic")
    print("ECE:", ece)

    print("\nFor a real run, see the module docstring above for how to wire")
    print("auroc_from_results() / auroc_hallucination_detection() into")
    print("eval_compare.py, and sample_from_model() into semantic_consistency_score().")
