"""Shared helpers for the v3 pipeline: text cleanup, correctness, calibration metrics.

Pure Python + numpy so it imports on a laptop and in Colab without a GPU.
"""

import json
import re
import string

import numpy as np

SYSTEM_PROMPT = (
    "You are a question-answering assistant for Vellore Institute of Technology (VIT) curricula, "
    "academic regulations and campus life. Answer in one short sentence and state the key fact first. "
    "If the answer is not something you reliably know, say that you do not have enough information."
)

IDK_ANSWER = "I don't have enough information in my knowledge base to answer this question accurately."

# Anything that matches this counts as a refusal / abstention.
IDK_RE = re.compile(
    r"don't know|I don't have|cannot verify|cannot check|cannot submit|"
    r"cannot see|I cannot|not in my knowledge|do not have access|"
    r"don't have enough information|do not have enough information|"
    r"do not have a live feed|real-time|moment to moment",
    re.IGNORECASE,
)

# Source-CSV rows are marked unanswerable with this broader pattern (kept identical to v1/v2).
SOURCE_IDK_PATTERN = (
    "don't know|I don't|cannot verify|cannot check|cannot submit|"
    "cannot see|real-time|moment to moment|I cannot|not in my knowledge"
)

COURSE_CODE = re.compile(r"\b[A-Z]{2,6}\d{3,4}[A-Z]?\b")
NUMBER = re.compile(r"\d+(?:\.\d+)?")
_MOJIBAKE = re.compile(r"[Â-ô][^\x00-\x7f]{1,3}")


def repair_text(text):
    """Undo UTF-8 text that was decoded as latin-1/cp1252 (shows up as 'Â' + control chars)."""
    def fix(match):
        chunk = match.group(0)
        for codec in ("latin-1", "cp1252"):
            try:
                return chunk.encode(codec).decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
        return chunk
    text = _MOJIBAKE.sub(fix, text)
    # leftover replacement/control junk next to a dash, e.g. "Yes �\x80\x94 posting"
    text = re.sub(r"\s*[�\x80-\x9f]+\s*", lambda m: " — " if "\x94" in m.group(0) or "�" in m.group(0) else " ", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def normalize(s):
    s = s.lower()
    s = re.sub(f"[{re.escape(string.punctuation)}]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def token_f1(pred, ref):
    p, r = normalize(pred).split(), normalize(ref).split()
    if not p or not r:
        return float(p == r)
    same = sum(min(p.count(t), r.count(t)) for t in set(p))
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(r)
    return 2 * prec * rec / (prec + rec)


def refused(text):
    return bool(IDK_RE.search(text))


def key_facts(text):
    """Course codes and numbers in a string -- the facts that make an answer right or wrong."""
    facts = {c.lower() for c in COURSE_CODE.findall(text)}
    facts |= set(NUMBER.findall(text))
    return facts


def keyfact_correct(pred, ref, f1_fallback=0.5):
    """Answerable question: every code/number in the reference must appear in the prediction.
    If the reference has none (a descriptive answer), fall back to token F1."""
    ref_facts = key_facts(ref)
    if ref_facts:
        return int(ref_facts <= key_facts(pred))
    return int(token_f1(pred, ref) >= f1_fallback)


def correctness(pred, ref, is_idk, f1_threshold=0.5):
    """Returns (strict_f1_correct, keyfact_correct). Refusal-aware in both variants."""
    if is_idk:
        ok = int(refused(pred))
        return ok, ok
    if refused(pred):
        return 0, 0
    return int(token_f1(pred, ref) >= f1_threshold), keyfact_correct(pred, ref, f1_threshold)


def same_answer(a, b, f1_threshold=0.6):
    """Do two sampled answers say the same thing? Used for self-consistency."""
    ra, rb = refused(a), refused(b)
    if ra or rb:
        return ra and rb
    fa, fb = key_facts(a), key_facts(b)
    if fa or fb:
        return fa == fb
    return token_f1(a, b) >= f1_threshold


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- calibration metrics

def ece(conf, correct, n_bins=10):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    if len(conf) == 0:
        return 0.0
    idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def brier(conf, correct):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    return float(np.mean((conf - correct) ** 2)) if len(conf) else 0.0


def auroc(conf, correct):
    from sklearn.metrics import roc_auc_score
    correct = np.asarray(correct)
    if len(set(correct.tolist())) < 2:
        return None
    return float(roc_auc_score(correct, conf))


def reliability_bins(conf, correct, n_bins=10):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        out.append((float(conf[m].mean()), float(correct[m].mean()), int(m.sum())) if m.any() else (None, None, 0))
    return out
