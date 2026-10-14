import argparse
import json
import re
import string

from rag_context import Retriever

IDK_RE = re.compile(
    r"don't know|I don't have|cannot verify|cannot check|cannot submit|"
    r"cannot see|I cannot|not in my knowledge|do not have access|"
    r"don't have enough information|do not have a live feed|real-time|moment to moment",
    re.IGNORECASE,
)


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


def coverage(context_answers, ref):
    ref_toks = set(normalize(ref).split())
    if not ref_toks:
        return 0.0
    ctx = set(normalize(" ".join(context_answers)).split())
    return len(ref_toks & ctx) / len(ref_toks)


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--f1-threshold", type=float, default=0.5)
    a = p.parse_args()

    train, test = load(a.train), load(a.test)
    kb = [(r["raw_question"], r["answer"]) for r in train if not r["is_idk"]]
    retr = Retriever(kb)

    rows = []
    for r in test:
        hits = retr.search(r["raw_question"], k=a.k)
        rows.append((r, hits))

    answerable = [(r, h) for r, h in rows if not r["is_idk"]]
    idk = [(r, h) for r, h in rows if r["is_idk"]]

    top1_f1 = [token_f1(kb[h[0][0]][1], r["answer"]) for r, h in answerable]
    top1_hit = sum(f >= a.f1_threshold for f in top1_f1) / len(answerable)
    cov_hit = sum(coverage([kb[i][1] for i, _ in h], r["answer"]) >= 0.8 for r, h in answerable) / len(answerable)
    print(f"test rows {len(test)}  answerable {len(answerable)}  idk {len(idk)}")
    print(f"answer fully recoverable from top-{a.k} notes (>=80% of gold tokens): {cov_hit:.3f}")
    print(f"top-1 note alone matches gold at F1>={a.f1_threshold}: {top1_hit:.3f}")

    best = None
    for thr in [x / 100 for x in range(10, 91, 2)]:
        correct = 0
        for r, h in rows:
            top_score = h[0][1]
            abstain = top_score < thr
            if r["is_idk"]:
                correct += int(abstain)
            elif not abstain:
                correct += int(token_f1(kb[h[0][0]][1], r["answer"]) >= a.f1_threshold)
        acc = correct / len(rows)
        if best is None or acc > best[0]:
            best = (acc, thr)
    print(f"retrieval-only overall accuracy with tuned abstain threshold: {best[0]:.3f} (threshold {best[1]:.2f}, tuned on this test set, so optimistic)")


if __name__ == "__main__":
    main()
