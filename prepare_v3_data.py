"""Build the v3 train / valid / test splits.

Differences from v1/v2:
  * mojibake repaired and exact-duplicate questions dropped *before* splitting
  * one canonical "I don't have enough information..." answer for every unanswerable row
  * optional paraphrase augmentation of the TRAIN split only (valid/test stay untouched)
  * prompts are NOT baked in; the training/eval scripts render them with the model's chat template
  * writes stats.json including how many test questions have a near-duplicate in train

Run twice: once plain (so augment_v3.py knows the train questions), then again with --paraphrases.
The split is deterministic for a given --seed, so both runs produce the same valid/test sets.

    python prepare_v3_data.py
    python augment_v3.py --train data_v3/train.jsonl --out data_v3/paraphrases.jsonl
    python prepare_v3_data.py --paraphrases data_v3/paraphrases.jsonl --max-para 3
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from v3_common import (COURSE_CODE, IDK_ANSWER, SOURCE_IDK_PATTERN, normalize, read_jsonl,
                       repair_text, write_jsonl)

FINAL_QA_CSV = Path("final_qa_dataset.csv")
COMBINED_CSV = Path("combined_qa_finetuning.csv")


def build_groups(questions, answers):
    """Union rows that share a course code so a whole course lands in one split."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    keys = []
    for i, (q, a) in enumerate(zip(questions, answers)):
        codes = COURSE_CODE.findall(q) or COURSE_CODE.findall(a)[:1]
        if not codes:
            keys.append(f"row{i}")
            continue
        for c in codes[1:]:
            parent[find(codes[0])] = find(c)
        keys.append(codes[0])
    return [find(k) for k in keys]


def split_frame(pool, mode, seed):
    n = len(pool)
    n_train, n_valid = int(n * 0.70), int(n * 0.15)
    if mode == "question":
        return (pool.iloc[:n_train], pool.iloc[n_train:n_train + n_valid], pool.iloc[n_train + n_valid:])
    keys = build_groups(pool["question"].tolist(), pool["answer"].tolist())
    pool = pool.assign(group=keys)
    groups = list(pool.groupby("group").indices.items())
    random.Random(seed).shuffle(groups)
    buckets = {"train": [], "valid": [], "test": []}
    for _, positions in groups:
        if len(buckets["train"]) < n_train:
            buckets["train"].extend(positions)
        elif len(buckets["valid"]) < n_valid:
            buckets["valid"].extend(positions)
        else:
            buckets["test"].extend(positions)
    return tuple(pool.iloc[buckets[k]] for k in ("train", "valid", "test"))


def near_duplicate_rate(train_q, test_q, threshold=0.9):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(train_q + test_q)
    sims = (vec.transform(test_q) @ vec.transform(train_q).T).max(axis=1).toarray().ravel()
    return float((sims >= threshold).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--idk-ratio", type=float, default=0.15, help="share of unanswerable rows in each split")
    p.add_argument("--split-by", choices=["question", "course"], default="question")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--paraphrases", help="jsonl from augment_v3.py; adds rewrites of TRAIN questions only")
    p.add_argument("--max-para", type=int, default=3)
    p.add_argument("--extra", help="extra_course_qa.jsonl from make_course_qa.py; appended to TRAIN only")
    p.add_argument("--extra-kinds", default="credits,ltpc,title,code",
                   help="which kinds of extra rows to use (credits, ltpc, title, code)")
    p.add_argument("--max-extra", type=int, default=0, help="cap on extra rows (0 = all of the chosen kinds)")
    p.add_argument("--out-dir", default="data_v3")
    a = p.parse_args()

    final = pd.read_csv(FINAL_QA_CSV, encoding="utf-8-sig")
    combined = pd.read_csv(COMBINED_CSV, encoding="utf-8-sig")
    df = pd.concat([final, combined], ignore_index=True)
    df["question"] = df["question"].astype(str).map(repair_text)
    df["answer"] = df["answer"].astype(str).map(repair_text)
    df = df[(df["question"] != "") & (df["answer"] != "")]

    before = len(df)
    df = df.assign(_k=df["question"].map(normalize)).drop_duplicates("_k").drop(columns="_k")
    print(f"rows: {before} -> {len(df)} after dropping duplicate questions")

    df["is_idk"] = df["answer"].str.contains(SOURCE_IDK_PATTERN, case=False, na=False)
    df.loc[df["is_idk"], "answer"] = IDK_ANSWER
    pool = df.sample(frac=1, random_state=a.seed).reset_index(drop=True)

    splits = {}
    for name, part in zip(("train", "valid", "test"), split_frame(pool, a.split_by, a.seed)):
        reg, idk = part[~part["is_idk"]], part[part["is_idk"]]
        k = min(int(round(len(reg) * a.idk_ratio / (1 - a.idk_ratio))), len(idk))
        picked = idk.sample(n=k, random_state=a.seed) if k else idk.iloc[0:0]
        if name == "train":
            spare_idk = idk.drop(picked.index)  # unused unanswerable rows, used to top up when extra rows are added
        idk = picked
        splits[name] = pd.concat([reg, idk]).sample(frac=1, random_state=a.seed).reset_index(drop=True)

    out = Path(a.out_dir)
    out.mkdir(exist_ok=True)

    paras = {}
    if a.paraphrases:
        for r in read_jsonl(a.paraphrases):
            paras[r["question"]] = r["paraphrases"][:a.max_para]

    extra_rows = []
    if a.extra:
        kinds = set(a.extra_kinds.split(","))
        blocked = {normalize(q) for part in splits.values() for q in part["question"]}  # never duplicate any existing question
        for r in read_jsonl(a.extra):
            if r["kind"] in kinds and normalize(r["question"]) not in blocked:
                extra_rows.append({"question": r["question"], "answer": r["answer"], "is_idk": False,
                                   "source": "pdf_table"})
        random.Random(a.seed).shuffle(extra_rows)
        if a.max_extra:
            extra_rows = extra_rows[:a.max_extra]
        print(f"extra course-table rows for train: {len(extra_rows)} (kinds: {sorted(kinds)})")

    stats = {"split_by": a.split_by, "idk_ratio": a.idk_ratio, "seed": a.seed}
    for name, part in splits.items():
        rows = [{"question": q, "answer": ans, "is_idk": bool(i), "source": "original"}
                for q, ans, i in zip(part["question"], part["answer"], part["is_idk"])]
        if name == "train" and paras:
            extra = [{"question": pq, "answer": r["answer"], "is_idk": r["is_idk"], "source": "paraphrase"}
                     for r in rows for pq in paras.get(r["question"], [])]
            rows += extra
            print(f"train: added {len(extra)} paraphrased rows")
        if name == "train" and extra_rows:
            rows += extra_rows
            # keep the unanswerable share at --idk-ratio so abstention is not diluted by the extra rows
            have = sum(r["is_idk"] for r in rows)
            want = int(round((len(rows) - have) * a.idk_ratio / (1 - a.idk_ratio)))
            take = max(0, min(want - have, len(spare_idk)))
            topup = spare_idk.sample(n=take, random_state=a.seed) if take else spare_idk.iloc[0:0]
            rows += [{"question": q, "answer": ans, "is_idk": True, "source": "original"}
                     for q, ans in zip(topup["question"], topup["answer"])]
            print(f"train: topped up with {take} more unanswerable rows "
                  f"(wanted {max(want - have, 0)}, {len(spare_idk)} available)")
        if name == "train":
            random.Random(a.seed).shuffle(rows)
        write_jsonl(out / f"{name}.jsonl", rows)
        n_idk = sum(r["is_idk"] for r in rows)
        stats[name] = {"rows": len(rows), "idk": n_idk, "answerable": len(rows) - n_idk,
                       "paraphrased": sum(r["source"] == "paraphrase" for r in rows),
                       "pdf_table": sum(r["source"] == "pdf_table" for r in rows)}
        print(f"{name}: {len(rows)} rows ({n_idk} unanswerable)")

    tq = [normalize(q) for q in splits["train"]["question"]]
    for name in ("valid", "test"):
        rate = near_duplicate_rate(tq, [normalize(q) for q in splits[name]["question"]])
        stats[name]["near_duplicate_of_train"] = rate
        print(f"{name}: {rate:.1%} of questions have a near-duplicate (cosine >= 0.9) in train")

    with open(out / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"wrote {out}/")


if __name__ == "__main__":
    main()
