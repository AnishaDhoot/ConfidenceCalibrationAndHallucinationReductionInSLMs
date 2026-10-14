import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd

from rag_context import COURSE_CODE, IDK_ANSWER, Retriever, build_prompt

FINAL_QA_CSV = Path("final_qa_dataset.csv")
COMBINED_CSV = Path("combined_qa_finetuning.csv")

IDK_PATTERN = (
    "don't know|I don't|cannot verify|cannot check|cannot submit|"
    "cannot see|real-time|moment to moment|I cannot|not in my knowledge"
)

BARE_IDK = re.compile(r"^I don't know\s*[—-]\s*that information isn't in the curriculum documents I have\.?$")

MOJIBAKE = re.compile(r"[\u00c2-\u00f4][^\x00-\x7f]{1,3}")


def repair_text(text):
    def fix(match):
        chunk = match.group(0)
        for codec in ("latin-1", "cp1252"):
            try:
                return chunk.encode(codec).decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
        return chunk
    return MOJIBAKE.sub(fix, text)


def normalize_idk(answer, style):
    if style == "canonical":
        return IDK_ANSWER
    if BARE_IDK.match(answer.strip()):
        return IDK_ANSWER
    return answer.strip()


def build_groups(questions, answers):
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    keys = []
    for i, (q, a) in enumerate(zip(questions, answers)):
        codes = COURSE_CODE.findall(q)
        if not codes:
            codes = COURSE_CODE.findall(a)[:1]
        if not codes:
            keys.append(f"row{i}")
            continue
        for c in codes[1:]:
            union(codes[0], c)
        keys.append(codes[0])
    return [find(k) for k in keys]


def split_frame(full, mode, seed):
    n = len(full)
    n_train, n_valid = int(n * 0.70), int(n * 0.15)
    if mode == "question":
        return (
            full.iloc[:n_train].reset_index(drop=True),
            full.iloc[n_train:n_train + n_valid].reset_index(drop=True),
            full.iloc[n_train + n_valid:].reset_index(drop=True),
        )
    keys = build_groups(full["question"].astype(str).tolist(), full["answer"].astype(str).tolist())
    full = full.assign(group=keys)
    sizes = full["group"].value_counts()
    print(f"course groups: {len(sizes)}, largest group {sizes.iloc[0]} rows")
    groups = list(full.groupby("group").indices.items())
    random.Random(seed).shuffle(groups)
    buckets = {"train": [], "valid": [], "test": []}
    for _, positions in groups:
        if len(buckets["train"]) < n_train:
            buckets["train"].extend(positions)
        elif len(buckets["valid"]) < n_valid:
            buckets["valid"].extend(positions)
        else:
            buckets["test"].extend(positions)
    return tuple(full.iloc[buckets[name]].reset_index(drop=True) for name in ("train", "valid", "test"))


def question_codes(df):
    codes = set()
    for q in df["question"].astype(str):
        codes.update(COURSE_CODE.findall(q))
    return codes


def build(idk_ratio, style, rag_k, split_by, seed):
    random.seed(seed)
    final = pd.read_csv(FINAL_QA_CSV, encoding="utf-8-sig")
    combined = pd.read_csv(COMBINED_CSV, encoding="utf-8-sig")

    for df in (final, combined):
        df["question"] = df["question"].astype(str).map(repair_text)
        df["answer"] = df["answer"].astype(str).map(repair_text)

    mask_f = final["answer"].str.contains(IDK_PATTERN, case=False, na=False)
    mask_c = combined["answer"].str.contains(IDK_PATTERN, case=False, na=False)
    regular = pd.concat([final[~mask_f], combined[~mask_c]], ignore_index=True)
    idk_pool = pd.concat([final[mask_f], combined[mask_c]], ignore_index=True).copy()
    idk_pool["answer"] = idk_pool["answer"].map(lambda a: normalize_idk(str(a), style))

    pool = pd.concat(
        [regular.assign(is_idk=False), idk_pool.assign(is_idk=True)], ignore_index=True
    ).sample(frac=1, random_state=seed).reset_index(drop=True)
    parts = split_frame(pool, split_by, seed)

    splits = {}
    for name, part in zip(("train", "valid", "test"), parts):
        reg = part[~part["is_idk"]]
        idk = part[part["is_idk"]]
        k = min(int(round(len(reg) * idk_ratio / (1 - idk_ratio))), len(idk))
        idk_pick = idk.sample(n=k, random_state=seed) if k else idk.iloc[0:0]
        splits[name] = pd.concat([reg, idk_pick]).sample(frac=1, random_state=seed).reset_index(drop=True)
    train, valid, test = splits["train"], splits["valid"], splits["test"]
    full = pd.concat(splits.values(), ignore_index=True)

    print(f"rows: {len(full)}  split_by: {split_by}  style: {style}  rag_k: {rag_k}")
    for name, df in splits.items():
        print(f"  {name}: {len(df)} rows, idk {df['is_idk'].mean():.1%}")
    print(f"train/test question overlap: {len(set(train['question']) & set(test['question']))}")
    print(f"course codes shared by train and test questions: {len(question_codes(train) & question_codes(test))}")
    print(f"course codes shared by train and valid questions: {len(question_codes(train) & question_codes(valid))}")

    kb = train[~train["is_idk"]].reset_index(drop=True)
    kb_pairs = list(zip(kb["question"].astype(str), kb["answer"].astype(str)))
    kb_index = {q: i for i, (q, _) in enumerate(kb_pairs)}
    retriever = Retriever(kb_pairs) if rag_k > 0 else None

    suffix = f"idk{int(idk_ratio * 100)}_{style}_rag{rag_k}" + ("_course" if split_by == "course" else "")
    for name, df in splits.items():
        path = Path(f"expc2_{suffix}_{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                q, a = str(row["question"]).strip(), str(row["answer"]).strip()
                if not q or not a:
                    continue
                prompt = q
                if retriever is not None:
                    hits = retriever.search(q, k=rag_k, exclude=kb_index.get(q))
                    prompt = build_prompt(q, hits, kb_pairs)
                f.write(json.dumps(
                    {"question": prompt, "answer": a, "raw_question": q, "is_idk": bool(row["is_idk"])},
                    ensure_ascii=False) + "\n")
        print(f"wrote {path}")
    return suffix


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--idk-ratio", type=float, default=0.15)
    p.add_argument("--idk-style", choices=["keep", "canonical"], default="keep")
    p.add_argument("--rag-k", type=int, default=0)
    p.add_argument("--split-by", choices=["question", "course"], default="question")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    build(a.idk_ratio, a.idk_style, a.rag_k, a.split_by, a.seed)
