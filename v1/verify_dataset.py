import json
import re
from pathlib import Path

import pandas as pd

# Quick sanity checks on the dataset before training/eval:
#   1. every IDK row in the processed splits should be the one canonical refusal
#   2. the pool-selection regex shouldn't be dragging real answers into the IDK
#      pool just because "real-time" / "moment to moment" appear mid-sentence

CANONICAL = "I don't have enough information in my knowledge base to answer this question accurately."

# current pattern used to build the IDK pool
CURRENT = re.compile(
    r"don't know|I don't|cannot verify|cannot check|cannot submit|cannot see|"
    r"real-time|moment to moment|I cannot|not in my knowledge", re.IGNORECASE)

# same thing minus the two tokens that fire inside genuine answers. the live-status
# refusals we actually want are still caught (they say "I cannot" / "do not have access").
FIXED = re.compile(
    r"don't know|I don't have|cannot verify|cannot check|cannot submit|cannot see|"
    r"I cannot|not in my knowledge|do not have access|don't have enough information",
    re.IGNORECASE)


def check_split(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    idk = [r for r in rows if CURRENT.search(r["answer"])]
    off = [r for r in idk if r["answer"].strip() != CANONICAL]
    status = "ok" if not off else f"{len(off)} not canonical"
    print(f"  {Path(path).name}: {len(rows)} rows, {len(idk)} IDK, "
          f"{len(idk) - len(off)}/{len(idk)} canonical ({status})")


def check_source(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    a = df["answer"].astype(str)
    cur, fix = a.str.contains(CURRENT, na=False), a.str.contains(FIXED, regex=True, na=False)
    rt = a.str.contains(r"real-time|moment to moment", case=False, na=False)

    lost = df[cur & ~fix]        # real refusals the fix would drop (want this to be 0)
    removed = df[rt & ~fix]      # false positives the fix cleans up
    print(f"  {Path(path).name}: IDK pool {cur.sum()} -> {fix.sum()}  "
          f"(false positives removed: {len(removed)}, real refusals lost: {len(lost)})")
    for _, r in removed.iterrows():
        print(f"     fixed: {str(r['question'])[:80]}")


if __name__ == "__main__":
    print("IDK labels in the splits:")
    for f in ["expc_idk15_train.jsonl", "expc_idk15_valid.jsonl", "expc_idk15_test.jsonl"]:
        if Path(f).exists():
            check_split(f)

    print("\nSource pool false positives (drop 'real-time' / 'moment to moment'):")
    for f in ["final_qa_dataset.csv", "combined_qa_finetuning.csv"]:
        if Path(f).exists():
            check_source(f)
