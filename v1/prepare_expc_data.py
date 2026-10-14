"""
Exp C dataset builder -- Windows/CUDA port of your teammate's mlx_lm version.

Same experimental design, different execution stack:
  - Your teammate's version: mlx_lm, Apple Silicon only, "prompt"/"completion" JSONL
  - This version: plain question/answer JSONL, feeds directly into your existing
    train_lora.py (which already handles the Phi-3 chat-template formatting itself)

WHAT THIS DOES (mirrors teammate's build_dataset() exactly):
  1. Loads final_qa_dataset.csv (regular QA) and combined_qa_finetuning.csv (mixed)
  2. Splits each into "regular" vs "IDK" rows using the same detection pattern
  3. Normalizes every IDK answer to ONE canonical string (not varied phrasing --
     this matters for measuring refusal precision/recall cleanly)
  4. Samples exactly enough IDK rows to hit your target ratio (e.g. 15%), rather
     than just dumping in whatever IDK rows happen to exist
  5. Shuffles, splits 70/15/15 train/valid/test, checks for train/test leakage
  6. Writes question/answer JSONL files ready for train_lora.py

USAGE:
    python prepare_expc_data.py --dry-run              # 50 rows, sanity check first
    python prepare_expc_data.py --idk-ratio 0.15        # full run, 15% IDK (primary)
    python prepare_expc_data.py --idk-ratio 0.10        # ablation
    python prepare_expc_data.py --idk-ratio 0.20        # ablation
"""

import json
import random
import argparse
import pandas as pd
from pathlib import Path

FINAL_QA_CSV = Path("final_qa_dataset.csv")
COMBINED_CSV = Path("combined_qa_finetuning.csv")
DATA_DIR = Path(".")  # output files land next to your scripts, matching train_lora.py's expectations

# Canonical IDK answer -- every abstention example gets exactly this string.
# Deliberately different from the varied-phrasing approach used earlier: for
# *measuring* refusal precision/recall cleanly, one fixed target is easier to
# detect reliably than several paraphrases. (Trade-off: the model may learn
# this as a slightly more rote response than the varied version would have --
# worth comparing against your earlier varied-phrasing run if you want to see
# which generalizes better.)
IDK_ANSWER = "I don't have enough information in my knowledge base to answer this question accurately."

# Same detection pattern as your teammate's script, so both pipelines agree
# on which existing rows count as "already IDK" before normalization.
IDK_PATTERN = (
    "don't know|I don't|cannot verify|cannot check|cannot submit|"
    "cannot see|real-time|moment to moment|I cannot|not in my knowledge"
)


def build_dataset(idk_ratio: float, dry_run: bool = False, seed: int = 42):
    print("=" * 60)
    print(f"BUILDING DATASET | IDK ratio: {idk_ratio:.0%} | dry_run: {dry_run}")
    print("=" * 60)

    random.seed(seed)

    if not FINAL_QA_CSV.exists() or not COMBINED_CSV.exists():
        raise FileNotFoundError(
            f"Need both {FINAL_QA_CSV} and {COMBINED_CSV} in the current folder."
        )

    final = pd.read_csv(FINAL_QA_CSV, encoding="utf-8-sig")
    combined = pd.read_csv(COMBINED_CSV, encoding="utf-8-sig")

    idk_mask_f = final["answer"].str.contains(IDK_PATTERN, case=False, na=False)
    idk_mask_c = combined["answer"].str.contains(IDK_PATTERN, case=False, na=False)

    regular_rows = pd.concat([final[~idk_mask_f], combined[~idk_mask_c]], ignore_index=True)
    idk_pool = pd.concat([final[idk_mask_f], combined[idk_mask_c]], ignore_index=True)

    print(f"  Regular QA pool : {len(regular_rows)} rows")
    print(f"  IDK pool        : {len(idk_pool)} rows")

    idk_pool = idk_pool.copy()
    idk_pool["answer"] = IDK_ANSWER  # normalize to the one canonical string

    n_regular = len(regular_rows)
    n_idk_needed = int(n_regular * idk_ratio / (1 - idk_ratio))
    n_idk_needed = min(n_idk_needed, len(idk_pool))

    if n_idk_needed < int(n_regular * idk_ratio / (1 - idk_ratio)):
        print(f"  WARNING: wanted {int(n_regular * idk_ratio / (1 - idk_ratio))} IDK rows "
              f"for a clean {idk_ratio:.0%} ratio, but only {len(idk_pool)} exist. "
              f"Actual ratio will be lower than requested.")

    idk_sample = idk_pool.sample(n=n_idk_needed, random_state=seed)

    full_df = pd.concat([regular_rows, idk_sample], ignore_index=True)
    full_df = full_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"  Final dataset   : {len(full_df)} rows")
    print(f"  IDK actual      : {n_idk_needed} ({n_idk_needed/len(full_df)*100:.1f}%)")

    if dry_run:
        full_df = full_df.head(50)
        print(f"  DRY RUN: trimmed to {len(full_df)} rows")

    n = len(full_df)
    n_train = int(n * 0.70)
    n_valid = int(n * 0.15)

    train_df = full_df.iloc[:n_train]
    valid_df = full_df.iloc[n_train:n_train + n_valid]
    test_df = full_df.iloc[n_train + n_valid:]

    print(f"  Train: {len(train_df)}  Valid: {len(valid_df)}  Test: {len(test_df)}")

    train_qs = set(train_df["question"].tolist())
    test_qs = set(test_df["question"].tolist())
    overlap = train_qs & test_qs
    if overlap:
        print(f"  WARNING: {len(overlap)} questions appear in both train and test.")
    else:
        print("  OK: no train/test contamination.")

    suffix = "dry" if dry_run else f"idk{int(idk_ratio*100)}"

    for split_name, split_df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        path = DATA_DIR / f"expc_{suffix}_{split_name}.jsonl"
        written = 0
        with open(path, "w", encoding="utf-8") as f:
            for _, row in split_df.iterrows():
                q, a = str(row["question"]).strip(), str(row["answer"]).strip()
                if not q or not a:
                    continue
                f.write(json.dumps({"question": q, "answer": a}, ensure_ascii=False) + "\n")
                written += 1
        print(f"  Wrote {written} rows to {path.name}")

    print(f"\nDone. Point train_lora.py's DATA_PATH at: expc_{suffix}_train.jsonl")
    print(f"Its own 90/10 split inside train_lora.py isn't needed here -- this file")
    print(f"is already your train split; the eval_compare.py held-out set should")
    print(f"instead use expc_{suffix}_test.jsonl directly (see compare_idk_stats.py).")
    return suffix


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="50 rows, quick sanity check")
    parser.add_argument("--idk-ratio", type=float, default=0.15, help="0.10 / 0.15 / 0.20")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_dataset(args.idk_ratio, dry_run=args.dry_run, seed=args.seed)
