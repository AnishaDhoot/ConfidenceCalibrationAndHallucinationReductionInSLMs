"""
Experiment C orchestrator: builds the data for a given IDK ratio (via
prepare_expc_data.py) and then runs train_lora.py against it, so the whole
"one ratio -> one adapter" pipeline is a single command instead of two
copy-pasted ones.

This just shells out to the two scripts rather than importing them --
train_lora.py runs its training loop at import time (not wrapped in a
function), so importing it would kick off training as a side effect of the
import itself, which is fragile. Subprocess keeps each stage's GPU memory
cleanly freed when it exits, too.

USAGE:
    python exp_c_train.py --idk-ratio 0.15
    python exp_c_train.py --idk-ratio 0.0 --output-dir phi3-vit-lora-expc
    python exp_c_train.py --idk-ratio 0.15 --skip-data-prep   # data already built, just train

On Colab: run this from the repo root with the two CSVs and the *.py files
present; the adapter comes out under --output-dir, which you'll want to
copy to Drive before the runtime disconnects.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"Command failed (exit {result.returncode}): {' '.join(cmd)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idk-ratio", type=float, required=True,
                         help="0.0 for the 'normal' fine-tune, 0.15 for the primary IDK-controlled run, etc.")
    parser.add_argument("--output-dir", type=str, default=None,
                         help="Adapter output folder. Defaults to phi3-vit-lora-expc-idk<N>.")
    parser.add_argument("--skip-data-prep", action="store_true",
                         help="Skip prepare_expc_data.py and train directly on the existing expc_idk<N>_train.jsonl")
    args = parser.parse_args()

    suffix = f"idk{int(args.idk_ratio * 100)}"
    train_path = Path(f"expc_{suffix}_train.jsonl")
    output_dir = args.output_dir or f"phi3-vit-lora-expc-{suffix}"

    if not args.skip_data_prep:
        run([sys.executable, "prepare_expc_data.py", "--idk-ratio", str(args.idk_ratio)])
    elif not train_path.exists():
        sys.exit(f"--skip-data-prep was set but {train_path} doesn't exist. Run prepare_expc_data.py first.")

    run([sys.executable, "train_lora.py", "--data", str(train_path), "--output-dir", output_dir])

    print(f"\nDone. Adapter for idk_ratio={args.idk_ratio} saved to ./{output_dir}")
    print(f"Evaluate it with:")
    print(f"  python eval_compare.py --normal-adapter phi3-vit-lora-expc "
          f"--idk-adapter {output_dir} --test expc_{suffix}_test.jsonl")
    print(f"  python compare_idk_stats.py --normal-adapter phi3-vit-lora-expc "
          f"--idk-adapter {output_dir} --test expc_{suffix}_test.jsonl")


if __name__ == "__main__":
    main()
