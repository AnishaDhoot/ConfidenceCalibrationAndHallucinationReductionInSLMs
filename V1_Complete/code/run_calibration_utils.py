"""
Runs your REAL evaluation results (from eval_compare.py) through
calibration_eval_utils.py's compute_auroc() and reliability_diagram().

This does NOT re-run the model -- it reuses the confidence/correctness
arrays eval_compare.py already computed and exported, so it's essentially
instant, no GPU work involved.

USAGE:
    python eval_compare.py --test expc_idk15_test.jsonl   # produces the export
    python run_calibration_utils.py --input results/calibration_export.json
"""

import json
import argparse
from calibration_eval_utils import compute_auroc, reliability_diagram


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results/calibration_export.json")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        print("No models found in the export -- check eval_compare.py's output for the NOTE lines "
              "about missing per-example data (usually a stale cache).")
        return

    print(f"Models in export: {list(data.keys())}\n")

    for label, d in data.items():
        confidences = d["confidences"]
        corrects = d["corrects"]
        safe_label = label.lower().replace(" ", "_")

        print(f"=== {label} ===")
        auroc = compute_auroc(
            confidences, corrects, plot=True,
            title=f"Hallucination Detection ROC -- {label}",
        )
        # compute_auroc always saves to "roc_curve.png" internally; rename
        # per-model right after so successive models don't overwrite each other.
        import os
        if os.path.exists("roc_curve.png"):
            os.replace("roc_curve.png", f"results/roc_curve_{safe_label}.png")

        ece = reliability_diagram(
            confidences, corrects, n_bins=10, title=label,
            save_path=f"results/reliability_diagram_{safe_label}.png",
        )
        print(f"  AUROC (hallucination detection): {auroc:.3f}")
        print(f"  ECE (via calibration_eval_utils): {ece:.3f}")
        print()

    print("Note on AUROC here: it measures whether the model's OWN confidence")
    print("score can tell apart its correct answers from its wrong ones -- a")
    print("high AUROC means low confidence is a good warning sign of a bad")
    print("answer, even before you know the true answer. This is a different")
    print("question from ECE (is confidence numerically well-calibrated) and")
    print("from the refusal precision/recall in compare_idk_stats.py (does")
    print("the model explicitly SAY 'I don't know' at the right times).")


if __name__ == "__main__":
    main()
