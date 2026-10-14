"""Score the generated answers, calibrate confidence on VALID, report on TEST (CPU only).

Everything that is tuned (the logistic calibrator and the abstention threshold) is fit on the valid
split. The test split is only ever used to report numbers.

    python calibrate_v3.py --valid results_v3/rows_valid.jsonl --test results_v3/rows_test.jsonl

Confidence signals:
  token   mean token probability of the greedy answer
  agree   fraction of sampled answers that say the same thing as the greedy answer (self-consistency)
  calib   logistic regression on [token, min token prob, agree, refused] fit on valid -> P(correct)

Final system = abstain when the model refuses OR calibrated confidence < tau (tau chosen on valid to
maximise overall accuracy). Abstaining on an answerable question counts as wrong (false refusal);
answering an unanswerable one counts as wrong (hallucination).
"""

import argparse
import json
from pathlib import Path

import numpy as np

from v3_common import (auroc, brier, correctness, ece, read_jsonl, refused, reliability_bins, same_answer)


def featurize(rows):
    X, strict, keyfact = [], [], []
    for r in rows:
        s_strict, s_key = correctness(r["greedy"], r["reference"], r["is_idk"])
        samples = r["samples"]
        agree = sum(same_answer(r["greedy"], s) for s in samples) / len(samples) if samples else 0.0
        X.append([r["token_conf"], r["min_prob"], agree, float(refused(r["greedy"]))])
        strict.append(s_strict)
        keyfact.append(s_key)
    return np.array(X), np.array(strict), np.array(keyfact)


def fit_calibrator(X, y):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=1.0, max_iter=1000).fit(X, y)


def pick_threshold(p_valid, rows, correct_if_answered):
    """tau maximising overall accuracy on valid, where abstaining is right only for unanswerable rows."""
    is_idk = np.array([r["is_idk"] for r in rows])
    ref = np.array([refused(r["greedy"]) for r in rows])
    best = (-1.0, 0.0)
    for tau in np.unique(np.concatenate([[0.0], np.round(p_valid, 3)])):
        abstain = ref | (p_valid < tau)
        acc = np.where(abstain, is_idk, correct_if_answered & ~ref).mean()
        if acc > best[0] + 1e-9:
            best = (float(acc), float(tau))
    return best[1]


def final_metrics(p, rows, correct_if_answered, tau):
    is_idk = np.array([r["is_idk"] for r in rows])
    ref = np.array([refused(r["greedy"]) for r in rows])
    abstain = ref | (p < tau)
    correct = np.where(abstain, is_idk, correct_if_answered & ~ref)
    answered = ~abstain
    halluc = float((answered & is_idk).sum() / max(is_idk.sum(), 1))
    false_ref = float((abstain & ~is_idk).sum() / max((~is_idk).sum(), 1))
    sel_acc = float(correct[answered].mean()) if answered.any() else None
    return {"overall_accuracy": float(correct.mean()), "coverage": float(answered.mean()),
            "selective_accuracy": sel_acc, "hallucination_rate": halluc, "false_refusal_rate": false_ref}


def risk_coverage(p, correct_if_answered, ref, is_idk):
    """Accuracy among kept answers as we keep only the most confident X% (refusals included in ranking)."""
    order = np.argsort(-p)
    correct = np.where(ref, is_idk, correct_if_answered)[order]
    out = []
    for cov in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        k = max(1, int(round(cov * len(order))))
        out.append({"coverage": cov, "accuracy": float(correct[:k].mean())})
    return out


def plots(out_dir, name, curves):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    for label, bins in curves.items():
        pts = [(c, a) for c, a, n in bins if n]
        ax.plot(*zip(*pts), marker="o", label=label)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"reliability_{name}.png", dpi=200)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--valid", default="results_v3/rows_valid.jsonl")
    p.add_argument("--test", default="results_v3/rows_test.jsonl")
    p.add_argument("--metric", choices=["keyfact", "f1"], default="keyfact",
                   help="correctness used for calibration targets and the headline numbers")
    p.add_argument("--out-dir", default="results_v3")
    p.add_argument("--name", default="")
    a = p.parse_args()

    valid, test = read_jsonl(a.valid), read_jsonl(a.test)
    Xv, sv, kv = featurize(valid)
    Xt, st, kt = featurize(test)
    yv, yt = (kv, kt) if a.metric == "keyfact" else (sv, st)

    cal = fit_calibrator(Xv, yv)
    pv, pt = cal.predict_proba(Xv)[:, 1], cal.predict_proba(Xt)[:, 1]
    tau = pick_threshold(pv, valid, yv.astype(bool))

    sources = {"token prob": Xt[:, 0], "self-consistency": Xt[:, 2], "calibrated (valid-fit)": pt}
    report = {"metric": a.metric, "n_valid": len(valid), "n_test": len(test), "tau": tau,
              "accuracy_greedy_keyfact": float(kt.mean()), "accuracy_greedy_f1": float(st.mean()),
              "answerable_accuracy_keyfact": float(kt[[not r["is_idk"] for r in test]].mean()),
              "calibration": {}, "reliability": {}}
    for label, conf in sources.items():
        report["calibration"][label] = {"ece": ece(conf, yt), "brier": brier(conf, yt), "auroc": auroc(conf, yt)}
        report["reliability"][label] = reliability_bins(conf, yt)
    ref_t = np.array([refused(r["greedy"]) for r in test])
    idk_t = np.array([r["is_idk"] for r in test])
    report["final_system"] = final_metrics(pt, test, yt.astype(bool), tau)
    report["risk_coverage"] = risk_coverage(pt, yt.astype(bool), ref_t, idk_t)

    out = Path(a.out_dir)
    out.mkdir(exist_ok=True)
    (out / f"report{a.name}.json").write_text(json.dumps(report, indent=2))
    plots(out, a.name or "main", report["reliability"])

    f = report["final_system"]
    print(f"\nTEST ({len(test)} questions, {idk_t.sum()} unanswerable)  correctness = {a.metric}")
    print(f"  greedy accuracy       key-fact {kt.mean():.3f}   token-F1>=0.5 {st.mean():.3f}")
    print(f"  answerable accuracy   {report['answerable_accuracy_keyfact']:.3f}")
    print(f"  {'confidence source':<26}{'ECE':>8}{'Brier':>8}{'AUROC':>8}")
    for label, m in report["calibration"].items():
        print(f"  {label:<26}{m['ece']:>8.3f}{m['brier']:>8.3f}{(m['auroc'] or 0):>8.3f}")
    print(f"  final system (tau={tau:.3f}, fit on valid)")
    print(f"    overall accuracy {f['overall_accuracy']:.3f}   coverage {f['coverage']:.3f}   "
          f"selective accuracy {f['selective_accuracy'] if f['selective_accuracy'] is None else round(f['selective_accuracy'], 3)}")
    print(f"    hallucination rate {f['hallucination_rate']:.3f}   false-refusal rate {f['false_refusal_rate']:.3f}")
    print("  risk-coverage: " + "  ".join(f"{c['coverage']:.0%}->{c['accuracy']:.3f}" for c in report["risk_coverage"]))


if __name__ == "__main__":
    main()
