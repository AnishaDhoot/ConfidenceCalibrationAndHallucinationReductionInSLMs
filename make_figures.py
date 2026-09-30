"""
Builds the per-example calibration figures from results/calibration_export.json
(written by eval_compare_fixed.py). No GPU, no re-eval -- runs in seconds.

  python make_figures.py

Outputs into results/figures/:
  reliability.png   - calibration curve (confidence vs accuracy) for all models
  roc.png           - ROC curves for hallucination detection, AUROC in legend
  confidence_hist.png - confidence on correct vs wrong answers, per model
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import roc_auc_score, roc_curve

mpl.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})

C = {"Base": "#0072B2", "Normal FT": "#E69F00", "IDK FT": "#009E73"}
IN = "results/calibration_export.json"
OUT = "results/figures/"
os.makedirs(OUT, exist_ok=True)

data = json.load(open(IN))
models = [m for m in ["Base", "Normal FT", "IDK FT"] if m in data]


def arrays(m):
    d = data[m]
    return np.asarray(d["confidences"], float), np.asarray(d["corrects"], int)


# --- 1. Reliability diagram (calibration curve) ---
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], "--", color="#999", lw=1.5, label="Perfect calibration")
n_bins = 10
edges = np.linspace(0, 1, n_bins + 1)
for m in models:
    conf, corr = arrays(m)
    ids = np.digitize(conf, edges[1:-1])
    xs, ys = [], []
    for b in range(n_bins):
        mask = ids == b
        if mask.sum():
            xs.append(conf[mask].mean()); ys.append(corr[mask].mean())
    ax.plot(xs, ys, "o-", color=C[m], lw=2, markersize=6, label=m)
ax.set_xlabel("Mean predicted confidence"); ax.set_ylabel("Actual accuracy")
ax.set_title("Reliability diagram (closer to diagonal = better calibrated)")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(frameon=False)
fig.savefig(OUT + "reliability.png"); plt.close(fig); print("wrote reliability.png")

# --- 2. ROC curves (hallucination / error detection) ---
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], "--", color="#999", lw=1.5, label="Random (0.500)")
for m in models:
    conf, corr = arrays(m)
    if len(set(corr.tolist())) < 2:
        continue
    fpr, tpr, _ = roc_curve(corr, conf)
    auc = roc_auc_score(corr, conf)
    ax.plot(fpr, tpr, color=C[m], lw=2, label=f"{m} (AUROC={auc:.3f})")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("Can confidence flag wrong answers? (higher AUROC = better)")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(frameon=False, loc="lower right")
fig.savefig(OUT + "roc.png"); plt.close(fig); print("wrote roc.png")

# --- 3. Confidence histograms: correct vs wrong, per model ---
fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.2), sharey=True)
if len(models) == 1:
    axes = [axes]
bins = np.linspace(0, 1, 21)
for ax, m in zip(axes, models):
    conf, corr = arrays(m)
    ax.hist(conf[corr == 1], bins=bins, color=C[m], alpha=0.85, label="correct", zorder=3)
    ax.hist(conf[corr == 0], bins=bins, color="#C1121F", alpha=0.45, label="wrong", zorder=2)
    ax.set_title(m); ax.set_xlabel("Confidence")
    ax.legend(frameon=False, fontsize=9)
axes[0].set_ylabel("Count")
fig.suptitle("Confidence on correct vs wrong answers (more separation = better)",
             fontweight="bold", y=1.02)
fig.savefig(OUT + "confidence_hist.png"); plt.close(fig); print("wrote confidence_hist.png")

print("Done -> results/figures/")
