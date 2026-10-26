"""Metrics + bootstrap CIs for the course-split run, from calibration_export.json and the regenerated test split.

    python results_v2/analyze_v2.py
The test split comes from `prepare_expc_data_v2.py --split-by course --idk-ratio 0.15` (deterministic, seed 42).
"""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

H = Path(__file__).parent
test = [json.loads(l) for l in open(H / "test_course_split.jsonl", encoding="utf-8")]
idk = np.array([t["is_idk"] for t in test], bool)
data = json.load(open(H / "calibration_export.json"))
M = ["Base", "Normal FT", "IDK FT"]


def ece(c, y, bins=10):
    ids = np.minimum((c * bins).astype(int), bins - 1)
    return sum((ids == b).mean() * abs(c[ids == b].mean() - y[ids == b].mean()) for b in range(bins) if (ids == b).any())


def metrics(c, y, m):
    return dict(acc=y.mean(), ece=ece(c, y), brier=((c - y) ** 2).mean(), auroc=roc_auc_score(y, c),
                hr=1 - y[m].mean(), ans_acc=y[~m].mean())


rng = np.random.default_rng(42)
print(f"test: {len(test)} rows, {idk.sum()} unanswerable, {(~idk).sum()} answerable")
res = {}
for m in M:
    c, y = np.array(data[m]["confidences"]), np.array(data[m]["corrects"])
    res[m] = metrics(c, y, idk)
    boots = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2 or idk[i].sum() == 0:
            continue
        boots.append(metrics(c[i], y[i], idk[i]))
    res[m]["ci"] = {k: (np.percentile([b[k] for b in boots], 2.5), np.percentile([b[k] for b in boots], 97.5))
                    for k in res[m] if k != "ci"}
    print(f"\n{m}: unanswerable correctly refused {int(y[idk].sum())}/{idk.sum()}; answerable correct {int(y[~idk].sum())}/{(~idk).sum()}")
    for k in ["acc", "ece", "brier", "auroc", "hr", "ans_acc"]:
        lo, hi = res[m]["ci"][k]
        print(f"  {k:8s} {res[m][k]:.3f}  [{lo:.3f}, {hi:.3f}]")
# paired bootstrap IDK FT - Normal FT
c_n, y_n = np.array(data["Normal FT"]["confidences"]), np.array(data["Normal FT"]["corrects"])
c_i, y_i = np.array(data["IDK FT"]["confidences"]), np.array(data["IDK FT"]["corrects"])
d = {"ece": [], "brier": [], "auroc": []}
for _ in range(2000):
    i = rng.integers(0, len(y_n), len(y_n))
    if len(set(y_n[i])) < 2 or len(set(y_i[i])) < 2:
        continue
    d["ece"].append(ece(c_i[i], y_i[i]) - ece(c_n[i], y_n[i]))
    d["brier"].append(((c_i[i] - y_i[i]) ** 2).mean() - ((c_n[i] - y_n[i]) ** 2).mean())
    d["auroc"].append(roc_auc_score(y_i[i], c_i[i]) - roc_auc_score(y_n[i], c_n[i]))
print("\nIDK FT minus Normal FT (paired bootstrap, 95% CI):")
for k, v in d.items():
    print(f"  {k:7s} {np.mean(v):+.3f}  [{np.percentile(v, 2.5):+.3f}, {np.percentile(v, 97.5):+.3f}]")
# confidence of refusals vs answers on IDK rows for IDK FT
print("\nIDK FT mean confidence: unanswerable rows", round(c_i[idk].mean(), 3), "| answerable rows", round(c_i[~idk].mean(), 3))
