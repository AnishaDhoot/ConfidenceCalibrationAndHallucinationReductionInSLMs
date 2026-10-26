# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Ellipse, FancyArrowPatch

OUT = r"C:\Anisha\VITV\Sem 7\7th sem project\semrpoject\V1_Complete\diagrams" + "\\"
plt.rcParams["font.family"] = "Times New Roman"
NAVY, GOLD, GREY, LIGHT = "#002147", "#ffc000", "#444444", "#eef2f8"


def box(ax, x, y, w, h, text, fc=LIGHT, ec=NAVY, fs=10, bold=False, round_=True):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08" if round_ else "square,pad=0",
                       fc=fc, ec=ec, lw=1.4)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal",
            color="#111", wrap=True)


def store(ax, x, y, w, h, text, fs=9.5):
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec=NAVY, lw=1.4))
    ax.plot([x + 0.12, x + 0.12], [y, y + h], color=NAVY, lw=1.4)
    ax.text(x + w / 2 + 0.05, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, p1, p2, label=None, off=(0, 0.08), fs=8.5, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=12, lw=1.2, color=GREY,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls))
    if label:
        ax.text((p1[0] + p2[0]) / 2 + off[0], (p1[1] + p2[1]) / 2 + off[1], label, ha="center", va="bottom",
                fontsize=fs, color="#222", style="italic", bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9))


# ------------------------------------------------------------- data flow diagram
fig, ax = plt.subplots(figsize=(11, 6.2))
ax.set_xlim(0, 11); ax.set_ylim(0, 6.2); ax.axis("off")
box(ax, 0.15, 2.6, 1.5, 0.9, "Researcher", fc="#fff3cc", bold=True, round_=False)
box(ax, 2.2, 4.6, 2.1, 0.95, "1.0  Build dataset\nmixtures", bold=True)
box(ax, 5.3, 4.6, 2.1, 0.95, "2.0  Fine-tune LoRA\nadapters (QLoRA)", bold=True)
box(ax, 8.4, 4.6, 2.3, 0.95, "3.0  Generate answers\n+ confidence", bold=True)
box(ax, 8.4, 2.5, 2.3, 0.95, "4.0  Compute refusal and\ncalibration metrics", bold=True)
box(ax, 5.3, 2.5, 2.1, 0.95, "5.0  Plot figures\nand tables", bold=True)
store(ax, 2.2, 0.6, 2.2, 0.7, "D1  QA source files (CSV)")
store(ax, 5.1, 0.6, 2.3, 0.7, "D2  Train / valid / test (JSONL)")
store(ax, 8.2, 0.6, 2.5, 0.7, "D3  LoRA adapters")
store(ax, 2.2, 2.7, 2.2, 0.7, "D4  Per-example results")
arrow(ax, (3.3, 1.3), (3.25, 4.6), "QA rows", off=(-0.55, 0))
arrow(ax, (3.4, 4.6), (5.8, 1.3), "splits", off=(0.35, 0.35), rad=-0.1)
arrow(ax, (6.0, 1.3), (6.1, 4.6), "train file", off=(0.45, 0.4))
arrow(ax, (7.4, 5.1), (8.4, 5.1), None)
arrow(ax, (6.3, 4.6), (9.3, 1.3), "adapter", off=(0.9, 0.3), rad=0.15)
arrow(ax, (9.55, 1.3), (9.55, 4.6), "load adapter", off=(0.55, -0.5))
arrow(ax, (9.55, 4.6), (9.55, 3.45), "answers,\nconfidence", off=(0.7, -0.1))
arrow(ax, (8.4, 2.97), (7.4, 2.97), "metrics", off=(0, 0.08))
arrow(ax, (5.3, 2.97), (4.4, 3.05), None)
arrow(ax, (8.9, 3.45), (4.4, 3.3), "per-example confidence, correctness", off=(1.2, 0.18), rad=-0.08)
arrow(ax, (1.65, 3.05), (2.2, 3.05), "figures", off=(0, 0.1))
arrow(ax, (0.9, 3.5), (2.3, 4.7), "run ratio", off=(-0.3, 0.1), rad=-0.1)
ax.text(5.5, 5.95, "Level-1 data flow diagram of the evaluation pipeline", ha="center", fontsize=12, fontweight="bold", color=NAVY)
fig.savefig(OUT + "dfd_level1.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ------------------------------------------------------------- use case diagram
fig, ax = plt.subplots(figsize=(10, 6.4))
ax.set_xlim(0, 10); ax.set_ylim(0, 6.4); ax.axis("off")
ax.add_patch(Rectangle((2.6, 0.3), 5.0, 5.7, fc="white", ec=NAVY, lw=1.6))
ax.text(5.1, 5.75, "Calibration and abstention system", ha="center", fontsize=11.5, fontweight="bold", color=NAVY)
cases = [(3.9, "Build dataset mixtures"), (3.1, "Train adapters"), (2.3, "Evaluate systems"), (1.5, "View calibration plots"),
         (0.9, None)]
uc = [("Build dataset mixtures", 5.15, 5.0), ("Train Normal FT and IDK FT adapters", 5.15, 4.15), ("Evaluate refusal and calibration", 5.15, 3.3),
      ("View plots and tables", 5.15, 2.45), ("Ask a question", 5.15, 1.6), ("See answer, confidence or abstention", 5.15, 0.9)]
for text, x, y in uc:
    ax.add_patch(Ellipse((x, y), 3.7, 0.62, fc=LIGHT, ec=NAVY, lw=1.3))
    ax.text(x, y, text, ha="center", va="center", fontsize=9.5)


def actor(x, y, name):
    ax.add_patch(plt.Circle((x, y + 0.55), 0.17, fc="white", ec="#111", lw=1.4))
    ax.plot([x, x], [y + 0.38, y - 0.05], color="#111", lw=1.4)
    ax.plot([x - 0.28, x + 0.28], [y + 0.25, y + 0.25], color="#111", lw=1.4)
    ax.plot([x, x - 0.25], [y - 0.05, y - 0.45], color="#111", lw=1.4)
    ax.plot([x, x + 0.25], [y - 0.05, y - 0.45], color="#111", lw=1.4)
    ax.text(x, y - 0.75, name, ha="center", fontsize=10.5, fontweight="bold")


actor(1.1, 3.3, "Researcher")
actor(8.9, 1.5, "Student / user")
for _, x, y in uc[:4]:
    ax.plot([1.45, 3.3], [3.3, y], color=GREY, lw=1)
for _, x, y in uc[4:]:
    ax.plot([8.55, 7.0], [1.55, y], color=GREY, lw=1)
fig.savefig(OUT + "use_case.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ------------------------------------------------------------- sequence diagram
fig, ax = plt.subplots(figsize=(11, 6.6))
ax.set_xlim(0, 11); ax.set_ylim(0, 6.6); ax.axis("off")
names = ["Student", "Web page\n(chat.js)", "Flask API\n/api/chat", "Answer backend\n(lookup or LoRA)", "Phi-3 + adapter\n(LoRA mode only)"]
xs = [1.0, 3.3, 5.6, 7.9, 10.0]
for x, n in zip(xs, names):
    box(ax, x - 0.85, 5.55, 1.7, 0.7, n, bold=True, fs=9.5)
    ax.plot([x, x], [0.4, 5.55], color="#888", lw=1, ls="--")
msgs = [(0, 1, 5.0, "1  types a question"),
        (1, 2, 4.4, "2  POST question"),
        (2, 3, 3.8, "3  answer(question)"),
        (3, 4, 3.2, "4  generate + token probabilities"),
        (4, 3, 2.7, "5  answer, mean token probability"),
        (3, 2, 2.1, "6  {answer, confidence, abstained}"),
        (2, 1, 1.5, "7  JSON response"),
        (1, 0, 0.9, "8  shows answer, confidence bar or abstention")]
for a, b, y, t in msgs:
    arrow(ax, (xs[a], y), (xs[b], y), None)
    ax.text((xs[a] + xs[b]) / 2, y + 0.07, t, ha="center", va="bottom", fontsize=9, style="italic")
ax.text(5.5, 6.45, "Sequence of one chat query", ha="center", fontsize=12, fontweight="bold", color=NAVY)
ax.text(5.5, 0.15, "In lookup mode steps 4 and 5 are skipped and the confidence is the retrieval score", ha="center", fontsize=8.5, color="#555")
fig.savefig(OUT + "sequence_chat.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("diagrams ok")
