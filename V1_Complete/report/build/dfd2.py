# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

OUT = r"C:\Anisha\VITV\Sem 7\7th sem project\semrpoject\V1_Complete\diagrams" + "\\"
plt.rcParams["font.family"] = "Times New Roman"
NAVY, GREY, LIGHT = "#002147", "#444444", "#eef2f8"

fig, ax = plt.subplots(figsize=(11.5, 5.4))
ax.set_xlim(0, 11.5); ax.set_ylim(0, 5.4); ax.axis("off")


def proc(x, text):
    ax.add_patch(FancyBboxPatch((x, 2.0), 1.9, 0.95, boxstyle="round,pad=0.02,rounding_size=0.1", fc=LIGHT, ec=NAVY, lw=1.5))
    ax.text(x + 0.95, 2.475, text, ha="center", va="center", fontsize=9.6, fontweight="bold")


def store(cx, text):
    x = cx - 0.775
    ax.add_patch(Rectangle((x, 4.15), 1.55, 0.7, fc="white", ec=NAVY, lw=1.5))
    ax.plot([x + 0.14, x + 0.14], [4.15, 4.85], color=NAVY, lw=1.5)
    ax.text(cx + 0.07, 4.5, text, ha="center", va="center", fontsize=8.3)


def ent(x, y, text):
    ax.add_patch(Rectangle((x, y), 1.7, 0.7, fc="#fff3cc", ec="#111", lw=1.5))
    ax.text(x + 0.85, y + 0.35, text, ha="center", va="center", fontsize=10, fontweight="bold")


def arr(p1, p2, label=None, dx=0.0, dy=0.0, ha="center"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13, lw=1.3, color=GREY))
    if label:
        ax.text((p1[0] + p2[0]) / 2 + dx, (p1[1] + p2[1]) / 2 + dy, label, fontsize=8.6, style="italic", ha=ha, va="center",
                bbox=dict(fc="white", ec="none", pad=0.8))


px = [0.3, 2.5, 4.7, 6.9, 9.1]
proc(px[0], "1.0  Build dataset\nmixtures")
proc(px[1], "2.0  Fine-tune LoRA\nadapters (QLoRA)")
proc(px[2], "3.0  Generate answers\n+ confidence")
proc(px[3], "4.0  Refusal and\ncalibration metrics")
proc(px[4], "5.0  Plot figures\nand tables")
store(0.95, "D1  QA source CSVs")
store(2.75, "D2  Splits (JSONL)")
store(4.75, "D3  LoRA adapters")
store(8.85, "D4  Results (per example)")
arr((1.0, 4.15), (1.0, 2.95), "QA rows", dx=-0.4)
arr((1.9, 2.95), (2.4, 4.15), "splits", dx=-0.45, dy=0.0)
arr((3.2, 4.15), (3.2, 2.95), "train file", dx=0.4)
arr((4.1, 2.95), (4.4, 4.15), "adapter", dx=0.55, dy=0.1)
arr((5.3, 4.15), (5.3, 2.95), "load adapter", dx=0.6)
arr((6.6, 2.475), (6.9, 2.475))
ax.text(6.75, 1.85, "answers, confidence", fontsize=8.6, style="italic", ha="center", va="center")
arr((8.2, 2.95), (8.55, 4.15), "per-example confidence,\ncorrectness", dx=-1.0, dy=0.0)
arr((9.6, 4.15), (9.8, 2.95), "results", dx=0.55)
ent(0.3, 0.45, "Researcher")
ent(9.4, 0.45, "Researcher")
arr((1.15, 1.15), (1.15, 2.0), "chooses ratio", dx=0.75, dy=-0.05)
arr((10.25, 2.0), (10.25, 1.15), "figures, tables", dx=-0.85)
ax.text(5.75, 5.2, "Level-1 data flow diagram of the evaluation pipeline", ha="center", fontsize=12, fontweight="bold", color=NAVY)
ax.text(4.2, 1.1, "D2 also supplies the test file to process 3.0", fontsize=8.6, style="italic", color="#444")
fig.savefig(OUT + "dfd_level1.png", dpi=200, bbox_inches="tight", facecolor="white")
print("dfd ok")
