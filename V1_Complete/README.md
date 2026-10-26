# V1 Project Bundle — Confidence Calibration and Hallucination Reduction in Small Language Models

Everything for the v1 experiment in one folder: code, data, results, paper, report, review slides, demo site and a review guide.

Start with `docs/REVIEW_GUIDE.md`.

| Folder | What is inside |
|---|---|
| `docs/` | `REVIEW_GUIDE.md` (full explanation, speaking notes, likely questions, known inconsistencies) and `PAPER_WORDING_FIXES.md` (paste-ready corrections for the paper) |
| `report/` | `BCSE497J_Project_I_Report_Final.docx` (open in Word, click Yes to update fields so the contents and figure/table lists fill in) |
| `review/` | `Review3_Presentation.pptx` (18 slides) |
| `paper/` | Final Springer paper PDF (`Confidence_Calibration_SLMs_Springer.pdf`); `paper.docx`, drafts and build script are OLDER versions |
| `code/` | v1 scripts: data prep, QLoRA training, evaluation, plots, verification |
| `data/` | Source CSVs, JSONL, splits for 0/10/15/20% abstention, source-inventory workbook |
| `results/` | `comparison_fixed.md` and `figures/` (the originals used in the paper) |
| `diagrams/` | Architecture diagrams and Gantt chart. `superseded_earlier_runs/` = old screenshots with different numbers, do not present |
| `notebooks/` | Early Colab notebooks |
| `chatbot_site/` | Flask demo (`python chatbot_site/app.py`) |

Headline result (449 test queries, 79 unanswerable): hallucination rate 0.506 (base) → 1.000 (normal fine-tune) → 0.139 (15% abstention); ECE 0.622 → 0.408 → 0.304; AUROC 0.557 → 0.764 → 0.810.

Not included: trained adapters and the v1 per-example calibration export (not found on disk), later-version files (v2, v3), and the raw curriculum PDFs.
