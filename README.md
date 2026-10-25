# Confidence Calibration and Hallucination Detection

Controlled abstention fine-tuning of a small language model (Phi-3-mini, QLoRA) on VIT curriculum question
answering: dataset preparation, LoRA fine-tuning, calibration and hallucination evaluation, a paper, and a Flask
chatbot demo site.

## Headline results (v1, 449 test queries, 79 unanswerable)

| System | Hallucination rate | IDK F1 | Answerable acc. | ECE | AUROC |
|---|---|---|---|---|---|
| Base Phi-3-mini | 0.506 | 0.333 | 0.227 | 0.622 | 0.557 |
| Standard fine-tune (0% IDK) | 1.000 | 0.000 | 0.451 | 0.408 | 0.764 |
| Abstention-aware fine-tune (15% IDK) | **0.139** | **0.840** | 0.408 | **0.304** | **0.810** |

The paper is in `paper_content/` (section drafts, figures, `build_paper.py`, and the built `paper.docx`).
Rebuild it with `python paper_content/build_paper.py --template <splnproc2510.docm>` (needs `pillow`).

## Layout

| Path | What it is |
|---|---|
| root `*_v3.py`, `colab_v3.ipynb` | **Current pipeline (v3)** |
| `final_qa_dataset.csv`, `combined_qa_finetuning.csv` | Source data used by v3 and the demo site |
| `v1/` | Earlier experiments (v1 scripts and datasets, plus `v1/v2_files/`) kept for reference |
| `chatbot_site/` | Flask demo site |
| `paper_content/` | Paper: section drafts, figures, `build_paper.py` (Springer LNCS Word template), `paper.docx` |

## v3 pipeline (Google Colab T4)

Open `colab_v3.ipynb` in Colab, or run the steps yourself:

```bash
pip install -r requirements_v3.txt
python prepare_v3_data.py                                   # splits (repair text, dedupe, canonical IDK answer)
python augment_v3.py                                        # paraphrase train questions with a local model (no API key)
python prepare_v3_data.py --paraphrases data_v3/paraphrases.jsonl --max-para 3
python train_lora_v3.py --output-dir adapters/qwen3b        # QLoRA, best epoch chosen by validation accuracy
python eval_v3.py --adapter adapters/qwen3b/best            # greedy answer + token prob + 5 sampled answers
python calibrate_v3.py                                      # calibrate on valid, report on test
```

What changed from v1/v2:
- Chat-template prompts with a fixed system prompt, used identically in training and evaluation.
- Train-only paraphrase augmentation; valid and test are never augmented.
- Checkpoint selection by validation accuracy instead of eval loss.
- Correctness uses a key-fact check (course codes and numbers) alongside token F1; both are reported.
- Confidence = self-consistency over sampled answers plus token probability, then calibrated on the validation
  split. The abstention threshold is also chosen on validation. Test data is used only for reporting.
- Reports ECE, Brier, AUROC, hallucination rate, false-refusal rate, selective accuracy and a risk-coverage table.

## Extra training rows from the curriculum PDFs (optional, train split only)

The existing dataset is never modified. `extract_courses.py` reads the course tables (code, title, L-T-P-C,
credits) out of the curriculum PDFs by rule, and `make_course_qa.py` turns them into extra Q&A rows
(`extra_course_qa.jsonl`). `prepare_v3_data.py --extra ...` appends a chosen number of them to the TRAIN split
only; valid and test stay identical, and any question already in valid/test is excluded. Unanswerable rows are
topped up so the IDK share stays near `--idk-ratio`.

```bash
python extract_courses.py "<folder with the PDFs>" --out course_tables.jsonl   # needs: pip install pymupdf
python make_course_qa.py                                                       # -> extra_course_qa.jsonl
python prepare_v3_data.py --extra extra_course_qa.jsonl --extra-kinds credits --max-extra 4000
```

## Run the chatbot site

```powershell
pip install -r chatbot_site/requirements.txt
python chatbot_site/app.py
```

Open `http://127.0.0.1:5000`. The default backend is a dataset lookup. To use a LoRA adapter instead, set
`CHATBOT_BACKEND=lora` and `ADAPTER_PATH`. The site's `lora` backend still targets the v1 Phi-3 adapters; it has
not been updated for v3 adapters.
