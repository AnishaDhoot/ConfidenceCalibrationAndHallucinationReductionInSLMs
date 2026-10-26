# Project Review Guide — Confidence Calibration and Hallucination Reduction in Small Language Models

Team: Anisha Dhoot, Nataliya Singh, Yash Hirani · Guide: Prof. Swathi J N · BCSE497J Project-I · Review 3 (28 Oct 2026)

This guide explains the whole v1 project so that you can answer questions on any part of it. Every number is checked against the paper, the v1 data files and the v1 scripts. Section 11 lists the places where the paper, data or code do not fully agree. Read that section before the review: an examiner who looks closely is most likely to find those.

---

## 1. The project in one minute

**Problem.** A small language model (SLM) fine-tuned on a university's documents answers every question, including ones it cannot know, such as "Is my re-FAT application approved?", "Is the library open right now?" or "Credits of course XYZ1234" (a code that does not exist). It sounds just as sure when it is wrong. For students this is worse than "I don't know".

**Idea.** Fine-tune the model with a controlled share (15%) of examples whose correct answer is a fixed refusal ("I don't have enough information in my knowledge base to answer this question accurately."), next to normal question-answer examples. This is refusal-aware tuning (R-Tuning style). Change only the training data. Add no calibration loss, no temperature scaling and no retrieval.

**Experiment.** Three systems on the same 449-question test set (370 answerable, 79 unanswerable):

| System | What it is |
|---|---|
| Base | Phi-3-mini-4k-instruct, no fine-tuning |
| Normal FT | LoRA adapter trained on answerable questions only (0% IDK) |
| IDK FT | LoRA adapter trained on the same answerable questions plus 15% refusal examples |

**Result.**

| | Base | Normal FT | IDK FT |
|---|---|---|---|
| Hallucination rate (HR) | 0.506 | 1.000 | **0.139** |
| IDK F1 | 0.333 | 0.000 | **0.840** |
| Answerable accuracy | 0.227 | **0.451** | 0.408 |
| ECE | 0.622 | 0.408 | **0.304** |
| Brier | 0.480 | 0.351 | **0.285** |
| AUROC | 0.557 | 0.764 | **0.810** |

**Take-away.** Fine-tuning gives knowledge but removes abstention (HR rises from 0.506 to 1.000). Abstention is not a free by-product; it must be supervised. Fifteen percent refusal data cuts HR to 0.139 and improves calibration, for 15 false refusals out of 370 answerable questions (4.1%).

**Honest caveats.** One model, one ratio, one seed and one institution. All systems are still over-confident. Part of the calibration gain comes from the model repeating one fixed sentence with very high confidence.

---

## 2. Background concepts (what you must be able to explain)

### 2.1 Small Language Model (SLM)
A language model of roughly 1–4 billion parameters. It is cheap enough to run on-premise or on a laptop. The cost is limited memory for facts, so it fills gaps with plausible text. Phi-3-mini has 3.8 B parameters and is trained on carefully filtered "textbook-quality" data. That is why it performs well for its size.

### 2.2 Hallucination
Here the definition is deliberately narrow. **A hallucination is an answer attempt to a question that has no answer in the source documents.** Formally `hallucination = z(q)·(1 − a(r))`, where `z(q) = 1` if the question is unanswerable and `a(r) = 1` if the model abstained.
- **Hallucination rate** `HR = FN / (TP + FN)` is the share of unanswerable questions that the model answers. It equals `1 − IDK recall`.
- A *wrong answer to an answerable question* is an **error**, not counted as a hallucination, because it may be a paraphrase or partial answer that word overlap cannot judge.

### 2.3 Confidence and calibration
- **Confidence** `c(q)` is a number in [0,1] that the system attaches to its answer.
- **Calibrated** means that among answers given with confidence 0.8, about 80% are correct.
- **Over-confident** means accuracy is below confidence (the curve lies under the diagonal). This is the usual failure of language models.
- Calibration alone is not enough. A model that always says "0.45", which is its average accuracy, is perfectly calibrated and useless. Confidence must also **discriminate** (correct answers get higher scores than wrong ones). AUROC measures this.

### 2.4 The metrics
| Metric | Meaning | Good direction |
|---|---|---|
| **ECE** (Expected Calibration Error) | Split confidences into 10 equal bins. For each bin take \|mean confidence − accuracy\|. Average weighted by bin size. | lower |
| **Brier score** | Mean of (confidence − correct)². No binning. Mixes calibration and sharpness. | lower |
| **AUROC** | Probability that a random correct answer has higher confidence than a random wrong one. 0.5 = no information. | higher |
| **NLL** (reference) | Teacher-forced negative log-likelihood of the reference answer. How "surprised" the model is by the true answer. | lower |
| **Token F1** | SQuAD-style word-overlap F1 between answer and reference (lower-cased, punctuation removed). | higher |
| **Exact match** | Normalised strings identical. Almost always 0 for free-form answers, hence F1. | higher |
| **IDK precision / recall / F1** | Treat refusing as a classifier. Precision = of refusals, how many were needed. Recall = of unanswerable questions, how many were refused. | higher |
| **False refusal rate (FRR)** | Refused answerable questions / answerable questions. The cost of abstention. | lower |
| **Reliability diagram** | Plot of accuracy against confidence per bin; the diagonal is perfect. Shows over- versus under-confidence, which ECE hides. | closer to diagonal |

**Worked ECE intuition.** If every answer has confidence 0.9 and 60% are correct, ECE = 0.30. The base model's ECE of 0.622 means that on average it is about 62 points too confident.

### 2.5 Refusal-aware correctness (the key definition)
One binary label `y` is needed to relate confidence to outcomes for both kinds of question:

- unanswerable question → correct **iff the model refused**
- answerable question → correct **iff it did not refuse AND token-F1(answer, reference) ≥ 0.5**

Check on lazy strategies. A model that refuses everything is right only on the unanswerable share, 79/449 = **17.6%**. A model that never refuses is capped by its answerable accuracy. Neither lazy strategy scores well.

### 2.6 Sequence-level confidence
For greedy decoding of up to T tokens:

`c(q) = exp( (1/T) · Σ log p(token_t | previous tokens, question) )`

This is the **geometric mean of the token probabilities** (length-normalised sequence likelihood, Malinin & Gales). Without the 1/T a long correct answer would score lower than a one-word wrong one. It costs nothing extra and needs no second model.

### 2.7 Abstention, selective prediction, R-Tuning
- **Selective prediction** (Chow 1970; Geifman & El-Yaniv 2017): a model may reject inputs; there is a trade-off between error and coverage.
- **R-Tuning** (Zhang et al., NAACL 2024): fine-tune an LLM so that it says "I don't know" on questions beyond its knowledge. The refusal skill generalises.
- **Over-abstention** (Wen et al. survey): refusing questions the model could answer. This is why false refusals are measured here.
- This project differs from R-Tuning in three ways: (1) the abstention share is a controlled variable; (2) a sub-4 B model on a 4 GB laptop GPU; (3) hallucination, calibration and false-refusal cost are measured together.

### 2.8 Parameter-efficient fine-tuning
- **LoRA**: freeze the weights W and learn a low-rank update ΔW = B·A (rank r). Only A and B are trained. There is no extra inference latency, and the adapter file is small.
- **QLoRA**: store the frozen base in 4-bit and train LoRA on top. It combines three ideas:
  - **NF4** (4-bit NormalFloat, a quantisation grid matched to normally distributed weights);
  - **double quantisation** (the quantisation constants are quantised too);
  - **paged optimisers** (optimiser state is paged to CPU memory when the GPU is short). The project uses the paged 8-bit AdamW.
- **Memory tricks used here**: batch size 1 with 8 gradient-accumulation steps (effective batch 8); gradient checkpointing (recompute activations); KV cache off during training; bfloat16 compute.
- Settings: r = 8, α = 16, dropout 0.05, on `qkv_proj, o_proj, gate_up_proj, down_proj` (all attention and feed-forward projections of Phi-3).

### 2.9 Other terms
- **Chat template.** Phi-3 format: `<|user|>\n{question}<|end|>\n<|assistant|>\n{answer}<|end|>`. The same format is used in training and testing.
- **Greedy decoding.** Always take the most probable next token. Deterministic, so the results are reproducible.
- **Closed-book QA.** The model answers from what it memorised during fine-tuning; nothing is retrieved at inference.

---

## 3. Data (Module 1–2)

### 3.1 Sources
| File | Rows | Role |
|---|---|---|
| `data/final_qa_dataset.csv` | 2,349 | Knowledge QA from public VIT documents |
| `data/combined_qa_finetuning.csv` | 980 | Supplementary set, mostly "I don't know" style rows |

- The source inventory `data/VIT_QA_Resource_Breakdown.xlsx` accounts for all 2,349 rows. **1,930 (82%) come from curriculum and syllabus PDFs.** The rest come from regulations (74), the student-life handbook (108), the code of conduct (54), the academic calendar (41), Ph.D. admission and research guides (130), and fee and admission information (12).
- **Non-public rows were removed**: 898 from the knowledge file (3,224 → 2,349 after 23 additions) and 380 from the supplementary file (1,360 → 980). Mostly these were company-specific placement-drive details.
- The five categories of unanswerable query are: personal or record-specific (attendance, re-FAT status, seating); time-sensitive or live (is FAT out, is the gate locked); unsupported specifics (valid detail absent from the documents); non-existent entities (fake course codes); out-of-domain.

### 3.2 Building a mixture (`code/prepare_expc_data.py`)
1. Load both CSVs.
2. Flag refusal rows with a regex: `don't know | I don't | cannot verify | cannot check | cannot submit | cannot see | real-time | moment to moment | I cannot | not in my knowledge`.
   - Result: 3 rows from the knowledge file and 789 from the supplementary file are flagged, so the IDK pool has 792 rows.
   - The remaining 2,346 + 191 = **2,537** rows form the answerable pool.
3. Rewrite every flagged answer to **one canonical string**: *"I don't have enough information in my knowledge base to answer this question accurately."*
4. For a target ratio ρ sample `n_idk = floor(N_reg · ρ / (1 − ρ))` refusal rows without replacement (seed 42). For ρ = 0.15: floor(2537 · 0.15 / 0.85) = **447**.
5. Shuffle with seed 42 and split 70/15/15. Check that no question appears in both train and test.

| Mixture | Total | Train | Valid | Test |
|---|---|---|---|---|
| IDK-0 (Normal FT) | 2,537 | 1,775 (0) | 380 (0) | 382 (0) |
| IDK-15 (IDK FT) | 2,984 | 2,088 (312) | 447 (56) | 449 (79) |

IDK share: train 14.9%, valid 12.5%, test 17.6%. The splits are random, not stratified, so the shares differ a little. I checked these numbers against the `data/splits` files and they match.

Files for 10% and 20% (`expc_idk10_*`, `expc_idk20_*`) were also generated. **No results for them are reported in the paper.** They are prepared for the future ratio ablation.

### 3.3 Why one canonical refusal?
- Measurement: varied refusals ("please check VTOP", "contact your proctor") make refusal detection noisy.
- Safety: several hedge styles risk teaching the model to hedge on answerable questions.
- Cost: the refusal becomes a memorised sentence rather than a measure of uncertainty. This is stated as a limitation, and it affects how the confidence results are read (Section 6.4).

---

## 4. Training (Module 3, `code/train_lora.py`, `code/exp_c_train.py`)

Flow: load Phi-3-mini in 4-bit → `prepare_model_for_kbit_training` → attach LoRA → format each row with the chat template → truncate to 512 tokens → SFTTrainer.

| Setting | Value |
|---|---|
| Epochs | 3 |
| Batch | 1 × 8 accumulation = 8 |
| Learning rate | 2e-4, linear decay |
| Optimiser | paged AdamW 8-bit |
| Precision | bf16 compute, 4-bit weights |
| Hold-out | 10% of the training file for per-epoch validation loss (so ≈ 235 optimiser steps/epoch for IDK-15: 2088 × 0.9 / 8) |
| Seed | 42 |
| GPU | RTX 3050 Ti Laptop, 4 GB |

- Rows starting with "NOT FOUND" are skipped (placeholder for unsourced facts). A rote non-answer would harm calibration.
- `exp_c_train.py` is a wrapper: build data for a ratio, then train. Normal FT = ratio 0.0, IDK FT = ratio 0.15.
- The two adapters were trained under identical conditions, so any difference comes from the abstention data.
- The trained adapters are **not in this bundle** (they were produced on your machine or on Colab). The scripts reproduce them.

---

## 5. Evaluation (Module 4–5)

Systems are evaluated one after another because two 4-bit copies of Phi-3-mini do not fit in 4 GB. The CUDA cache is cleared between runs.

| Script | What it produces | Max new tokens |
|---|---|---|
| `eval_compare_fixed.py` | Refusal-aware accuracy, EM, token F1, **ECE, Brier, NLL, AUROC**, reliability bins, per-example export | 64 |
| `compare_idk_stats.py` | Refusal rate, IDK precision/recall/F1, **FP, FN (hallucinations)**, answerable accuracy | 100 |
| `calibration_eval_utils.py`, `run_calibration_utils.py` | AUROC and reliability-diagram helpers, applied to the export | – |
| `make_figures.py` | Combined reliability diagram, ROC curves, confidence histograms | – |
| `verify_dataset.py` | Label/leakage checks | – |

The test set for all three systems is `expc_idk15_test.jsonl` (449 rows), because the IDK-0 test set has no unanswerable questions. Consequence: **15 of the 449 test questions are also in the IDK-0 training set** (the two mixtures were split independently), which slightly favours Normal FT. I verified the number 15 against the files. There is no overlap between IDK-15 train and IDK-15 test.

**Two correctness definitions are in the paper. Do not mix them.**
- Table 2 (`compare_idk_stats.py`): "answerable accuracy" = answer shares more than 40% of the reference *words* (set overlap). Values 0.227 / 0.451 / 0.408.
- Table 3 (`eval_compare_fixed.py`): "accuracy" = refusal-aware correctness with F1 ≥ 0.5 on *all* queries. Values 0.100 / 0.327 / 0.477.
They measure different things, so 0.451 and 0.327 are not contradictory.

---

## 6. Results explained

### 6.1 Hallucination (Table 2)
| System | Refusal rate | Precision | Recall | F1 | HR | FN | FP | Answerable acc. |
|---|---|---|---|---|---|---|---|---|
| Base | 0.345 | 0.252 | 0.494 | 0.333 | 0.506 | 40 | 116 | 0.227 |
| Normal FT | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 79 | 0 | 0.451 |
| IDK FT | 0.185 | 0.819 | 0.861 | 0.840 | 0.139 | 11 | 15 | 0.408 |

- **Base**: refuses 155 of 449 questions, but 116 of those are answerable. So about 75% of its refusals are unneeded. Its recall of 0.494 is not knowledge-awareness: it uses generic "I can't access institutional information" replies.
- **Normal FT**: refuses nothing and answers all 79 unanswerable questions. Fine-tuning made hallucination worse than doing nothing.
- **IDK FT**: refuses 83 times (68 correct, 15 wrong). HR drops 86% against Normal FT and 73% against Base. Its refusal rate of 18.5% is close to the true 17.6%.
- Arithmetic checks: 68 / 83 = 0.819; 68 / 79 = 0.861; 11 = 79 − 68; 15 / 370 = 4.1%; 116 / 370 = 31.4%.

### 6.2 Calibration (Table 3)
| Metric | Base | Normal FT | IDK FT |
|---|---|---|---|
| Accuracy (refusal-aware) | 0.100 | 0.327 | 0.477 |
| Exact match | 0.000 | 0.051 | 0.189 |
| Token F1 | 0.168 | 0.391 | 0.515 |
| ECE | 0.622 | 0.408 | 0.304 |
| Brier | 0.480 | 0.351 | 0.285 |
| Reference NLL | 2.405 | 1.663 | 0.965 |
| AUROC | 0.557 | 0.764 | 0.810 |

- ECE falls 51% overall: (0.622 − 0.304) / 0.622. About 25% of the original comes from abstention data: (0.408 − 0.304) / 0.408.
- Brier shows the same trend, so it is not a binning artefact.
- Reliability diagram: all curves are below the diagonal (over-confident). Base is almost flat (accuracy ≈ 0–0.15 whatever the confidence). Normal FT plateaus near 0.55 accuracy above 0.8 confidence. IDK FT rises monotonically to ≈ 0.80 accuracy at ≈ 0.97 confidence, but at ≈ 0.65 confidence it is correct only 19% of the time.
- **Two caveats the paper itself states.**
  1. Much of the gain comes from unanswerable questions: correct refusals are the exact same sentence, so they get near-perfect token F1 and very high confidence. Exact matches rise from 23 to 85 of 449, mostly for this reason, and the NLL drop partly reflects 79 identical references.
  2. The rise in accuracy (0.327 → 0.477, about 67 more correct responses) is about equal to the number of unanswerable questions now answered correctly. On answerable questions the quality is about the same (token F1 about 0.39 vs 0.40).

### 6.3 Confidence separation
- Base: correct and wrong answers have nearly identical confidence (0.55–0.85). AUROC 0.557.
- Normal FT: correct answers move toward 0.7–1.0, wrong ones sit at 0.55–0.75, but the wrong ones have a tail to 0.95 (confident hallucinations). AUROC 0.764.
- IDK FT: AUROC 0.810. A tall peak (over 80 correct answers) at 0.95–1.0, mostly canonical refusals. The ROC reaches about 0.28 TPR at almost zero FPR (Normal FT about 0.05). At FPR 0.1 it keeps about 43% of correct answers (Normal FT 25%). Curves converge above FPR ≈ 0.4.
- The same peak contains about 15 wrong answers: the false refusals. A fixed sentence can be produced with high confidence even when it is the wrong action.

### 6.4 Cost of abstention
- FRR 4.1% (15 / 370); Base 31.4%; Normal FT 0%.
- Answerable accuracy 0.451 → 0.408 (167 → 151 correct, a loss of 16, about the number of false refusals). On the questions IDK FT attempted: 151 / 355 = 42.5% against Normal FT 167 / 370 = 45.1%. So the loss comes from refusing, not from answering worse.
- Trade-off: lose 16 correct answers, avoid 68 hallucinations, about four per correct answer lost.
- Normal FT's accuracy is slightly overstated because of the 15 overlapping test questions.

### 6.5 Ablation by stage (Table 4)
| | Domain adaptation (Base → Normal FT) | Abstention data (Normal FT → IDK FT) |
|---|---|---|
| HR | +0.494 | −0.861 |
| IDK F1 | −0.333 | +0.840 |
| FRR | −0.314 | +0.041 |
| Answerable acc. | +0.224 | −0.043 |
| Token F1 | +0.223 | +0.124 |
| ECE | −0.214 | −0.104 |
| Brier | −0.129 | −0.066 |
| AUROC | +0.207 | +0.046 |

Reading: domain adaptation supplies knowledge and the biggest single calibration gain; abstention data repairs what adaptation destroyed (refusal) and adds a further, smaller calibration gain at a small price.

### 6.6 Qualitative errors (IDK FT)
1. **11 missed abstentions.** The question looks normal (a plausible but fake course code, or a detail absent from every document) and the model invents a title or credits. Most dangerous because the answer is fluent.
2. **15 false refusals.** Probably triggered by surface wording shared with refusal examples ("my", "attendance", "schedule", "today") or by rarely seen programmes. Their confidence is above 0.95.
3. **Wrong answers to answerable questions.** The biggest group: wrong numbers (credits, codes) or repeating the question. Abstention training does not touch these; they come from limited memory of a 3.8 B model after about 1,800 training examples.

---

## 7. Demo chatbot (Module 6, `chatbot_site/`)
- Flask site with Home, Academics and Campus-life pages and a chat widget (`/api/chat`).
- **Default backend = retrieval over `final_qa_dataset.csv`** (sentence embeddings `all-MiniLM-L6-v2` blended with TF-IDF; a course-code bonus). It abstains with the canonical sentence if the best score is below a threshold (default 0.43), or if the question names an entity that appears nowhere in the data.
- **Optional backend = `lora`** (`CHATBOT_BACKEND=lora`, `ADAPTER_PATH=...`) loads Phi-3-mini + an adapter and shows the same mean-token-probability confidence as the evaluation.
- Every reply is labelled with the backend, so **do not present the default lookup as the fine-tuned model.** If asked "is the chatbot your model?", answer: the site supports both; the default demo is a lookup so that it runs without a GPU, and the LoRA mode needs the adapter.
- Run it from the bundle: `pip install -r chatbot_site/requirements.txt` then `python chatbot_site/app.py` (the data path inside `chatbot.py` was updated to `data/final_qa_dataset.csv` in this bundle).

---

## 8. Slide-by-slide speaking notes (`review/Review3_Presentation.pptx`, 18 slides)

| # | Slide | What to say (about 1 min each) |
|---|---|---|
| 1 | Title | Name the project, the three team members, the guide. |
| 2 | Approval mail | Template slide. **Replace the template's sample picture with the screenshot of your guide's approval mail** (the template asks for it before 21-10-2026). |
| 3 | Aim | Study whether controlled refusal data makes an SLM better calibrated and less hallucinating. Three systems, one test set, one 4 GB GPU. |
| 4 | Abstract | Problem → method → headline numbers → cost. Say "HR 1.000 → 0.139" and "ECE 0.622 → 0.304". |
| 5 | Literature review | R-Tuning and Cheng et al. (teach refusal); Guo and Jiang (miscalibration); Kapoor and Hu (training improves calibration/detection); Gekhman (fine-tuning on new facts raises hallucination); Wen (over-abstention open problem). |
| 6 | Research gap | Prior work: big models, abstention share fixed by data, calibration and hallucination separate, over-abstention rarely quantified. Ours: sub-4 B, controlled ratio, joint measure, FRR reported. |
| 7 | Objectives | Five objectives map to Base→Normal FT, Normal FT→IDK FT, calibration, cost, reproducibility. SDG 4 and 9; TRL 4 (lab-validated prototype). |
| 8 | Framework | Walk the four layers: data → training (3 systems) → inference (answer + confidence or IDK) → evaluation. |
| 9 | Functional requirements | FR1–FR8; each maps to a script. |
| 10 | Modules | Six modules, each with its files. |
| 11 | Results 1/5 | Dataset sizes and the setup; why IDK-15 test is used for all three systems. |
| 12 | Results 2/5 | Hallucination table: Base 0.506, Normal FT 1.000, IDK FT 0.139. |
| 13 | Results 3/5 | Calibration table and reliability diagram. Say plainly that everything is still over-confident. |
| 14 | Results 4/5 | Histograms/AUROC and the cost trade-off (16 correct lost, 68 hallucinations avoided). |
| 15 | Results 5/5 | Stage ablation, error types, limitations. |
| 16 | Conclusion | Three conclusions and the future work list. |
| 17 | Outcomes | **Update the conference-paper status** (submitted / accepted / published) and upload proof by 27 Oct. |
| 18 | References | Key references; the full list of 48 is in the report. |

---

## 9. Likely review questions and good answers

**Concept and motivation**
1. *Why not just use RAG?* RAG helps when a source exists. Many student questions have no document to retrieve (personal records, live status, fake codes). The model must still know to refuse. RAG plus abstention is listed as future work and should be complementary.
2. *Why a small model?* On-premise, low cost, runs on a consumer 4 GB GPU. The literature mostly studies 7 B+ models.
3. *Why is Normal FT's HR 1.000 a meaningful finding?* The training data contained only answerable questions, so the model learned "every question has an answer". It generalised that to unanswerable ones. Fine-tuning is worse than the base model here (0.506).
4. *What is novel?* The abstention share is controlled and varied (0 vs 15%), a sub-4 B model on a 4 GB GPU, and joint reporting of hallucination, calibration and false-refusal cost. Be modest: the technique itself (R-Tuning-style) is not new.

**Method**
5. *Why 15%?* A compromise between enough examples to teach refusal and keeping answerable accuracy. It was not tuned; the paper says so. 10% and 20% splits exist but were not evaluated.
6. *Why is the refusal a fixed sentence?* For clean detection and measurement; the cost is rote behaviour (Section 3.3).
7. *Why the 0.5 F1 threshold and the 40% coverage threshold?* Free-form answers rarely match exactly. They are convenient lexical proxies, stated as a limitation; they do not verify numbers.
8. *Why geometric mean of token probabilities?* It is length-normalised; without it long answers would be penalised. It is cheap and standard.
9. *Why LoRA on feed-forward layers too?* To let the model change behaviour based on content (answer or abstain), not only attention patterns.
10. *Did you use a validation set for anything?* In v1 training only a 10% hold-out of the training file (validation loss per epoch). The 15% `valid` split exists but was not used for model selection or for calibration. The final adapter is the one saved at the end of training (epoch 3), not a "best" checkpoint.
11. *Was there any calibration step (temperature scaling)?* No, deliberately. The question is what data alone can do.

**Results**
12. *Is the improvement just because refusals are easy?* Partly, and the paper says so: correct refusals are near-perfect token F1 and high confidence. On answerable questions IDK FT and Normal FT are about equal (F1 ≈ 0.39 vs 0.40). The real gain is on unanswerable questions, which is what the project targets.
13. *Is 4.1% false refusals acceptable?* In this setting a wrong credit or deadline is costlier than a pointer to VTOP or the proctor. The trade is 16 correct answers for 68 avoided hallucinations.
14. *Are the results statistically significant?* The test set is small. HR = 11 / 79 has a 95% interval of about 0.08–0.23. Differences of a few questions should not be over-read. One seed only.
15. *Why is AUROC only 0.81?* The model remains over-confident, false refusals carry high confidence, and wrong answers to answerable questions still get medium-high confidence.
16. *Base model recall 0.494: does it know what it does not know?* No. 116 of its 155 refusals were on answerable questions; the refusals are generic.
17. *Why different "answerable accuracy" numbers (0.451 vs 0.327)?* Different definitions (Section 5). Say which table each came from.

**Limitations and validity**
18. *Is there train/test leakage?* No exact-match question overlap between IDK-15 train and test (checked, 0). 15 test questions are in the IDK-0 training set (favours Normal FT). Near-duplicate or paraphrased questions across splits are possible and not measured, because the split is random, not by course.
19. *Does it generalise?* Unknown: one institution, one model, synthetic unanswerable categories.
20. *Would a different phrasing of refusal be detected?* Not by the phrase list. This is why Section 11 items 2 and 3 matter.
21. *What would you do next?* Ratio sweep with several seeds; varied context-aware refusals; temperature scaling and a confidence threshold; NLI or judge-based correctness; retrieval plus abstention; other models and institutions.

---

## 10. What is in the bundle (`V1_Complete/`)

| Folder | Contents |
|---|---|
| `code/` | The 9 v1 scripts (training, data prep, evaluation, plots, verification) |
| `data/` | The two source CSVs, the JSONL version, the source-inventory workbook, `splits/` (IDK-0/10/15/20 train/valid/test + `eval_set.jsonl`), `source_datasets/` (raw earlier dataset files) |
| `results/` | `comparison_fixed.md` (Table 3 numbers) and `figures/` (reliability, ROC, histogram, refusal-metric, error-breakdown PNGs, per-model reliability diagrams) |
| `diagrams/` | Architecture diagrams (vertical and horizontal), Gantt chart; `superseded_earlier_runs/` holds old console screenshots **with different numbers** (see Section 11) |
| `paper/` | Springer LNCS paper PDF and DOCX, section drafts, figures, build script, template |
| `notebooks/` | The two early Colab notebooks |
| `chatbot_site/` | Flask demo site |
| `report/` | `BCSE497J_Project_I_Report_Final.docx` (filled template) |
| `review/` | `Review3_Presentation.pptx` (filled template, 18 slides) |
| `docs/` | This guide and `README.md` |

**Left out on purpose.** `v1/v2_files/`, `results_v2/` and the root `*_v3` files belong to later versions, not v1. The 96 source PDFs (curricula, eligibility and admission documents; about 529 MB, `Downloads/project/Questions/`) are the raw sources; the bundle keeps the inventory workbook instead.

**Re-running (Windows or Colab).** The scripts use relative paths. Put `code/*.py`, `data/final_qa_dataset.csv`, `data/combined_qa_finetuning.csv` and the needed `data/splits/*.jsonl` in one working folder, then:
```
pip install torch transformers peft trl datasets bitsandbytes accelerate scikit-learn pandas matplotlib
python prepare_expc_data.py --idk-ratio 0.0
python prepare_expc_data.py --idk-ratio 0.15
python exp_c_train.py --idk-ratio 0.0 --output-dir phi3-vit-lora-expc-idk0
python exp_c_train.py --idk-ratio 0.15
python eval_compare_fixed.py --normal-adapter phi3-vit-lora-expc-idk0 --idk-adapter phi3-vit-lora-expc-idk15 --test expc_idk15_test.jsonl
python compare_idk_stats.py --normal-adapter phi3-vit-lora-expc-idk0 --idk-adapter phi3-vit-lora-expc-idk15 --test expc_idk15_test.jsonl
```
(`exp_c_train.py` with ratio 0.0 names the folder `phi3-vit-lora-expc-idk0`; its closing hint mentions the older script name `eval_compare.py`, which is `eval_compare_fixed.py` now.)

**Missing from v1 (not recoverable from the files I found).**
- The v1 trained adapters and the base-model cache files.
- The v1 per-example `calibration_export.json` (449 queries). The only export on disk has 439 queries and different numbers (ECE 0.621 / 0.378 / 0.266); it belongs to a later run. So `make_figures.py` cannot be re-run for v1. The PNG figures in `results/figures/` are the originals and are identical to the paper's figures.

---

## 11. Inconsistencies and weak spots to know about (be ready, or fix the paper)

I did not change your paper; paste-ready replacement wording for items 1, 2, 3, 4 and 5 is in `docs/PAPER_WORDING_FIXES.md` (the final PDF has no editable source in the project). The report and slides use the paper's numbers, and I checked every table value, percentage and count in the paper against the data and arithmetic: all agree. These are the places where the *description* or the *setup* is weaker than the numbers.

1. **"Three models are trained" (abstract).** Two adapters are trained (Normal FT, IDK FT); the base model is not trained. Suggested wording: "three systems are compared".
2. **Refusal-style rows that the detector does not catch.** About 191 of the 980 supplementary rows (and some others) are refusal-style answers with different wording ("I do not have any confirmed information…", "I am not able to do that for you…", "That is specific to your own record, and I have no way to check it"). The pattern misses them, so they stay in the answerable pool with their long original answer.
   - My rough count with a broader pattern: about 89 of 1,775 IDK-0 training rows (about 5%) and about 16 of the 370 "answerable" IDK-15 test rows. These are estimates from a quick regex, not exact figures.
   - Consequences: (a) "Normal FT never sees abstention" is not strictly true; about 5% of its training answers are refusal-like. (b) Some of IDK FT's 15 "false refusals" may be questions that should have been refused, which would mean the true FRR is lower than 4.1%. (c) A varied refusal by Normal FT would be counted as an answer, so "Normal FT refused 0 of 449" depends on the detector.
   - The paper already admits that the pattern is conservative and gives one example. The honest answer in the review is: "this is a known limitation; the effect is on the order of 5%, and we list a better detector or labelling pass as future work." Do not claim the 16 and the 15 are the same rows; that was not checked (no per-example predictions for v1 are in the files).
3. **Two refusal detectors and two accuracy definitions.** `compare_idk_stats.py` (Table 2) uses the broad training-time pattern (includes "I don't", "real-time"); `eval_compare_fixed.py` (Table 3) uses a narrower one (adds "do not have access", "don't have enough information"). Paper Section 5.2 lists phrases of the narrower one. Both flag exactly the same 79 reference refusals, so the labels agree; the model outputs could still be classified slightly differently between the two tables. Table 2 and Table 3 also use different token limits (100 vs 64) and different correctness rules (Section 5).
4. **Three core rows wrongly became refusals.** The regex flagged 3 core QA rows: the PSO question for B.Tech CSE, "Does the MBT curriculum include Real-Time PCR experiments?" and an M.Arch professional-training question. They are genuine answers (they contain "real-time" or similar words), and were rewritten to the canonical refusal. `verify_dataset.py` shows this. It is 3 of 792 refusal rows, so the effect is tiny, but the paper's "3 pairs of core QA were flagged as abstention" reads as if they were real refusals.
5. **Dataset description.** The paper says the 2,349 pairs are derived from "VIT's official curriculum documents". By your own inventory, 82% are curriculum; the rest are regulations, handbook, code of conduct, calendar, research guides and fees. The report and slides say "public VIT documents (82% curriculum)". Also, the 980 supplementary rows are not all unanswerable: 789 were flagged; 191 are in the answerable pool (see item 2).
6. **IDK-0 vs IDK-15 test overlap.** 15 answerable test questions are in the IDK-0 training set. Already disclosed in the paper; the ablation favours Normal FT slightly.
7. **The 15% `valid` split is unused in v1.** v1 training used its own 10% hold-out and kept the last-epoch adapter; calibration was not fitted on valid. (v3 changed this.) Do not say "best checkpoint chosen on validation" for v1.
8. **Stale files.** `data/splits/eval_set.jsonl` (239 rows) is the first 10% split exported by `train_lora.py` in an early run and is not the test set used in the paper. `diagrams/superseded_earlier_runs/` holds old console screenshots (ECE 0.716 / 0.660 / 0.575 on 499 + 105 rows, "accuracy 0.000" for Base) from before the evaluation was fixed. **Do not show these;** they contradict the paper. `data/source_datasets/final_qa_dataset.csv` is an older raw file that differs from the one used (different hash).
9. **Reg. numbers, HoD name and signature dates are blank in the report; the status of the conference paper (slide 17) must be filled by you.** I did not invent them.
10. **Confidence on refusals.** The high-confidence peak is largely the memorised refusal sentence. Do not describe it as the model "knowing" it does not know; say it "has learned to produce the refusal".

---

## 12. Numbers cheat-sheet

- Corpus: 2,349 + 980 = 3,329 rows. Pools: 2,537 answerable + 792 IDK. IDK-15 mixture: 2,537 + 447 = 2,984 rows.
- Test: 449 (370 answerable, 79 unanswerable = 17.6%). Train/valid/test = 2,088/447/449 (IDK-15) and 1,775/380/382 (IDK-0).
- Model: Phi-3-mini-4k-instruct, 3.8 B; QLoRA r = 8, α = 16, dropout 0.05; 3 epochs; lr 2e-4; effective batch 8; seed 42; RTX 3050 Ti 4 GB.
- HR 0.506 → 1.000 → 0.139; IDK F1 0.333 → 0.000 → 0.840; FRR 31.4% → 0% → 4.1%.
- ECE 0.622 → 0.408 → 0.304 (−51%); Brier 0.480 → 0.351 → 0.285; AUROC 0.557 → 0.764 → 0.810.
- Cost: 167 → 151 correct answerable (−16); 68 of 79 unanswerable refused; ≈ 4 hallucinations avoided per correct answer lost.
- HR 95% interval ≈ 0.08–0.23 (11 of 79).
