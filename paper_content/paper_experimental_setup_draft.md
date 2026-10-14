# 5 Experimental Setup

## 5.1 Configurations and Baselines

Three systems are compared. All three share identical base weights, the same 4-bit quantisation, and the same inference-time prompt format.

1. **Base model.** Phi-3-mini-4k-instruct evaluated zero-shot without an adapter. Each query is presented in the standard chat template, with no system prompt, retrieved context, or in-context examples. This baseline characterises the model's calibration and hallucination behaviour on institution-specific queries when it relies solely on knowledge acquired during pre-training and instruction tuning.
2. **Standard fine-tuning (Normal FT).** The IDK-0 adapter, trained exclusively on answerable queries (Section 4.2). This model acquires domain knowledge but is never exposed to abstention as a valid response. It serves to determine whether domain fine-tuning alone improves calibration and reduces hallucination, or whether it merely increases the model's fluency and confidence on queries it cannot answer.
3. **Abstention-aware fine-tuning (IDK FT).** The IDK-15 adapter, trained on the same answerable queries with 15% canonical abstention examples included. This is the proposed configuration.

The comparison between the base model and Normal FT isolates the effect of domain adaptation, while the comparison between Normal FT and IDK FT isolates the effect of abstention data on calibration and hallucination, since these two adapters were trained under identical conditions and differ only in their training mixture.

All three systems are evaluated on the **IDK-15 test partition**, which comprises 449 queries: 370 answerable and 79 unanswerable. This partition is used for all systems because the IDK-0 test partition contains no unanswerable queries and therefore cannot be used to measure hallucination on such queries. Because the two mixtures were shuffled independently, 15 of the 449 test queries, all of them answerable, also appear in the IDK-0 training data. This overlap affects approximately 3% of the test set and, if anything, favours the Normal FT baseline.

## 5.2 Hallucination Metrics

The hallucination analysis assesses whether each model answers the queries it should answer and declines those it cannot reliably answer. For every test query, it is recorded whether the model abstained, using the same lexical refusal pattern applied to the reference answers (Section 4.1). Each query is thereby assigned to one of four outcome categories:

| | Model abstains | Model answers |
|---|---|---|
| **Unanswerable** | True positive (TP) | Missed abstention (FN) |
| **Answerable** | False refusal (FP) | Answer attempt (TN) |

The following metrics are derived from these counts.

**Hallucination rate.** The primary hallucination metric is the proportion of unanswerable queries for which the model produced an answer (Section 3.2):

$$
\text{HR} = \frac{\text{FN}}{\text{TP} + \text{FN}}.
$$

As established in Section 3.2, each such response constitutes a hallucination by construction. Although HR is equal to 1 − IDK recall, it is reported separately because it most directly reflects the deployment risk of the system, namely the frequency with which a user would receive a fabricated answer.

**IDK precision, recall, and F1.** The model's decision to abstain is further evaluated as a binary classification task:

$$
P_{\text{IDK}} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \qquad
R_{\text{IDK}} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \qquad
F1_{\text{IDK}} = \frac{2\,P_{\text{IDK}}\,R_{\text{IDK}}}{P_{\text{IDK}} + R_{\text{IDK}}}.
$$

Precision reflects over-caution, decreasing when the model refuses answerable queries, whereas recall reflects under-caution, decreasing when the model answers unanswerable queries. Since either measure can be maximised trivially in isolation (a model that always refuses attains perfect recall, and one that never refuses produces no false refusals), F1 is used as the summary measure. The raw counts of false refusals and missed abstentions, together with the overall refusal rate, are also reported, since models with similar F1 scores may exhibit markedly different error profiles.

**Answer quality on answerable queries.** A reduction in hallucination is of practical value only if answer quality on answerable queries is maintained. **Token-level F1** between each generated response and its reference is therefore reported, computed as in SQuAD [40]: both strings are lower-cased, punctuation is removed, the text is tokenised on whitespace, and F1 is computed as the harmonic mean of token-level precision and recall over the shared bag of tokens. Token-level F1 is preferred to exact match because the models generate free-form text; a response such as *"The MSB programme requires 80 credits in total"* is correct but would receive a score of zero under exact matching against a differently worded reference. Exact match is nevertheless reported for completeness. The hallucination analysis additionally reports **answerable accuracy**, under which an answer attempt is considered correct if it covers more than 40% of the distinct words in the reference. This coverage-based criterion does not penalise responses that include additional explanation.

## 5.3 Calibration Metrics

The calibration analysis assesses whether each model's confidence score (Section 4.4) is informative, that is, whether higher confidence corresponds to a higher likelihood of correctness. All metrics in this section are computed from the confidence score c_i of each response and the refusal-aware correctness label y_i ∈ {0, 1} defined in Section 3.2, under which a correct abstention is treated as a correct response and a false refusal as an error.

**Expected Calibration Error (ECE)** [14,34]. The confidence interval [0, 1] is partitioned into M = 10 equal-width bins B₁, …, B_M, and the mean confidence in each bin is compared with the empirical accuracy:

$$
\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N}\,\bigl|\,\text{acc}(B_m) - \text{conf}(B_m)\,\bigr|.
$$

A perfectly calibrated model, which is correct 70% of the time when it reports a confidence of 0.7, attains an ECE of zero. Reliability diagrams plotting per-bin accuracy against mean confidence are also presented, since ECE summarises the full calibration curve in a single value and does not indicate whether a model is over- or under-confident.

**Brier score** [4]. The mean squared error between confidence and correctness:

$$
\text{Brier} = \frac{1}{N}\sum_{i=1}^{N} (c_i - y_i)^2.
$$

Unlike ECE, the Brier score is independent of the choice of bins. It also rewards *sharpness* in addition to calibration: a model that uniformly reports a confidence of 0.5 may obtain a low ECE but will nonetheless receive a poor Brier score.

**AUROC.** The area under the receiver operating characteristic curve obtained when confidence is used to discriminate correct from incorrect responses. AUROC equals the probability that a randomly selected correct response is assigned a higher confidence than a randomly selected incorrect one, so a value of 0.5 indicates that confidence carries no information about correctness. Calibration and discrimination are distinct properties: a model may be well calibrated on average while assigning nearly identical confidence to its correct and incorrect responses. AUROC therefore determines whether a confidence threshold would be practically effective for filtering hallucinated responses at deployment time.

The mean teacher-forced negative log-likelihood (NLL) of the reference answers (Section 4.4) is reported as a supplementary measure.

## 5.4 Implementation Details

The pipeline is implemented in Python using PyTorch [36] and the Hugging Face ecosystem: `transformers` [44] for model loading and generation, `peft` [33] for the LoRA adapters, `trl` [43] (`SFTTrainer`) for supervised fine-tuning, `bitsandbytes` [7,8] for 4-bit quantisation and 8-bit paged optimisation, and `datasets` for data handling. AUROC is computed with scikit-learn [37], while ECE, the Brier score, token-level F1, and the abstention confusion counts are computed using custom implementations. Figures are produced with Matplotlib.

All training and evaluation were performed on a single NVIDIA RTX 3050 Ti laptop GPU with 4 GB of VRAM. Since two 4-bit instances of Phi-3-mini cannot be accommodated in this memory simultaneously, models are evaluated sequentially: each model is loaded, evaluated on the complete test set, and released, and the CUDA cache is cleared before the next model is loaded. Adapters are maintained separately from the frozen base model and attached at load time, ensuring that all systems share identical quantised base weights. Base-model outputs are cached per test file so that subsequent adapter evaluations do not require the base model to be re-run.

At inference time, each model receives the query in the training-time chat template (`<|user|>\n{question}<|end|>\n<|assistant|>\n`) and decodes greedily without sampling, so all outputs are deterministic. The hallucination analysis (Section 5.2) permits up to 100 new tokens to avoid truncating longer answers before scoring. The calibration analysis (Section 5.3) limits generation to 64 new tokens and records per-token log-probabilities during decoding in order to compute the confidence score. The refusal pattern used to label model outputs is consistent with that used to label the reference answers; on the IDK-15 test set, the patterns used during data preparation and evaluation identify the same 79 unanswerable queries.

All stochastic operations, including dataset shuffling, abstention sampling, the train/validation/test partitioning, and the internal validation hold-out, use a fixed random seed of 42. Given the two source files, the provided scripts reproduce every partition reported in Table 1 exactly.
