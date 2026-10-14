# 6 Results and Analysis

This section reports the effect of each configuration on the two principal objectives of the study, hallucination reduction (Section 6.1) and confidence calibration (Sections 6.2 and 6.3), followed by an analysis of the associated cost in false refusals (Section 6.4) and an ablation isolating the contribution of the abstention data (Section 6.5). All results are obtained on the IDK-15 test partition, comprising 449 queries (370 answerable and 79 unanswerable). Refusal-based metrics are drawn from the 100-token hallucination analysis, and calibration metrics from the 64-token calibration analysis (Section 5.4).

## 6.1 Hallucination Reduction

Table 2 summarises the refusal behaviour of each system, and Figure 1 presents the same metrics graphically.

**Table 2.** Hallucination and refusal metrics on the IDK-15 test set. HR denotes the hallucination rate (missed abstentions divided by the 79 unanswerable queries). Answerable accuracy is computed using the >40% reference-coverage criterion (Section 5.2).

| System | Refusal rate | IDK precision | IDK recall | IDK F1 | HR ↓ | Missed abstentions (FN) | False refusals (FP) | Answerable acc. |
|---|---|---|---|---|---|---|---|---|
| Base | 0.345 | 0.252 | 0.494 | 0.333 | 0.506 | 40 | 116 | 0.227 |
| Normal FT | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 79 | 0 | **0.451** |
| IDK FT | 0.185 | **0.819** | **0.861** | **0.840** | **0.139** | **11** | 15 | 0.408 |

The most notable finding concerns the effect of standard fine-tuning on hallucination. Prior to fine-tuning, the base model declines approximately half of the unanswerable queries (HR = 0.506). After fine-tuning on answerable domain data alone, it declines **none**: the Normal FT model did not refuse any of the 449 test queries. It produced an answer to all 79 unanswerable queries, including those concerning individual student records, real-time schedules, and non-existent courses, each of which is necessarily fabricated. Domain adaptation therefore improved the model's domain knowledge, approximately doubling answerable accuracy from 0.227 to 0.451, while simultaneously conditioning it to answer every query regardless of whether a reliable answer was available. With respect to hallucination on unanswerable queries, standard fine-tuning produced a worse outcome than no fine-tuning at all.

The inclusion of 15% abstention data reverses this effect. The IDK FT model abstains on 68 of the 79 unanswerable queries, reducing the hallucination rate to 0.139, which corresponds to a relative reduction of 86% compared with Normal FT and 73% compared with the base model. Its refusals are also well targeted: 68 of its 83 refusals occur on queries that warrant abstention (precision 0.819), yielding an IDK F1 of 0.840. The overall refusal rate of 18.5% closely approximates the true proportion of unanswerable queries in the test set (17.6%), indicating that the model has learned not only the form of an abstention but also an appropriate frequency of abstaining.

The base model's apparent recall of 0.494 should not be interpreted as evidence of knowledge-aware behaviour. Of its 155 refusals, 116 occur on answerable queries, so three out of every four refusals are misplaced. These refusals consist largely of generic disclaimers regarding access to institution-specific information and are issued with little regard to whether a query is in fact answerable. Figure 2 contrasts the error profiles of the three systems: the base model errs predominantly through over-caution, Normal FT exclusively through hallucination, and IDK FT exhibits a small number of both error types (15 false refusals and 11 missed abstentions).

> **Figure 1.** Comparison of refusal behaviour (refusal rate, IDK precision, recall, F1, and answerable accuracy) for the base model, Normal FT, and IDK FT. *(refusal_metrics.png)*
>
> **Figure 2.** Error breakdown showing false refusals (over-caution) and missed abstentions (hallucinations) for each system. *(error_breakdown.png)*

## 6.2 Calibration Improvement

Table 3 reports the calibration metrics together with the answer-quality measures obtained in the same evaluation.

**Table 3.** Calibration and answer quality on the IDK-15 test set, using refusal-aware correctness with a token-F1 threshold of 0.5.

| Metric | Base | Normal FT | IDK FT |
|---|---|---|---|
| Accuracy (refusal-aware) ↑ | 0.100 | 0.327 | **0.477** |
| Exact match ↑ | 0.000 | 0.051 | **0.189** |
| Token F1 ↑ | 0.168 | 0.391 | **0.515** |
| ECE ↓ | 0.622 | 0.408 | **0.304** |
| Brier score ↓ | 0.480 | 0.351 | **0.285** |
| Reference NLL ↓ | 2.405 | 1.663 | **0.965** |
| AUROC ↑ | 0.557 | 0.764 | **0.810** |

Calibration improves at each stage. ECE decreases from 0.622 for the base model to 0.408 following standard fine-tuning and to 0.304 following abstention-aware fine-tuning, an overall reduction of 51%, of which a 25% reduction relative to Normal FT is attributable to the abstention data alone. The Brier score follows the same trend (0.480 → 0.351 → 0.285), confirming that the improvement is not an artefact of the binning scheme.

The reliability diagram in Figure 3 clarifies the source of these gains. All three curves lie below the diagonal across the full confidence range, indicating that **every system remains over-confident** and that no configuration eliminates over-confidence entirely. The shape of the curves, however, differs considerably. The base model's curve is essentially flat: its accuracy remains between 0 and 0.15 whether it reports a confidence of 0.57 or 0.93, and its highest-confidence bin has an accuracy of zero, so its confidence is largely uninformative. The Normal FT curve increases with confidence but plateaus at an accuracy of approximately 0.55 above a confidence of 0.8, meaning that the model remains incorrect in almost half of its most confident responses. The IDK FT curve increases monotonically across the full range and reaches an accuracy of 0.80 at a mean confidence of approximately 0.97, placing its highest-confidence bin closest to the diagonal among the three systems. A substantial gap nonetheless persists in the mid-range; at a confidence of approximately 0.65, IDK FT is correct in only about 19% of cases.

Two qualifications are necessary when interpreting these results. First, part of the aggregate improvement arises from the unanswerable queries themselves. Because IDK FT reproduces the canonical abstention verbatim, each correct abstention constitutes an exact match, achieves a near-perfect token F1, and is produced with high confidence. This accounts for most of the increase in exact match (from 23 to 85 of 449 responses) and contributes to the reduction in reference NLL, since all 79 unanswerable references share a single string that the model has learned to predict. Second, the gain in refusal-aware accuracy (0.327 → 0.477, approximately 67 additional correct responses) corresponds closely to the number of unanswerable queries that IDK FT handles correctly. Assuming that IDK FT behaves on unanswerable queries in this evaluation as it does in the hallucination analysis, its accuracy on answerable queries under the F1 criterion is approximately 0.39, essentially equivalent to the 0.40 obtained by Normal FT. Abstention-aware training therefore improves calibration primarily by converting confidently incorrect answers into confidently correct abstentions, rather than by improving the quality of the model's factual answers.

> **Figure 3.** Reliability diagram for the base model, Normal FT, and IDK FT (10 equal-width bins). The dashed diagonal denotes perfect calibration. *(reliability.png)*

## 6.3 Confidence Separation

Whereas calibration measures whether confidence is accurate *on average*, deployment additionally requires that confidence *separate* correct from incorrect responses sufficiently well to support a threshold. Figure 4 shows the confidence distributions for correct and incorrect responses, and Figure 5 shows the corresponding ROC curves.

For the base model, the two distributions overlap almost entirely. Both correct and incorrect responses fall within a narrow band between approximately 0.55 and 0.85, and the small number of correct responses (approximately 45) is indistinguishable from the roughly 400 incorrect ones. No threshold separates the two groups, which is reflected in an AUROC of 0.557, only marginally above chance.

Fine-tuning separates the two distributions. For Normal FT, correct responses shift predominantly into the 0.7–1.0 range, whereas incorrect responses peak between 0.55 and 0.75. The incorrect distribution nevertheless retains a long tail extending to 0.95, consisting of confidently incorrect responses that include the model's fabricated answers to unanswerable queries. Its AUROC is 0.764.

IDK FT achieves the clearest separation (AUROC 0.810). Its incorrect responses are concentrated around a confidence of 0.6, whereas its correct responses lie predominantly above 0.7, with a pronounced peak of more than 80 correct responses in the 0.95–1.0 bin, consisting largely of canonical abstentions. This peak is of practical significance. The IDK FT ROC curve rises almost vertically from the origin, reaching a true-positive rate of approximately 0.28 at a false-positive rate close to zero. In other words, roughly a quarter of IDK FT's correct responses can be accepted at a high confidence threshold while admitting almost no incorrect responses, compared with approximately 0.05 for Normal FT at the same operating point. The advantage persists throughout the low false-positive region: at a false-positive rate of 0.1, IDK FT retains approximately 43% of its correct responses, compared with approximately 25% for Normal FT. Beyond a false-positive rate of about 0.4, the two curves converge. The improvement is therefore concentrated in the high-confidence region, which is precisely where a practical acceptance threshold would be set.

The same peak also contains a small number of incorrect responses (approximately 15 in the 0.95–1.0 bin), a count that coincides with the 15 false refusals reported in Table 2. This observation is consistent with the concern raised in Section 4.4: because the canonical abstention is a fixed phrase, the model can produce it with high confidence even when abstention is inappropriate. Such errors cannot be identified through confidence alone.

> **Figure 4.** Confidence distributions for correct and incorrect responses for each system. *(confidence_hist.png)*
>
> **Figure 5.** ROC curves for discriminating correct from incorrect responses using sequence-level confidence. *(roc.png)*

## 6.4 Cost of Abstention: False Refusals

Training a model to abstain introduces a corresponding risk that it will also decline queries it is able to answer. This cost is quantified directly.

IDK FT refuses 15 of the 370 answerable queries, a false refusal rate of 4.1%. This is substantially lower than the base model's rate of 31.4% (116 false refusals) and only marginally higher than the 0% observed for Normal FT. Under the coverage criterion, answerable accuracy decreases from 0.451 to 0.408, a reduction of 4.3 percentage points corresponding to 16 fewer correct answers (167 → 151).

The magnitude of this reduction is informative, since the 16 lost correct answers closely match the number of false refusals. When only the answerable queries actually attempted by each model are considered, IDK FT is correct on 151 of 355 (42.5%) and Normal FT on 167 of 370 (45.1%). The similarity of these figures indicates that the cost arises predominantly from the model declining a small number of answerable queries, rather than from any general degradation in answer quality. The inclusion of abstention data therefore does not appear to have impaired the domain knowledge acquired during fine-tuning.

On balance, the trade-off strongly favours abstention-aware training. In exchange for 16 fewer correct answers, IDK FT avoids 68 fabricated ones, preventing approximately four hallucinations for every correct answer forgone. In a student-facing assistant, a fabricated answer regarding examination dates, credit requirements, or registration status may lead to consequential errors, whereas an unnecessary abstention at most redirects the student to VTOP or their proctor. The latter is a considerably less costly failure mode.

One factor marginally favours the baseline in this comparison. As noted in Section 5.1, 15 of the answerable test queries also appear in the training data of Normal FT but not in that of IDK FT. Normal FT's answerable accuracy may therefore be slightly overstated, and the true difference between the two systems may be smaller than 4.3 percentage points.

## 6.5 Ablation: Effect of Abstention Data

Since Normal FT and IDK FT share the same base model, answerable data, hyperparameters, and random seed, the comparison between them isolates the effect of the abstention data. Similarly, the comparison between the base model and Normal FT isolates the effect of domain adaptation. Table 4 presents both comparisons.

**Table 4.** Changes attributable to each stage. Arrows indicate the direction of improvement.

| Metric | Domain adaptation (Base → Normal FT) | Abstention data (Normal FT → IDK FT) |
|---|---|---|
| Hallucination rate ↓ | +0.494 (0.506 → 1.000) | **−0.861** (1.000 → 0.139) |
| IDK F1 ↑ | −0.333 | **+0.840** |
| False refusal rate ↓ | **−0.314** | +0.041 |
| Answerable accuracy ↑ | **+0.224** | −0.043 |
| Token F1 ↑ | +0.223 | +0.124 |
| ECE ↓ | **−0.214** | −0.104 |
| Brier ↓ | **−0.129** | −0.066 |
| AUROC ↑ | **+0.207** | +0.046 |

The two stages make complementary contributions. Domain adaptation accounts for most of the improvement in domain knowledge: it approximately doubles answerable accuracy, eliminates the base model's indiscriminate refusals, and yields the largest individual improvements in ECE, Brier score, and AUROC. In isolation, however, it eliminates abstention entirely and results in hallucination on every unanswerable query. The abstention data addresses precisely this deficiency, reducing the hallucination rate from 1.000 to 0.139 and providing further improvements in calibration and discrimination beyond those achieved through domain adaptation, at a cost of 4.1 percentage points in false refusals and 4.3 percentage points in answerable accuracy. Neither stage alone produces a model that is both knowledgeable and appropriately calibrated, whereas their combination does.

Two limitations of this ablation should be noted. First, only a single abstention ratio was evaluated, so the optimality of 15% cannot be established. A lower ratio might further reduce false refusals while retaining most of the reduction in hallucination, whereas a higher ratio might increase IDK recall at the cost of greater over-caution. Second, each configuration was trained with a single random seed. Characterising this trade-off across multiple ratios and estimating variance across seeds are natural directions for future work.
