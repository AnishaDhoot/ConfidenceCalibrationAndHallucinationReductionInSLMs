# 4 Methodology

This study investigates whether the confidence calibration of a small instruction-tuned language model can be improved, and its tendency to hallucinate reduced, through the composition of its fine-tuning data. The intervention examined is the controlled inclusion of abstention examples, which teach the model to decline queries that fall outside its knowledge base while continuing to answer those within it. To this end, a domain-specific question-answering corpus covering the academic programmes of Vellore Institute of Technology (VIT) was constructed. Two training mixtures were derived from this corpus, differing solely in the proportion of abstention examples (0% and 15%). Phi-3-mini was fine-tuned on each mixture using QLoRA, and every generated response was assigned a sequence-level confidence score, enabling hallucination behaviour and calibration to be assessed jointly. Each stage of this pipeline is described in the following subsections.

## 4.1 Dataset Construction and Splits

The corpus is drawn from two sources. The first, referred to as the *core QA set*, comprises 2,349 question–answer pairs derived from VIT's official curriculum documents. These documents specify programme credit structures, course codes and credit values, category-wise credit distributions (e.g., Discipline Core, Discipline Elective, Open Elective, and Projects and Internship), and the academic regulations governing undergraduate and postgraduate programmes. The questions vary in complexity, from direct factual retrieval (e.g., *"How many credits are allotted to Professional Elective Courses in the M.Tech in Applied Computational Fluid Dynamics program?"*) to cross-programme comparisons and student-style requests (e.g., *"I want a Discipline Elective that combines lecture and lab work on computer vision for medical use. Which course code should I register for?"*). Questions have a mean length of 18.9 words, and answers a mean length of 16.7 words (median 14). Answers were intentionally kept concise and factual so that generated responses can be reliably compared against a reference.

The second source is a supplementary set of 980 pairs constructed specifically to represent queries that the curriculum documents cannot answer. These queries fall into five principal categories:

- **Personal or record-specific queries**, which depend on an individual student's data (e.g., attendance, re-FAT application status, or examination seating) and can be resolved only through the institutional portal (VTOP) or a faculty proctor.
- **Time-sensitive queries**, which require real-time information (e.g., whether the FAT schedule has been released at the time of asking, or the library's closing time on a given day).
- **Unsupported specifics**, which request plausible but undocumented details (e.g., the exact notice period for an outing pass, or the research group associated with a particular elective).
- **Non-existent entities**, which refer to course codes or courses absent from all curricula and are intended to elicit fabricated responses.
- **Out-of-domain queries**, which bear no relation to the institution.

The supplementary set also contains a number of answerable pairs, predominantly cross-programme comparisons, which are treated as standard QA data.

Abstention examples were separated from answerable examples using a case-insensitive lexical pattern that matches common refusal expressions (e.g., *"don't know"*, *"I cannot"*, *"cannot verify"*, *"real-time"*, and *"not in my knowledge"*). Applied to both sources, the pattern identifies 3 core pairs and 789 supplementary pairs as abstentions, yielding an **answerable pool of 2,537 pairs** and an **abstention pool of 792 pairs**. Because the pattern is deliberately conservative, a small number of hedged responses expressed in other terms (e.g., *"That is specific to your own record, and I have no way to check it…"*) are not detected and remain in the answerable pool with their original wording. This is acknowledged as a limitation of the present study.

Each mixture constructed from these pools (Section 4.2) is shuffled using a fixed random seed (42) and partitioned into training, validation, and test sets in a 70/15/15 ratio. Prior to partitioning, an exact-match check confirmed that no question appears in both the training and test sets of either mixture and that all questions within each mixture are unique. The resulting partition sizes are reported in Table 1.

**Table 1.** Partition sizes for the two training mixtures. The number of abstention examples in each partition is given in parentheses.

| Mixture | ρ (target) | Total | Train | Validation | Test |
|---|---|---|---|---|---|
| IDK-0 | 0.00 | 2,537 | 1,775 (0) | 380 (0) | 382 (0) |
| IDK-15 | 0.15 | 2,984 | 2,088 (312) | 447 (56) | 449 (79) |

Since the target ratio is enforced over the complete mixture prior to partitioning, the proportion within each individual partition varies slightly. For IDK-15, abstention examples constitute 14.9% of the training set, 12.5% of the validation set, and 17.6% of the test set. Random rather than stratified partitioning was retained so that both mixtures are processed identically. To prevent this variation from distorting aggregate measures, refusal behaviour is evaluated separately over answerable and unanswerable queries (Section 5.2).

## 4.2 Controlled Abstention Data Mixing

Abstention examples are the mechanism through which this study seeks to reduce hallucination and improve calibration. Exposure to explicit examples of declining unanswerable queries is intended both to suppress fabricated answers and to provide the model with a learned response for low-knowledge situations, thereby sharpening the relationship between its confidence and its correctness. A straightforward approach would be to include all available refusal examples in the training data. However, this makes the effective abstention rate dependent on the number of such examples that happen to have been collected. In the present work, the abstention rate is therefore treated as an explicitly controlled variable. For a target ratio ρ, the number of abstention examples sampled is

$$
n_{\text{idk}} = \left\lfloor N_{\text{reg}} \cdot \frac{\rho}{1-\rho} \right\rfloor,
$$

where N_reg = 2,537 denotes the size of the answerable pool. This ensures that abstention examples constitute exactly a fraction ρ of the combined dataset. The examples are sampled without replacement from the 792-example abstention pool using a fixed random seed. For ρ = 0.15, this yields 447 abstention examples, which lies well within the available pool, so the target ratio is achieved exactly. The value of 15% was selected so that abstention examples form a sufficiently large share of the data for the behaviour to be learned, without outweighing the answerable examples on which the model must retain its accuracy.

Prior to mixing, **all abstention targets are normalised to a single canonical response**:

> *"I don't have enough information in my knowledge base to answer this question accurately."*

The original supplementary answers exhibited considerable variation in phrasing, and several redirected the user to external resources (e.g., *"please check VTOP"* or *"contact your proctor"*). Although varied phrasing may appear more natural, it introduces two difficulties in a controlled experimental setting. First, it reduces the reliability of refusal detection during evaluation, introducing noise into every refusal-dependent metric. Second, it allows the model to acquire multiple loosely defined hedging styles that may carry over into responses to legitimate queries. A single canonical string defines abstention as one well-specified behaviour that can be measured precisely. The corresponding trade-off is that the model may learn abstention as a fixed phrase rather than as a graded expression of uncertainty. The implications of this for confidence estimation are discussed in Section 4.4.

A separate adapter is trained for each mixture. IDK-0 contains no abstention examples and serves as the *standard fine-tuning* baseline, while IDK-15 constitutes the proposed abstention-aware configuration. All other factors are held constant across the two conditions, namely the answerable pool, the random seed, the partition proportions, and the training hyperparameters. Observed differences in hallucination behaviour and calibration can therefore be attributed to the inclusion of abstention data.

## 4.3 QLoRA Fine-Tuning of Phi-3-mini

**Phi-3-mini-4k-instruct** [1], a 3.8-billion-parameter model, is adopted as the base model. Its size permits fine-tuning on consumer-grade hardware, and its prior instruction tuning enables it to follow a conversational format without additional alignment. All experiments were conducted within a 4 GB VRAM budget on an NVIDIA RTX 3050 Ti laptop GPU, and this constraint informed most of the configuration choices described below.

**Quantisation.** Following QLoRA [8], the base model is loaded in 4-bit NormalFloat (NF4) precision with double quantisation, and computation is performed in bfloat16. The quantised base weights remain frozen throughout training, and gradients are propagated only to the low-rank adapter parameters.

**Adapter configuration.** LoRA [16] adapters with rank r = 8, scaling factor α = 16, and dropout 0.05 are applied to all attention and feed-forward projection layers (`qkv_proj`, `o_proj`, `gate_up_proj`, and `down_proj`). Adapting the feed-forward blocks in addition to the attention layers increases the model's capacity to alter the content of its outputs, not only the distribution of its attention. This capacity is relevant when the target behaviour involves deciding between answering and abstaining. The resulting configuration contains 12,582,912 trainable parameters, corresponding to 0.33% of the model's 3.83 billion total parameters.

**Input formatting.** Each question–answer pair is formatted using the native Phi-3 chat template:

```
<|user|>
{question}<|end|>
<|assistant|>
{answer}<|end|>
```

Adopting the template used during the model's original instruction tuning avoids any discrepancy between training-time and inference-time prompt formats. Sequences exceeding 512 tokens are truncated, although in practice almost no examples approach this limit given the brevity of the answers. Any example whose answer begins with the placeholder *"NOT FOUND"*, a marker introduced during dataset construction for facts that could not be sourced, is excluded. Training on such examples would encourage the model to produce an uninformative, refusal-like string unrelated to the query.

**Optimisation.** Training is performed for three epochs with a per-device batch size of 1 and eight gradient-accumulation steps, giving an effective batch size of 8. A learning rate of 2 × 10⁻⁴ with linear decay is used together with the paged 8-bit AdamW optimiser [7,8]. Gradient checkpointing is enabled, with the key–value cache disabled during training, in order to keep activation memory within the available budget. Ten percent of each mixture's training partition is held out using a fixed seed to compute validation loss at the end of each epoch, and a checkpoint is saved after every epoch. For IDK-15, this corresponds to approximately 235 optimiser steps per epoch. The base model, adapter configuration, and hyperparameters are identical for both mixtures, so the two training runs differ only in their training data.

## 4.4 Sequence-Level Confidence Estimation

Assessing whether fine-tuning alters the extent to which the model *knows what it knows* requires a confidence score for every generated response that can be compared against that response's correctness. The model's own token probabilities are used for this purpose, since they are obtained at no additional computational cost and do not require an auxiliary model or repeated sampling.

Given a query q, the model generates a response r = (r₁, …, r_T) by greedy decoding, with a maximum of 64 new tokens. At each decoding step, the log-probability assigned to the emitted token is recorded. The confidence score is defined as the exponentiated mean of these log-probabilities:

$$
c(q) = \exp\!\left(\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(r_t \mid r_{<t}, q)\right),
$$

which corresponds to the geometric mean of the per-token probabilities, a standard length-normalised likelihood measure for autoregressive models [31]. Length normalisation is necessary because the unnormalised sequence probability ∏ₜ p_θ(r_t | r_{<t}, q) decreases monotonically with each additional token. Without normalisation, a correct multi-sentence answer could receive a lower score than an incorrect single-word answer, reflecting response length rather than model certainty. The normalised score lies in [0, 1] and is comparable across responses of different lengths, including the fixed-length canonical refusal.

Each confidence score is paired with the refusal-aware correctness label y(q, r) defined in Section 3.2. Under this definition, abstention is counted as correct for an unanswerable query and as a false refusal for an answerable query, and an answer attempt on an answerable query is counted as correct when its token-level F1 score against the reference is at least 0.5. The resulting (confidence, correctness) pairs form the input to the calibration metrics described in Section 5.3. As a complementary, generation-independent measure, the teacher-forced negative log-likelihood of the *reference* answer is also computed for each model. This quantifies the probability mass the model assigns to the correct answer, irrespective of whether the model would generate that answer itself.

An interaction with the design choice described in Section 4.2 should be noted. Because all abstention targets share a single canonical string, a model trained on this data may reproduce the refusal with very high token probabilities. A high-confidence refusal therefore indicates that the model has learned the phrase reliably. It does not, on its own, demonstrate that the model has recognised a gap in its knowledge. For this reason, the aggregate calibration metrics are interpreted alongside the refusal analysis, which distinguishes correct abstentions from false refusals (Section 6).
