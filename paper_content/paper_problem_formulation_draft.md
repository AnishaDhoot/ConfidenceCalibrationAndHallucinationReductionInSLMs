# 3 Problem Formulation

## 3.1 Task Definition

This study addresses closed-book question answering within a single institutional domain. The model is required to respond to student queries about VIT's academic programmes using only the knowledge acquired during fine-tuning, without access to retrieval or supporting documents at inference time. Let 𝒟 denote the source material that the assistant is intended to cover, namely VIT's official curriculum documents and academic regulations. Every incoming query q belongs to exactly one of two disjoint sets:

$$
\mathcal{Q} = \mathcal{Q}_A \,\cup\, \mathcal{Q}_U, \qquad \mathcal{Q}_A \cap \mathcal{Q}_U = \varnothing .
$$

A query is **answerable** (q ∈ 𝒬_A) if a correct answer can be derived from 𝒟, in which case it is associated with a reference answer r*. A query is **unanswerable** (q ∈ 𝒬_U) if 𝒟 does not support any correct answer. Unanswerable queries in this study fall into five categories: requests concerning an individual student's records, queries requiring real-time information, requests for specific details not stated in the documents, references to non-existent entities (such as invented course codes), and queries outside the institutional domain. The answerability label is denoted z(q) = 𝟙[q ∈ 𝒬_U].

Answerability is defined *relative to the scope of the assistant* rather than to the full extent of the model's parametric knowledge. A pre-trained model may correctly answer an out-of-domain general-knowledge question, yet such a question is nonetheless labelled unanswerable, since a response falls outside the source material for which the assistant can be held accountable. From the user's perspective, an unverifiable answer issued with confidence is indistinguishable from a fabricated one.

For a given query, the model f_θ produces a response r = f_θ(q) and an associated scalar confidence c(q) ∈ [0, 1], as defined in Section 4.4. Each response is either an answer attempt or an **abstention**. The abstention indicator a(r) ∈ {0, 1} takes the value 1 when r declines to answer. During training, every abstention target is the canonical string r_∅ (Section 4.2). During evaluation, a(r) is determined by a lexical refusal detector, so that abstentions expressed in the model's own wording are also recognised.

The study pursues two principal objectives: **reducing hallucination** and **improving confidence calibration**. The desired behaviour follows directly from these objectives. The model should answer correctly when q ∈ 𝒬_A, decline to answer when q ∈ 𝒬_U, and assign a confidence score that accurately reflects the likelihood that its response is correct. The answerability label z(q) is not available to the model, which must therefore infer from the query alone whether a reliable answer lies within its knowledge. Abstention thus serves as the behavioural mechanism through which hallucination is avoided, while confidence provides a continuous signal of response reliability.

## 3.2 Defining Hallucination and Correctness

Combining the answerability of a query with the model's decision to answer or abstain yields four possible outcomes:

| | a(r) = 1 (abstains) | a(r) = 0 (answers) |
|---|---|---|
| **q ∈ 𝒬_U** | correct abstention | **hallucination** |
| **q ∈ 𝒬_A** | false refusal | answer attempt (correct or incorrect) |

**Hallucination.** A hallucination is defined as *an answer attempt on an unanswerable query*:

$$
\text{hallucination}(q, r) = z(q)\,\bigl(1 - a(r)\bigr).
$$

This definition is intentionally conservative. For q ∈ 𝒬_U, the source material contains no correct answer, so any substantive response must be fabricated, regardless of how it is phrased. An incorrect response to an *answerable* query is more ambiguous: it may be fabricated, but it may also be a valid paraphrase not recognised by lexical metrics, or a partially correct answer. Such responses are therefore counted as errors in the correctness evaluation but are not classified as hallucinations. The hallucination rate is accordingly defined as the proportion of unanswerable queries that receive an answer attempt:

$$
\text{HR} = \frac{\sum_{q} z(q)\,\bigl(1 - a(r)\bigr)}{\sum_{q} z(q)} .
$$

**Refusal-aware correctness.** Relating confidence to outcomes requires a single binary correctness label that applies to both answerable and unanswerable queries. This label is defined as

$$
y(q, r) =
\begin{cases}
a(r), & \text{if } q \in \mathcal{Q}_U,\\[4pt]
\bigl(1 - a(r)\bigr)\cdot \mathbb{1}\!\left[\mathrm{F1}(r, r^{*}) \ge \tau\right], & \text{if } q \in \mathcal{Q}_A,
\end{cases}
$$

where F1(r, r*) denotes the token-level F1 score between the response and the reference after normalisation (Section 5.2), and τ = 0.5. For an unanswerable query, abstention is the only correct response. For an answerable query, abstention is incorrect, and an answer attempt is considered correct only if its overlap with the reference meets the threshold.

This formulation precludes both degenerate strategies. A model that always abstains attains y = 1 on every unanswerable query and y = 0 on every answerable query, so its accuracy equals the proportion of unanswerable queries (17.6% on the test set used in this study). A model that never abstains can at most attain the proportion of answerable queries and receives no credit on unanswerable ones. High accuracy is achievable only by deciding correctly, on a per-query basis, whether to answer. A token-overlap threshold is preferred to exact string matching because the models generate free-form text, and correct answers phrased differently from the reference would otherwise be marked as incorrect.

## 3.3 Calibration Objective

Confidence calibration is a primary objective of this study. A model is perfectly calibrated with respect to the correctness label y if

$$
\Pr\bigl(y(q, r) = 1 \;\big|\; c(q) = p\bigr) = p \qquad \text{for all } p \in [0, 1],
$$

that is, among all responses produced with confidence 0.8, exactly 80% are correct. Calibration alone, however, is not sufficient. A model that assigns a constant confidence equal to its overall accuracy to every response is perfectly calibrated yet provides no basis for distinguishing reliable responses from unreliable ones. Confidence must therefore also be **discriminative**, in that correct responses should receive higher confidence than incorrect ones:

$$
\Pr\bigl(c(q_i) > c(q_j) \;\big|\; y_i = 1,\; y_j = 0\bigr) \to 1 ,
$$

which is the quantity estimated by AUROC. When both properties are satisfied, a deployed system can apply a confidence threshold below which it withholds its answer or refers the user to an authoritative source, providing a second safeguard against hallucination in addition to the abstentions the model learns to produce.

The overall objective of this study is to obtain adapter parameters θ that jointly:

1. **improve confidence calibration**, as measured by lower ECE and Brier score;
2. **improve confidence discrimination**, as measured by higher AUROC;
3. **reduce hallucination**, as measured by a lower hallucination rate on 𝒬_U; and
4. **preserve utility**, as measured by a low false refusal rate and high answer quality on 𝒬_A.

None of these objectives is optimised directly. Training employs only the standard next-token cross-entropy loss over each templated training sequence x (Section 4.3):

$$
\mathcal{L}(\theta) = -\sum_{x \in \mathcal{D}_{\text{train}}} \sum_{t} \log p_\theta\bigl(x_t \mid x_{<t}\bigr),
$$

with only the LoRA parameters updated. The sole experimental variable is the composition of 𝒟_train, specifically the proportion of unanswerable queries paired with the canonical abstention r_∅. The central research question is therefore whether a change to the training *data* alone, without a calibration-specific loss, temperature scaling, or post-hoc recalibration, can improve calibration and reduce hallucination simultaneously, and at what cost to utility. Objectives 3 and 4 are in tension, since a higher proportion of abstention examples is expected to reduce hallucination while increasing the risk of false refusals. This trade-off is examined in Sections 6.1 and 6.4.
