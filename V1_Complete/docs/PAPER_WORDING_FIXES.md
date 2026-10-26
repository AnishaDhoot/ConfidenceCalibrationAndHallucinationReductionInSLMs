# Paper wording fixes (numbers unchanged)

The final paper (`paper/Confidence_Calibration_SLMs_Springer.pdf`, three authors) is newer than every editable copy in the project: `paper/paper.docx` and `paper/drafts/` are older versions (two authors, different abstract). A PDF cannot be edited, so these are exact replacements to paste into your current source. Section numbers follow the PDF. No table value, count or percentage changes.

## 1. Abstract: "three models"
Original: "Three models are trained and compared: the base model with no fine-tuning, a conventionally fine-tuned model ..., and a refusal-aware model ..."
Replace with: "Three systems are compared: the base model with no fine-tuning, and two LoRA-adapted models, one fine-tuned conventionally on answerable questions only and one refusal-aware model trained with a controlled proportion of unanswerable questions."

## 2. Dataset description
**Section 4.1, first source.** Original: "...2,349 question-answer pairs derived from VIT's official curriculum documents. These documents specify programme credit structures..."
Replace with: "...2,349 question-answer pairs derived from public VIT documents. Of these, 1,930 (82%) come from official curriculum and syllabus documents, which specify programme credit structures, course codes and their credit values, category-wise credit distributions and the academic regulations governing undergraduate and postgraduate programmes; the remainder come from academic regulations, the code of conduct, the academic calendar, research-admission guides, the student-life handbook and the fee structure. Rows that relied on non-public sources were removed before use."

**Section 3.1.** Original: "...in this case, it refers to the oﬃcial curriculum documents of VIT."
Replace with: "...in this case, it refers to the public VIT documents used to build the dataset, chiefly the official curriculum documents."

**Section 4.1, second source.** Original: "...a set of 980 questions constructed specifically to represent queries that the curriculum documents cannot answer."
Replace with: "...a supplementary set of 980 questions constructed to represent queries that the documents cannot answer. The refusal pattern below flags 789 of them; the other 191 are not matched by it (see the note on refusal wording below)."

**Section 4.1, after "3 pairs of core QA and 789 supplementary pairs were flagged as abstention."** Add: "The 3 core pairs are false positives: they are genuine answers that contain words such as 'real-time' and were relabelled with the canonical refusal. They are 3 of the 792 abstention examples."

## 3. Refusal-wording limitation
**Section 4.1, after "This is acknowledged as the limitation of the current study."** Add: "A broader pattern suggests that this affects roughly 5% of the IDK-0 training answers and roughly 4% of the answerable-labelled test queries (approximate counts). Three consequences follow. First, the standard fine-tuned model is not completely free of refusal-style targets. Second, some of the IDK FT false refusals may in fact be appropriate refusals, so the reported false refusal rate may be an upper bound. Third, a refusal that a model words differently is counted as an answer."

**Section 4.2 / 5.1.** Original: "...never sees abstention as an appropriate answer."
Replace with: "...sees almost no abstention targets, since only refusal-style rows that the pattern misses remain in its training data (Section 4.1)."

**Section 6.1.** Original: "the Normal FT model rejected none of the 449 test questions."
Replace with: "the Normal FT model rejected none of the 449 test questions according to the refusal pattern."

## 4. Refusal-detector note
**Section 5.2.** Replace the sentence "For every test query, we check whether the model refused to answer, using a pattern of refusal phrases such as 'I don't have', 'I cannot', and 'don't have enough information'." with:
"For every test query, we check whether the model refused to answer with a lexical pattern. The hallucination analysis (100 new tokens) uses the training-time pattern ('don't know', 'I don't', 'I cannot', 'cannot verify', 'real-time', 'not in my knowledge' and similar). The calibration analysis (64 new tokens) uses a narrower pattern ('I don't have', 'I cannot', 'do not have access', 'don't have enough information' and similar). Both identify the same 79 reference refusals in the test set."

**Section 5.4, last paragraph before "All the random functions".** Replace "Patterns of refusals used to label the output from the models are identical to the refusal patterns used to label the reference answers; ..." with: "Within each analysis, the pattern used to label model outputs is the same as the one used to label the reference answers; for the IDK-15 test set both patterns identify the same 79 unanswerable questions."

**Section 5.2, end (add):** "Answerable accuracy in Table 2 counts an answer as correct when it covers more than 40% of the reference words; accuracy in Table 3 is refusal-aware correctness with a token-F1 threshold of 0.5 over all queries. The two are therefore not directly comparable."
