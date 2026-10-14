# 8 Conclusion and Future Work

This study examined whether the confidence calibration of a small language model can be improved, and its hallucination rate reduced, through the composition of its fine-tuning data alone. Phi-3-mini was fine-tuned with QLoRA for institutional question answering under a 4 GB VRAM budget, and three systems were compared: the base model, a standard domain fine-tune, and an abstention-aware fine-tune in which 15% of the training data consisted of abstention examples sharing a single canonical response.

Three principal conclusions emerge. First, **standard domain fine-tuning is insufficient and, on unanswerable queries, increases hallucination.** Fine-tuning on answerable data alone approximately doubled accuracy on answerable queries but eliminated the model's capacity to decline: the standard fine-tune answered all 79 unanswerable test queries, raising the hallucination rate from 0.506 for the base model to 1.000. Second, **the controlled inclusion of abstention data substantially reduces hallucination.** With 15% abstention examples, the hallucination rate fell to 0.139 and IDK F1 reached 0.840. The associated cost was limited to 15 false refusals among 370 answerable queries (4.1%) and a 4.3-percentage-point reduction in answerable accuracy, almost entirely attributable to those refusals rather than to degraded answer quality. Third, **the same modification to the training data improved calibration and discrimination without any calibration-specific objective.** ECE decreased from 0.622 (base) to 0.408 (standard fine-tuning) and 0.304 (abstention-aware fine-tuning), while AUROC increased from 0.557 to 0.764 and 0.810, with the largest gains concentrated in the high-confidence region where a deployment threshold would typically be set.

The analysis also identifies the limits of these findings. All systems remain over-confident across the full confidence range. Part of the calibration improvement derives from the model reproducing a fixed abstention phrase with high confidence, and the same mechanism renders some false refusals highly confident. The results are based on a single abstention ratio, a single random seed, one base model, and data from a single institution, and correctness is assessed using lexical-overlap measures that cannot always distinguish a valid paraphrase from an error.

These limitations suggest several directions for future work:

- **Characterising the abstention-ratio trade-off.** Training at a range of ratios (e.g., 5–30%), each with multiple random seeds, would establish how calibration, hallucination rate, and false refusals vary with the proportion of abstention data, and whether 15% lies near the optimum.
- **Varied versus canonical abstentions.** Training with varied, context-specific abstentions (e.g., directing the user to VTOP or a proctor) would indicate whether abstention can be learned as a graded expression of uncertainty rather than a memorised phrase, and how this affects the confidence assigned to refusals.
- **Post-hoc recalibration and selective prediction.** Temperature scaling fitted on the validation partition could reduce the residual over-confidence. A confidence threshold could then be tuned and evaluated using risk–coverage curves, establishing confidence-based filtering as an explicit second safeguard against hallucination.
- **Alternative uncertainty signals.** Comparing token-probability confidence with sampling-based semantic consistency, verbalised confidence, and lightweight probes of hidden representations would identify which signal is most effective for small models under strict computational constraints.
- **More robust correctness assessment.** NLI-based or LLM-as-judge evaluation, complemented by human assessment of a subset of responses, would reduce reliance on token overlap and clarify the proportion of responses marked incorrect that are in fact acceptable paraphrases.
- **Integration with retrieval.** Combining the abstention-aware model with retrieval over the curriculum documents could improve accuracy on answerable queries while preserving the ability to decline queries that no document addresses.
- **Broader evaluation.** Replicating the study with other small models (e.g., sub-4B variants of Llama, Qwen, or Gemma) and with data from other institutions would assess the generality of the findings, and examining calibration as curricula are revised across academic years would inform long-term maintenance of such systems.

Overall, the results indicate that a small, inexpensive model can be made measurably better calibrated and substantially less prone to hallucination within a narrow domain through a modest change to its training data, without modification of the training procedure. In student-facing applications, where a confident but incorrect answer regarding credits or deadlines can have real consequences, ensuring that a model's confidence reflects the reliability of its answers, and that it declines when it lacks the requisite knowledge, deserves as much attention as improving the answers themselves.

---

## Acknowledgments

> *[Fill in — keep it to one or two sentences. Typical content:]*
> We thank [supervisor's name and designation], [School], Vellore Institute of Technology, for guidance throughout this project. We also thank [anyone who helped build or validate the dataset, if applicable].
> *[If you received any funding or institutional compute grant, name it here; if not, omit funding entirely.]*

## Disclosure of Interests

The authors have no competing interests to declare that are relevant to the content of this article.

---

## References

1. Abdin, M., et al.: Phi-3 technical report: A highly capable language model locally on your phone. arXiv preprint arXiv:2404.14219 (2024)
2. Biderman, D., et al.: LoRA learns less and forgets less. Transactions on Machine Learning Research (2024)
3. Brahman, F., et al.: The art of saying no: Contextual noncompliance in language models. In: Advances in Neural Information Processing Systems (Datasets and Benchmarks Track) (2024)
4. Brier, G.W.: Verification of forecasts expressed in terms of probability. Monthly Weather Review **78**(1), 1–3 (1950)
5. Cheng, Q., Sun, T., Liu, X., Zhang, W., Yin, Z., Li, S., Li, L., He, Z., Chen, K., Qiu, X.: Can AI assistants know what they don't know? In: Proceedings of the 41st International Conference on Machine Learning (ICML) (2024)
6. Chow, C.K.: On optimum recognition error and reject tradeoff. IEEE Transactions on Information Theory **16**(1), 41–46 (1970)
7. Dettmers, T., Lewis, M., Shleifer, S., Zettlemoyer, L.: 8-bit optimizers via block-wise quantization. In: International Conference on Learning Representations (ICLR) (2022)
8. Dettmers, T., Pagnoni, A., Holtzman, A., Zettlemoyer, L.: QLoRA: Efficient finetuning of quantized LLMs. In: Advances in Neural Information Processing Systems (NeurIPS) (2023)
9. El-Yaniv, R., Wiener, Y.: On the foundations of noise-free selective classification. Journal of Machine Learning Research **11**, 1605–1641 (2010)
10. Feng, S., Shi, W., Wang, Y., Ding, W., Balachandran, V., Tsvetkov, Y.: Don't hallucinate, abstain: Identifying LLM knowledge gaps via multi-LLM collaboration. In: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL) (2024)
11. Geifman, Y., El-Yaniv, R.: Selective classification for deep neural networks. In: Advances in Neural Information Processing Systems (NeurIPS) (2017)
12. Gekhman, Z., Yona, G., Aharoni, R., Eyal, M., Feder, A., Reichart, R., Herzig, J.: Does fine-tuning LLMs on new knowledge encourage hallucinations? In: Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP) (2024)
13. Gunasekar, S., et al.: Textbooks are all you need. arXiv preprint arXiv:2306.11644 (2023)
14. Guo, C., Pleiss, G., Sun, Y., Weinberger, K.Q.: On calibration of modern neural networks. In: Proceedings of the 34th International Conference on Machine Learning (ICML). PMLR, vol. 70, pp. 1321–1330 (2017)
15. Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., de Laroussilhe, Q., Gesmundo, A., Attariyan, M., Gelly, S.: Parameter-efficient transfer learning for NLP. In: Proceedings of the 36th International Conference on Machine Learning (ICML). PMLR, vol. 97, pp. 2790–2799 (2019)
16. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W.: LoRA: Low-rank adaptation of large language models. In: International Conference on Learning Representations (ICLR) (2022)
17. Hu, X., Zhang, Y., Wei, S., Zhao, C., Li, Q., Li, B., Chen, F.: Small updates, big doubts: Does parameter-efficient fine-tuning enhance hallucination detection? arXiv preprint arXiv:2602.11166 (2026)
18. Huang, L., et al.: A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. ACM Transactions on Information Systems **43**(2) (2025)
19. Ji, Z., et al.: Survey of hallucination in natural language generation. ACM Computing Surveys **55**(12), 1–38 (2023)
20. Jiang, Z., Araki, J., Ding, H., Neubig, G.: How can we know when language models know? On the calibration of language models for question answering. Transactions of the Association for Computational Linguistics **9**, 962–977 (2021)
21. Kadavath, S., et al.: Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221 (2022)
22. Kalai, A.T., Nachum, O., Vempala, S.S., Zhang, E.: Why language models hallucinate. arXiv preprint arXiv:2509.04664 (2025)
23. Kalai, A.T., Vempala, S.S.: Calibrated language models must hallucinate. In: Proceedings of the 56th Annual ACM Symposium on Theory of Computing (STOC) (2024)
24. Kamath, A., Jia, R., Liang, P.: Selective question answering under domain shift. In: Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL) (2020)
25. Kapoor, S., et al.: Large language models must be taught to know what they don't know. In: Advances in Neural Information Processing Systems (NeurIPS) (2024)
26. Kuhn, L., Gal, Y., Farquhar, S.: Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. In: International Conference on Learning Representations (ICLR) (2023)
27. Lester, B., Al-Rfou, R., Constant, N.: The power of scale for parameter-efficient prompt tuning. In: Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP) (2021)
28. Lewis, P., et al.: Retrieval-augmented generation for knowledge-intensive NLP tasks. In: Advances in Neural Information Processing Systems (NeurIPS) (2020)
29. Li, X.L., Liang, P.: Prefix-tuning: Optimizing continuous prompts for generation. In: Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics (ACL), pp. 4582–4597 (2021)
30. Lin, S., Hilton, J., Evans, O.: TruthfulQA: Measuring how models mimic human falsehoods. In: Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL), pp. 3214–3252 (2022)
31. Malinin, A., Gales, M.: Uncertainty estimation in autoregressive structured prediction. In: International Conference on Learning Representations (ICLR) (2021)
32. Manakul, P., Liusie, A., Gales, M.J.F.: SelfCheckGPT: Zero-resource black-box hallucination detection for generative large language models. In: Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP) (2023)
33. Mangrulkar, S., Gugger, S., Debut, L., Belkada, Y., Paul, S., Bossan, B.: PEFT: State-of-the-art parameter-efficient fine-tuning methods. https://github.com/huggingface/peft (2022)
34. Naeini, M.P., Cooper, G.F., Hauskrecht, M.: Obtaining well calibrated probabilities using Bayesian binning. In: Proceedings of the AAAI Conference on Artificial Intelligence, vol. 29 (2015)
35. OpenAI: GPT-4 technical report. arXiv preprint arXiv:2303.08774 (2023)
36. Paszke, A., et al.: PyTorch: An imperative style, high-performance deep learning library. In: Advances in Neural Information Processing Systems (NeurIPS) (2019)
37. Pedregosa, F., et al.: Scikit-learn: Machine learning in Python. Journal of Machine Learning Research **12**, 2825–2830 (2011)
38. Porwal, S., Prasath S, S., Bi, X., Scandlen, M.: Evaluating hallucinations in domain-adapted large language models. arXiv preprint arXiv:2606.07521 (2026)
39. Rajpurkar, P., Jia, R., Liang, P.: Know what you don't know: Unanswerable questions for SQuAD. In: Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL), pp. 784–789 (2018)
40. Rajpurkar, P., Zhang, J., Lopyrev, K., Liang, P.: SQuAD: 100,000+ questions for machine comprehension of text. In: Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 2383–2392 (2016)
41. Tian, K., Mitchell, E., Zhou, A., Sharma, A., Rafailov, R., Yao, H., Finn, C., Manning, C.D.: Just ask for calibration: Strategies for eliciting calibrated confidence scores from language models fine-tuned with human feedback. In: Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP) (2023)
42. Wen, B., Yao, J., Feng, S., Xu, C., Tsvetkov, Y., Howe, B., Wang, L.L.: Know your limits: A survey of abstention in large language models. Transactions of the Association for Computational Linguistics (2025)
43. von Werra, L., et al.: TRL: Transformer reinforcement learning. https://github.com/huggingface/trl (2020)
44. Wolf, T., et al.: Transformers: State-of-the-art natural language processing. In: Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pp. 38–45 (2020)
45. Xiong, M., Hu, Z., Lu, X., Li, Y., Fu, J., He, J., Hooi, B.: Can LLMs express their uncertainty? An empirical evaluation of confidence elicitation in LLMs. In: International Conference on Learning Representations (ICLR) (2024)
46. Yang, Y., Chern, E., Qiu, X., Neubig, G., Liu, P.: Alignment for honesty. In: Advances in Neural Information Processing Systems (NeurIPS) (2024)
47. Zhang, C., Zhu, Z., Wei, Y., Tian, B., Liu, J., Wang, H., Wang, X., Liu, Y.: Confidence-calibrated small-large language model collaboration for cost-efficient reasoning. arXiv preprint arXiv:2603.03752 (2026)
48. Zhang, H., Diao, S., Lin, Y., Fung, Y.R., Lian, Q., Wang, X., Chen, Y., Ji, H., Zhang, T.: R-Tuning: Instructing large language models to say 'I don't know'. In: Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL) (2024)

---

### LaTeX note (if you use `llncs.cls`)

The current LNCS template puts the two end sections inside a `credits` environment, just before the bibliography:

```latex
\begin{credits}
\subsubsection{\ackname} We thank ...

\subsubsection{\discintname}
The authors have no competing interests to declare that are
relevant to the content of this article.
\end{credits}
```
