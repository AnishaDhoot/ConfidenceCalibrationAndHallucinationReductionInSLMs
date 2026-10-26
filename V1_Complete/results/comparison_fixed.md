# Experiment comparison (semantic correctness)

IDK rows count as correct when the model refuses; answerable rows when it answers with token-F1 >= 0.5.

| Metric | Base | Normal FT | IDK FT |
|---|---|---|---|
| Accuracy (semantic) | 0.100 | 0.327 | 0.477 |
| Exact Match | 0.000 | 0.051 | 0.189 |
| Token F1 | 0.168 | 0.391 | 0.515 |
| ECE | 0.622 | 0.408 | 0.304 |
| Brier | 0.480 | 0.351 | 0.285 |
| NLL | 2.405 | 1.663 | 0.965 |
| AUROC | 0.557 | 0.764 | 0.810 |