# Experiment comparison (semantic correctness)

IDK rows count as correct when the model refuses; answerable rows when it answers with token-F1 >= 0.5.

| Metric | Base | Normal FT | IDK FT |
|---|---|---|---|
| Accuracy (semantic) | 0.082 | 0.374 | 0.499 |
| Exact Match | 0.000 | 0.082 | 0.155 |
| Token F1 | 0.135 | 0.430 | 0.513 |
| ECE | 0.621 | 0.378 | 0.266 |
| Brier | 0.468 | 0.332 | 0.269 |
| NLL | 2.535 | 1.340 | 0.914 |
| AUROC | 0.435 | 0.795 | 0.786 |