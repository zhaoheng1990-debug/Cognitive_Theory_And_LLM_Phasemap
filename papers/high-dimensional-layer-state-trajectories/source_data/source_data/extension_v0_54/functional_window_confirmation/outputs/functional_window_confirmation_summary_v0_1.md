# Functional-window prospective confirmation v0.1

Verdict: **failed three-checkpoint role recurrence**.

| Checkpoint | Topology separation (95% CI) | Macro F1 | Label-null 95% | Downstream r | Target-null 95% | Frozen rank | Role | Location |
|---|---:|---:|---:|---:|---:|---:|---|---|
| qwen | -0.185 [-0.189, -0.181] | 0.739 | 0.521 | 0.659 | 0.623 | 58/112 (49.1%) | False | False |
| llama | -0.092 [-0.092, -0.091] | 0.662 | 0.510 | 0.411 | 0.452 | 38/40 (7.5%) | False | False |
| gemma | -0.090 [-0.091, -0.089] | 0.649 | 0.475 | 0.624 | 0.404 | 5/100 (96.0%) | False | True |

The confirmation set was not used for PCA fitting, feature scaling, classifier fitting, ridge fitting or window selection.
A positive result is restricted to new graph instantiations within the same controlled task family.
