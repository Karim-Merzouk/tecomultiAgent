# TelcoAgent Multi-Agent — Results

### System comparison (all) — TelcoAgent-Bench metrics

| Config | IRA | MSC | EAP | SAS | GPC-0 | GPC-1 | SD | BRS | RA | esc% |
|---|---|---|---|---|---|---|---|---|---|---|
| full_system_reconcile_real | 0.273 | 0.884 | 0.984 | 0.868 | 0.687 | 0.863 | 0.126 | 0.777 | 0.501 | 29.1 |

_Embedder: hashed-ngram-fallback._

### System comparison (en) — TelcoAgent-Bench metrics

| Config | IRA | MSC | EAP | SAS | GPC-0 | GPC-1 | SD | BRS | RA | esc% |
|---|---|---|---|---|---|---|---|---|---|---|
| full_system_reconcile_real | 0.309 | 0.887 | 0.983 | 0.870 | 0.682 | 0.878 | 0.120 | 0.781 | 0.542 | 29.1 |

_Embedder: hashed-ngram-fallback._

### System comparison (ar) — TelcoAgent-Bench metrics

| Config | IRA | MSC | EAP | SAS | GPC-0 | GPC-1 | SD | BRS | RA | esc% |
|---|---|---|---|---|---|---|---|---|---|---|
| full_system_reconcile_real | 0.236 | 0.882 | 0.985 | 0.867 | 0.693 | 0.848 | 0.132 | 0.774 | 0.460 | 29.1 |

_Embedder: hashed-ngram-fallback._

### Communication overhead (new metric)

| Config | msgs/case | tool_calls/case | tokens/case | repair_rounds/case |
|---|---|---|---|---|
| full_system_reconcile_real | 9.09 | 3.48 | 5148 | 1.34 |

### Auto-generated findings

- `full_system_reconcile_real`: 2940 cases, 855 escalated, cross-language report consistency 0.447.

### LaTeX (all languages)
```latex
\begin{tabular}{lrrrrrrrrr}
\toprule
Config & IRA & MSC & EAP & SAS & GPC-0 & GPC-1 & SD & BRS & RA \\
\midrule
full system reconcile real & 0.273 & 0.884 & 0.984 & 0.868 & 0.687 & 0.863 & 0.126 & 0.777 & 0.501 \\
\bottomrule
\end{tabular}
```