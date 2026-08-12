# TelcoAgent Multi-Agent — Results

### System comparison (all) — TelcoAgent-Bench metrics

| Config | IRA | MSC | EAP | SAS | GPC-0 | GPC-1 | SD | BRS | RA | esc% |
|---|---|---|---|---|---|---|---|---|---|---|
| full_system_forced_real | 0.268 | 0.823 | 0.977 | 0.800 | 0.629 | 0.787 | 0.176 | 0.715 | 0.559 | 0.0 |

_Embedder: hashed-ngram-fallback._

### System comparison (en) — TelcoAgent-Bench metrics

| Config | IRA | MSC | EAP | SAS | GPC-0 | GPC-1 | SD | BRS | RA | esc% |
|---|---|---|---|---|---|---|---|---|---|---|
| full_system_forced_real | 0.305 | 0.840 | 0.973 | 0.813 | 0.626 | 0.805 | 0.149 | 0.725 | 0.594 | 0.0 |

_Embedder: hashed-ngram-fallback._

### System comparison (ar) — TelcoAgent-Bench metrics

| Config | IRA | MSC | EAP | SAS | GPC-0 | GPC-1 | SD | BRS | RA | esc% |
|---|---|---|---|---|---|---|---|---|---|---|
| full_system_forced_real | 0.230 | 0.805 | 0.982 | 0.787 | 0.631 | 0.769 | 0.204 | 0.706 | 0.524 | 0.0 |

_Embedder: hashed-ngram-fallback._

### Communication overhead (new metric)

| Config | msgs/case | tool_calls/case | tokens/case | repair_rounds/case |
|---|---|---|---|---|
| full_system_forced_real | 9.50 | 3.49 | 5330 | 1.34 |

### Auto-generated findings

- `full_system_forced_real`: 2940 cases, 0 escalated, cross-language report consistency 0.339.

### LaTeX (all languages)
```latex
\begin{tabular}{lrrrrrrrrr}
\toprule
Config & IRA & MSC & EAP & SAS & GPC-0 & GPC-1 & SD & BRS & RA \\
\midrule
full system forced real & 0.268 & 0.823 & 0.977 & 0.800 & 0.629 & 0.787 & 0.176 & 0.715 & 0.559 \\
\bottomrule
\end{tabular}
```