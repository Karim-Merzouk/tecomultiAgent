# TelcoAgent Multi-Agent — Results

### System comparison (all) — TelcoAgent-Bench metrics

| Config | IRA | MSC | EAP | SAS | GPC-0 | GPC-1 | SD | BRS | RA | esc% |
|---|---|---|---|---|---|---|---|---|---|---|
| full_system_real | 0.271 | 0.780 | 0.880 | 0.691 | 0.321 | 0.607 | 0.000 | 0.543 | 0.369 | 64.3 |
| full_system_reconcile_real | 0.286 | 0.750 | 0.967 | 0.717 | 0.393 | 0.643 | 0.000 | 0.589 | 0.426 | 57.1 |
| no_checker_real | 0.281 | 0.518 | 0.863 | 0.458 | 0.000 | 0.321 | 0.000 | 0.296 | 0.564 | 0.0 |
| single_agent_real | 0.294 | 0.309 | 1.000 | 0.309 | 0.000 | 0.071 | 0.000 | 0.221 | 0.549 | 0.0 |

_Embedder: hashed-ngram-fallback._

### System comparison (en) — TelcoAgent-Bench metrics

| Config | IRA | MSC | EAP | SAS | GPC-0 | GPC-1 | SD | BRS | RA | esc% |
|---|---|---|---|---|---|---|---|---|---|---|
| full_system_real | 0.287 | 0.774 | 0.860 | 0.670 | 0.286 | 0.571 | 0.000 | 0.514 | 0.454 | 64.3 |
| full_system_reconcile_real | 0.306 | 0.786 | 0.958 | 0.744 | 0.429 | 0.714 | 0.000 | 0.629 | 0.493 | 57.1 |
| no_checker_real | 0.295 | 0.512 | 0.845 | 0.452 | 0.000 | 0.286 | 0.000 | 0.286 | 0.595 | 0.0 |
| single_agent_real | 0.314 | 0.309 | 1.000 | 0.309 | 0.000 | 0.071 | 0.000 | 0.221 | 0.604 | 0.0 |

_Embedder: hashed-ngram-fallback._

### System comparison (ar) — TelcoAgent-Bench metrics

| Config | IRA | MSC | EAP | SAS | GPC-0 | GPC-1 | SD | BRS | RA | esc% |
|---|---|---|---|---|---|---|---|---|---|---|
| full_system_real | 0.254 | 0.786 | 0.901 | 0.712 | 0.357 | 0.643 | 0.000 | 0.571 | 0.285 | 64.3 |
| full_system_reconcile_real | 0.266 | 0.714 | 0.976 | 0.691 | 0.357 | 0.571 | 0.000 | 0.550 | 0.359 | 57.1 |
| no_checker_real | 0.267 | 0.524 | 0.881 | 0.464 | 0.000 | 0.357 | 0.000 | 0.307 | 0.532 | 0.0 |
| single_agent_real | 0.274 | 0.309 | 1.000 | 0.309 | 0.000 | 0.071 | 0.000 | 0.221 | 0.493 | 0.0 |

_Embedder: hashed-ngram-fallback._

### Communication overhead (new metric)

| Config | msgs/case | tool_calls/case | tokens/case | repair_rounds/case |
|---|---|---|---|---|
| full_system_real | 10.04 | 5.50 | 6130 | 1.68 |
| full_system_reconcile_real | 9.54 | 3.50 | 5271 | 1.54 |
| no_checker_real | 5.00 | 3.14 | 3068 | 0.00 |
| single_agent_real | 3.00 | 1.61 | 1952 | 0.00 |

### Auto-generated findings

- `full_system_real`: 28 cases, 18 escalated, cross-language report consistency 0.663.
- `full_system_reconcile_real`: 28 cases, 16 escalated, cross-language report consistency 0.617.
- `no_checker_real`: 28 cases, 0 escalated, cross-language report consistency 0.324.
- `single_agent_real`: 28 cases, 0 escalated, cross-language report consistency 0.299.

### LaTeX (all languages)
```latex
\begin{tabular}{lrrrrrrrrr}
\toprule
Config & IRA & MSC & EAP & SAS & GPC-0 & GPC-1 & SD & BRS & RA \\
\midrule
full system real & 0.271 & 0.780 & 0.880 & 0.691 & 0.321 & 0.607 & 0.000 & 0.543 & 0.369 \\
full system reconcile real & 0.286 & 0.750 & 0.967 & 0.717 & 0.393 & 0.643 & 0.000 & 0.589 & 0.426 \\
no checker real & 0.281 & 0.518 & 0.863 & 0.458 & 0.000 & 0.321 & 0.000 & 0.296 & 0.564 \\
single agent real & 0.294 & 0.309 & 1.000 & 0.309 & 0.000 & 0.071 & 0.000 & 0.221 & 0.549 \\
\bottomrule
\end{tabular}
```