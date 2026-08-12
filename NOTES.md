# NOTES — decisions, deviations, TODOs

## Reconciliations against the design prompt / taxonomy (authoritative = the real dataset)

1. **20 intents, not 15.** The real TelcoAgent-Bench repo (present locally at
   `../TelcoAgent-Bench`) contains 20 intent folders, 49 blueprints, 1470
   dialogues. `vocab.INTENTS` lists all 20 with human labels and a curated
   7-category grouping (the dataset carries no `category` field, so the mapping
   is ours and flagged here).

2. **Tool registry is a union.** The README "core tools" table lists
   `recommend_param_change`, but the gold `gold_path` data actually uses
   `optimize_cell` and `optimize_slice` as the corrective actions. To guarantee
   no real gold path is rejected at parse time, `ToolName` is the union:
   7 README core + `optimize_cell` + `optimize_slice` + 6 distractors = 15 tools.
   Segment classification: diagnostic = {oss_query, kpi_timeseries, coverage_map,
   neighbor_audit}; corrective = {recommend_param_change, optimize_cell,
   optimize_slice, push_config, create_ticket}.

3. **Architecture attachment.** The prompt referenced
   `telcoagent_multiagent_architecture.mermaid`; the actual attachment was a PNG.
   Design was taken from the PNG + taxonomy doc, which agree with this prompt.

## Design choices realized (all from the prompt's settled decisions)
- G0 is deterministic Python (FSM in `orchestrator/coordinator.py`), not an LLM.
- Single trajectory writer: agents plan `ToolCallSpec`s; only the coordinator
  executes against the simulator and appends `TrajectoryStep`s.
- G3 is a separate agent; G7 escalation is a coordinator routing rule.
- Loop bounds: 2 evidence loops, 2 repair rounds/phase, 1 report round, 15 tool
  calls/case; any breach triggers escalation (ticket + handoff summary).
- Inter-agent messages are English + enum-typed; only the G6 report is localized.
- Intent confidence gating (`tau_intent`) fires when a backend supplies an
  envelope confidence; the offline heuristic backend signals ambiguity via
  `clarification_needed` instead.

## Offline "heuristic" backend
`llm/client.py::HeuristicOracleClient` makes the whole architecture runnable and
**produce metric tables without a served model**. It reads gold signals from
`context['_oracle']` and injects seeded noise (intent errors, dropped/swapped
diagnostic or corrective steps, distractor insertions) so metrics are realistic
(never trivially 1.0) and G5 has real defects to catch. It is an oracle-style
reference backend for smoke runs and pipeline demonstration — **real experiments
set `backend.kind: openai` (vLLM/Ollama serving Granite/Qwen/Llama) or
`anthropic`.** Noise knobs live in the config and sweep difficulty.

## Embeddings
IRA/RA use `paraphrase-multilingual-mpnet-base-v2` when
`sentence-transformers` is installed; otherwise a deterministic hashed
char-n-gram cosine fallback (weaker; recorded as `embedder` in every summary).

## TODO
- Phase 2: swap per-agent checkpoints via `backend` per agent (interface ready).
- Add W/o List IRA sweep to the runner (prompt file already exists).
- Escalation precision needs synthesized out-of-scope cases (~50) — generator
  stub is the corruption `relabel_intent` path; a dedicated out-of-taxonomy
  generator is still TODO.
