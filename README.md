# Multi-Agent TelcoAgent — Phase 1 (Prompted Baseline)

A modular multi-agent system for bilingual (En/Ar) telecom RAN troubleshooting,
extending **TelcoAgent-Bench**. Every LLM agent shares one base model with a
role-specific prompt; the coordinator (G0) is deterministic Python. Structured
so Phase-2 fine-tuned checkpoints can be dropped in per agent.

```
Engineer ──query──▶ G0 Coordinator (FSM)
   G1 Intent ▶ G2 Acquire ⇄ G3 Diagnose ▶ G4 Act ⇄ G5 Check ▶ G6 Report
   G5 also checks the draft report; G7 escalation is a G0 routing rule.
```

## Install
```bash
cd telco-multiagent
pip install -e .            # + [embed] for real multilingual embeddings, [dev] for tests
```

## Run (offline — no served model needed)
The default `heuristic` backend produces results immediately:
```bash
telco-ma run-case  --config full_system --language en
telco-ma run-eval  --config full_system --split test --languages en,ar --n-samples-per-blueprint 10
telco-ma run-eval  --config no_checker  --split test --languages en,ar --n-samples-per-blueprint 10
telco-ma run-eval  --config single_agent --split test --languages en,ar --n-samples-per-blueprint 10
telco-ma make-corruptions
telco-ma defect-eval --config full_system
telco-ma report                    # aggregates results/ -> results/REPORT.md
```
(or `python -m telco_multiagent.cli ...`)

## Run with a real model
Serve Granite/Qwen/Llama via vLLM/Ollama/LM Studio, then set in the config:
```yaml
backend: { kind: openai, model: granite3.3:8b, base_url: http://localhost:11434/v1 }
```
`kind: anthropic` uses the Anthropic API (`ANTHROPIC_API_KEY`).

## Layout
- `schema/` — inter-agent protocol + case state (Pydantic v2)
- `data/` — loader (real TelcoAgent-Bench), records, synthetic fallback, A10 corruptions
- `tools/` — registry (7+2 core, 6 distractors) + blueprint-driven simulator
- `agents/` — BaseAgent (JSON parse/retry) + G1–G6; prompts in `prompts/*.md`
- `orchestrator/` — G0 FSM, budgets, escalation, ablation switches
- `eval/` — metrics (IRA/MSC/EAP/SAS/GPC-0/1/SD/BRS/RA), new metrics, runner, report
- `configs/` — `full_system`, `no_checker`, `single_agent`

See `NOTES.md` for how the code reconciles with the real dataset (20 intents,
tool-name union, offline backend).

## Tests
```bash
pytest        # schema, FSM (no LLM), simulator, metrics, corruptions
```
