"""Typer CLI: run-case, run-eval, make-corruptions, defect-eval, report."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from .data.corruptions import generate_corruptions
from .data.loader import load_dataset
from .eval.defect_eval import run_defect_eval
from .eval.report import full_report
from .eval.runner import run_eval
from .llm.factory import make_client
from .orchestrator.config import RunConfig
from .orchestrator.coordinator import Coordinator

app = typer.Typer(add_completion=False, help="Multi-Agent TelcoAgent (Phase-1 baseline).")
console = Console()

_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


def _load_config(name_or_path: str) -> RunConfig:
    p = Path(name_or_path)
    if p.exists():
        return RunConfig.from_yaml(p)
    for cand in (_CONFIG_DIR / f"{name_or_path}.yaml", _CONFIG_DIR / "ablations" / f"{name_or_path}.yaml"):
        if cand.exists():
            return RunConfig.from_yaml(cand)
    raise typer.BadParameter(f"config not found: {name_or_path}")


@app.command("run-case")
def run_case(
    config: str = typer.Option("full_system", "--config"),
    language: str = typer.Option("en", "--language"),
    sample_id: str = typer.Option("", "--sample-id", help="empty = first sample"),
):
    """Run a single case end-to-end and print the trace summary."""
    cfg = _load_config(config)
    ds = load_dataset()
    client = make_client(cfg.backend, seed=cfg.seed)
    coord = Coordinator(client, ds, cfg)
    sample = next((s for s in ds.samples if s.id == sample_id), ds.samples[0])
    state = coord.run_case(sample, language)
    console.print(f"[bold]Case[/bold] {state.case_id}")
    console.print(f"intent={state.intent_key} status={state.status.value} "
                  f"phase={state.phase.value}")
    console.print(f"gold : {sample.gold_tools()}")
    console.print(f"agent: {state.agent_sequence()}")
    console.print(f"tool_calls={state.budgets.tool_calls_used} "
                  f"repairs={state.budgets.repair_rounds_used} "
                  f"evidence_loops={state.budgets.evidence_loops_used}")
    console.print(f"report: {state.report_text[:200]}")


@app.command("run-eval")
def run_eval_cmd(
    config: str = typer.Option("full_system", "--config"),
    split: str = typer.Option("test", "--split"),
    languages: str = typer.Option("en,ar", "--languages"),
    n_samples_per_blueprint: int = typer.Option(30, "--n-samples-per-blueprint"),
    seed: int = typer.Option(42, "--seed"),
    results_dir: str = typer.Option("results", "--results-dir"),
    no_traces: bool = typer.Option(False, "--no-traces"),
):
    """Evaluate a config over a split; write traces, per-blueprint CSV, summary."""
    cfg = _load_config(config)
    cfg.seed = seed
    langs = [x.strip() for x in languages.split(",") if x.strip()]
    res = run_eval(cfg, langs, split, n_samples_per_blueprint,
                   results_dir=results_dir, write_traces=not no_traces)
    console.print(f"[bold green]{cfg.name}[/bold green] cases={res.n_cases} "
                  f"escalated={res.n_escalated} embedder={res.embedder}")
    console.print_json(json.dumps(res.summary))
    console.print(f"comm: {res.comm}")
    console.print(f"cross-language consistency: {res.cross_lang}")


@app.command("make-corruptions")
def make_corruptions(
    seed: int = typer.Option(42, "--seed"),
    out: str = typer.Option("results/corruptions.json", "--out"),
):
    """Generate versioned, blueprint-split A10 corruptions."""
    ds = load_dataset()
    corr = generate_corruptions(ds, seed=seed)
    payload = {k: [r.to_dict() for r in v] for k, v in corr.items()}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    console.print(f"train={len(corr['train'])} eval={len(corr['eval'])} "
                  f"positives={len(corr['positives'])} -> {out}")


@app.command("defect-eval")
def defect_eval_cmd(config: str = typer.Option("full_system", "--config")):
    """Defect detection P/R/F1 for G5 on held-out corruptions."""
    cfg = _load_config(config)
    ds = load_dataset()
    prf = run_defect_eval(cfg, ds)
    for dtype, m in prf.items():
        console.print(f"{dtype:16s} P={m.precision:.3f} R={m.recall:.3f} F1={m.f1:.3f} "
                      f"(tp={m.tp} fp={m.fp} fn={m.fn})")


@app.command("report")
def report_cmd(
    results_dir: str = typer.Option("results", "--results-dir"),
    out: str = typer.Option("results/REPORT.md", "--out"),
):
    """Aggregate results/ into paper-style tables and a findings summary."""
    md = full_report(results_dir)
    Path(out).write_text(md, encoding="utf-8")
    console.print(md)
    console.print(f"\n[green]written[/green] {out}")


if __name__ == "__main__":
    app()
