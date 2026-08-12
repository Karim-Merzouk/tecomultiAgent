"""Aggregate results/ into paper-style tables (markdown + LaTeX) and findings."""

from __future__ import annotations

import json
from pathlib import Path

_METRIC_COLS = ["ira", "msc", "eap", "sas", "gpc0", "gpc1", "sd", "brs", "ra"]
_HEADERS = ["IRA", "MSC", "EAP", "SAS", "GPC-0", "GPC-1", "SD", "BRS", "RA"]


def _load_summaries(results_dir: str | Path) -> list[dict]:
    out = []
    for sub in sorted(Path(results_dir).glob("*/summary.json")):
        out.append(json.loads(sub.read_text(encoding="utf-8")))
    return out


def markdown_table(results_dir: str | Path, lang: str = "all") -> str:
    summaries = _load_summaries(results_dir)
    lines = [
        f"### System comparison ({lang}) — TelcoAgent-Bench metrics",
        "",
        "| Config | " + " | ".join(_HEADERS) + " | esc% |",
        "|" + "---|" * (len(_HEADERS) + 2),
    ]
    for s in summaries:
        row = s["summary"].get(lang, {})
        vals = " | ".join(f"{row.get(c, 0):.3f}" for c in _METRIC_COLS)
        esc = 100.0 * s["n_escalated"] / max(1, s["n_cases"])
        lines.append(f"| {s['config']} | {vals} | {esc:.1f} |")
    lines += ["", f"_Embedder: {summaries[0]['embedder'] if summaries else 'n/a'}._"]
    return "\n".join(lines)


def comm_table(results_dir: str | Path) -> str:
    summaries = _load_summaries(results_dir)
    lines = [
        "### Communication overhead (new metric)",
        "",
        "| Config | msgs/case | tool_calls/case | tokens/case | repair_rounds/case |",
        "|---|---|---|---|---|",
    ]
    for s in summaries:
        c = s["comm_overhead"]
        lines.append(
            f"| {s['config']} | {c['msgs_per_case']:.2f} | {c['tool_calls_per_case']:.2f} "
            f"| {c['tokens_per_case']:.0f} | {c['repair_rounds_per_case']:.2f} |"
        )
    return "\n".join(lines)


def latex_table(results_dir: str | Path, lang: str = "all") -> str:
    summaries = _load_summaries(results_dir)
    cols = "l" + "r" * len(_HEADERS)
    lines = [
        r"\begin{tabular}{" + cols + "}",
        r"\toprule",
        "Config & " + " & ".join(_HEADERS) + r" \\",
        r"\midrule",
    ]
    for s in summaries:
        row = s["summary"].get(lang, {})
        vals = " & ".join(f"{row.get(c, 0):.3f}" for c in _METRIC_COLS)
        lines.append(f"{s['config'].replace('_', ' ')} & {vals} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def findings(results_dir: str | Path) -> str:
    summaries = {s["config"]: s for s in _load_summaries(results_dir)}
    out = ["### Auto-generated findings", ""]
    full = summaries.get("full_system")
    noch = summaries.get("no_checker")
    single = summaries.get("single_agent")
    if full and noch:
        d_gpc0 = full["summary"]["all"].get("gpc0", 0) - noch["summary"]["all"].get("gpc0", 0)
        d_sas = full["summary"]["all"].get("sas", 0) - noch["summary"]["all"].get("sas", 0)
        out.append(
            f"- Enabling the G5 checker changes GPC-0 by {d_gpc0:+.3f} and SAS by "
            f"{d_sas:+.3f} versus the checking-free pipeline."
        )
    if full and single:
        d_brs = full["summary"]["all"].get("brs", 0) - single["summary"]["all"].get("brs", 0)
        out.append(
            f"- The full multi-agent protocol changes BRS by {d_brs:+.3f} versus the "
            f"single-agent baseline."
        )
    for name, s in summaries.items():
        out.append(
            f"- `{name}`: {s['n_cases']} cases, {s['n_escalated']} escalated, "
            f"cross-language report consistency {s['cross_language_consistency']:.3f}."
        )
    return "\n".join(out)


def full_report(results_dir: str | Path) -> str:
    parts = [
        "# TelcoAgent Multi-Agent — Results",
        "",
        markdown_table(results_dir, "all"),
        "",
        markdown_table(results_dir, "en"),
        "",
        markdown_table(results_dir, "ar"),
        "",
        comm_table(results_dir),
        "",
        findings(results_dir),
        "",
        "### LaTeX (all languages)",
        "```latex",
        latex_table(results_dir, "all"),
        "```",
    ]
    return "\n".join(parts)
