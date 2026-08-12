"""Batch evaluation over dataset splits -> traces, per-blueprint CSV, summary."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..data.loader import Dataset, load_dataset
from ..data.records import DialogueSample
from ..llm.factory import make_client
from ..orchestrator.config import RunConfig
from ..orchestrator.coordinator import Coordinator
from ..schema.case_state import CaseState
from ..vocab import intent_label_str, tool_segment
from . import metrics as M
from . import new_metrics as NM
from .embed import backend_name


@dataclass
class CaseRun:
    sample: DialogueSample
    language: str
    state: CaseState


def split_blueprints(dataset: Dataset, split: str, seed: int, test_fraction: float = 0.3) -> set[str]:
    bps = sorted(dataset.by_blueprint().keys())
    rng = random.Random(seed)
    rng.shuffle(bps)
    n_test = max(1, int(len(bps) * test_fraction))
    test = set(bps[:n_test])
    if split == "test":
        return test
    if split == "train":
        return set(bps) - test
    return set(bps)


def select_samples(
    dataset: Dataset, split: str, seed: int, n_per_blueprint: int
) -> list[DialogueSample]:
    keep = split_blueprints(dataset, split, seed)
    rng = random.Random(seed + 1)
    out: list[DialogueSample] = []
    for bp, samples in sorted(dataset.by_blueprint().items()):
        if bp not in keep:
            continue
        chosen = sorted(samples, key=lambda s: s.id)
        rng.shuffle(chosen)
        out.extend(chosen[:n_per_blueprint])
    return out


def _seg(seq: list[str], segment: str) -> list[str]:
    return [t for t in seq if tool_segment(t) == segment]


def _blueprint_metrics(runs: list[CaseRun]) -> M.BlueprintMetrics:
    seqs = [r.state.agent_sequence() for r in runs]
    golds = [r.sample.gold_tools() for r in runs]
    msc = sum(M.msc(s, g) for s, g in zip(seqs, golds, strict=False)) / len(runs)
    eap = sum(M.eap(s, g) for s, g in zip(seqs, golds, strict=False)) / len(runs)
    sas = sum(M.sas(s, g) for s, g in zip(seqs, golds, strict=False)) / len(runs)
    msc_d = sum(M.msc(_seg(s, "diagnostic"), _seg(g, "diagnostic")) for s, g in zip(seqs, golds, strict=False)) / len(runs)
    msc_c = sum(M.msc(_seg(s, "corrective"), _seg(g, "corrective")) for s, g in zip(seqs, golds, strict=False)) / len(runs)
    g0 = M.gpc0(seqs, golds)
    g1 = M.gpc1(seqs, golds)
    sd = M.sequence_diversity(seqs)
    brs = M.brs(g0, g1, sd)
    ira = sum(
        M.ira(r.state.intent_free_text or r.sample.intent, intent_label_str(r.sample.intent))
        for r in runs
    ) / len(runs)
    ra = sum(M.ra(r.state.report_text, r.sample.gold_summary(r.language)) for r in runs) / len(runs)
    return M.BlueprintMetrics(
        blueprint_id=runs[0].sample.blueprint_id,
        intent=runs[0].sample.intent,
        language=runs[0].language,
        n=len(runs),
        msc=msc, eap=eap, sas=sas, msc_diag=msc_d, msc_corr=msc_c,
        gpc0=g0, gpc1=g1, sd=sd, brs=brs, ira=ira, ra=ra,
    )


@dataclass
class EvalResult:
    config_name: str
    embedder: str
    per_blueprint: list[M.BlueprintMetrics] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    comm: dict = field(default_factory=dict)
    cross_lang: float = 0.0
    n_cases: int = 0
    n_escalated: int = 0


def _aggregate(pbs: list[M.BlueprintMetrics]) -> dict:
    def by_lang(lang: str) -> dict:
        rows = [p for p in pbs if p.language == lang]
        if not rows:
            return {}
        keys = ["msc", "eap", "sas", "msc_diag", "msc_corr", "gpc0", "gpc1", "sd", "brs", "ira", "ra"]
        return {k: round(sum(getattr(r, k) for r in rows) / len(rows), 4) for k in keys}

    return {"en": by_lang("en"), "ar": by_lang("ar"), "all": _mean_all(pbs)}


def _mean_all(pbs: list[M.BlueprintMetrics]) -> dict:
    if not pbs:
        return {}
    keys = ["msc", "eap", "sas", "msc_diag", "msc_corr", "gpc0", "gpc1", "sd", "brs", "ira", "ra"]
    return {k: round(sum(getattr(r, k) for r in pbs) / len(pbs), 4) for k in keys}


def run_eval(
    config: RunConfig,
    languages: list[str],
    split: str = "test",
    n_per_blueprint: int = 30,
    dataset: Dataset | None = None,
    results_dir: str | Path = "results",
    write_traces: bool = True,
) -> EvalResult:
    dataset = dataset or load_dataset()
    client = make_client(config.backend, seed=config.seed)
    coord = Coordinator(client, dataset, config)
    samples = select_samples(dataset, split, config.seed, n_per_blueprint)

    out_dir = Path(results_dir) / config.name
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = out_dir / "traces"
    if write_traces:
        trace_dir.mkdir(exist_ok=True)

    runs: list[CaseRun] = []
    for sample in samples:
        for lang in languages:
            state = coord.run_case(sample, lang)
            runs.append(CaseRun(sample, lang, state))
            if write_traces:
                (trace_dir / f"{state.case_id.replace(':', '_')}.json").write_text(
                    state.model_dump_json(indent=1), encoding="utf-8"
                )

    # per (blueprint, language)
    groups: dict[tuple[str, str], list[CaseRun]] = {}
    for r in runs:
        groups.setdefault((r.sample.blueprint_id, r.language), []).append(r)
    pbs = [_blueprint_metrics(g) for g in groups.values()]
    pbs.sort(key=lambda p: (p.intent, p.blueprint_id, p.language))

    # cross-language consistency: pair en/ar reports of same sample
    by_sample: dict[str, dict[str, str]] = {}
    for r in runs:
        by_sample.setdefault(r.sample.id, {})[r.language] = r.state.report_text
    pairs = [(v["en"], v["ar"]) for v in by_sample.values() if "en" in v and "ar" in v]

    states = [r.state for r in runs]
    reasons: dict[str, int] = {}
    for s in states:
        if s.escalation_reason:
            reasons[s.escalation_reason] = reasons.get(s.escalation_reason, 0) + 1
    comm = NM.communication_overhead(states)
    result = EvalResult(
        config_name=config.name,
        embedder=backend_name(),
        per_blueprint=pbs,
        summary=_aggregate(pbs),
        comm=asdict(comm),
        cross_lang=round(NM.cross_language_consistency(pairs), 4),
        n_cases=len(states),
        n_escalated=sum(1 for s in states if s.status.value == "escalated"),
    )

    _write_csv(out_dir / "per_blueprint.csv", pbs)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": config.name,
                "embedder": result.embedder,
                "n_cases": result.n_cases,
                "n_escalated": result.n_escalated,
                "summary": result.summary,
                "comm_overhead": result.comm,
                "cross_language_consistency": result.cross_lang,
                "escalation_reasons": reasons,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return result


def _write_csv(path: Path, pbs: list[M.BlueprintMetrics]) -> None:
    import csv

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["intent", "blueprint_id", "language", "n", "MSC", "EAP", "SAS",
             "MSC_diag", "MSC_corr", "GPC0", "GPC1", "SD", "BRS", "IRA", "RA"]
        )
        for p in pbs:
            w.writerow(
                [p.intent, p.blueprint_id, p.language, p.n,
                 f"{p.msc:.4f}", f"{p.eap:.4f}", f"{p.sas:.4f}", f"{p.msc_diag:.4f}",
                 f"{p.msc_corr:.4f}", f"{p.gpc0:.4f}", f"{p.gpc1:.4f}", f"{p.sd:.4f}",
                 f"{p.brs:.4f}", f"{p.ira:.4f}", f"{p.ra:.4f}"]
            )
