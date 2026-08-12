"""Run configuration loaded from configs/*.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class BackendConfig:
    kind: str = "heuristic"  # heuristic | openai | anthropic
    model: str = "granite3.3:8b"
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "not-needed"
    temperature: float = 0.0
    # heuristic-only noise knobs
    intent_error_rate: float = 0.05
    drop_rate: float = 0.08
    swap_rate: float = 0.08
    distractor_rate: float = 0.10


@dataclass
class RunConfig:
    name: str = "full_system"
    seed: int = 42
    tau_intent: float = 0.5
    with_list: bool = True

    # ablation switches
    single_agent: bool = False
    enable_diagnosis: bool = True
    enable_checker: bool = True
    # repair strategy: True = apply G5's structural critique deterministically
    # (drop flagged extraneous, dedup, insert missing, fix order) instead of
    # re-prompting the same temp=0 planner, which re-emits the identical plan.
    reconcile_repair: bool = False
    # When False, the system never escalates/hands off: on any budget breach or
    # parse failure it COMMITS its best-so-far plan and reports, so it attempts
    # 100% of cases like a single-model baseline (apples-to-apples comparison).
    # Escalation (True) is a refuse-capable feature, reported only as an ablation.
    escalation_enabled: bool = True

    # loop bounds
    max_evidence_loops: int = 2
    max_repair_rounds: int = 2
    max_report_rounds: int = 1
    max_tool_calls: int = 15

    backend: BackendConfig = field(default_factory=BackendConfig)

    @staticmethod
    def from_yaml(path: str | Path) -> RunConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        backend = BackendConfig(**(raw.pop("backend", {}) or {}))
        return RunConfig(backend=backend, **raw)
