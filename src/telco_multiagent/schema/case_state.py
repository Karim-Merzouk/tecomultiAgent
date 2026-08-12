"""Case state, trajectory, and event log used by the coordinator."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from .messages import (
    AgentId,
    AnomalyFlag,
    Hypothesis,
    Language,
    ToolCallSpec,
)


class Phase(StrEnum):
    INIT = "INIT"
    INTENT = "INTENT"
    ACQUIRE = "ACQUIRE"
    DIAGNOSE = "DIAGNOSE"
    ACT = "ACT"
    CHECK = "CHECK"
    REPORT = "REPORT"
    REPORT_CHECK = "REPORT_CHECK"
    DONE = "DONE"
    ESCALATED = "ESCALATED"


class CaseStatus(StrEnum):
    active = "active"
    done = "done"
    escalated = "escalated"


class TrajectoryStep(BaseModel):
    idx: int
    tool_call: ToolCallSpec
    simulator_return: dict[str, object] = Field(default_factory=dict)
    provenance: AgentId
    phase_segment: Literal["diagnostic", "corrective"]
    wall_clock_s: float = 0.0
    tokens: int = 0


class TraceEvent(BaseModel):
    """One replayable event: envelope, tool exec, transition, or parse issue."""

    seq: int
    kind: Literal[
        "envelope",
        "tool_exec",
        "transition",
        "parse_rejection",
        "parse_failure",
        "escalation",
        "reconcile",
        "forced_completion",
    ]
    detail: dict[str, object] = Field(default_factory=dict)
    wall_clock_s: float = 0.0


class Budgets(BaseModel):
    evidence_loops_used: int = 0
    repair_rounds_used: int = 0
    report_rounds_used: int = 0
    tool_calls_used: int = 0
    max_evidence_loops: int = 2
    max_repair_rounds: int = 2
    max_report_rounds: int = 1
    max_tool_calls: int = 15


class CaseState(BaseModel):
    case_id: str
    language: Language
    blueprint_id: str
    intent_key: str | None = None
    intent_free_text: str = ""
    category: str = ""
    problem_statement: str = ""

    evidence_summary: str = ""
    anomaly_flags: list[AnomalyFlag] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    selected_hypothesis: str = ""
    requested_evidence: list[str] = Field(default_factory=list)
    repair_hint: str = ""

    trajectory: list[TrajectoryStep] = Field(default_factory=list)
    events: list[TraceEvent] = Field(default_factory=list)

    phase: Phase = Phase.INIT
    status: CaseStatus = CaseStatus.active
    budgets: Budgets = Field(default_factory=Budgets)

    ticket_ref: str | None = None
    escalation_reason: str | None = None
    report_text: str = ""
    root_cause: str = ""
    actions: list[str] = Field(default_factory=list)

    total_tokens: int = 0
    total_cost: float = 0.0
    wall_clock_s: float = 0.0

    # --- convenience -------------------------------------------------------

    def agent_sequence(self) -> list[str]:
        return [s.tool_call.tool_name.value for s in self.trajectory]

    def diagnostic_sequence(self) -> list[str]:
        return [
            s.tool_call.tool_name.value
            for s in self.trajectory
            if s.phase_segment == "diagnostic"
        ]

    def corrective_sequence(self) -> list[str]:
        return [
            s.tool_call.tool_name.value
            for s in self.trajectory
            if s.phase_segment == "corrective"
        ]

    def digest(self) -> str:
        payload = {
            "intent": self.intent_key,
            "seq": self.agent_sequence(),
            "phase": self.phase.value,
        }
        raw = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha1(raw).hexdigest()[:12]

    def log(self, kind: str, detail: dict[str, object], wall_clock_s: float = 0.0) -> None:
        self.events.append(
            TraceEvent(seq=len(self.events), kind=kind, detail=detail, wall_clock_s=wall_clock_s)
        )
