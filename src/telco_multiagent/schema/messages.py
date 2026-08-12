"""Inter-agent message protocol (Pydantic v2).

Every inter-agent message is an :class:`Envelope` carrying exactly one typed
payload. All LLM agents must emit JSON validating against their payload model;
hallucinated tool names are rejected at parse time (they become logged events
feeding the EAP analysis). Inter-agent fields are English and enum-typed; only
the engineer-facing report is localized.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

from ..vocab import ALL_TOOLS, IntentLabel, ToolName

Language = Literal["en", "ar"]


class AgentId(StrEnum):
    G0 = "G0"
    G1 = "G1"
    G2 = "G2"
    G3 = "G3"
    G4 = "G4"
    G5 = "G5"
    G6 = "G6"
    G7 = "G7"


class AnomalyFlag(StrEnum):
    coverage_hole = "coverage_hole"
    overshoot = "overshoot"
    beam_offset = "beam_azimuth_offset"
    high_prb = "high_prb_util"
    low_throughput = "low_throughput"
    high_bler = "high_bler"
    rlf_spike = "rlf_spike"
    pci_conflict = "pci_conflict"
    neighbor_misconfig = "neighbor_misconfig"
    latency_violation = "latency_violation"
    slice_congestion = "slice_congestion"
    backhaul_degraded = "backhaul_degraded"
    outage = "outage"
    none = "none"


class DefectType(StrEnum):
    missing = "missing"
    misordered = "misordered"
    extraneous = "extraneous"
    intent_mismatch = "intent_mismatch"
    report_inconsistency = "report_inconsistency"


class ToolCallSpec(BaseModel):
    """A planned tool invocation. Rejects out-of-registry tool names."""

    tool_name: ToolName
    arguments: dict[str, object] = Field(default_factory=dict)

    @field_validator("tool_name", mode="before")
    @classmethod
    def _reject_hallucinated(cls, v: object) -> object:
        if isinstance(v, str) and v not in ALL_TOOLS:
            raise ValueError(f"hallucinated tool name: {v!r}")
        return v


# --- per-agent payloads -----------------------------------------------------


_INTENT_VALUES = {e.value for e in IntentLabel}


class G1Payload(BaseModel):
    kind: Literal["G1"] = "G1"
    intent_free_text: str
    intent_label: IntentLabel
    category: str = ""
    clarification_needed: bool = False

    @field_validator("intent_label", mode="before")
    @classmethod
    def _coerce_label(cls, v: object) -> object:
        # accept only taxonomy keys; map off-taxonomy free text to `unknown`
        if isinstance(v, IntentLabel):
            return v
        s = str(v).strip().upper().replace(" ", "_").replace("-", "_")
        return s if s in _INTENT_VALUES else "unknown"


_ANOMALY_VALUES = {e.value for e in AnomalyFlag}


class G2Payload(BaseModel):
    kind: Literal["G2"] = "G2"
    planned_diagnostic_sequence: list[ToolCallSpec] = Field(default_factory=list)
    evidence_summary: str = ""
    missing_information: list[str] = Field(default_factory=list)
    anomaly_flags: list[AnomalyFlag] = Field(default_factory=list)

    @field_validator("anomaly_flags", mode="before")
    @classmethod
    def _keep_known_flags(cls, v: object) -> object:
        if not isinstance(v, list):
            return v
        return [f for f in v if isinstance(f, AnomalyFlag) or str(f) in _ANOMALY_VALUES]


class Hypothesis(BaseModel):
    root_cause: str
    supporting_evidence: str = ""
    confidence: float = Field(ge=0.0, le=1.0)


class G3Payload(BaseModel):
    kind: Literal["G3"] = "G3"
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    selected_hypothesis: str = ""
    needs_more_evidence: bool = False
    requested_evidence: list[str] = Field(default_factory=list)


class G4Payload(BaseModel):
    kind: Literal["G4"] = "G4"
    planned_corrective_sequence: list[ToolCallSpec] = Field(default_factory=list)
    justification_per_action: dict[str, str] = Field(default_factory=dict)
    expected_effect: str = ""


class Defect(BaseModel):
    type: DefectType
    # `step` is a locator only; real models emit ints, strings ("2"), tool
    # names, or "N/A" here, so coerce leniently to int-or-None rather than
    # rejecting an otherwise-valid verdict.
    step: int | None = None
    detail: str = ""

    @field_validator("step", mode="before")
    @classmethod
    def _coerce_step(cls, v: object) -> int | None:
        if v is None or isinstance(v, int):
            return v
        digits = "".join(ch for ch in str(v) if ch.isdigit())
        return int(digits) if digits else None


_DEFECT_VALUES = {e.value for e in DefectType}


class G5Payload(BaseModel):
    kind: Literal["G5"] = "G5"
    verdict: Literal["pass", "fail"]
    defects: list[Defect] = Field(default_factory=list)
    repair_hint: str = ""

    @field_validator("verdict", mode="before")
    @classmethod
    def _norm_verdict(cls, v: object) -> str:
        s = str(v).strip().lower()
        if s.startswith("fail"):
            return "fail"
        if s.startswith("pass"):
            return "pass"
        return "fail" if s not in ("ok", "true", "valid") else "pass"

    @field_validator("defects", mode="before")
    @classmethod
    def _drop_unknown_defects(cls, v: object) -> object:
        if not isinstance(v, list):
            return v
        kept = []
        for d in v:
            if isinstance(d, Defect):
                kept.append(d)
            elif isinstance(d, dict) and str(d.get("type")) in _DEFECT_VALUES:
                kept.append(d)
        return kept


class G6Payload(BaseModel):
    kind: Literal["G6"] = "G6"
    report_text: str
    root_cause: str = ""
    actions: list[str] = Field(default_factory=list)
    ticket_ref: str | None = None
    report_language: Language = "en"


Payload = Annotated[
    G1Payload | G2Payload | G3Payload | G4Payload | G5Payload | G6Payload,
    Field(discriminator="kind"),
]


class Envelope(BaseModel):
    case_id: str
    msg_id: str
    sender: AgentId
    recipient: AgentId
    next_agent: AgentId | None = None
    language: Language = "en"
    confidence: float | None = None
    state_digest: str = ""
    payload: Payload
