"""FSM tests with a scripted stub client (no real LLM)."""

from __future__ import annotations

import json

from telco_multiagent.data.synthetic import build_synthetic_dataset
from telco_multiagent.llm.client import GenResult
from telco_multiagent.orchestrator.config import RunConfig
from telco_multiagent.orchestrator.coordinator import Coordinator


class ScriptedClient:
    """Returns queued JSON per role; repeats the last response when drained."""

    name = "scripted"

    def __init__(self, scripts: dict[str, list[dict]]):
        self.scripts = {k: list(v) for k, v in scripts.items()}
        self.last: dict[str, dict] = {}

    def generate(self, role, system, user, context) -> GenResult:
        q = self.scripts.get(role, [])
        obj = q.pop(0) if q else self.last.get(role, {"kind": role})
        self.last[role] = obj
        return GenResult(text=json.dumps(obj), tokens=10)


def _happy_scripts():
    return {
        "G1": [{"kind": "G1", "intent_free_text": "coverage hole",
                "intent_label": "COVERAGE_HOLE", "category": "Coverage & Beam",
                "clarification_needed": False}],
        "G2": [{"kind": "G2",
                "planned_diagnostic_sequence": [
                    {"tool_name": "coverage_map", "arguments": {}},
                    {"tool_name": "oss_query", "arguments": {}}],
                "evidence_summary": "low rsrp", "missing_information": [],
                "anomaly_flags": ["coverage_hole"]}],
        "G3": [{"kind": "G3",
                "hypotheses": [{"root_cause": "sparse grid", "supporting_evidence": "rsrp",
                                "confidence": 0.9}],
                "selected_hypothesis": "sparse grid", "needs_more_evidence": False,
                "requested_evidence": []}],
        "G4": [{"kind": "G4",
                "planned_corrective_sequence": [{"tool_name": "create_ticket", "arguments": {}}],
                "justification_per_action": {"create_ticket": "log"}, "expected_effect": "fix"}],
        "G5": [{"kind": "G5", "verdict": "pass", "defects": [], "repair_hint": ""},
               {"kind": "G5", "verdict": "pass", "defects": [], "repair_hint": ""}],
        "G6": [{"kind": "G6", "report_text": "done", "root_cause": "sparse grid",
                "actions": ["create_ticket"], "ticket_ref": "TT-1", "report_language": "en"}],
    }


def _phases(state):
    return [e.detail["to"] for e in state.events if e.kind == "transition"]


def _run(scripts, **cfg_over):
    ds = build_synthetic_dataset()
    cfg = RunConfig(**cfg_over)
    coord = Coordinator(ScriptedClient(scripts), ds, cfg)
    sample = next(s for s in ds.samples if s.intent == "COVERAGE_HOLE")
    return coord.run_case(sample, "en")


def test_happy_path_phase_order():
    state = _run(_happy_scripts())
    ph = _phases(state)
    assert ph == ["INTENT", "ACQUIRE", "DIAGNOSE", "ACT", "CHECK", "REPORT",
                  "REPORT_CHECK", "DONE"]
    assert state.status.value == "done"
    assert state.agent_sequence() == ["coverage_map", "oss_query", "create_ticket"]


def test_intent_clarification_escalates():
    s = _happy_scripts()
    s["G1"] = [{"kind": "G1", "intent_free_text": "?", "intent_label": "unknown",
                "category": "", "clarification_needed": True}]
    state = _run(s)
    assert state.status.value == "escalated"
    assert state.escalation_reason == "intent_clarification_needed"


def test_evidence_loop_bound_escalates():
    s = _happy_scripts()
    s["G3"] = [{"kind": "G3", "hypotheses": [], "selected_hypothesis": "",
                "needs_more_evidence": True, "requested_evidence": ["more"]}]
    state = _run(s, max_evidence_loops=2)
    assert state.status.value == "escalated"
    assert state.escalation_reason == "evidence_loop_exhausted"
    assert state.budgets.evidence_loops_used == 2


def test_repair_bound_escalates():
    s = _happy_scripts()
    s["G5"] = [{"kind": "G5", "verdict": "fail",
                "defects": [{"type": "missing", "step": 1, "detail": "missing create_ticket"}],
                "repair_hint": "add ticket"}]
    state = _run(s, max_repair_rounds=2)
    assert state.status.value == "escalated"
    assert state.escalation_reason == "repair_budget_exhausted"
    assert state.budgets.repair_rounds_used == 2


def test_no_checker_skips_check_phase():
    state = _run(_happy_scripts(), enable_checker=False)
    ph = _phases(state)
    assert "CHECK" not in ph
    assert "REPORT_CHECK" not in ph
    assert state.status.value == "done"


def test_single_agent_no_diagnose_or_check():
    s = _happy_scripts()
    s["G2"] = [{"kind": "G2", "planned_diagnostic_sequence": [
        {"tool_name": "coverage_map", "arguments": {}},
        {"tool_name": "oss_query", "arguments": {}},
        {"tool_name": "create_ticket", "arguments": {}}],
        "evidence_summary": "x", "missing_information": [], "anomaly_flags": ["none"]}]
    state = _run(s, single_agent=True, enable_diagnosis=False, enable_checker=False)
    ph = _phases(state)
    assert "DIAGNOSE" not in ph and "CHECK" not in ph
    assert state.status.value == "done"


def test_tool_budget_exhaustion_escalates():
    s = _happy_scripts()
    s["G2"] = [{"kind": "G2", "planned_diagnostic_sequence":
                [{"tool_name": "oss_query", "arguments": {}}] * 5,
                "evidence_summary": "x", "missing_information": [], "anomaly_flags": ["none"]}]
    state = _run(s, max_tool_calls=3)
    assert state.status.value == "escalated"
    assert state.escalation_reason == "tool_budget_exhausted"
