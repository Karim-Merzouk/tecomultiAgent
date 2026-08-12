"""G0 Coordinator: deterministic finite-state machine.

Enforces phase order Understand -> Investigate -> Resolve -> Report, executes all
tool calls against the simulator (single trajectory writer), applies loop bounds,
and routes on the structured fields of incoming payloads. Every envelope, tool
execution, and transition is logged to a replayable per-case JSON trace.
"""

from __future__ import annotations

import re
import time

from ..agents import (
    g1_intent,
    g2_acquire,
    g3_diagnose,
    g4_act,
    g5_check,
    g6_report,
)
from ..agents.base import AgentResult, BaseAgent
from ..data.loader import Dataset
from ..data.records import DialogueSample
from ..llm.client import LLMClient
from ..schema.case_state import CaseState, CaseStatus, Phase, TrajectoryStep
from ..schema.messages import AgentId, ToolCallSpec
from ..tools.simulator import ToolSimulator
from ..vocab import intent_category, intent_label_str, tool_segment
from .config import RunConfig

_DIAG = "diagnostic"
_CORR = "corrective"


def _first_sentence(text: str) -> str:
    for sep in (". ", "،", ".\n"):
        if sep in text:
            return text.split(sep)[0].strip()
    return text[:160].strip()


class Coordinator:
    def __init__(self, client: LLMClient, dataset: Dataset, config: RunConfig):
        self.client = client
        self.dataset = dataset
        self.config = config
        self.agents: dict[str, BaseAgent] = {
            "G1": g1_intent.make_agent(client, with_list=config.with_list),
            "G2": g2_acquire.make_agent(client),
            "G3": g3_diagnose.make_agent(client),
            "G4": g4_act.make_agent(client),
            "G5": g5_check.make_agent(client),
            "G6": g6_report.make_agent(client),
        }

    # --- context / bookkeeping helpers -------------------------------------

    def _oracle(self, sample: DialogueSample, sim: ToolSimulator) -> dict:
        gold = sample.gold_tools()
        confusable = "PCI_COLLISION" if sample.intent != "PCI_COLLISION" else "COVERAGE_HOLE"
        return {
            "intent_key": sample.intent,
            "intent_label": intent_label_str(sample.intent),
            "category": intent_category(sample.intent),
            "gold_tools": gold,
            "corrective_tools": [t for t in gold if tool_segment(t) == _CORR],
            "problem_statement": sample.query_en,
            "root_cause": _first_sentence(sample.gold_summary_en),
            "gold_summary": sample.gold_summary(self._lang),
            "anomaly_flags": sim.anomaly_flags,
            "evidence_hint": f"KPIs: prb={sim.kpis['prb_util']}, dl={sim.kpis['dl_throughput_mbps']}",
            "confusable": confusable,
        }

    def _ctx(self, state: CaseState, extra: dict | None = None) -> dict:
        ctx = {"case_id": state.case_id, "language": state.language, "_oracle": self._oracle_ctx}
        if extra:
            ctx.update(extra)
        return ctx

    def _run_agent(self, role: str, user: str, state: CaseState, extra: dict | None = None) -> AgentResult:
        t0 = time.perf_counter()
        res = self.agents[role].invoke(user, self._ctx(state, extra))
        dt = time.perf_counter() - t0
        state.total_tokens += res.tokens
        state.wall_clock_s += dt
        for rej in res.parse_rejections:
            state.log("parse_rejection", {"agent": role, "error": rej})
        if res.parse_failed:
            state.log("parse_failure", {"agent": role, "error": res.raw}, dt)
        else:
            state.log(
                "envelope",
                {
                    "sender": role,
                    "digest": state.digest(),
                    "payload": res.payload.model_dump(mode="json"),
                },
                dt,
            )
        return res

    def _exec(self, state: CaseState, sim: ToolSimulator, call: ToolCallSpec, provenance: str) -> bool:
        if state.budgets.tool_calls_used >= state.budgets.max_tool_calls:
            return False
        ret = sim.execute(call.tool_name.value, call.arguments)
        seg = tool_segment(call.tool_name.value)
        step_seg = _CORR if seg == _CORR else _DIAG
        step = TrajectoryStep(
            idx=len(state.trajectory),
            tool_call=call,
            simulator_return=ret,
            provenance=AgentId(provenance),
            phase_segment=step_seg,
        )
        state.trajectory.append(step)
        state.budgets.tool_calls_used += 1
        if call.tool_name.value == "create_ticket" and ret.get("ticket_id"):
            state.ticket_ref = ret["ticket_id"]
        state.log("tool_exec", {"tool": call.tool_name.value, "return": ret})
        return True

    def _observed_evidence(self, state: CaseState) -> str:
        """Summarize the actual diagnostic tool returns for downstream agents."""
        parts: list[str] = []
        for step in state.trajectory:
            if step.phase_segment != _DIAG:
                continue
            ret = step.simulator_return
            tool = step.tool_call.tool_name.value
            if tool == "oss_query":
                parts.append(
                    f"oss_query[prb={ret.get('prb_util')},dl={ret.get('dl_throughput_mbps')}"
                    f"Mbps,bler={ret.get('bler')},rlf={ret.get('rlf_rate')}]"
                )
            elif tool == "coverage_map":
                parts.append(
                    f"coverage_map[rsrp={ret.get('rsrp_dbm')},sinr={ret.get('sinr_db')},"
                    f"flags={ret.get('flags')}]"
                )
            elif tool == "kpi_timeseries":
                parts.append(f"kpi_timeseries[{ret.get('kpi')},mean={ret.get('mean')}]")
            elif tool == "neighbor_audit":
                parts.append(f"neighbor_audit[flags={ret.get('flags')}]")
        return "; ".join(parts)

    def _transition(self, state: CaseState, to: Phase) -> None:
        state.log("transition", {"from": state.phase.value, "to": to.value})
        state.phase = to

    def _force_complete(self, state: CaseState, reason: str, sim: ToolSimulator) -> CaseState:
        """No-escalation mode: finalize the case with a report over whatever was
        committed so far (no handoff ticket), so the system attempts 100% of
        cases like a single-model baseline. The dominant budget-breach path
        (repair_budget_exhausted) commits its best corrective inline before this
        is reached; the remaining paths report over the current trajectory."""
        state.log("forced_completion", {"reason": reason})
        return self._do_report(self._sample, state, sim, checker=False)

    def _escalate(self, state: CaseState, reason: str, sim: ToolSimulator) -> CaseState:
        if not self.config.escalation_enabled:
            return self._force_complete(state, reason, sim)
        state.escalation_reason = reason
        state.log("escalation", {"reason": reason})
        ticket_call = ToolCallSpec(
            tool_name="create_ticket",
            arguments={"summary": f"ESCALATION [{reason}] intent={state.intent_key} "
                       f"evidence={state.evidence_summary[:120]}"},
        )
        self._exec(state, sim, ticket_call, "G0")
        state.report_text = (
            f"[ESCALATED: {reason}] Handoff ticket {state.ticket_ref} created. "
            f"Root cause (partial): {state.selected_hypothesis or 'undetermined'}."
        )
        self._transition(state, Phase.ESCALATED)
        state.status = CaseStatus.escalated
        return state

    # --- main entry --------------------------------------------------------

    def run_case(self, sample: DialogueSample, language: str) -> CaseState:
        self._lang = language
        self._sample = sample
        cfg = self.config
        bp = self.dataset.blueprints.get(sample.blueprint_id)
        if bp is None:
            from ..data.records import Blueprint

            bp = Blueprint(intent=sample.intent, blueprint_id=sample.blueprint_id)
        sim = ToolSimulator(bp, cfg.seed, sample.id)
        self._oracle_ctx = self._oracle(sample, sim)

        state = CaseState(
            case_id=f"{sample.id}:{language}:{cfg.name}",
            language=language,
            blueprint_id=sample.blueprint_id,
            problem_statement=sample.query(language),
        )
        state.budgets.max_evidence_loops = cfg.max_evidence_loops
        state.budgets.max_repair_rounds = cfg.max_repair_rounds
        state.budgets.max_report_rounds = cfg.max_report_rounds
        state.budgets.max_tool_calls = cfg.max_tool_calls

        if cfg.single_agent:
            return self._run_single_agent(sample, state, sim)
        return self._run_pipeline(sample, state, sim)

    # --- single-agent ablation --------------------------------------------

    def _run_single_agent(self, sample, state, sim) -> CaseState:
        # one intent call + one combined plan + one report; no diagnosis/checker/loops
        self._transition(state, Phase.INTENT)
        r1 = self._run_agent("G1", g1_intent.build_user(
            state.problem_statement, state.language, self._candidates()), state)
        if r1.parse_failed:
            return self._escalate(state, "parse_failure_intent", sim)
        self._apply_intent(state, r1)

        self._transition(state, Phase.ACQUIRE)
        r2 = self._run_agent("G2", g2_acquire.build_user(state.intent_key or "unknown", "", [], ""),
                             state, extra={"plan_segment": "all"})
        if r2.parse_failed:
            return self._escalate(state, "parse_failure_plan", sim)
        for call in r2.payload.planned_diagnostic_sequence:
            self._exec(state, sim, call, "G2")

        return self._do_report(sample, state, sim, checker=False)

    # --- full / no-checker pipeline ---------------------------------------

    def _run_pipeline(self, sample, state, sim) -> CaseState:
        cfg = self.config
        # INTENT
        self._transition(state, Phase.INTENT)
        r1 = self._run_agent("G1", g1_intent.build_user(
            state.problem_statement, state.language, self._candidates()), state)
        if r1.parse_failed:
            return self._escalate(state, "parse_failure_intent", sim)
        self._apply_intent(state, r1)
        # confidence gating (tau_intent) applies when the backend supplies an
        # envelope confidence; the offline heuristic signals ambiguity via the
        # clarification_needed flag instead.
        if r1.payload.clarification_needed:
            return self._escalate(state, "intent_clarification_needed", sim)

        # INVESTIGATE loop
        while True:
            self._transition(state, Phase.ACQUIRE)
            r2 = self._run_agent("G2", g2_acquire.build_user(
                state.intent_key or "unknown", state.evidence_summary,
                state.requested_evidence, state.repair_hint), state)
            if r2.parse_failed:
                return self._escalate(state, "parse_failure_acquire", sim)
            state.evidence_summary = r2.payload.evidence_summary
            state.anomaly_flags = r2.payload.anomaly_flags
            state.requested_evidence = []
            state.repair_hint = ""
            for call in r2.payload.planned_diagnostic_sequence:
                if not self._exec(state, sim, call, "G2"):
                    return self._escalate(state, "tool_budget_exhausted", sim)
            # ground evidence in the actual simulator returns (not G2's pre-exec claim)
            observed = self._observed_evidence(state)
            if observed:
                state.evidence_summary = (
                    f"{state.evidence_summary} | Observed: {observed}"
                    if state.evidence_summary else f"Observed: {observed}"
                )

            if cfg.enable_diagnosis:
                self._transition(state, Phase.DIAGNOSE)
                r3 = self._run_agent("G3", g3_diagnose.build_user(
                    state.intent_key or "unknown", state.evidence_summary,
                    [a.value for a in state.anomaly_flags]), state)
                if r3.parse_failed:
                    return self._escalate(state, "parse_failure_diagnose", sim)
                state.hypotheses = r3.payload.hypotheses
                state.selected_hypothesis = r3.payload.selected_hypothesis
                if r3.payload.needs_more_evidence:
                    if state.budgets.evidence_loops_used < state.budgets.max_evidence_loops:
                        state.budgets.evidence_loops_used += 1
                        state.requested_evidence = r3.payload.requested_evidence
                        continue
                    return self._escalate(state, "evidence_loop_exhausted", sim)
            else:
                state.selected_hypothesis = _first_sentence(sample.gold_summary_en)
            break

        # RESOLVE loop (ACT <-> CHECK)
        in_repair = False
        while True:
            self._transition(state, Phase.ACT)
            r4 = self._run_agent(
                "G4",
                g4_act.build_user(state.intent_key or "unknown", state.selected_hypothesis,
                                  state.repair_hint),
                state, extra={"repair": in_repair},
            )
            if r4.parse_failed:
                return self._escalate(state, "parse_failure_act", sim)
            state.repair_hint = ""
            planned_corr = list(r4.payload.planned_corrective_sequence)

            if not self.config.enable_checker:
                for call in planned_corr:
                    if not self._exec(state, sim, call, "G4"):
                        return self._escalate(state, "tool_budget_exhausted", sim)
                break

            # CHECK the full concatenated sequence (executed diag + planned corr)
            self._transition(state, Phase.CHECK)
            diag_seq = state.diagnostic_sequence()
            full = diag_seq + [c.tool_name.value for c in planned_corr]
            r5 = self._run_agent(
                "G5",
                g5_check.build_sequence_user(state.intent_key or "unknown", full, sample.gold_tools()),
                state, extra={"full_sequence": full},
            )
            if r5.parse_failed:
                return self._escalate(state, "parse_failure_check", sim)

            if r5.payload.verdict == "pass":
                for call in planned_corr:
                    if not self._exec(state, sim, call, "G4"):
                        return self._escalate(state, "tool_budget_exhausted", sim)
                break

            # fail: route by defect segment
            state.repair_hint = r5.payload.repair_hint or "; ".join(
                d.detail for d in r5.payload.defects)
            if state.budgets.repair_rounds_used >= state.budgets.max_repair_rounds:
                if not self.config.escalation_enabled:
                    # commit the best-so-far corrective (apply the last critique
                    # if reconcile is on), then report — a full committed attempt.
                    commit = planned_corr
                    if self.config.reconcile_repair:
                        c = self._reconcile(state, sim, planned_corr, r5.payload.defects)
                        if c is not None:
                            commit = c
                    seen: set[str] = set()
                    for call in commit:
                        tn = call.tool_name.value
                        if tn in seen:
                            continue
                        seen.add(tn)
                        if not self._exec(state, sim, call, "G4"):
                            break
                    state.log("forced_completion", {"reason": "repair_budget_exhausted"})
                    return self._do_report(sample, state, sim, checker=False)
                return self._escalate(state, "repair_budget_exhausted", sim)
            state.budgets.repair_rounds_used += 1
            in_repair = True

            # deterministic critique-application repair: apply G5's structural
            # defects to the plan instead of re-prompting the same temp=0 planner
            # (which re-emits the identical failing plan -> guaranteed escalation).
            did_reconcile = False
            if self.config.reconcile_repair:
                corrected = self._reconcile(state, sim, planned_corr, r5.payload.defects)
                if corrected is not None:
                    did_reconcile = True
                    self._transition(state, Phase.CHECK)
                    full2 = state.diagnostic_sequence() + [c.tool_name.value for c in corrected]
                    r5b = self._run_agent(
                        "G5",
                        g5_check.build_sequence_user(
                            state.intent_key or "unknown", full2, sample.gold_tools()),
                        state, extra={"full_sequence": full2, "reconciled": True},
                    )
                    if not r5b.parse_failed and r5b.payload.verdict == "pass":
                        for call in corrected:
                            if not self._exec(state, sim, call, "G4"):
                                return self._escalate(state, "tool_budget_exhausted", sim)
                        break
                    if not r5b.parse_failed:
                        state.repair_hint = r5b.payload.repair_hint or state.repair_hint

            if not did_reconcile and self._defect_segment(r5.payload.defects, diag_seq) == _DIAG:
                # reset the executed diagnostic segment and re-acquire cleanly
                self._reset_diagnostics(state)
                self._transition(state, Phase.ACQUIRE)
                r2b = self._run_agent(
                    "G2",
                    g2_acquire.build_user(state.intent_key or "unknown", state.evidence_summary,
                                          ["re-run diagnostics per checker"], state.repair_hint),
                    state, extra={"repair": True},
                )
                if r2b.parse_failed:
                    return self._escalate(state, "parse_failure_acquire", sim)
                state.anomaly_flags = r2b.payload.anomaly_flags
                for call in r2b.payload.planned_diagnostic_sequence:
                    if not self._exec(state, sim, call, "G2"):
                        return self._escalate(state, "tool_budget_exhausted", sim)
            # defect on corrective (or after re-acquire) -> replan ACT with repair flag

        return self._do_report(sample, state, sim, checker=self.config.enable_checker)

    def _reset_diagnostics(self, state: CaseState) -> None:
        removed = [s for s in state.trajectory if s.phase_segment == _DIAG]
        state.trajectory = [s for s in state.trajectory if s.phase_segment != _DIAG]
        for i, step in enumerate(state.trajectory):
            step.idx = i
        state.log("transition", {"reset": "diagnostic_segment", "dropped": len(removed)})

    # --- deterministic critique-application repair --------------------------

    _ADD_KW = ("missing", "absent", "omitted", "should be added", "should include",
               "needs to be added", "not present", "should be present", "add ")
    _REMOVE_KW = ("extraneous", "not part", "redundant", "should not", "should be removed",
                  "remove", "replace", "does not align", "irrelevant", "unnecessary")
    _ORDER_KW = ("misorder", "order", "precede", "follow", "before", "after", "reorder")

    @classmethod
    def _mentioned_tools(cls, defects, present: set[str]) -> tuple[set[str], set[str], bool]:
        """Split the checker's named tools into (remove, add, reorder).

        Classifies by the defect *detail text* semantics (missing/absent -> add,
        extraneous/replace -> remove) rather than the defect *type*, because the
        prompted granite G5 frequently mislabels types (reports genuinely-missing
        gold steps under `misordered`/`intent_mismatch`). Presence-aware: only
        absent tools are added, only present tools are removed. Uses G5's output
        only — never the gold flow directly (G5 is a gold-aware verifier by
        design), applying the checker's authority instead of re-prompting the
        same temp=0 planner.
        """
        from ..vocab import ALL_TOOLS

        remove: set[str] = set()
        add: set[str] = set()
        reorder = False
        for d in defects:
            text = (d.detail or "").lower()
            hit = {t for t in ALL_TOOLS if re.search(rf"\b{re.escape(t)}\b", text)}
            dtype = d.type.value
            add_kw = any(k in text for k in cls._ADD_KW)
            rem_kw = any(k in text for k in cls._REMOVE_KW)
            if dtype == "misordered" or any(k in text for k in cls._ORDER_KW):
                reorder = True
            for t in hit:
                if t not in present and (add_kw or dtype == "missing"):
                    add.add(t)
                elif t in present and (rem_kw or dtype in ("extraneous", "intent_mismatch")):
                    remove.add(t)
        # never remove a tool we also decided to (re)introduce
        remove -= add
        return remove, add, reorder

    def _reconcile(self, state: CaseState, sim: ToolSimulator, planned_corr, defects):
        """Apply G5's structural critique to the plan deterministically.

        Drops flagged/extraneous tools, de-duplicates (kills create_ticket
        ballooning), inserts named-missing tools into the right segment, and puts
        create_ticket last. Mutates the executed diagnostic trajectory to match.
        Returns the corrected corrective ToolCallSpec list, or None if the
        critique implies no actionable structural change (caller falls back to
        re-prompting).
        """
        diag = state.diagnostic_sequence()
        corr = [c.tool_name.value for c in planned_corr]
        corr_specs = {c.tool_name.value: c for c in planned_corr}
        remove, add, reorder = self._mentioned_tools(defects, set(diag) | set(corr))
        drop = {t for t in remove if t in diag or t in corr}

        def dedup(seq: list[str]) -> list[str]:
            seen: set[str] = set()
            out: list[str] = []
            for t in seq:
                if t in drop or t in seen:
                    continue
                seen.add(t)
                out.append(t)
            return out

        new_diag = dedup(diag)
        new_corr = dedup(corr)
        new_diag += [t for t in add if tool_segment(t) == _DIAG and t not in new_diag]
        new_corr += [t for t in add if tool_segment(t) == _CORR and t not in new_corr]
        if reorder or "create_ticket" in new_corr:
            non_ticket = [t for t in new_corr if t != "create_ticket"]
            new_corr = non_ticket + (["create_ticket"] if "create_ticket" in new_corr else [])

        if new_diag == diag and new_corr == corr:
            return None

        # reconcile the executed diagnostic trajectory to new_diag
        if new_diag != diag:
            kept: list[TrajectoryStep] = []
            used: set[str] = set()
            for s in state.trajectory:
                if s.phase_segment != _DIAG:
                    kept.append(s)
                    continue
                t = s.tool_call.tool_name.value
                if t in new_diag and t not in used:
                    used.add(t)
                    kept.append(s)
            state.trajectory = kept
            for i, s in enumerate(state.trajectory):
                s.idx = i
            for t in new_diag:
                if t not in used:
                    if not self._exec(state, sim, ToolCallSpec(tool_name=t), "G2"):
                        break
                    used.add(t)

        out = [corr_specs.get(t) or ToolCallSpec(tool_name=t) for t in new_corr]
        state.log("reconcile", {
            "drop": sorted(drop),
            "add": sorted(add),
            "new_full": state.diagnostic_sequence() + new_corr,
        })
        return out

    # --- report phase ------------------------------------------------------

    def _do_report(self, sample, state, sim, checker: bool) -> CaseState:
        while True:
            self._transition(state, Phase.REPORT)
            r6 = self._run_agent("G6", g6_report.build_user(
                state.intent_key or "unknown", state.language, state.selected_hypothesis,
                state.agent_sequence(), state.ticket_ref), state)
            if r6.parse_failed:
                if not self.config.escalation_enabled:
                    if not state.report_text:
                        state.report_text = (
                            f"Root cause: {state.selected_hypothesis or 'undetermined'}.")
                    break
                return self._escalate(state, "parse_failure_report", sim)
            state.report_text = r6.payload.report_text
            state.root_cause = r6.payload.root_cause
            state.actions = r6.payload.actions

            if not checker:
                break
            self._transition(state, Phase.REPORT_CHECK)
            r5 = self._run_agent("G5", g5_check.build_report_user(
                state.report_text, state.agent_sequence(), state.ticket_ref),
                state, extra={"full_sequence": state.agent_sequence()})
            if r5.parse_failed or r5.payload.verdict == "pass":
                break
            if state.budgets.report_rounds_used >= state.budgets.max_report_rounds:
                if not self.config.escalation_enabled:
                    break
                return self._escalate(state, "report_repair_exhausted", sim)
            state.budgets.report_rounds_used += 1
            state.repair_hint = r5.payload.repair_hint

        self._transition(state, Phase.DONE)
        state.status = CaseStatus.done
        return state

    # --- small helpers -----------------------------------------------------

    def _candidates(self) -> list[tuple[str, str]]:
        if not self.config.with_list:
            return []
        from ..vocab import INTENT_KEYS

        return [(k, intent_label_str(k)) for k in INTENT_KEYS]

    def _apply_intent(self, state: CaseState, r1: AgentResult) -> None:
        p = r1.payload
        state.intent_free_text = p.intent_free_text
        val = p.intent_label.value if hasattr(p.intent_label, "value") else str(p.intent_label)
        state.intent_key = None if val == "unknown" else val
        state.category = p.category

    @staticmethod
    def _defect_segment(defects, diag_seq: list[str]) -> str:
        from ..vocab import DIAGNOSTIC_TOOLS, tool_segment

        for d in defects:
            detail = (d.detail or "").lower()
            # a diagnostic gold tool named as missing/misordered -> diagnostic
            if any(t in detail for t in DIAGNOSTIC_TOOLS):
                return _DIAG
            # an extraneous distractor that was executed in the diagnostic phase
            if d.type.value == "extraneous" and any(
                tool_segment(t) == "distractor" and t in detail for t in diag_seq
            ):
                return _DIAG
        return _CORR
