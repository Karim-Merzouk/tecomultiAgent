"""The projection operator  Pi : Sigma* -> C(i_hat).

Pi is the enforcement half of the contract layer. An agent *proposes* an action
sequence; Pi *disposes* -- mapping any proposal onto the admissible set before it
reaches the next agent or the network. The LLM is retained on the proposal path
(which intent, which cell, which parameter values) and removed from the
structural decision path (which tools, in which order).

Two modes, which form the paper's central ablation axis:

``min``     -- *subtractive*. Removes contract violations (distractors,
               extraneous tools, duplicates, phase-order inversions, order
               inversions, out-of-envelope parameters) but never inserts a step
               the proposal omitted. Injects no domain knowledge; isolates the
               value of defect removal alone.

``strict``  -- *subtractive + completive*. Additionally inserts runbook steps the
               proposal omitted. Structurally equivalent to emitting the runbook
               chain for the predicted intent, with argument bags carried over
               from the proposal wherever the proposal supplied them.

Properties (see tests/test_projection_properties.py, which checks all four
exhaustively over the generated proposal space):

    P1 Soundness        Pi(a) satisfies every active structural contract clause.
    P2 Idempotence      Pi(Pi(a)) == Pi(a).
    P3 Boundedness      |Pi(a)| <= |R(i_hat)|  (strict: equality when R is known).
    P4 Termination      because of P2, a propose-check-project loop converges in
                        at most one repair round; it cannot cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..vocab import DISTRACTOR_TOOLS, tool_segment
from .contracts import ContractSet, Violation, _numeric_params

_DIAG = "diagnostic"
_CORR = "corrective"

Mode = Literal["min", "strict"]


@dataclass
class ProjectionReport:
    """What Pi changed. Feeds both the audit certificate and the CVR metric."""

    mode: Mode = "strict"
    intent: str | None = None
    proposed_diag: tuple[str, ...] = ()
    proposed_corr: tuple[str, ...] = ()
    projected_diag: tuple[str, ...] = ()
    projected_corr: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    inserted: tuple[str, ...] = ()
    reordered: bool = False
    clamped_params: tuple[str, ...] = ()
    violations: tuple[Violation, ...] = ()

    @property
    def acted(self) -> bool:
        """True iff the raw proposal violated a contract (drives CVR)."""
        return bool(self.removed or self.inserted or self.reordered or self.clamped_params)

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "intent": self.intent,
            "proposed": {"diagnostic": list(self.proposed_diag),
                         "corrective": list(self.proposed_corr)},
            "projected": {"diagnostic": list(self.projected_diag),
                          "corrective": list(self.projected_corr)},
            "removed": list(self.removed),
            "inserted": list(self.inserted),
            "reordered": self.reordered,
            "clamped_params": list(self.clamped_params),
            "violations": [v.as_dict() for v in self.violations],
            "acted": self.acted,
        }


def _dedup(seq) -> list:
    seen: set = set()
    out: list = []
    for t in seq:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def project(
    *,
    contracts: ContractSet,
    intent: str | None,
    proposed_diag: list[str],
    proposed_corr_calls: list,
    acquired: set[str] | None = None,
    mode: Mode = "strict",
    current_params: dict[str, float] | None = None,
) -> tuple[list[str], list, ProjectionReport]:
    """Project a proposal onto the admissible set.

    Returns ``(projected_diagnostic_tools, projected_corrective_calls, report)``.
    ``proposed_corr_calls`` carries argument bags; those are preserved for every
    tool that survives projection, and clamped into the C3 envelope.
    """
    acquired = acquired or set()
    corr_names = [
        getattr(getattr(c, "tool_name", None), "value", None) or str(getattr(c, "tool_name", c))
        for c in proposed_corr_calls
    ]
    call_by_name = {
        n: c for n, c in zip(corr_names, proposed_corr_calls, strict=False)
    }

    violations = contracts.check_all(
        intent=intent, diag=proposed_diag, corr_calls=proposed_corr_calls, acquired=acquired
    )

    ref = contracts.runbook.chain(intent) if contracts.c1 else ()
    ref_diag = [t for t in ref if tool_segment(t) == _DIAG]
    ref_corr = [t for t in ref if tool_segment(t) == _CORR]

    # ---- subtractive stage (always applied) --------------------------------
    def keep(t: str) -> bool:
        if t in DISTRACTOR_TOOLS:
            return False
        if ref and t not in ref:          # extraneous wrt the runbook
            return False
        return True

    kept_diag = [t for t in _dedup(proposed_diag) if keep(t)]
    kept_corr = [t for t in _dedup(corr_names) if keep(t)]
    removed = [t for t in _dedup(list(proposed_diag) + corr_names)
               if t not in kept_diag and t not in kept_corr]

    # ---- completive stage (strict only) ------------------------------------
    inserted: list[str] = []
    if mode == "strict" and ref:
        for t in ref_diag:
            if t not in kept_diag:
                kept_diag.append(t)
                inserted.append(t)
        for t in ref_corr:
            if t not in kept_corr:
                kept_corr.append(t)
                inserted.append(t)

    # ---- ordering stage ----------------------------------------------------
    # Canonical order is the runbook order; tools outside the runbook (only
    # possible when the runbook is unknown) keep their proposed relative order.
    def canonical(seq: list[str], reference: list[str]) -> list[str]:
        if not reference:
            return seq
        rank = {t: i for i, t in enumerate(reference)}
        return sorted(seq, key=lambda t: (rank.get(t, len(rank)), seq.index(t)))

    ordered_diag = canonical(kept_diag, ref_diag)
    ordered_corr = canonical(kept_corr, ref_corr)
    reordered = (ordered_diag != kept_diag) or (ordered_corr != kept_corr)

    # phase order is structural: diagnostics are a separate list from
    # correctives, so "diagnostic after corrective" is unrepresentable by
    # construction once projected.

    # ---- C3 parameter clamping --------------------------------------------
    clamped: list[str] = []
    out_calls = []
    for t in ordered_corr:
        call = call_by_name.get(t)
        if call is None:
            out_calls.append(_bare_call(t))
            continue
        params = _numeric_params(getattr(call, "arguments", None))
        if contracts.c3 and params:
            fixed = contracts.envelope.clamp(params, current_params)
            changed = [k for k in params
                       if k not in fixed or float(fixed[k]) != float(params[k])]
            if changed:
                clamped.extend(changed)
                call = _with_params(call, fixed)
        out_calls.append(call)

    report = ProjectionReport(
        mode=mode,
        intent=intent,
        proposed_diag=tuple(proposed_diag),
        proposed_corr=tuple(corr_names),
        projected_diag=tuple(ordered_diag),
        projected_corr=tuple(ordered_corr),
        removed=tuple(removed),
        inserted=tuple(inserted),
        reordered=reordered,
        clamped_params=tuple(sorted(set(clamped))),
        violations=tuple(violations),
    )
    return ordered_diag, out_calls, report


# --- call helpers -----------------------------------------------------------
# Kept local so the operator does not hard-depend on the message schema; any
# object exposing ``tool_name`` / ``arguments`` works, which keeps Pi portable
# across orchestration frameworks (the portability claim in the paper).

def _bare_call(tool: str):
    from ..schema.messages import ToolCallSpec
    return ToolCallSpec(tool_name=tool)


def _with_params(call, params: dict[str, float]):
    args = getattr(call, "arguments", None)
    if isinstance(args, dict):
        new = dict(args)
        new.update(params)
    elif isinstance(args, list):
        keys = set(params)
        new = [a for a in args
               if not (isinstance(a, str) and a.partition("=")[0].strip() in keys)]
        new += [f"{k}={v}" for k, v in params.items()]
    else:
        new = params
    try:
        return call.model_copy(update={"arguments": new})
    except AttributeError:
        from ..schema.messages import ToolCallSpec
        name = getattr(getattr(call, "tool_name", None), "value", None) or str(call)
        return ToolCallSpec(tool_name=name, arguments=new)
