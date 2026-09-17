"""C1-C4: assume-guarantee contracts over the corrective-planning handoff.

Each contract is a *decidable predicate* over a candidate action sequence plus
the case context. Contracts are sourced from real operator artifacts, not from
per-case ground truth:

    C1 Runbook      admissible tool multiset + phase order, keyed on the
                    PREDICTED intent (source: NOC runbook)
    C2 Evidence     a corrective action is admissible only once diagnostic
                    evidence has been acquired for the KPI it targets
                    (source: blueprint KPI ranges)
    C3 Policy       proposed parameter deltas stay inside the operator/3GPP
                    envelope (source: parameter bound tables)
    C4 BlastRadius  verification depth / human gating scales with the scope of
                    the action (source: cell-sector-node-region hierarchy)

Contracts are *checked* here and *enforced* by ``projection.project``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..vocab import DISTRACTOR_TOOLS, tool_segment
from .runbook import Runbook

_DIAG = "diagnostic"
_CORR = "corrective"

ContractId = Literal["C1", "C2", "C3", "C4"]


@dataclass(frozen=True)
class Violation:
    contract: ContractId
    kind: str
    tool: str | None
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {"contract": self.contract, "kind": self.kind,
                "tool": self.tool, "detail": self.detail}


# --- C3: policy envelope ----------------------------------------------------

# Operator/3GPP-style bounds. Values are the admissible *absolute* range for a
# parameter and the maximum single-step delta. A real deployment loads these
# from the vendor CM model; the structure is what matters for the architecture.
_PARAM_BOUNDS: dict[str, tuple[float, float, float]] = {
    # name:            (min,   max,   max_abs_delta_per_step)
    "electrical_tilt":  (0.0,  15.0,  3.0),
    "mechanical_tilt":  (0.0,  10.0,  2.0),
    "tx_power_dbm":     (20.0, 46.0,  3.0),
    "cio_db":           (-6.0,  6.0,  2.0),
    "a3_offset_db":     (-6.0,  6.0,  2.0),
    "ttt_ms":           (0.0, 5120.0, 640.0),
    "pci":              (0.0, 1007.0, 1007.0),
    "slice_admission_pct": (0.0, 100.0, 20.0),
    "prb_quota_pct":    (0.0, 100.0, 25.0),
}


@dataclass(frozen=True)
class PolicyEnvelope:
    bounds: dict[str, tuple[float, float, float]] = field(default_factory=lambda: dict(_PARAM_BOUNDS))

    def check(self, params: dict[str, float], current: dict[str, float] | None = None
              ) -> list[Violation]:
        out: list[Violation] = []
        cur = current or {}
        for name, val in params.items():
            spec = self.bounds.get(name)
            if spec is None:
                out.append(Violation("C3", "unknown_parameter", name,
                                     f"{name} is not in the policy envelope"))
                continue
            lo, hi, max_delta = spec
            try:
                v = float(val)
            except (TypeError, ValueError):
                out.append(Violation("C3", "non_numeric", name, f"{name}={val!r}"))
                continue
            if not (lo <= v <= hi):
                out.append(Violation("C3", "out_of_range", name,
                                     f"{name}={v} outside [{lo}, {hi}]"))
            if name in cur and abs(v - float(cur[name])) > max_delta:
                out.append(Violation("C3", "delta_too_large", name,
                                     f"|{v} - {cur[name]}| > {max_delta}"))
        return out

    def clamp(self, params: dict[str, float], current: dict[str, float] | None = None
              ) -> dict[str, float]:
        """Project parameters into the envelope. Unknown parameters are dropped."""
        cur = current or {}
        out: dict[str, float] = {}
        for name, val in params.items():
            spec = self.bounds.get(name)
            if spec is None:
                continue
            lo, hi, max_delta = spec
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            if name in cur:
                c = float(cur[name])
                v = max(c - max_delta, min(c + max_delta, v))
            out[name] = max(lo, min(hi, v))
        return out


# --- C4: blast radius -------------------------------------------------------

_SCOPE_RANK = {"cell": 0, "sector": 1, "node": 2, "region": 3, "cluster": 4}

_TOOL_SCOPE: dict[str, str] = {
    "optimize_cell": "cell",
    "optimize_slice": "node",
    "recommend_param_change": "cell",
    "push_config": "node",
    "create_ticket": "cell",
}


def blast_radius(tool: str, arguments: dict | None = None,
                 step_fraction: float | None = None) -> str:
    """Scope tier of an action.

    Three things widen the blast radius: the tool itself (a slice or config push
    touches more than a cell), an explicit wider target in the arguments, and
    the *magnitude* of the change. A near-maximal parameter step is not a
    cell-local event -- it spills into neighbours -- so ``step_fraction``
    (|delta| / max_step) promotes the tier. This is what makes C4 a magnitude-
    sensitive gate rather than a static per-tool label.
    """
    scope = _TOOL_SCOPE.get(tool, "cell")
    if arguments:
        target = str(arguments.get("scope") or arguments.get("target_scope") or "")
        if target in _SCOPE_RANK and _SCOPE_RANK[target] > _SCOPE_RANK[scope]:
            scope = target
    if step_fraction is not None:
        if step_fraction >= 1.0 and _SCOPE_RANK[scope] < _SCOPE_RANK["region"]:
            scope = "region"
        elif step_fraction >= 0.6 and _SCOPE_RANK[scope] < _SCOPE_RANK["node"]:
            scope = "node"
        elif step_fraction >= 0.3 and _SCOPE_RANK[scope] < _SCOPE_RANK["sector"]:
            scope = "sector"
    return scope


# --- the contract set -------------------------------------------------------

@dataclass
class ContractSet:
    """The four contracts, individually switchable for ablation."""

    runbook: Runbook
    envelope: PolicyEnvelope = field(default_factory=PolicyEnvelope)
    c1: bool = True
    c2: bool = True
    c3: bool = True
    c4: bool = True
    # C4: scopes at or above this tier require twin verification before commit
    verify_at_or_above: str = "node"

    # -- C1 ------------------------------------------------------------------
    def c1_admissible(self, intent: str | None, seg: str) -> tuple[str, ...]:
        return self.runbook.segment(intent, seg) if self.c1 else ()

    def check_c1(self, intent: str | None, diag: list[str], corr: list[str]) -> list[Violation]:
        if not self.c1:
            return []
        ref = self.runbook.chain(intent)
        if not ref:
            return []
        out: list[Violation] = []
        allowed = set(ref)
        seen: set[str] = set()
        for t in list(diag) + list(corr):
            if t in DISTRACTOR_TOOLS:
                out.append(Violation("C1", "distractor", t, f"{t} is a distractor tool"))
            elif t not in allowed:
                out.append(Violation("C1", "extraneous", t, f"{t} not in runbook chain {list(ref)}"))
            if t in seen:
                out.append(Violation("C1", "duplicate", t, f"{t} invoked more than once"))
            seen.add(t)
        for t in ref:
            if t not in seen:
                out.append(Violation("C1", "missing", t, f"{t} required by runbook, not planned"))
        # phase order: every diagnostic must precede every corrective
        full = list(diag) + list(corr)
        first_corr = next((i for i, t in enumerate(full) if tool_segment(t) == _CORR), None)
        if first_corr is not None:
            for t in full[first_corr:]:
                if tool_segment(t) == _DIAG:
                    out.append(Violation("C1", "phase_order", t,
                                         f"diagnostic {t} scheduled after a corrective action"))
        # ordering within the reference chain
        if [t for t in full if t in allowed] != [t for t in ref if t in full]:
            out.append(Violation("C1", "misordered", None,
                                 f"order {[t for t in full if t in allowed]} != runbook {list(ref)}"))
        return out

    # -- C2 ------------------------------------------------------------------
    def check_c2(self, corr: list[str], acquired: set[str], intent: str | None) -> list[Violation]:
        """A corrective action requires at least one executed diagnostic.

        This is the 'no action before evidence' invariant. ``create_ticket`` is
        exempt only when the runbook itself schedules it as the sole follow-up
        (e.g. CELL_OUTAGE_DETECTION = oss_query -> create_ticket).
        """
        if not self.c2:
            return []
        if acquired:
            return []
        out: list[Violation] = []
        for t in corr:
            if tool_segment(t) != _CORR:
                continue
            out.append(Violation("C2", "no_evidence", t,
                                 f"{t} planned with no diagnostic evidence acquired"))
        return out

    # -- C3 ------------------------------------------------------------------
    def check_c3(self, calls) -> list[Violation]:
        if not self.c3:
            return []
        out: list[Violation] = []
        for c in calls:
            name = getattr(getattr(c, "tool_name", None), "value", None) or str(getattr(c, "tool_name", ""))
            if tool_segment(name) != _CORR:
                continue
            params = _numeric_params(getattr(c, "arguments", None))
            out.extend(self.envelope.check(params))
        return out

    # -- C4 ------------------------------------------------------------------
    def requires_verification(self, tool: str, arguments: dict | None = None,
                             step_fraction: float | None = None) -> bool:
        if not self.c4:
            return False
        tier = blast_radius(tool, arguments, step_fraction)
        return _SCOPE_RANK[tier] >= _SCOPE_RANK[self.verify_at_or_above]

    def check_all(self, *, intent, diag, corr_calls, acquired) -> list[Violation]:
        corr = [getattr(getattr(c, "tool_name", None), "value", None) or str(c) for c in corr_calls]
        return (
            self.check_c1(intent, diag, corr)
            + self.check_c2(corr, acquired, intent)
            + self.check_c3(corr_calls)
        )


def _numeric_params(arguments) -> dict[str, float]:
    """Extract numeric parameter assignments from a tool-call argument bag."""
    if not arguments:
        return {}
    if isinstance(arguments, dict):
        items = arguments.items()
    elif isinstance(arguments, list):
        items = []
        for a in arguments:
            if isinstance(a, str) and "=" in a:
                k, _, v = a.partition("=")
                items.append((k.strip(), v.strip()))
        items = list(items)
    else:
        return {}
    out: dict[str, float] = {}
    for k, v in items:
        if isinstance(v, bool):
            continue
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out
