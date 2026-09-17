"""Closed-loop metrics: CLRR, MAR, UAR, RBR, AC.

These are the metrics a runbook lookup cannot score on, because they depend on
the *parameters* of an action and on what the network did in response.
"""

from __future__ import annotations

from dataclasses import dataclass

from .twin import Outcome


@dataclass
class ClosedLoopMetrics:
    n: int = 0
    clrr: float = 0.0      # Closed-Loop Resolution Rate
    mar: float = 0.0       # Mean Actions to Resolution (resolved cases only)
    uar: float = 0.0       # Unsafe Action Rate (unsafe actions / all actions)
    case_uar: float = 0.0  # fraction of cases with >=1 unsafe action
    rbr: float = 0.0       # RollBack Rate (rollbacks / action)
    ndr: float = 0.0       # Neighbour Degradation Rate
    ac: float = 0.0        # Auditability Coverage
    kpi_gain: float = 0.0  # mean normalised KPI improvement

    def row(self) -> dict[str, object]:
        return {"n": self.n, "CLRR": round(self.clrr, 4), "MAR": round(self.mar, 3),
                "UAR": round(self.uar, 4), "case-UAR": round(self.case_uar, 4),
                "RBR": round(self.rbr, 4), "NDR": round(self.ndr, 4),
                "AC": round(self.ac, 4), "KPI-gain": round(self.kpi_gain, 4)}


def aggregate_cl(outcomes: list[Outcome], *, certified: list[bool] | None = None
                 ) -> ClosedLoopMetrics:
    if not outcomes:
        return ClosedLoopMetrics()
    n = len(outcomes)
    resolved = [o for o in outcomes if o.resolved]
    total_actions = sum(o.actions for o in outcomes) or 1
    cert = certified or [False] * n
    return ClosedLoopMetrics(
        n=n,
        clrr=len(resolved) / n,
        mar=(sum(o.actions for o in resolved) / len(resolved)) if resolved else float("nan"),
        uar=sum(o.unsafe_actions for o in outcomes) / total_actions,
        case_uar=sum(1 for o in outcomes if o.unsafe_actions) / n,
        rbr=sum(o.rollbacks for o in outcomes) / total_actions,
        ndr=sum(1 for o in outcomes if o.neighbour_degraded) / n,
        ac=sum(1 for c in cert if c) / n,
        kpi_gain=sum(_norm_gain(o) for o in outcomes) / n,
    )


def _norm_gain(o: Outcome) -> float:
    """Normalised movement of the target KPI toward health, clipped to [0, 1]."""
    if not o.kpi_trace:
        return 0.0
    span = abs(o.kpi_before) + 1e-9
    return max(0.0, min(1.0, abs(o.kpi_after - o.kpi_before) / span))
