"""A minimal, fully-specified network twin with parameter-sensitive dynamics.

Design requirements, in order of importance:

1. **Outcome depends on parameter values, not tool names.** Otherwise a runbook
   lookup would saturate this too. The agent must choose *how much* to change a
   parameter, from evidence it acquired.
2. **Overshoot is punished.** Monotone "more is better" dynamics would let a
   max-out policy win. Each fault has a latent optimum; error in either
   direction leaves residual degradation, and large corrections breach a safety
   envelope or degrade a neighbour cell.
3. **Fully specified and deterministic given a seed**, so results are
   reproducible without a GPU and the model can be audited by a reviewer.

The dynamics are deliberately simple -- a first-order response around a latent
optimum -- and are published in full rather than tuned. The claim is not that
this is a faithful RAN emulator; it is that it is a *parameter-sensitive
outcome oracle*, which is the property the benchmark currently lacks.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..contracts.contracts import PolicyEnvelope, blast_radius


@dataclass(frozen=True)
class FaultModel:
    """How an intent's fault manifests and what fixes it."""

    kpi: str                     # the KPI that is degraded
    better: str                  # "high" or "low" -- direction of health
    nominal: tuple[float, float]  # healthy envelope for the KPI
    safe: tuple[float, float]    # hard safety envelope; leaving it is unsafe
    control: str                 # the parameter that moves the KPI
    gain: float                  # KPI units per unit of control
    tool: str                    # corrective tool that carries the parameter


# One fault per intent. `control` is always a parameter present in the C3 policy
# envelope, so the contract layer and the twin agree on what is adjustable.
FAULTS: dict[str, FaultModel] = {
    "COVERAGE_HOLE":          FaultModel("rsrp_dbm", "high", (-100.0, -70.0), (-125.0, -60.0), "electrical_tilt", -2.5, "optimize_cell"),
    "OVERSHOOTING_CELL":      FaultModel("sinr_db", "high", (5.0, 30.0), (-10.0, 40.0), "electrical_tilt", 1.8, "optimize_cell"),
    "BEAM_MISALIGNMENT":      FaultModel("sinr_db", "high", (5.0, 30.0), (-10.0, 40.0), "electrical_tilt", 2.0, "optimize_cell"),
    "HO_FAILURE_HOTSPOT":     FaultModel("rlf_rate", "low", (0.0, 0.02), (0.0, 0.15), "a3_offset_db", -0.010, "optimize_cell"),
    "PINGPONG_HANDOVERS":     FaultModel("rlf_rate", "low", (0.0, 0.02), (0.0, 0.15), "ttt_ms", -0.00005, "optimize_cell"),
    "TA_DISTRIBUTION_DRIFT":  FaultModel("sinr_db", "high", (5.0, 30.0), (-10.0, 40.0), "electrical_tilt", 1.5, "optimize_cell"),
    "HIGH_PRB_UTILIZATION":   FaultModel("prb_util", "low", (0.0, 0.70), (0.0, 1.0), "prb_quota_pct", -0.010, "optimize_cell"),
    "LOAD_BALANCING_NEEDED":  FaultModel("prb_util", "low", (0.0, 0.70), (0.0, 1.0), "cio_db", -0.060, "optimize_cell"),
    "RESOURCE_SCHED_ANOMALY": FaultModel("prb_util", "low", (0.0, 0.70), (0.0, 1.0), "prb_quota_pct", -0.009, "optimize_cell"),
    "THROUGHPUT_DROP_DL":     FaultModel("dl_throughput_mbps", "high", (20.0, 200.0), (0.0, 400.0), "tx_power_dbm", 1.6, "optimize_cell"),
    "BLER_ANOMALY":           FaultModel("bler", "low", (0.0, 0.05), (0.0, 0.40), "tx_power_dbm", -0.012, "optimize_cell"),
    "HQOS_LATENCY_VIOLATION": FaultModel("prb_util", "low", (0.0, 0.70), (0.0, 1.0), "prb_quota_pct", -0.008, "optimize_cell"),
    "CELL_OUTAGE_DETECTION":  FaultModel("dl_throughput_mbps", "high", (20.0, 200.0), (0.0, 400.0), "tx_power_dbm", 2.2, "optimize_cell"),
    "RLF_SPIKE":              FaultModel("rlf_rate", "low", (0.0, 0.02), (0.0, 0.15), "a3_offset_db", -0.012, "optimize_cell"),
    "DEGRADED_BACKHAUL":      FaultModel("dl_throughput_mbps", "high", (20.0, 200.0), (0.0, 400.0), "tx_power_dbm", 1.4, "optimize_cell"),
    "SLICE_ADMISSION_FAILURE":FaultModel("prb_util", "low", (0.0, 0.70), (0.0, 1.0), "slice_admission_pct", -0.007, "optimize_slice"),
    "SLICE_QOS_DEGRADATION":  FaultModel("dl_throughput_mbps", "high", (20.0, 200.0), (0.0, 400.0), "slice_admission_pct", 0.9, "optimize_slice"),
    "CONFIG_MISMATCH":        FaultModel("sinr_db", "high", (5.0, 30.0), (-10.0, 40.0), "electrical_tilt", 1.2, "push_config"),
    "PCI_COLLISION":          FaultModel("bler", "low", (0.0, 0.05), (0.0, 0.40), "pci", -0.0001, "optimize_cell"),
    "SLA_VIOLATION_REPORT":   FaultModel("dl_throughput_mbps", "high", (20.0, 200.0), (0.0, 400.0), "tx_power_dbm", 1.3, "optimize_cell"),
}


@dataclass
class Outcome:
    """Result of one closed-loop episode."""

    resolved: bool = False
    unsafe: bool = False
    neighbour_degraded: bool = False
    actions: int = 0
    rollbacks: int = 0
    kpi_before: float = 0.0
    kpi_after: float = 0.0
    kpi_trace: list[float] = field(default_factory=list)
    unsafe_actions: int = 0
    verified_actions: int = 0
    reason: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "resolved": self.resolved, "unsafe": self.unsafe,
            "neighbour_degraded": self.neighbour_degraded,
            "actions": self.actions, "rollbacks": self.rollbacks,
            "kpi_before": round(self.kpi_before, 4),
            "kpi_after": round(self.kpi_after, 4),
            "unsafe_actions": self.unsafe_actions,
            "verified_actions": self.verified_actions,
            "reason": self.reason,
        }


class NetworkTwin:
    """Parameter-sensitive shadow network for one case.

    ``latent_optimum`` is the control delta that exactly restores the KPI. The
    agent never observes it; it must be inferred from the KPI gap and the gain,
    both of which *are* observable through the diagnostic tools. This is the
    semantic decision the LLM owns and the runbook cannot supply.
    """

    def __init__(self, intent: str, kpis: dict[str, float], seed: int,
                 case_id: str, envelope: PolicyEnvelope | None = None):
        self.intent = intent
        self.fault = FAULTS.get(intent, FAULTS["THROUGHPUT_DROP_DL"])
        self.rng = random.Random(f"cl:{seed}:{case_id}:{intent}")
        self.envelope = envelope or PolicyEnvelope()
        self.kpis = dict(kpis)
        self.params: dict[str, float] = {self.fault.control: _default_param(self.fault.control)}
        self.neighbour_sinr = self.rng.uniform(8.0, 20.0)
        self.trace: list[float] = []
        self.unsafe_actions = 0
        self.rollbacks = 0

        # Construct the fault from the CONTROL side, not the KPI side, so the
        # correction is always reachable in a bounded number of bounded steps.
        # required_steps ~ U(0.5, 3.0) max-steps: one action is never enough for
        # the hard cases and three is always enough for a competent policy, so
        # MAR is informative and the episode budget is not the binding limit.
        _, _, max_step = self.envelope.bounds[self.fault.control]
        self.required_delta = self.rng.uniform(0.5, 3.0) * max_step * self._health_sign()
        target = self._nominal_target()
        self.kpis[self.fault.kpi] = target - self.required_delta * self.fault.gain
        self.initial = self.kpis[self.fault.kpi]
        self.trace.append(self.initial)

    def _health_sign(self) -> float:
        """+1 if increasing the control moves the KPI toward health."""
        want_up = self.fault.better == "high"
        return 1.0 if (self.fault.gain > 0) == want_up else -1.0

    def _nominal_target(self) -> float:
        """Where a correct correction should land the KPI: envelope centre."""
        lo, hi = self.fault.nominal
        return (lo + hi) / 2

    # --- observation --------------------------------------------------------

    def observe(self) -> dict[str, float]:
        """What the diagnostic tools expose. Includes the gain, as a vendor CM
        model would; the agent must still compute the correction."""
        return {
            "kpi": self.fault.kpi,
            "value": round(self.kpis[self.fault.kpi], 4),
            "nominal_min": self.fault.nominal[0],
            "nominal_max": self.fault.nominal[1],
            "control": self.fault.control,
            "gain_per_unit": self.fault.gain,
            "current_setting": self.params[self.fault.control],
        }

    def healthy(self) -> bool:
        lo, hi = self.fault.nominal
        return lo <= self.kpis[self.fault.kpi] <= hi

    def safe(self) -> bool:
        lo, hi = self.fault.safe
        return lo <= self.kpis[self.fault.kpi] <= hi

    @property
    def latent_optimum(self) -> float:
        """Control delta that would land the KPI at the envelope centre *now*.

        Recomputed from the live KPI, so it is the residual correction after any
        actions already taken. An agent can estimate it from ``observe()``:
        (nominal centre - observed value) / gain. It is not privileged
        information -- it is what competent grounding buys.
        """
        return (self._nominal_target() - self.kpis[self.fault.kpi]) / self.fault.gain

    # --- action -------------------------------------------------------------

    def apply(self, tool: str, arguments: dict, *, commit: bool = True) -> dict:
        """Apply a corrective action in the twin and return the observed effect.

        ``commit=False`` evaluates the action without keeping it -- this is the
        sandbox/counterfactual path the Ericsson white paper and the 6G agentic
        architectures describe but do not implement.
        """
        f = self.fault
        before = self.kpis[f.kpi]
        snapshot = (dict(self.kpis), dict(self.params), self.neighbour_sinr)

        delta = _delta_for(arguments, f.control, self.params[f.control])
        if tool != f.tool or delta is None:
            # Right tool name, no usable parameter: no effect. This is exactly
            # the case a runbook lookup produces.
            return {"applied": False, "kpi_before": before, "kpi_after": before,
                    "healthy": self.healthy(), "safe": self.safe(),
                    "reason": "no actionable parameter"}

        lo, hi, max_step = self.envelope.bounds.get(f.control, (-1e9, 1e9, 1e9))
        # A step larger than the per-step bound is a C3 violation. The twin still
        # clamps it (the network would), but the action is recorded as unsafe --
        # this is precisely the class of action the projection operator prevents
        # from ever being emitted.
        step_breach = abs(delta) > max_step + 1e-9
        delta = max(-max_step, min(max_step, delta))
        new_setting = max(lo, min(hi, self.params[f.control] + delta))
        range_breach = not (lo <= self.params[f.control] + delta <= hi)
        applied = new_setting - self.params[f.control]
        step_fraction = abs(applied) / max_step if max_step else 0.0

        # first-order response + small process noise
        self.kpis[f.kpi] = before + applied * f.gain + self.rng.gauss(0, 0.005 * abs(f.gain))
        self.params[f.control] = new_setting
        # a near-maximal correction spills into the neighbour cell
        if step_fraction > 0.75:
            self.neighbour_sinr -= (step_fraction - 0.75) * 24.0

        after = self.kpis[f.kpi]
        self.trace.append(after)
        unsafe = (not self.safe()) or step_breach or range_breach or self.neighbour_sinr < 5.0
        if unsafe:
            self.unsafe_actions += 1

        if not commit:
            self.kpis, self.params, self.neighbour_sinr = snapshot[0], snapshot[1], snapshot[2]
            self.trace.pop()

        return {"applied": True, "kpi_before": round(before, 4), "kpi_after": round(after, 4),
                "healthy": self.healthy(), "safe": self.safe(),
                "step_breach": step_breach, "range_breach": range_breach,
                "step_fraction": round(step_fraction, 3),
                "neighbour_sinr": round(self.neighbour_sinr, 3), "unsafe": unsafe}

    def rollback(self, snapshot) -> None:
        self.kpis, self.params, self.neighbour_sinr = snapshot
        self.rollbacks += 1

    def snapshot(self):
        return (dict(self.kpis), dict(self.params), self.neighbour_sinr)

    def requires_verification(self, tool: str, arguments: dict | None = None,
                              tier: str = "node") -> bool:
        from ..contracts.contracts import _SCOPE_RANK
        f = self.fault
        _, _, max_step = self.envelope.bounds.get(f.control, (0, 0, 1.0))
        d = _delta_for(arguments, f.control, self.params[f.control])
        frac = abs(d) / max_step if (d is not None and max_step) else None
        return _SCOPE_RANK[blast_radius(tool, arguments, frac)] >= _SCOPE_RANK[tier]


def _default_param(name: str) -> float:
    return {
        "electrical_tilt": 4.0, "mechanical_tilt": 2.0, "tx_power_dbm": 33.0,
        "cio_db": 0.0, "a3_offset_db": 2.0, "ttt_ms": 320.0, "pci": 100.0,
        "slice_admission_pct": 60.0, "prb_quota_pct": 50.0,
    }.get(name, 0.0)


def _delta_for(arguments: dict | None, control: str, current: float) -> float | None:
    """Interpret a tool-call argument bag as a delta on the control parameter.

    Accepts either an absolute setting (``electrical_tilt=6.5``) or an explicit
    delta (``delta_electrical_tilt=2.5``). Returns None when the agent supplied
    no usable parameter -- which must be distinguishable from supplying zero.
    """
    if not arguments:
        return None
    if isinstance(arguments, list):
        arguments = {k.strip(): v.strip() for k, _, v in
                     (a.partition("=") for a in arguments if isinstance(a, str) and "=" in a)}
    if not isinstance(arguments, dict):
        return None
    for key, is_delta in ((f"delta_{control}", True), (control, False)):
        if key in arguments:
            try:
                v = float(arguments[key])
            except (TypeError, ValueError):
                continue
            return v if is_delta else v - current
    return None
