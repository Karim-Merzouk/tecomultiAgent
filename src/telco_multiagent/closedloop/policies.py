"""Reference closed-loop policies, in increasing order of competence.

These are the controls the paper reports alongside any agentic system. Each
consumes the same observation interface, so the comparison is exact.

``RunbookLookupPolicy``   emits the runbook chain with NO parameters. Scores 1.0
                          on every structural metric of TelcoAgent-Metrics and
                          CLRR = 0 here. This single pair of numbers is the
                          argument for the closed-loop extension.
``FixedStepPolicy``       always applies the maximum permitted step in the
                          helpful direction. Tests whether monotone dynamics
                          would let a trivial policy win (they must not).
``ProportionalPolicy``    computes the correction from the observed KPI gap and
                          the published gain -- the competent reference an agent
                          should match. Upper bound for a well-grounded agent.
``NoisyProportionalPolicy`` proportional with multiplicative error, used to
                          characterise how CLRR degrades with grounding quality.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .twin import NetworkTwin, Outcome


@dataclass
class _Base:
    max_actions: int = 4
    verify_before_commit: bool = True
    tier: str = "node"

    def parameters(self, twin: NetworkTwin, obs: dict) -> dict | None:
        raise NotImplementedError

    def run(self, twin: NetworkTwin) -> Outcome:
        out = Outcome(kpi_before=twin.initial)
        for _ in range(self.max_actions):
            if twin.healthy():
                break
            obs = twin.observe()
            params = self.parameters(twin, obs)
            if params is None:
                out.reason = "no parameters proposed"
                break
            tool = twin.fault.tool
            out.actions += 1

            snap = twin.snapshot()
            if self.verify_before_commit and twin.requires_verification(tool, params, self.tier):
                trial = twin.apply(tool, params, commit=False)
                out.verified_actions += 1
                if trial.get("unsafe"):
                    out.rollbacks += 1
                    out.unsafe_actions += 1
                    continue                       # rejected before it reached the network
            res = twin.apply(tool, params, commit=True)
            if res.get("unsafe"):
                out.unsafe_actions += 1
                twin.rollback(snap)
                out.rollbacks += 1
        out.resolved = twin.healthy()
        out.unsafe = out.unsafe_actions > 0
        out.neighbour_degraded = twin.neighbour_sinr < 5.0
        out.kpi_after = twin.kpis[twin.fault.kpi]
        out.kpi_trace = list(twin.trace)
        return out


def _clip_step(twin, ctrl: str, delta: float, *, respect_bound: bool = True) -> float:
    """Clip a proposed delta to the per-step policy bound.

    ``respect_bound=False`` models an agent that ignores C3 -- the behaviour the
    projection operator is there to prevent. Comparing the two is the C3
    ablation.
    """
    _, _, max_step = twin.envelope.bounds[ctrl]
    return max(-max_step, min(max_step, delta)) if respect_bound else delta


class RunbookLookupPolicy(_Base):
    """Correct tool names, no parameters: the structural-metric champion."""

    def parameters(self, twin, obs):
        return {}


@dataclass
class FixedStepPolicy(_Base):
    """Always the maximum permitted step toward health. Tests that the twin is
    not monotone-exploitable: it overshoots, so it must not beat a grounded
    policy."""

    def parameters(self, twin, obs):
        ctrl = obs["control"]
        _, _, max_step = twin.envelope.bounds[ctrl]
        gap = obs["nominal_max"] - obs["value"] if twin.fault.better == "high" \
            else obs["value"] - obs["nominal_min"]
        sign = 1.0 if (gap > 0) == (obs["gain_per_unit"] > 0) else -1.0
        return {f"delta_{ctrl}": sign * max_step}


@dataclass
class ProportionalPolicy(_Base):
    """Computes the correction from the observed KPI gap and the published gain,
    then respects the per-step bound. The competent reference."""

    damping: float = 1.0
    respect_bound: bool = True

    def parameters(self, twin, obs):
        ctrl = obs["control"]
        want = ((obs["nominal_min"] + obs["nominal_max"]) / 2 - obs["value"]) \
            / obs["gain_per_unit"]
        return {f"delta_{ctrl}": _clip_step(twin, ctrl, want * self.damping,
                                            respect_bound=self.respect_bound)}


@dataclass
class NoisyProportionalPolicy(_Base):
    """Proportional with multiplicative grounding error. Characterises how CLRR
    degrades as evidence interpretation degrades -- the axis on which an agent
    can actually differ from a lookup table."""

    sigma: float = 0.4
    seed: int = 0
    respect_bound: bool = True

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    def parameters(self, twin, obs):
        ctrl = obs["control"]
        want = ((obs["nominal_min"] + obs["nominal_max"]) / 2 - obs["value"]) \
            / obs["gain_per_unit"]
        return {f"delta_{ctrl}": _clip_step(twin, ctrl, want * self._rng.gauss(1.0, self.sigma),
                                            respect_bound=self.respect_bound)}
