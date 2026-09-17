"""The coordination tax: why LLM-mediated decomposition degrades structure.

Model
-----
Let a case be resolved by a pipeline of ``N`` LLM-mediated handoffs. Model each
handoff as independently injecting a *structural* defect (a dropped mandatory
tool, an inserted distractor, a swapped pair) with probability ``p``. Structural
correctness of the whole pipeline is then

    Pr[exact gold path]  =  (1 - p) ** N                                    (1)

which is strictly decreasing in ``N``. Decomposition therefore *degrades* exact
process correctness relative to a monolithic agent (``N = 1``) whenever handoffs
are LLM-mediated -- regardless of how much each specialist improves its own
sub-task. Specialisation buys semantic accuracy; it is paid for in compounding
stochasticity. We call the resulting gap the **coordination tax**

    CTX(M)  =  M_single  -  M_multi          (matched backbone and budget)     (2)

with CTX > 0 meaning decomposition *lost*.

Under contract-governed handoffs the structural component of every proposal is
projected onto the admissible set (see ``contracts.projection``). Structural
defects are absorbed at each boundary rather than propagated, so the effective
per-handoff injection rate at the *interface* is zero and (1) collapses to

    Pr[exact gold path]  =  Pr[i_hat correct]                               (3)

which is independent of ``N``. This is the paper's central prediction: an
N-sweep is *decreasing* without contracts and *flat* with them.

This module provides
  * :func:`predicted_exact`      -- equation (1)
  * :func:`coordination_tax`     -- equation (2)
  * :func:`fit_p`                -- recover p from an observed (N, score) pair
  * :class:`PipelineSimulator`   -- a controlled harness that injects defects at
    a *known* rate over the real dataset, with and without projection, so the
    compositional claim can be tested without an LLM in the loop.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..contracts import ContractSet, project
from ..contracts.runbook import Runbook
from ..schema.messages import ToolCallSpec
from ..vocab import DISTRACTOR_TOOLS, tool_segment
from . import metrics as M

_DIAG = "diagnostic"
_CORR = "corrective"


def predicted_exact(p: float, n: int) -> float:
    """Equation (1): expected exact-path rate for n LLM-mediated handoffs."""
    return (1.0 - p) ** n


def coordination_tax(m_single: float, m_multi: float) -> float:
    """Equation (2). Positive => decomposition lost against a single agent."""
    return m_single - m_multi


def fit_p(observed_exact: float, n: int) -> float:
    """Invert (1): the per-handoff defect rate implied by an observed GPC-0."""
    if n <= 0:
        return 0.0
    observed_exact = min(max(observed_exact, 1e-9), 1.0)
    return 1.0 - observed_exact ** (1.0 / n)


@dataclass
class SweepPoint:
    n_handoffs: int
    p: float
    contracts: bool
    projection_mode: str
    msc: float
    eap: float
    sas: float
    gpc0: float
    gpc1: float
    sd: float
    brs: float
    predicted_gpc0: float
    cvr: float          # contract violation rate: fraction of handoffs Pi fixed

    def row(self) -> dict[str, object]:
        return {
            "N": self.n_handoffs, "p": round(self.p, 3),
            "contracts": self.contracts, "mode": self.projection_mode,
            "MSC": round(self.msc, 4), "EAP": round(self.eap, 4),
            "SAS": round(self.sas, 4), "GPC-0": round(self.gpc0, 4),
            "GPC-1": round(self.gpc1, 4), "SD": round(self.sd, 4),
            "BRS": round(self.brs, 4),
            "GPC-0_pred": round(self.predicted_gpc0, 4),
            "CVR": round(self.cvr, 4),
        }


class PipelineSimulator:
    """Controlled N-handoff pipeline with a *known* per-handoff defect rate.

    Each handoff perturbs the running plan with probability ``p`` by one of the
    three structural defect types the benchmark actually penalises (drop a
    mandatory tool, insert a distractor or an off-runbook core tool, swap an
    adjacent pair). This isolates the compositional question -- how defects
    accumulate across handoffs and whether projection absorbs them -- from LLM
    idiosyncrasies. The perturbation model is deliberately simple and stated in
    full so the result is reproducible without a GPU.
    """

    DEFECTS = ("drop", "insert", "swap")

    def __init__(self, runbook: Runbook, *, intent_accuracy: float = 1.0,
                 alphas: tuple[float, float, float] = (0.5, 0.3, 0.2)):
        self.runbook = runbook
        self.intent_accuracy = intent_accuracy
        self.alphas = alphas
        self._off_runbook = [t for t in ("neighbor_audit", "coverage_map",
                                         "push_config", "optimize_slice")]

    def _perturb(self, seq: list[str], rng: random.Random) -> list[str]:
        kind = rng.choice(self.DEFECTS)
        s = list(seq)
        if kind == "drop" and s:
            s.pop(rng.randrange(len(s)))
        elif kind == "insert":
            pool = list(DISTRACTOR_TOOLS) + self._off_runbook
            s.insert(rng.randrange(len(s) + 1), rng.choice(pool))
        elif kind == "swap" and len(s) >= 2:
            i = rng.randrange(len(s) - 1)
            s[i], s[i + 1] = s[i + 1], s[i]
        return s

    def run(self, samples, *, n_handoffs: int, p: float, contracts: ContractSet | None,
            mode: str = "strict", seed: int = 0) -> SweepPoint:
        rng = random.Random(seed)
        keys = list(self.runbook.chains)
        by_bp: dict[str, list[tuple[list[str], list[str]]]] = {}
        n_acted = n_handoff_total = 0

        for s in samples:
            gold = list(self.runbook.chain(s.intent)) or s.gold_tools()
            pred_intent = s.intent if rng.random() < self.intent_accuracy \
                else rng.choice([k for k in keys if k != s.intent])
            plan = list(self.runbook.chain(pred_intent))

            for _ in range(n_handoffs):
                n_handoff_total += 1
                if rng.random() < p:
                    plan = self._perturb(plan, rng)
                if contracts is not None:
                    diag = [t for t in plan if tool_segment(t) == _DIAG]
                    corr = [ToolCallSpec(tool_name=t) for t in plan
                            if tool_segment(t) == _CORR]
                    d, c, rep = project(contracts=contracts, intent=pred_intent,
                                        proposed_diag=diag, proposed_corr_calls=corr,
                                        acquired=set(diag) or {"oss_query"}, mode=mode)
                    plan = list(d) + [x.tool_name.value for x in c]
                    n_acted += int(rep.acted)

            by_bp.setdefault(s.blueprint_id, []).append((plan, gold))

        mscs, eaps, sass, g0s, g1s, sds = [], [], [], [], [], []
        for pairs in by_bp.values():
            seqs = [a for a, _ in pairs]; golds = [g for _, g in pairs]
            mscs += [M.msc(a, g) for a, g in pairs]
            eaps += [M.eap(a, g) for a, g in pairs]
            sass += [M.sas(a, g) for a, g in pairs]
            g0s.append(M.gpc0(seqs, golds)); g1s.append(M.gpc1(seqs, golds))
            sds.append(M.sequence_diversity(seqs))
        mean = lambda xs: sum(xs) / len(xs) if xs else 0.0      # noqa: E731
        g0, g1, sd = mean(g0s), mean(g1s), mean(sds)

        return SweepPoint(
            n_handoffs=n_handoffs, p=p, contracts=contracts is not None,
            projection_mode=mode if contracts is not None else "none",
            msc=mean(mscs), eap=mean(eaps), sas=mean(sass),
            gpc0=g0, gpc1=g1, sd=sd, brs=M.brs(g0, g1, sd, self.alphas),
            predicted_gpc0=predicted_exact(p, n_handoffs) * self.intent_accuracy,
            cvr=(n_acted / n_handoff_total) if n_handoff_total else 0.0,
        )

    def sweep(self, samples, *, ns=(1, 2, 3, 4, 5, 6, 7), p: float = 0.15,
              contracts: ContractSet | None = None, mode: str = "strict",
              seed: int = 0) -> list[SweepPoint]:
        return [self.run(samples, n_handoffs=n, p=p, contracts=contracts,
                         mode=mode, seed=seed + n) for n in ns]
