"""LB: the non-agentic ``classify -> runbook lookup`` control.

The single most important control missing from the telecom-agent literature. It
emits ``Runbook[predicted_intent]`` verbatim and does no reasoning, no tool
selection and no planning. Because the emitted chain is a deterministic function
of one categorical variable, the control is:

  * exact on every structural metric when the intent is right;
  * perfectly repeatable, so SD = 0 and the (1-SD) term of BRS is maximal;
  * free of extra tools, so EAP ~ 1 by construction.

Any architecture claiming a process-level result on TelcoAgent-Bench must be
compared against this curve, parameterised by intent accuracy. We report it at
the accuracy the system under test actually achieves, and as a sweep.

Two confusion models are provided. ``uniform`` draws a wrong intent uniformly and
is deliberately *conservative* (it ignores that real confusions concentrate
inside collision classes, where the emitted chain is unchanged and the error is
free). ``collision_aware`` draws wrong intents preferentially from the true
intent's own collision class and category, and is the realistic upper bound.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..contracts.runbook import Runbook
from ..eval import metrics as M
from ..vocab import intent_category


@dataclass
class LookupResult:
    intent_accuracy: float
    confusion: str
    n: int
    msc: float
    eap: float
    sas: float
    gpc0: float
    gpc1: float
    sd: float
    brs: float

    def row(self) -> dict[str, object]:
        return {
            "intent_accuracy": round(self.intent_accuracy, 3),
            "confusion": self.confusion,
            "n": self.n,
            "MSC": round(self.msc, 4), "EAP": round(self.eap, 4),
            "SAS": round(self.sas, 4), "GPC-0": round(self.gpc0, 4),
            "GPC-1": round(self.gpc1, 4), "SD": round(self.sd, 4),
            "BRS": round(self.brs, 4),
        }


@dataclass
class LookupBaseline:
    runbook: Runbook
    alphas: tuple[float, float, float] = (0.5, 0.3, 0.2)
    confusion: str = "uniform"          # uniform | collision_aware
    _classes: dict[str, list[str]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        for cls in self.runbook.collision_classes().classes:
            for i in cls:
                self._classes[i] = list(cls)

    def _wrong_intent(self, true_intent: str, rng: random.Random) -> str:
        keys = [k for k in self.runbook.chains if k != true_intent]
        if not keys:
            return true_intent
        if self.confusion == "collision_aware":
            # 50% inside the collision class (free), 30% same category, 20% far
            same_chain = [k for k in self._classes.get(true_intent, []) if k != true_intent]
            cat = intent_category(true_intent)
            same_cat = [k for k in keys
                        if intent_category(k) == cat and k not in same_chain]
            u = rng.random()
            if same_chain and u < 0.5:
                return rng.choice(same_chain)
            if same_cat and u < 0.8:
                return rng.choice(same_cat)
        return rng.choice(keys)

    def evaluate(self, samples, intent_accuracy: float, *, seed: int = 0) -> LookupResult:
        """Score the control on the given samples at a stipulated intent accuracy.

        Metrics reuse the benchmark's own equations (eval.metrics), so the
        numbers are directly comparable to the published tables.
        """
        rng = random.Random(seed)
        by_bp: dict[str, list[tuple[list[str], list[str]]]] = {}
        for s in samples:
            gold = list(self.runbook.chain(s.intent)) or s.gold_tools()
            pred_intent = s.intent if rng.random() < intent_accuracy \
                else self._wrong_intent(s.intent, rng)
            emitted = list(self.runbook.chain(pred_intent))
            by_bp.setdefault(s.blueprint_id, []).append((emitted, gold))

        mscs, eaps, sass, g0s, g1s, sds = [], [], [], [], [], []
        for pairs in by_bp.values():
            seqs = [p for p, _ in pairs]
            golds = [g for _, g in pairs]
            mscs += [M.msc(p, g) for p, g in pairs]
            eaps += [M.eap(p, g) for p, g in pairs]
            sass += [M.sas(p, g) for p, g in pairs]
            g0s.append(M.gpc0(seqs, golds))
            g1s.append(M.gpc1(seqs, golds))
            sds.append(M.sequence_diversity(seqs))

        mean = lambda xs: sum(xs) / len(xs) if xs else 0.0     # noqa: E731
        g0, g1, sd = mean(g0s), mean(g1s), mean(sds)
        return LookupResult(
            intent_accuracy=intent_accuracy, confusion=self.confusion, n=len(samples),
            msc=mean(mscs), eap=mean(eaps), sas=mean(sass),
            gpc0=g0, gpc1=g1, sd=sd, brs=M.brs(g0, g1, sd, self.alphas),
        )


def lookup_sweep(
    samples,
    runbook: Runbook,
    *,
    accuracies=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0),
    confusion: str = "uniform",
    seed: int = 0,
    alphas: tuple[float, float, float] = (0.5, 0.3, 0.2),
) -> list[LookupResult]:
    lb = LookupBaseline(runbook=runbook, confusion=confusion, alphas=alphas)
    return [lb.evaluate(samples, a, seed=seed + i) for i, a in enumerate(accuracies)]


def implied_intent_accuracy(
    samples, runbook: Runbook, target_metric: str, target_value: float,
    *, confusion: str = "uniform", seed: int = 0, tol: float = 1e-3,
) -> float:
    """Invert the LB curve: what intent accuracy does an observed score imply?

    Reporting "system X achieves SAS 0.80, which the lookup control reaches at
    intent accuracy 0.62" is a far more informative statement than the raw score,
    because it prices the result in the only currency the structural metrics
    actually measure.
    """
    lb = LookupBaseline(runbook=runbook, confusion=confusion)
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        val = getattr(lb.evaluate(samples, mid, seed=seed), target_metric)
        if abs(val - target_value) < tol:
            return mid
        if val < target_value:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
