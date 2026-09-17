"""C1 source artifact: the per-intent operator runbook.

A runbook maps an *intent* to the canonical ordered tool chain an operator would
execute for it. This is the artifact a real NOC owns; it is NOT per-ticket ground
truth. The distinction matters for experimental validity: a system that consults
``Runbook[predicted_intent]`` uses only information available at inference time,
whereas one that consults ``sample.gold_tools()`` is handed the ground-truth
intent label whenever the intent classifier is wrong.

The runbook is *derived once, offline* from the training split and frozen. We
verify the derivation is well-posed (one chain per intent) rather than assuming
it -- see :func:`derive_runbook` and :meth:`Runbook.integrity`.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ..vocab import tool_segment

_DIAG = "diagnostic"
_CORR = "corrective"


@dataclass(frozen=True)
class CollisionClasses:
    """Intents whose runbook chains are identical.

    A confusion *inside* a collision class is free under every structural metric
    of TelcoAgent-Metrics (MSC, EAP, SAS, GPC-0, GPC-1, SD, BRS), because the
    emitted chain is byte-identical. Quantifying this is a prerequisite for
    interpreting any process-level score on this benchmark.
    """

    classes: tuple[tuple[str, ...], ...]

    @property
    def n_intents(self) -> int:
        return sum(len(c) for c in self.classes)

    @property
    def n_chains(self) -> int:
        return len(self.classes)

    @property
    def free_pair_fraction(self) -> float:
        """Fraction of unordered intent pairs that cost zero to confuse."""
        n = self.n_intents
        total = n * (n - 1) / 2
        same = sum(len(c) * (len(c) - 1) / 2 for c in self.classes)
        return same / total if total else 0.0

    def largest(self) -> tuple[str, ...]:
        return max(self.classes, key=len) if self.classes else ()


@dataclass
class Runbook:
    """Frozen intent -> canonical chain map."""

    chains: dict[str, tuple[str, ...]] = field(default_factory=dict)
    provenance: str = "derived"

    def __contains__(self, intent: str) -> bool:
        return intent in self.chains

    def chain(self, intent: str | None) -> tuple[str, ...]:
        """Canonical chain for an intent; empty tuple for unknown/None.

        An empty chain is the correct behaviour for an unrecognised intent: the
        contract set then imposes no C1 constraint and the system degrades to
        unguarded (LLM-only) planning rather than silently guessing a chain.
        """
        if intent is None:
            return ()
        return self.chains.get(intent, ())

    def segment(self, intent: str | None, seg: str) -> tuple[str, ...]:
        return tuple(t for t in self.chain(intent) if tool_segment(t) == seg)

    def collision_classes(self) -> CollisionClasses:
        inv: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for k, v in self.chains.items():
            inv[v].append(k)
        return CollisionClasses(tuple(tuple(sorted(v)) for v in inv.values()))

    def integrity(self) -> dict[str, object]:
        """Structural facts a reviewer can re-derive. Reported, not assumed."""
        cc = self.collision_classes()
        return {
            "n_intents": len(self.chains),
            "n_distinct_chains": cc.n_chains,
            "chain_lengths": dict(sorted(Counter(len(c) for c in self.chains.values()).items())),
            "free_pair_fraction": round(cc.free_pair_fraction, 4),
            "largest_collision_class": list(cc.largest()),
            "provenance": self.provenance,
        }

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {"provenance": self.provenance,
                 "chains": {k: list(v) for k, v in sorted(self.chains.items())}},
                indent=2,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def from_json(path: str | Path) -> Runbook:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return Runbook(
            chains={k: tuple(v) for k, v in raw["chains"].items()},
            provenance=raw.get("provenance", "file"),
        )


class AmbiguousRunbook(ValueError):
    """Raised when an intent does not admit a single canonical chain."""


def derive_runbook(samples, *, strict: bool = True) -> Runbook:
    """Derive the runbook from a set of samples (the *training* split).

    ``strict`` raises if any intent exhibits more than one distinct chain. With
    ``strict=False`` the modal chain is taken and the ambiguity is recorded in
    the provenance string -- which is the behaviour a real deployment needs, and
    which keeps the operator honest about how well-posed the artifact is.
    """
    obs: dict[str, Counter] = defaultdict(Counter)
    for s in samples:
        obs[s.intent][tuple(s.gold_tools())] += 1

    ambiguous = {k: len(v) for k, v in obs.items() if len(v) > 1}
    if ambiguous and strict:
        raise AmbiguousRunbook(f"intents with >1 chain: {ambiguous}")

    chains = {k: v.most_common(1)[0][0] for k, v in obs.items()}
    prov = f"derived n={len(samples)}" + (f" ambiguous={ambiguous}" if ambiguous else " unambiguous")
    return Runbook(chains=chains, provenance=prov)
