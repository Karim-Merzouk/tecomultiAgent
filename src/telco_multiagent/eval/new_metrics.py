"""New metrics this work adds: defect P/R, escalation precision, comm overhead,
cross-language report consistency."""

from __future__ import annotations

from dataclasses import dataclass

from ..schema.case_state import CaseState
from .embed import cosine


@dataclass
class PRF:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def _prf(tp: int, fp: int, fn: int) -> PRF:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return PRF(p, r, f, tp, fp, fn)


def defect_detection(
    predictions: list[set[str]],
    truths: list[set[str]],
) -> dict[str, PRF]:
    """Per-defect-type and overall precision/recall/F1.

    Each element is the set of defect types predicted / truly present for a
    trajectory (empty set == clean).
    """
    types = sorted({t for s in truths for t in s} | {t for s in predictions for t in s})
    out: dict[str, PRF] = {}
    o_tp = o_fp = o_fn = 0
    for t in types:
        tp = sum(1 for p, y in zip(predictions, truths, strict=False) if t in p and t in y)
        fp = sum(1 for p, y in zip(predictions, truths, strict=False) if t in p and t not in y)
        fn = sum(1 for p, y in zip(predictions, truths, strict=False) if t not in p and t in y)
        out[t] = _prf(tp, fp, fn)
        o_tp += tp
        o_fp += fp
        o_fn += fn
    out["overall"] = _prf(o_tp, o_fp, o_fn)
    return out


def escalation_metrics(escalated: list[bool], should_escalate: list[bool]) -> PRF:
    tp = sum(1 for e, s in zip(escalated, should_escalate, strict=False) if e and s)
    fp = sum(1 for e, s in zip(escalated, should_escalate, strict=False) if e and not s)
    fn = sum(1 for e, s in zip(escalated, should_escalate, strict=False) if not e and s)
    return _prf(tp, fp, fn)


@dataclass
class CommOverhead:
    n_cases: int
    msgs_per_case: float
    tool_calls_per_case: float
    tokens_per_case: float
    repair_rounds_per_case: float


def communication_overhead(cases: list[CaseState]) -> CommOverhead:
    n = len(cases) or 1
    msgs = sum(sum(1 for e in c.events if e.kind == "envelope") for c in cases)
    tools = sum(c.budgets.tool_calls_used for c in cases)
    tokens = sum(c.total_tokens for c in cases)
    repairs = sum(
        c.budgets.repair_rounds_used + c.budgets.evidence_loops_used + c.budgets.report_rounds_used
        for c in cases
    )
    return CommOverhead(len(cases), msgs / n, tools / n, tokens / n, repairs / n)


def cross_language_consistency(report_pairs: list[tuple[str, str]]) -> float:
    """Mean cosine similarity between En/Ar reports for the same case."""
    if not report_pairs:
        return 0.0
    return sum(cosine(en, ar) for en, ar in report_pairs) / len(report_pairs)
