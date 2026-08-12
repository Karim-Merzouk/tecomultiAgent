"""Run G5 over held-out A10 corruptions and compute defect detection P/R/F1."""

from __future__ import annotations

from ..data.corruptions import generate_corruptions
from ..data.loader import Dataset
from ..llm.factory import make_client
from ..orchestrator.config import RunConfig
from ..vocab import intent_category, intent_label_str, tool_segment
from . import new_metrics as NM

# defect types the sequence + intent auditor is responsible for
_SEQ_DEFECTS = {"missing", "misordered", "extraneous", "intent_mismatch"}


def _oracle_for(record) -> dict:
    return {
        "intent_key": record.intent,
        "intent_label": intent_label_str(record.intent),
        "category": intent_category(record.intent),
        "gold_tools": record.gold_sequence,
        "corrective_tools": [t for t in record.gold_sequence if tool_segment(t) == "corrective"],
    }


def run_defect_eval(config: RunConfig, dataset: Dataset) -> dict[str, NM.PRF]:
    client = make_client(config.backend, seed=config.seed)
    corr = generate_corruptions(dataset, seed=config.seed)
    eval_records = corr["eval"] + [r for r in corr["positives"] if r.provenance.get("split") == "eval"]

    predictions: list[set[str]] = []
    truths: list[set[str]] = []
    for rec in eval_records:
        ctx = {
            "case_id": rec.sample_id,
            "language": "en",
            "_oracle": _oracle_for(rec),
            "full_sequence": rec.corrupted_sequence,
            "declared_intent": rec.declared_intent,
        }
        res = client.generate("G5", "", "", ctx)
        import json

        payload = json.loads(res.text)
        pred = {d["type"] for d in payload.get("defects", [])} & _SEQ_DEFECTS
        truth = set(rec.defect_labels) & _SEQ_DEFECTS
        predictions.append(pred)
        truths.append(truth)

    return NM.defect_detection(predictions, truths)
