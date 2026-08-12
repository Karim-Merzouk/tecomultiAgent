import pytest
from pydantic import ValidationError

from telco_multiagent.schema.messages import (
    Envelope,
    G2Payload,
    ToolCallSpec,
)


def test_tool_call_accepts_core_and_distractor():
    assert ToolCallSpec(tool_name="oss_query").tool_name.value == "oss_query"
    assert ToolCallSpec(tool_name="optimize_cell").tool_name.value == "optimize_cell"
    assert ToolCallSpec(tool_name="subscriber_insight").tool_name.value == "subscriber_insight"


def test_tool_call_rejects_hallucinated():
    with pytest.raises(ValidationError) as e:
        ToolCallSpec(tool_name="teleport_cell")
    assert "hallucinated" in str(e.value)


def test_envelope_discriminated_union():
    env = Envelope(
        case_id="c1",
        msg_id="m1",
        sender="G2",
        recipient="G0",
        payload=G2Payload(evidence_summary="x"),
    )
    assert env.payload.kind == "G2"
    dumped = env.model_dump()
    assert dumped["payload"]["kind"] == "G2"


def test_g2_default_lists():
    p = G2Payload()
    assert p.planned_diagnostic_sequence == []
    assert p.missing_information == []
