"""G2 Information Acquisition agent."""

from __future__ import annotations

from ..llm.client import LLMClient
from ..schema.messages import G2Payload
from .base import BaseAgent


def make_agent(client: LLMClient) -> BaseAgent:
    return BaseAgent("G2", client, G2Payload, "g2_acquire")


def build_user(intent_key: str, evidence: str, requested: list[str], repair_hint: str) -> str:
    lines = [f"Intent: {intent_key}", f"Evidence so far: {evidence or '(none)'}"]
    if requested:
        lines.append("Diagnosis requested additional evidence: " + ", ".join(requested))
    if repair_hint:
        lines.append("Checker repair hint: " + repair_hint)
    lines.append(
        "\nPlan the diagnostic tool sequence (diagnostic tools only). "
        "Return a G2 JSON payload."
    )
    return "\n".join(lines)
