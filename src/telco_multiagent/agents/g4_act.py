"""G4 Action Recommendation agent."""

from __future__ import annotations

from ..llm.client import LLMClient
from ..schema.messages import G4Payload
from .base import BaseAgent


def make_agent(client: LLMClient) -> BaseAgent:
    return BaseAgent("G4", client, G4Payload, "g4_act")


def build_user(intent_key: str, selected_hypothesis: str, repair_hint: str) -> str:
    lines = [
        f"Intent: {intent_key}",
        f"Confirmed diagnosis: {selected_hypothesis or '(none)'}",
    ]
    if repair_hint:
        lines.append("Checker repair hint: " + repair_hint)
    lines.append(
        "\nPlan the corrective tool sequence (corrective tools only), justify each "
        "action. Return a G4 JSON payload."
    )
    return "\n".join(lines)
