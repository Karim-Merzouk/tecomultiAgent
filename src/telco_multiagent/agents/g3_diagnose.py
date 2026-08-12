"""G3 Diagnosis agent."""

from __future__ import annotations

from ..llm.client import LLMClient
from ..schema.messages import G3Payload
from .base import BaseAgent


def make_agent(client: LLMClient) -> BaseAgent:
    return BaseAgent("G3", client, G3Payload, "g3_diagnose")


def build_user(intent_key: str, evidence: str, anomaly_flags: list[str]) -> str:
    return (
        f"Intent: {intent_key}\n"
        f"Evidence summary: {evidence or '(none)'}\n"
        f"Anomaly flags: {', '.join(anomaly_flags) or 'none'}\n\n"
        "Produce ranked root-cause hypotheses. If evidence is insufficient, set "
        "needs_more_evidence and list requested_evidence. Return a G3 JSON payload."
    )
