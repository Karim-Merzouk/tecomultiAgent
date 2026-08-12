"""G6 Resolution / Reporting agent."""

from __future__ import annotations

from ..llm.client import LLMClient
from ..schema.messages import G6Payload
from .base import BaseAgent


def make_agent(client: LLMClient) -> BaseAgent:
    return BaseAgent("G6", client, G6Payload, "g6_report")


def build_user(
    intent_key: str,
    language: str,
    root_cause: str,
    executed_sequence: list[str],
    ticket_ref: str | None,
) -> str:
    return (
        f"Intent: {intent_key}\n"
        f"Root cause: {root_cause}\n"
        f"Executed trajectory: {executed_sequence}\n"
        f"Ticket ref: {ticket_ref}\n"
        f"Report language: {language}\n\n"
        "Write the engineer-facing resolution report grounded strictly in the "
        "executed trajectory. Return a G6 JSON payload."
    )
