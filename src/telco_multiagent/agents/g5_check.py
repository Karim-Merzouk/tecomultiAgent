"""G5 Verification / Checking agent."""

from __future__ import annotations

from ..llm.client import LLMClient
from ..schema.messages import G5Payload
from .base import BaseAgent


def make_agent(client: LLMClient) -> BaseAgent:
    return BaseAgent("G5", client, G5Payload, "g5_check")


def build_sequence_user(intent_key: str, full_sequence: list[str], reference: list[str]) -> str:
    return (
        f"Declared intent: {intent_key}\n"
        f"Full planned/executed sequence: {full_sequence}\n"
        f"Reference gold flow: {reference}\n\n"
        "Audit for missing / misordered / extraneous steps and intent mismatch. "
        "Return a G5 JSON payload (verdict pass|fail with defects)."
    )


def build_report_user(report_text: str, executed_sequence: list[str], ticket_ref: str | None) -> str:
    return (
        f"Draft report:\n{report_text}\n\n"
        f"Executed trajectory: {executed_sequence}\n"
        f"Ticket ref: {ticket_ref}\n\n"
        "Check the report against the trajectory for report_inconsistency (claims "
        "unexecuted actions or omits the ticket). Return a G5 JSON payload."
    )
