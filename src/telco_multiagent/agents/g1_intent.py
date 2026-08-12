"""G1 Intent Understanding agent."""

from __future__ import annotations

from ..llm.client import LLMClient
from ..schema.messages import G1Payload
from .base import BaseAgent


def make_agent(client: LLMClient, with_list: bool = True) -> BaseAgent:
    return BaseAgent(
        role="G1",
        client=client,
        payload_model=G1Payload,
        prompt_name="g1_intent_withlist" if with_list else "g1_intent_nolist",
    )


def build_user(problem_statement: str, language: str, candidates: list[tuple[str, str]]) -> str:
    lines = [f"Problem statement ({language}): {problem_statement}", ""]
    if candidates:
        lines.append("Candidate intents (key | label):")
        lines += [f"- {k} | {label}" for k, label in candidates]
    lines.append("\nReturn a G1 JSON payload.")
    return "\n".join(lines)
