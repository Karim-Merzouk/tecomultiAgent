"""LLMClient: one interface, three backends.

All agents call ``client.generate(role, system, user, context)`` and receive a
raw string (expected to be JSON). Real backends ignore ``role``/``context`` and
send ``system``+``user`` to the model. The offline ``heuristic`` backend uses
``role`` and ``context`` (including seeded gold signals under ``_oracle``) to
synthesize valid payloads with controlled noise, so the whole architecture is
runnable and produces metric tables without a served model.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..vocab import DISTRACTOR_TOOLS, tool_segment


@dataclass
class GenResult:
    text: str
    tokens: int = 0
    cost: float = 0.0


class LLMClient(Protocol):
    name: str

    def generate(self, role: str, system: str, user: str, context: dict[str, Any]) -> GenResult:
        ...


# --- real backends ----------------------------------------------------------


@dataclass
class OpenAICompatibleClient:
    """Works with vLLM / Ollama / LM Studio / OpenAI /v1/chat/completions."""

    model: str
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "not-needed"
    temperature: float = 0.0
    timeout: float = 120.0
    name: str = "openai-compatible"

    def generate(self, role: str, system: str, user: str, context: dict[str, Any]) -> GenResult:
        import httpx

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        with httpx.Client(timeout=self.timeout) as cli:
            r = cli.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            r.raise_for_status()
            data = r.json()
        text = data["choices"][0]["message"]["content"]
        tokens = int(data.get("usage", {}).get("total_tokens", 0))
        return GenResult(text=text, tokens=tokens)


@dataclass
class AnthropicClient:
    model: str = "claude-fable-5"
    api_key: str = ""
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout: float = 120.0
    name: str = "anthropic"

    def generate(self, role: str, system: str, user: str, context: dict[str, Any]) -> GenResult:
        import os

        import httpx

        key = self.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        with httpx.Client(timeout=self.timeout) as cli:
            r = cli.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "system": system + "\nRespond with a single JSON object only.",
                    "messages": [{"role": "user", "content": user}],
                },
            )
            r.raise_for_status()
            data = r.json()
        text = "".join(b.get("text", "") for b in data.get("content", []))
        usage = data.get("usage", {})
        tokens = int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0))
        return GenResult(text=text, tokens=tokens)


# --- offline heuristic-oracle backend --------------------------------------


@dataclass
class HeuristicOracleClient:
    """Deterministic offline backend for smoke runs and result generation.

    Reads gold signals from ``context['_oracle']`` and injects seeded noise so
    metrics are realistic (never trivially perfect) and the checker (G5) has real
    defects to catch. Noise rates are configurable to sweep difficulty.
    """

    seed: int = 42
    intent_error_rate: float = 0.05
    drop_rate: float = 0.08
    swap_rate: float = 0.08
    distractor_rate: float = 0.10
    name: str = "heuristic"
    _calls: int = field(default=0, repr=False)

    def _rng(self, role: str, context: dict[str, Any]) -> random.Random:
        cid = context.get("case_id", "")
        return random.Random(f"{self.seed}:{cid}:{role}")

    def generate(self, role: str, system: str, user: str, context: dict[str, Any]) -> GenResult:
        self._calls += 1
        oracle: dict[str, Any] = context.get("_oracle", {})
        rng = self._rng(role, context)
        handler = {
            "G1": self._g1,
            "G2": self._g2,
            "G3": self._g3,
            "G4": self._g4,
            "G5": self._g5,
            "G6": self._g6,
        }[role]
        obj = handler(oracle, context, rng)
        return GenResult(text=json.dumps(obj, ensure_ascii=False), tokens=len(user) // 4 + 40)

    # -- role synthesizers --

    def _g1(self, oracle, context, rng) -> dict:
        intent = oracle.get("intent_key", "unknown")
        label = oracle.get("intent_label", intent)
        category = oracle.get("category", "")
        clarify = False
        if rng.random() < self.intent_error_rate:
            wrong = oracle.get("confusable", "unknown")
            intent, label = wrong, wrong
            clarify = rng.random() < 0.5
        return {
            "kind": "G1",
            "intent_free_text": oracle.get("problem_statement", label)[:160],
            "intent_label": intent,
            "category": category,
            "clarification_needed": clarify,
        }

    def _plan(self, gold_tools: list[str], segment: str, rng, clean: bool = False) -> list[dict]:
        if segment == "all":
            tools = list(gold_tools)
        else:
            tools = [t for t in gold_tools if tool_segment(t) == segment]
        if not clean:
            # controlled drop
            if tools and rng.random() < self.drop_rate and len(tools) > 1:
                tools.pop(rng.randrange(len(tools)))
            # controlled swap
            if len(tools) > 1 and rng.random() < self.swap_rate:
                i = rng.randrange(len(tools) - 1)
                tools[i], tools[i + 1] = tools[i + 1], tools[i]
            # controlled distractor insertion
            if rng.random() < self.distractor_rate:
                tools.insert(rng.randrange(len(tools) + 1), rng.choice(DISTRACTOR_TOOLS))
        return [{"tool_name": t, "arguments": {"note": "auto"}} for t in tools]

    def _g2(self, oracle, context, rng) -> dict:
        gold = oracle.get("gold_tools", [])
        segment = context.get("plan_segment", "diagnostic")
        return {
            "kind": "G2",
            "planned_diagnostic_sequence": self._plan(
                gold, segment, rng, clean=context.get("repair", False)
            ),
            "evidence_summary": oracle.get("evidence_hint", "collected diagnostic KPIs"),
            "missing_information": [],
            "anomaly_flags": oracle.get("anomaly_flags", ["none"]),
        }

    def _g3(self, oracle, context, rng) -> dict:
        rc = oracle.get("root_cause", "root cause inferred from evidence")
        return {
            "kind": "G3",
            "hypotheses": [
                {"root_cause": rc, "supporting_evidence": "KPI signature match", "confidence": 0.8}
            ],
            "selected_hypothesis": rc,
            "needs_more_evidence": False,
            "requested_evidence": [],
        }

    def _g4(self, oracle, context, rng) -> dict:
        gold = oracle.get("gold_tools", [])
        seq = self._plan(gold, "corrective", rng, clean=context.get("repair", False))
        return {
            "kind": "G4",
            "planned_corrective_sequence": seq,
            "justification_per_action": {
                s["tool_name"]: "consistent with diagnosis" for s in seq
            },
            "expected_effect": "restore KPIs to nominal range",
        }

    def _g5(self, oracle, context, rng) -> dict:
        # audit executed+planned sequence vs gold
        agent = context.get("full_sequence", [])
        gold = oracle.get("gold_tools", [])
        defects = _audit(agent, gold)
        declared = context.get("declared_intent")
        true_intent = oracle.get("intent_key")
        if declared is not None and true_intent is not None and declared != true_intent:
            defects.append(
                {"type": "intent_mismatch", "step": None,
                 "detail": f"declared {declared} != {true_intent}"}
            )
        verdict = "fail" if defects else "pass"
        hint = defects[0]["detail"] if defects else ""
        return {"kind": "G5", "verdict": verdict, "defects": defects, "repair_hint": hint}

    def _g6(self, oracle, context, rng) -> dict:
        lang = context.get("language", "en")
        summary = oracle.get("gold_summary", "Issue diagnosed and resolved.")
        return {
            "kind": "G6",
            "report_text": summary,
            "root_cause": oracle.get("root_cause", ""),
            "actions": oracle.get("corrective_tools", []),
            "ticket_ref": context.get("ticket_ref"),
            "report_language": lang,
        }


def _audit(agent: list[str], gold: list[str]) -> list[dict]:
    """Rule-based sequence audit used by the heuristic G5 and by report checks."""
    defects: list[dict] = []
    agent_core = [t for t in agent if tool_segment(t) != "distractor"]
    for i, t in enumerate(agent):
        if tool_segment(t) == "distractor":
            defects.append({"type": "extraneous", "step": i + 1, "detail": f"distractor {t}"})
    for g in gold:
        if g not in agent_core:
            defects.append({"type": "missing", "step": None, "detail": f"missing {g}"})
    # order check on the common subsequence
    common = [t for t in agent_core if t in gold]
    if common and common != [t for t in gold if t in common]:
        defects.append({"type": "misordered", "step": None, "detail": "order deviates from gold"})
    return defects
