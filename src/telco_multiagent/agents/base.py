"""BaseAgent: prompt rendering + JSON-mode parsing with one validation retry."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from ..llm.client import LLMClient

P = TypeVar("P", bound=BaseModel)

_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(name: str) -> str:
    path = _PROMPT_DIR / f"{name}.md"
    if not path.exists():
        return f"You are agent {name}. Respond with a single JSON object only."
    return path.read_text(encoding="utf-8")


def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


@dataclass
class AgentResult(Generic[P]):
    payload: P | None
    tokens: int = 0
    retries: int = 0
    parse_failed: bool = False
    parse_rejections: list[str] = field(default_factory=list)
    raw: str = ""


@dataclass
class BaseAgent:
    role: str
    client: LLMClient
    payload_model: type[BaseModel]
    prompt_name: str

    @property
    def system_prompt(self) -> str:
        return load_prompt(self.prompt_name)

    def invoke(self, user: str, context: dict[str, Any]) -> AgentResult:
        system = self.system_prompt
        tokens = 0
        rejections: list[str] = []
        last_err = ""
        for attempt in range(2):
            prompt = user if attempt == 0 else (
                f"{user}\n\nYour previous reply failed validation: {last_err}\n"
                "Return corrected JSON only."
            )
            res = self.client.generate(self.role, system, prompt, context)
            tokens += res.tokens
            raw = _extract_json(res.text)
            try:
                data = json.loads(raw)
                data.setdefault("kind", self.role)
                payload = self.payload_model.model_validate(data)
                return AgentResult(
                    payload=payload, tokens=tokens, retries=attempt,
                    parse_rejections=rejections, raw=raw,
                )
            except (json.JSONDecodeError, ValidationError) as e:
                last_err = str(e)[:400]
                if "hallucinated tool" in last_err:
                    rejections.append(last_err)
        return AgentResult(
            payload=None, tokens=tokens, retries=1, parse_failed=True,
            parse_rejections=rejections, raw=last_err,
        )
