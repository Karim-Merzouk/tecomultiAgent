"""Build an LLMClient from a BackendConfig."""

from __future__ import annotations

from ..orchestrator.config import BackendConfig
from .client import (
    AnthropicClient,
    HeuristicOracleClient,
    LLMClient,
    OpenAICompatibleClient,
)


def make_client(cfg: BackendConfig, seed: int = 42) -> LLMClient:
    if cfg.kind == "openai":
        return OpenAICompatibleClient(
            model=cfg.model,
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            temperature=cfg.temperature,
        )
    if cfg.kind == "anthropic":
        return AnthropicClient(model=cfg.model, temperature=cfg.temperature)
    return HeuristicOracleClient(
        seed=seed,
        intent_error_rate=cfg.intent_error_rate,
        drop_rate=cfg.drop_rate,
        swap_rate=cfg.swap_rate,
        distractor_rate=cfg.distractor_rate,
    )
