"""Sentence embedder with a deterministic offline fallback.

Uses paraphrase-multilingual-mpnet-base-v2 (multilingual, needed for Arabic)
when sentence-transformers is installed; otherwise a hashed character-n-gram
bag-of-words cosine so IRA/RA remain computable offline (clearly weaker, flagged
in the report metadata).
"""

from __future__ import annotations

import math
import os
import re
from functools import lru_cache

_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"


@lru_cache(maxsize=1)
def _load_model():
    # TELCO_EMBED=fallback forces the offline hashed-ngram embedder (no model
    # download); the default tries the multilingual model and silently falls
    # back if it (or its weights) are unavailable.
    if os.environ.get("TELCO_EMBED", "").lower() == "fallback":
        return None
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(_MODEL_NAME)
    except Exception:
        return None


def backend_name() -> str:
    return _MODEL_NAME if _load_model() is not None else "hashed-ngram-fallback"


def _ngrams(text: str, n: int = 3) -> list[str]:
    text = re.sub(r"\s+", " ", text.lower().strip())
    if len(text) < n:
        return [text] if text else []
    return [text[i : i + n] for i in range(len(text) - n + 1)]


def _fallback_vec(text: str, dim: int = 512) -> list[float]:
    vec = [0.0] * dim
    for g in _ngrams(text):
        vec[hash(g) % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a: str, b: str) -> float:
    model = _load_model()
    if model is not None:
        import numpy as np

        va, vb = model.encode([a, b], normalize_embeddings=True)
        return float(np.clip(np.dot(va, vb), -1.0, 1.0))
    va, vb = _fallback_vec(a), _fallback_vec(b)
    return max(-1.0, min(1.0, sum(x * y for x, y in zip(va, vb, strict=False))))
