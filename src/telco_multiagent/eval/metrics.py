"""TelcoAgent-Bench metrics: IRA, MSC, EAP, SAS, GPC-0/1, SD, BRS, RA.

Formulas implemented exactly as specified:
  MSC = LCS(agent, gold) / len(gold)
  EAP = 1 - max(0, n - m) / n         (n = agent length, m = gold length)
  SAS = MSC * EAP
  GPC-0 = fraction of blueprint samples with exact gold-path match
  GPC-1 = fraction with Levenshtein <= 1
  SD    = mean pairwise normalized Levenshtein across a blueprint's samples
  BRS   = a1*GPC0 + a2*GPC1 + a3*(1 - SD),  default a = (0.5, 0.3, 0.2)
  IRA   = cosine(agent intent free-text, gold intent label)  [multilingual]
  RA    = cosine(generated summary, gold summary)             [per language]
"""

from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein

from .embed import cosine


def lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def msc(agent: list[str], gold: list[str]) -> float:
    if not gold:
        return 1.0
    return lcs_len(agent, gold) / len(gold)


def eap(agent: list[str], gold: list[str]) -> float:
    n, m = len(agent), len(gold)
    if n == 0:
        return 1.0
    return 1.0 - max(0, n - m) / n


def sas(agent: list[str], gold: list[str]) -> float:
    return msc(agent, gold) * eap(agent, gold)


def levenshtein(a: list[str], b: list[str]) -> int:
    return Levenshtein.distance(a, b)


def norm_levenshtein(a: list[str], b: list[str]) -> float:
    m = max(len(a), len(b))
    return levenshtein(a, b) / m if m else 0.0


def gpc0(sequences: list[list[str]], golds: list[list[str]]) -> float:
    if not sequences:
        return 0.0
    hits = sum(1 for s, g in zip(sequences, golds, strict=False) if s == g)
    return hits / len(sequences)


def gpc1(sequences: list[list[str]], golds: list[list[str]]) -> float:
    if not sequences:
        return 0.0
    hits = sum(1 for s, g in zip(sequences, golds, strict=False) if levenshtein(s, g) <= 1)
    return hits / len(sequences)


def sequence_diversity(sequences: list[list[str]]) -> float:
    if len(sequences) < 2:
        return 0.0
    tot, cnt = 0.0, 0
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            tot += norm_levenshtein(sequences[i], sequences[j])
            cnt += 1
    return tot / cnt if cnt else 0.0


def brs(g0: float, g1: float, sd: float, alphas: tuple[float, float, float] = (0.5, 0.3, 0.2)) -> float:
    a1, a2, a3 = alphas
    return a1 * g0 + a2 * g1 + a3 * (1.0 - sd)


def ira(agent_free_text: str, gold_label: str) -> float:
    return cosine(agent_free_text, gold_label)


def ra(generated: str, gold: str) -> float:
    return cosine(generated, gold)


@dataclass
class BlueprintMetrics:
    blueprint_id: str
    intent: str
    language: str
    n: int
    msc: float
    eap: float
    sas: float
    msc_diag: float
    msc_corr: float
    gpc0: float
    gpc1: float
    sd: float
    brs: float
    ira: float
    ra: float
