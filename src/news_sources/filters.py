"""
Shared noise-filtering utilities for all news sources.

A headline passes if it matches at least one finance pattern AND
does not match any noise pattern.
"""
from __future__ import annotations

import re

FINANCE_PATTERNS: dict[str, str] = {
    "spy":           r"\bspy\b",
    "s&p 500":       r"\bs\s*&\s*p\s*500\b",
    "nasdaq":        r"\bnasdaq\b",
    "dow":           r"\bdow\b|\bdow jones\b",
    "fed":           r"\bfed\b|\bfomc\b",
    "federal reserve": r"\bfederal reserve\b",
    "inflation":     r"\binflation\b",
    "rates":         r"\brates?\b|\binterest rates?\b",
    "earnings":      r"\bearnings?\b",
    "guidance":      r"\bguidance\b",
    "stocks":        r"\bstocks?\b",
    "shares":        r"\bshares?\b",
    "market":        r"\bmarkets?\b",
    "wall street":   r"\bwall street\b",
    "equities":      r"\bequities\b|\bequity\b",
    "index":         r"\bindex\b|\bindexes\b|\bindices\b",
    "futures":       r"\bfutures\b",
    "etf":           r"\betf\b|\betfs\b",
    "bonds":         r"\bbonds?\b",
    "central bank":  r"\bcentral bank\b",
    "volatility":    r"\bvolatility\b",
    "rally":         r"\brally\b",
    "selloff":       r"\bselloff\b|\bsell-off\b",
    "treasury":      r"\btreasury\b|\btreasuries\b",
    "yield":         r"\byields?\b",
    "oil":           r"\boil\b",
    "gold":          r"\bgold\b",
    "tariff":        r"\btariffs?\b",
    "recession":     r"\brecession\b",
    "economy":       r"\beconom(y|ic)\b",
    "gdp":           r"\bgdp\b",
    "ipo":           r"\bipo\b",
    "dividend":      r"\bdividends?\b",
    "hedge fund":    r"\bhedge funds?\b",
    "short":         r"\bshort interest\b|\bshort selling\b",
}

# Headlines matching ANY of these are noise — drop them regardless
NOISE_PATTERNS: list[str] = [
    r"\bcelebrit(y|ies)\b",
    r"\bsports?\b",
    r"\bsoccer\b|\bfootball\b|\bbasketball\b|\bbaseball\b",
    r"\bwedding\b|\bdivorce\b",
    r"\bhoroscope\b",
    r"\brecipe\b|\bcooking\b",
    r"\bcosmetic\b|\bmakeup\b|\bbeauty tips\b",
    r"\bvideogame\b|\bgaming tips\b",
    r"\bshares photo\b",          # tabloid "celebrity shares photo"
    r"\bspotted (again|in)\b",    # tabloid sightings
    r"\bstudio photos\b",
]

_NOISE_RE = re.compile("|".join(NOISE_PATTERNS), re.IGNORECASE)


def relevance_terms(text: str) -> list[str]:
    lowered = text.lower()
    return [label for label, pattern in FINANCE_PATTERNS.items()
            if re.search(pattern, lowered)]


def is_noise(text: str) -> bool:
    return bool(_NOISE_RE.search(text))


def is_relevant(text: str) -> bool:
    """True if headline matches finance terms AND is not noise."""
    if is_noise(text):
        return False
    return bool(relevance_terms(text))
