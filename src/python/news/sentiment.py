from __future__ import annotations

from typing import List
import re

# Minimal lexicon to avoid extra deps; replace with VADER/TextBlob as needed
POS_WORDS = {
    "beat", "beats", "surge", "surges", "soar", "soars", "record", "strong", "up", "gain", "gains", "bullish",
    "upgrade", "outperform", "tops", "profit", "profits", "growth",
}
NEG_WORDS = {
    "miss", "misses", "fall", "falls", "drop", "drops", "plunge", "plunges", "lawsuit", "probe", "downgrade",
    "loss", "losses", "weak", "bearish", "decline", "recall",
}


def score_text(text: str) -> float:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    pos = sum(1 for w in words if w in POS_WORDS)
    neg = sum(1 for w in words if w in NEG_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


def aggregate(headlines: List[str]) -> float:
    if not headlines:
        return 0.0
    scores = [score_text(h) for h in headlines]
    return sum(scores) / len(scores)


