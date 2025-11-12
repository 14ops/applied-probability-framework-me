"""
Consensus governor combines individual agent signals into a single action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .agent_orchestrator import AgentSignal
from src.python.agents.lelouch import Lelouch


@dataclass
class GovernorDecision:
    action: str
    confidence: float
    rationale: Dict[str, float] = field(default_factory=dict)


@dataclass
class ConsensusGovernor:
    """
    Aggregates agent signals using Lelouch-provided weights or a static mapping.
    """

    weights: Dict[str, float] = field(default_factory=lambda: Lelouch().weights)
    buy_threshold: float = 0.55
    sell_threshold: float = 0.55
    name: str = "consensus_governor"

    def decide(
        self,
        signals: List[AgentSignal],
        include_lelouch: bool = False,
    ) -> GovernorDecision:
        scores = {"buy": 0.0, "sell": 0.0, "hold": 0.0}

        for sig in signals:
            if sig.agent == "lelouch" and not include_lelouch:
                continue

            weight = self.weights.get(sig.agent, 1.0)
            scores.setdefault(sig.action, 0.0)
            scores[sig.action] += weight * sig.confidence

        # Normalize to keep scores interpretable
        total_weight = sum(self.weights.get(sig.agent, 1.0) for sig in signals if sig.agent != "lelouch" or include_lelouch)
        normalized = {k: (v / total_weight) if total_weight else 0.0 for k, v in scores.items()}

        action = "hold"
        confidence = 0.0

        buy_score = normalized.get("buy", 0.0)
        sell_score = normalized.get("sell", 0.0)

        if buy_score >= self.buy_threshold and buy_score >= sell_score:
            action = "buy"
            confidence = buy_score
        elif sell_score >= self.sell_threshold and sell_score > buy_score:
            action = "sell"
            confidence = sell_score
        else:
            action = "hold"
            confidence = max(normalized.get("hold", 0.0), buy_score, sell_score)

        return GovernorDecision(action=action, confidence=confidence, rationale=normalized)

