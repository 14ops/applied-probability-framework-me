"""Consensus governor combines agent signals into an actionable decision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..agents import Signal
from ..agents.base import Action


@dataclass
class ConsensusDecision:
    action: Action
    score: float
    supporting_agents: List[str]
    dissenting_agents: List[str]
    metadata: Dict[str, object]


class ConsensusGovernor:
    """
    Aggregates agent signals using weightings provided by the meta-optimizer.

    Decision rule:
        1. Abort if any signal raises a veto (typically Yuzu).
        2. Compute weighted scores for ``buy`` and ``sell`` actions.
        3. Require both a minimum number of agreeing agents and a weighted
           confidence threshold before allowing a trade.
    """

    def __init__(
        self,
        min_agreement: int = 2,
        confidence_threshold: float = 0.8,
    ) -> None:
        self.min_agreement = min_agreement
        self.confidence_threshold = confidence_threshold

    def decide(
        self,
        signals: Sequence[Signal],
        weights: Optional[Mapping[str, float]] = None,
    ) -> ConsensusDecision:
        weights = weights or {}

        # Detect vetoes first (Yuzu or any risk agent can set metadata["veto"]).
        for signal in signals:
            if signal.metadata.get("veto"):
                metadata = {"vetoed_by": signal.agent, **signal.metadata}
                return ConsensusDecision(
                    action="hold",
                    score=0.0,
                    supporting_agents=[],
                    dissenting_agents=[sig.agent for sig in signals],
                    metadata=metadata,
                )

        directional_scores: Dict[Action, float] = {"buy": 0.0, "sell": 0.0}
        supporters: Dict[Action, List[str]] = {"buy": [], "sell": []}

        total_weight = 0.0
        for signal in signals:
            if signal.action == "hold" or signal.confidence <= 0.0:
                continue
            action = signal.action
            weight = weights.get(signal.agent, 1.0)
            weighted_confidence = signal.confidence * weight
            directional_scores[action] += weighted_confidence
            supporters[action].append(signal.agent)
            total_weight += weight

        if total_weight <= 0.0:
            return ConsensusDecision(
                action="hold",
                score=0.0,
                supporting_agents=[],
                dissenting_agents=[sig.agent for sig in signals],
                metadata={"reason": "no_directional_support"},
            )

        # Determine strongest direction
        action = max(directional_scores, key=directional_scores.get)
        score = directional_scores[action] / total_weight
        supporting_agents = supporters[action]

        if len(supporting_agents) < self.min_agreement:
            return ConsensusDecision(
                action="hold",
                score=score,
                supporting_agents=supporting_agents,
                dissenting_agents=[sig.agent for sig in signals if sig.agent not in supporting_agents],
                metadata={"reason": "insufficient_agreement"},
            )

        if score < self.confidence_threshold:
            return ConsensusDecision(
                action="hold",
                score=score,
                supporting_agents=supporting_agents,
                dissenting_agents=[sig.agent for sig in signals if sig.agent not in supporting_agents],
                metadata={"reason": "confidence_below_threshold"},
            )

        metadata = {
            "directional_scores": directional_scores,
            "weights_used": dict(weights),
        }
        dissenters = [sig.agent for sig in signals if sig.agent not in supporting_agents]
        return ConsensusDecision(
            action=action,
            score=min(score, 1.0),
            supporting_agents=supporting_agents,
            dissenting_agents=dissenters,
            metadata=metadata,
        )


