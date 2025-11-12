"""Lelouch - Meta optimizer adjusting agent weights."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional


@dataclass
class LelouchState:
    weights: Dict[str, float] = field(default_factory=dict)
    total_adjustments: int = 0


class LelouchMetaOptimizer:
    """
    Maintains ensemble weights based on rolling performance statistics.

    The update rule nudges weights toward agents with higher Sharpe ratios and
    away from underperformers, while keeping the distribution normalized.
    """

    def __init__(
        self,
        agents: Iterable[str],
        learning_rate: float = 0.05,
        min_weight: float = 0.05,
    ) -> None:
        agents = list(agents)
        if not agents:
            raise ValueError("LelouchMetaOptimizer requires at least one agent name")
        self.learning_rate = learning_rate
        self.min_weight = min_weight
        uniform = 1.0 / len(agents)
        self.state = LelouchState(weights={agent: uniform for agent in agents})

    @property
    def weights(self) -> Dict[str, float]:
        return self.state.weights.copy()

    def update(self, sharpe_by_agent: Mapping[str, Optional[float]]) -> Dict[str, float]:
        """Adjust weights using the provided Sharpe ratios (None treated as 0)."""
        if not sharpe_by_agent:
            return self.weights

        weights = self.state.weights
        avg_sharpe = sum((sharpe or 0.0) for sharpe in sharpe_by_agent.values()) / max(
            len(sharpe_by_agent), 1
        )

        for agent, sharpe in sharpe_by_agent.items():
            if agent not in weights:
                continue
            sharpe = sharpe or 0.0
            delta = self.learning_rate * (sharpe - avg_sharpe)
            weights[agent] = max(self.min_weight, weights[agent] + delta)

        self._renormalize(weights)
        self.state.total_adjustments += 1
        return self.weights

    def _renormalize(self, weights: Dict[str, float]) -> None:
        total = sum(weights.values())
        if total <= 0:
            uniform = 1.0 / len(weights)
            for agent in weights:
                weights[agent] = uniform
            return

        for agent in weights:
            weights[agent] /= total


