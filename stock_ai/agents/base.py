"""
Agent framework for the character-based trading ensemble.

Each agent consumes a pandas ``DataFrame`` of OHLCV data (most-recent row is
assumed to be the active bar) and arbitrary context and emits a normalized
signal of the form ``(action, confidence)``.  Agents can optionally attach
metadata such as veto flags or diagnostic information.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

import pandas as pd

Action = Literal["buy", "sell", "hold"]


@dataclass
class Signal:
    """Standardized signal returned by every agent."""

    agent: str
    action: Action
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def clipped(self, minimum: float = 0.0, maximum: float = 1.0) -> "Signal":
        """Return a copy with confidence clipped into the provided bounds."""
        return Signal(
            agent=self.agent,
            action=self.action,
            confidence=max(minimum, min(self.confidence, maximum)),
            metadata=self.metadata.copy(),
        )


class Agent(ABC):
    """Base class providing a common interface and helper utilities."""

    name: str

    def __init__(self, name: str, min_history: int = 100) -> None:
        self.name = name
        self.min_history = min_history

    def __call__(
        self, data: pd.DataFrame, context: Optional[Dict[str, Any]] = None
    ) -> Signal:
        context = context or {}
        if data is None or data.empty or len(data) < self.min_history:
            return Signal(agent=self.name, action="hold", confidence=0.0)
        signal = self.generate_signal(data, context)
        if not isinstance(signal, Signal):
            raise TypeError(
                f"{self.__class__.__name__}.generate_signal must return Signal, "
                f"got {type(signal)}"
            )
        return signal.clipped()

    @abstractmethod
    def generate_signal(
        self, data: pd.DataFrame, context: Dict[str, Any]
    ) -> Signal:
        """Return the agent's decision for the most recent bar."""

    def on_trade_result(self, result: Dict[str, Any]) -> None:
        """Optional callback when a trade completes."""
        # Default: no-op

