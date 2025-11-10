from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal

import numpy as np
import pandas as pd

Action = Literal["buy", "sell", "hold"]


PolicyFn = Callable[[pd.DataFrame, pd.Timestamp, int], Action]


@dataclass
class EnsembleAgent:
    strategies: Dict[str, PolicyFn]
    tau: float = 0.2  # temperature for softmax weighting
    decay: float = 0.9  # EMA decay for performance
    perf: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        for name in self.strategies:
            self.perf.setdefault(name, 0.0)

    def decide(self, df: pd.DataFrame, ts: pd.Timestamp, position: int) -> Action:
        votes: Dict[Action, float] = {"buy": 0.0, "sell": 0.0, "hold": 0.0}
        weights = self._weights()
        for name, strat in self.strategies.items():
            w = weights.get(name, 0.0)
            try:
                a = strat(df, ts, position)
            except Exception:
                a = "hold"
            votes[a] += w
        # choose highest weighted action
        return max(votes.items(), key=lambda kv: kv[1])[0]

    def update_performance(self, realized_pnl_by_strategy: Dict[str, float]) -> None:
        for name, delta in realized_pnl_by_strategy.items():
            prev = self.perf.get(name, 0.0)
            self.perf[name] = self.decay * prev + (1.0 - self.decay) * delta

    def _weights(self) -> Dict[str, float]:
        scores = np.array([self.perf.get(n, 0.0) for n in self.strategies.keys()], dtype=float)
        # softmax with temperature
        if self.tau <= 0:
            self.tau = 0.2
        exps = np.exp(scores / self.tau)
        if exps.sum() == 0:
            w = np.ones_like(exps) / len(exps)
        else:
            w = exps / exps.sum()
        return {name: float(wi) for name, wi in zip(self.strategies.keys(), w)}


