"""Metrics collection and trade logging for the ensemble."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Set

import numpy as np


@dataclass
class TradeRecord:
    timestamp: str
    symbol: str
    action: str
    price: float
    quantity: float
    pnl: float
    supporting_agents: Set[str] = field(default_factory=set)

    @property
    def cost(self) -> float:
        return abs(self.price * self.quantity)

    @property
    def return_pct(self) -> float:
        base = self.cost or 1.0
        return self.pnl / base


class MetricsTracker:
    """Keeps rolling trade history and exposes performance statistics."""

    def __init__(self, max_history: int = 1000, sharpe_window: int = 100) -> None:
        self.trades: Deque[TradeRecord] = deque(maxlen=max_history)
        self.sharpe_window = sharpe_window

    def log_trade(
        self,
        timestamp: str,
        symbol: str,
        action: str,
        price: float,
        quantity: float,
        pnl: float,
        supporting_agents: Optional[Iterable[str]] = None,
    ) -> TradeRecord:
        record = TradeRecord(
            timestamp=timestamp,
            symbol=symbol,
            action=action,
            price=price,
            quantity=quantity,
            pnl=pnl,
            supporting_agents=set(supporting_agents or []),
        )
        self.trades.append(record)
        return record

    def _recent_trades(self) -> Sequence[TradeRecord]:
        if len(self.trades) <= self.sharpe_window:
            return list(self.trades)
        return list(self.trades)[-self.sharpe_window :]

    def sharpe_by_agent(self) -> Dict[str, float]:
        recent = self._recent_trades()
        returns: Dict[str, List[float]] = defaultdict(list)

        for trade in recent:
            for agent in trade.supporting_agents:
                returns[agent].append(trade.return_pct)

        sharpe: Dict[str, float] = {}
        for agent, series in returns.items():
            if len(series) < 2:
                sharpe[agent] = 0.0
                continue
            arr = np.array(series, dtype=float)
            mean = arr.mean()
            std = arr.std(ddof=1)
            if std == 0:
                sharpe[agent] = 0.0
                continue
            sharpe_ratio = mean / std * np.sqrt(len(arr))
            sharpe[agent] = float(sharpe_ratio)
        return sharpe

    def win_rate_by_agent(self) -> Dict[str, float]:
        recent = self._recent_trades()
        wins: Dict[str, int] = defaultdict(int)
        totals: Dict[str, int] = defaultdict(int)

        for trade in recent:
            won = trade.pnl > 0
            for agent in trade.supporting_agents:
                totals[agent] += 1
                if won:
                    wins[agent] += 1

        return {
            agent: (wins[agent] / totals[agent]) if totals[agent] else 0.0
            for agent in totals
        }


