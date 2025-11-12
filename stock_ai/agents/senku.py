"""Senku - RSI(2) mean reversion specialist."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from .base import Agent, Signal
from .technical import rsi, sma


class SenkuAgent(Agent):
    """
    Buys oversold bounces when the long-term trend is positive.

    - RSI(2) below ``oversold`` triggers interest.
    - Only acts when 50-period SMA >= 200-period SMA (uptrend filter).
    - Confidence scales with how deep the RSI is below the oversold threshold.
    """

    def __init__(
        self,
        oversold: float = 10.0,
        overbought: float = 90.0,
        trend_fast: int = 50,
        trend_slow: int = 200,
    ) -> None:
        super().__init__(name="senku", min_history=max(trend_slow, 2) + 5)
        self.oversold = oversold
        self.overbought = overbought
        self.trend_fast = trend_fast
        self.trend_slow = trend_slow

    def generate_signal(self, data: pd.DataFrame, context: Dict) -> Signal:
        closes = data["close"]
        rsi_series = rsi(closes, period=2)
        latest_rsi = float(rsi_series.iloc[-1])

        fast = sma(closes, self.trend_fast).iloc[-1]
        slow = sma(closes, self.trend_slow).iloc[-1]
        trend_positive = fast >= slow

        metadata = {
            "rsi_2": latest_rsi,
            "trend_fast": fast,
            "trend_slow": slow,
            "trend_positive": trend_positive,
        }

        if not trend_positive:
            return Signal(agent=self.name, action="hold", confidence=0.0, metadata=metadata)

        if latest_rsi <= self.oversold:
            confidence = min(1.0, (self.oversold - latest_rsi) / self.oversold)
            metadata["trigger"] = "oversold_bounce"
            return Signal(agent=self.name, action="buy", confidence=confidence, metadata=metadata)

        if latest_rsi >= self.overbought:
            # Take profits aggressively when the bounce runs hot.
            confidence = min(1.0, (latest_rsi - self.overbought) / (100 - self.overbought))
            metadata["trigger"] = "overbought_exit"
            return Signal(agent=self.name, action="sell", confidence=confidence, metadata=metadata)

        return Signal(agent=self.name, action="hold", confidence=0.0, metadata=metadata)


