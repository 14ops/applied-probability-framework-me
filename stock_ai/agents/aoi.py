"""Aoi - Dual SMA trend confirmation agent."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from .base import Agent, Signal
from .technical import ema, sma


class AoiAgent(Agent):
    """
    Uses dual moving averages to filter trend direction and confirm momentum moves.

    - Requires fast SMA above slow SMA to consider longs (and vice versa for shorts).
    - Confidence increases when price extends away from the baseline exponentially smoothed mean.
    """

    def __init__(
        self,
        fast_period: int = 21,
        slow_period: int = 55,
        baseline_period: int = 34,
        extension_multiple: float = 1.5,
    ) -> None:
        min_hist = max(fast_period, slow_period, baseline_period) + 5
        super().__init__(name="aoi", min_history=min_hist)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.baseline_period = baseline_period
        self.extension_multiple = extension_multiple

    def generate_signal(self, data: pd.DataFrame, context: Dict) -> Signal:
        closes = data["close"]
        fast = sma(closes, self.fast_period).iloc[-1]
        slow = sma(closes, self.slow_period).iloc[-1]
        baseline = ema(closes, self.baseline_period).iloc[-1]
        latest_price = float(closes.iloc[-1])

        deviation = latest_price - baseline
        baseline_vol = (
            closes.rolling(window=self.baseline_period, min_periods=self.baseline_period)
            .std()
            .iloc[-1]
        )
        baseline_vol = float(baseline_vol) if pd.notnull(baseline_vol) else 0.0
        denom = max(baseline_vol, 1e-6)

        trend_up = fast >= slow
        trend_down = fast <= slow
        extension_ratio = deviation / denom

        metadata = {
            "fast_sma": fast,
            "slow_sma": slow,
            "baseline": baseline,
            "extension_ratio": extension_ratio,
        }

        if trend_up and latest_price > baseline:
            confidence = min(1.0, max(0.0, extension_ratio / self.extension_multiple))
            metadata["trend"] = "up"
            return Signal(agent=self.name, action="buy", confidence=confidence, metadata=metadata)

        if trend_down and latest_price < baseline:
            confidence = min(1.0, max(0.0, -extension_ratio / self.extension_multiple))
            metadata["trend"] = "down"
            return Signal(agent=self.name, action="sell", confidence=confidence, metadata=metadata)

        return Signal(agent=self.name, action="hold", confidence=0.0, metadata=metadata)


