"""Takeshi - Donchian breakout trend rider."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from .base import Agent, Signal
from .technical import atr, donchian_channels


class TakeshiAgent(Agent):
    """
    Hunts for breakouts using Donchian channels and rides the resulting trends.

    - Break above the upper channel => buy.
    - Break below the lower channel => sell.
    - Confidence scales with distance from the channel relative to ATR.
    """

    def __init__(self, channel_period: int = 20, atr_period: int = 14) -> None:
        min_hist = max(channel_period, atr_period) + 5
        super().__init__(name="takeshi", min_history=min_hist)
        self.channel_period = channel_period
        self.atr_period = atr_period

    def generate_signal(self, data: pd.DataFrame, context: Dict) -> Signal:
        price = data["close"]
        channels = donchian_channels(price, period=self.channel_period)
        upper = channels["upper"].iloc[-1]
        lower = channels["lower"].iloc[-1]
        middle = channels["middle"].iloc[-1]
        latest_price = float(price.iloc[-1])

        atr_series = atr(data[["high", "low", "close"]], period=self.atr_period)
        latest_atr = float(atr_series.iloc[-1])
        atr_denominator = max(latest_atr, 1e-6)

        metadata = {
            "donchian_upper": upper,
            "donchian_lower": lower,
            "donchian_middle": middle,
            "atr": latest_atr,
        }

        if latest_price > upper:
            confidence = min(1.0, (latest_price - upper) / atr_denominator)
            metadata["trigger"] = "upper_breakout"
            return Signal(agent=self.name, action="buy", confidence=confidence, metadata=metadata)

        if latest_price < lower:
            confidence = min(1.0, (lower - latest_price) / atr_denominator)
            metadata["trigger"] = "lower_breakout"
            return Signal(agent=self.name, action="sell", confidence=confidence, metadata=metadata)

        # If price sits between upper/middle with positive momentum, lean bullish.
        if latest_price >= middle:
            distance = (latest_price - middle) / atr_denominator
            confidence = min(0.5, max(0.0, distance))
            metadata["bias"] = "bullish_trend_riding"
            return Signal(agent=self.name, action="buy", confidence=confidence, metadata=metadata)

        distance = (middle - latest_price) / atr_denominator
        confidence = min(0.5, max(0.0, distance))
        metadata["bias"] = "bearish_trend_riding"
        return Signal(agent=self.name, action="sell", confidence=confidence, metadata=metadata)


