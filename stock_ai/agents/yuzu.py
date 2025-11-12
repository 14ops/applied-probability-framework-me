"""Yuzu - Volatility guardian and risk sentinel."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from .base import Agent, Signal
from .technical import atr


class YuzuAgent(Agent):
    """
    Monitors volatility (ATR) and portfolio drawdowns. Can veto trades when risk
    stretches beyond acceptable levels.

    Yuzu does not directly propose buy/sell directions. Instead it emits a hold
    action with a confidence score indicating how comfortable the risk profile is.
    Metadata includes ``veto`` flag when trading should be halted.
    """

    def __init__(
        self,
        atr_period: int = 14,
        atr_threshold: float = 0.04,
        drawdown_buffer: float = 0.02,
    ) -> None:
        super().__init__(name="yuzu", min_history=atr_period + 5)
        self.atr_period = atr_period
        self.atr_threshold = atr_threshold
        self.drawdown_buffer = drawdown_buffer

    def generate_signal(self, data: pd.DataFrame, context: Dict) -> Signal:
        atr_series = atr(data[["high", "low", "close"]], period=self.atr_period)
        latest_atr = float(atr_series.iloc[-1])
        latest_price = float(data["close"].iloc[-1])
        atr_ratio = latest_atr / max(latest_price, 1e-6)

        current_drawdown = float(context.get("current_drawdown", 0.0))
        max_drawdown = float(context.get("max_drawdown", 0.20))
        daily_pnl = float(context.get("daily_pnl", 0.0))
        daily_stop = float(context.get("daily_stop", 0.03))

        nearing_drawdown_limit = current_drawdown >= (max_drawdown - self.drawdown_buffer)
        nearing_daily_stop = daily_pnl <= -max(daily_stop - self.drawdown_buffer, 0.0)
        high_volatility = atr_ratio >= self.atr_threshold

        risk_flags = {
            "atr": latest_atr,
            "atr_ratio": atr_ratio,
            "current_drawdown": current_drawdown,
            "max_drawdown": max_drawdown,
            "daily_pnl": daily_pnl,
            "daily_stop": daily_stop,
            "nearing_drawdown_limit": nearing_drawdown_limit,
            "nearing_daily_stop": nearing_daily_stop,
            "high_volatility": high_volatility,
        }

        if high_volatility or nearing_drawdown_limit or nearing_daily_stop:
            confidence = min(
                1.0,
                sum(
                    float(flag)
                    for flag in [high_volatility, nearing_drawdown_limit, nearing_daily_stop]
                )
                / 3.0,
            )
            risk_flags["veto"] = True
            return Signal(agent=self.name, action="hold", confidence=confidence, metadata=risk_flags)

        confidence = max(0.0, 1.0 - atr_ratio / self.atr_threshold)
        risk_flags["veto"] = False
        return Signal(agent=self.name, action="hold", confidence=confidence, metadata=risk_flags)


