from dataclasses import dataclass

import pandas as pd

from .base import Agent, PositionState
from .indicators import rsi, sma, atr


@dataclass
class Senku(Agent):
    name: str = "senku"

    def decide(
        self, hist: pd.DataFrame, pos: PositionState, equity: float, context: dict
    ):
        if len(hist) < 200:
            return ("hold", 0.0, False)

        df = hist.copy()
        df["rsi2"] = rsi(df["close"], 2)
        df["sma200"] = sma(df["close"], 200)
        df["atr14"] = atr(df, 14)

        c, r, sm, a = (
            df["close"].iloc[-1],
            df["rsi2"].iloc[-1],
            df["sma200"].iloc[-1],
            df["atr14"].iloc[-1],
        )

        p_up = min(0.9, max(0.1, 1 - r / 100))
        ev = p_up * (0.02 * c) - (1 - p_up) * (2 * a)

        if not pos.in_long:
            if c > sm and r < 5 and ev > 0:
                return ("buy", p_up, False)
            return ("hold", 0.0, False)

        if r > 60 or c < pos.entry_price - 2 * a or ev <= 0:
            return ("sell", 0.8, False)

        return ("hold", 0.0, False)

