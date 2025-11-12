from dataclasses import dataclass

import pandas as pd

from .base import Agent, PositionState
from .indicators import donchian_upper, donchian_lower, atr, avg_volume


@dataclass
class Takeshi(Agent):
    name: str = "takeshi"

    def decide(
        self, hist: pd.DataFrame, pos: PositionState, equity: float, context: dict
    ):
        if len(hist) < 20:
            return ("hold", 0.0, False)

        df = hist.copy()

        up = donchian_upper(df, 20).iloc[-1]
        lo = donchian_lower(df, 10).iloc[-1]
        a = atr(df, 14).iloc[-1]
        c = df["close"].iloc[-1]

        v = df.get("volume", pd.Series([1])).iloc[-1]
        vavg = avg_volume(df, 20).iloc[-1]

        if not pos.in_long and c > up and v > 1.5 * vavg:
            return ("buy", 0.7, False)

        if pos.in_long and (c < lo or c < pos.entry_price - 3 * a):
            return ("sell", 0.8, False)

        return ("hold", 0.0, False)

