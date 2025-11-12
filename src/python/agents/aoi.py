from dataclasses import dataclass

import pandas as pd

from .base import Agent, PositionState
from .indicators import sma


@dataclass
class Aoi(Agent):
    name: str = "aoi"

    def decide(
        self, hist: pd.DataFrame, pos: PositionState, equity: float, context: dict
    ):
        if len(hist) < 200:
            return ("hold", 0.0, False)

        df = hist.copy()
        fast, slow = sma(df["close"], 50), sma(df["close"], 200)

        if (
            not pos.in_long
            and fast.iloc[-2] <= slow.iloc[-2]
            and fast.iloc[-1] > slow.iloc[-1]
        ):
            return ("buy", 0.6, False)

        if (
            pos.in_long
            and fast.iloc[-2] >= slow.iloc[-2]
            and fast.iloc[-1] < slow.iloc[-1]
        ):
            return ("sell", 0.8, False)

        return ("hold", 0.0, False)

