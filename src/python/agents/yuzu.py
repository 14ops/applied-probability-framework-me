from dataclasses import dataclass

import pandas as pd

from .base import Agent, PositionState


@dataclass
class Yuzu(Agent):
    name: str = "yuzu"

    daily_stop: float = 0.03
    max_drawdown: float = 0.20
    max_exposure: float = 0.60

    def decide(
        self, hist: pd.DataFrame, pos: PositionState, equity: float, context: dict
    ):
        dd = context.get("drawdown", 0)
        day_pnl = context.get("day_pnl", 0)
        exposure = context.get("exposure", 0)

        if (
            day_pnl <= -self.daily_stop * equity
            or dd >= self.max_drawdown
            or exposure >= self.max_exposure
        ):
            return ("hold", 1.0, True)

        return ("hold", 0.0, False)

