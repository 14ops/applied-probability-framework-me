from dataclasses import dataclass
from typing import Tuple, Dict

import pandas as pd

Action = Tuple[str, float, bool]  # (buy/sell/hold, confidence [0..1], veto)


@dataclass
class PositionState:
    in_long: bool = False
    entry_price: float = 0.0
    size: float = 0.0


class Agent:
    name: str = "base"

    def decide(
        self, hist: pd.DataFrame, pos: PositionState, equity: float, context: Dict
    ) -> Action:
        return ("hold", 0.0, False)

