from typing import Literal

import pandas as pd

Action = Literal["buy", "sell", "hold"]


def policy(df: pd.DataFrame, ts, position: int) -> Action:
    rsi = df.loc[ts, "RSI14"]
    if position == 0 and rsi < 30:
        return "buy"
    if position == 1 and rsi > 55:
        return "sell"
    return "hold"


