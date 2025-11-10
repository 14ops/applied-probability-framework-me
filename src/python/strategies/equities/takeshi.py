from typing import Literal

import pandas as pd

Action = Literal["buy", "sell", "hold"]


def policy(df: pd.DataFrame, ts, position: int) -> Action:
    row = df.loc[ts]
    fast = row["SMA_Fast"]
    slow = row["SMA_Slow"]
    if position == 0 and fast > slow:
        return "buy"
    if position == 1 and fast < slow:
        return "sell"
    return "hold"


