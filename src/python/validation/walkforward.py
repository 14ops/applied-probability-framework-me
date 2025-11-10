from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict

import pandas as pd

from src.python.environments.equities_env import EquitiesEnv, EnvConfig, Risk
from src.python.indicators.ta import sma, rsi, atr
from src.python.metrics.metrics import summarize


def purged_time_series_splits(index: pd.DatetimeIndex, n_splits: int, purge_days: int, test_days: int) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    start = index.min()
    end = index.max()
    total_days = (end - start).days
    if n_splits < 1:
        return []
    split_len = max(1, (total_days - test_days) // n_splits)
    splits = []
    for i in range(n_splits):
        train_start = start + pd.Timedelta(days=i * split_len)
        train_end = train_start + pd.Timedelta(days=split_len - 1)
        test_start = train_end + pd.Timedelta(days=purge_days + 1)
        test_end = test_start + pd.Timedelta(days=test_days - 1)
        if test_end > end:
            break
        splits.append((train_start, train_end, test_start, test_end))
    return splits


def evaluate_sma_strategy(df: pd.DataFrame, fast: int, slow: int) -> Dict[str, float]:
    data = df.copy()
    data["SMA_Fast"] = sma(data["Close"], fast)
    data["SMA_Slow"] = sma(data["Close"], slow)
    data["RSI14"] = rsi(data["Close"], 14)
    data["ATR"] = atr(data["High"], data["Low"], data["Close"], 14)
    data = data.dropna()
    def policy(d: pd.DataFrame, ts, pos: int):
        row = d.loc[ts]
        if pos == 0 and row["SMA_Fast"] > row["SMA_Slow"]:
            return "buy"
        if pos == 1 and row["SMA_Fast"] < row["SMA_Slow"]:
            return "sell"
        return "hold"
    env = EquitiesEnv(data, EnvConfig(risk=Risk()))
    res = env.run(policy)
    return summarize(res.equity_curve, res.trades)


def grid_search_sma(df: pd.DataFrame, fast_values: Iterable[int], slow_values: Iterable[int]) -> pd.DataFrame:
    rows = []
    for f in fast_values:
        for s in slow_values:
            if f >= s:
                continue
            m = evaluate_sma_strategy(df, f, s)
            m.update({"fast": f, "slow": s})
            rows.append(m)
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False)


