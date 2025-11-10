from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def compute_returns(equity_curve: pd.Series) -> pd.Series:
    return equity_curve.pct_change().dropna()


def drawdown_curve(equity_curve: pd.Series) -> pd.Series:
    cummax = equity_curve.cummax()
    dd = equity_curve / cummax - 1.0
    return dd


def drawdown_table(equity_curve: pd.Series, top_n: int = 5) -> pd.DataFrame:
    dd = drawdown_curve(equity_curve)
    in_dd = False
    peaks = []
    troughs = []
    recovers = []
    for i in range(1, len(dd)):
        if not in_dd and dd.iloc[i] < 0:
            in_dd = True
            peaks.append(dd.index[i - 1])
        if in_dd and dd.iloc[i] == 0:
            in_dd = False
            recovers.append(dd.index[i])
    # find troughs within each peak->recover segment
    for i, peak in enumerate(peaks):
        end = recovers[i] if i < len(recovers) else dd.index[-1]
        seg = dd.loc[peak:end]
        if seg.empty:
            troughs.append(peak)
        else:
            troughs.append(seg.idxmin())
    rows = []
    for i, peak in enumerate(peaks):
        end = recovers[i] if i < len(recovers) else dd.index[-1]
        trough = troughs[i]
        dd_val = dd.loc[trough]
        rows.append({"peak": peak, "trough": trough, "recovery": end, "drawdown": dd_val})
    tbl = pd.DataFrame(rows).sort_values("drawdown").head(top_n)
    return tbl


def annualized_return(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    if len(equity_curve) < 2:
        return np.nan
    rets = compute_returns(equity_curve)
    mean = rets.mean()
    return (1 + mean) ** periods_per_year - 1


def sharpe_ratio(equity_curve: pd.Series, periods_per_year: int = 252, rf: float = 0.0) -> float:
    rets = compute_returns(equity_curve)
    excess = rets - rf / periods_per_year
    if len(excess) < 2 or excess.std(ddof=0) == 0:
        return np.nan
    return (excess.mean() / excess.std(ddof=0)) * np.sqrt(periods_per_year)


def sortino_ratio(equity_curve: pd.Series, periods_per_year: int = 252, rf: float = 0.0) -> float:
    rets = compute_returns(equity_curve)
    downside = rets[rets < 0]
    if len(downside) == 0 or downside.std(ddof=0) == 0:
        return np.nan
    mean_excess = rets.mean() - rf / periods_per_year
    return (mean_excess / downside.std(ddof=0)) * np.sqrt(periods_per_year)


def calmar_ratio(equity_curve: pd.Series) -> float:
    ann = annualized_return(equity_curve)
    dd = drawdown_curve(equity_curve).min()
    return np.nan if dd == 0 else ann / abs(dd)


def turnover(trades: pd.DataFrame, equity_curve: pd.Series) -> float:
    if trades.empty:
        return 0.0
    num_trades = len(trades)
    days = (equity_curve.index[-1] - equity_curve.index[0]).days or 1
    return num_trades / days


def summarize(equity_curve: pd.Series, trades: pd.DataFrame) -> Dict[str, float]:
    return {
        "total_return": equity_curve.iloc[-1] / equity_curve.iloc[0] - 1,
        "ann_return": annualized_return(equity_curve),
        "sharpe": sharpe_ratio(equity_curve),
        "sortino": sortino_ratio(equity_curve),
        "calmar": calmar_ratio(equity_curve),
        "max_drawdown": drawdown_curve(equity_curve).min(),
        "turnover": turnover(trades, equity_curve),
        "num_trades": float(len(trades)),
    }


