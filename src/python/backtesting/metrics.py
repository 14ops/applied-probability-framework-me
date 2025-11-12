"""
Performance metric helpers for backtest results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


@dataclass
class TradeRecord:
    """Serializable representation of a trade event."""

    timestamp: pd.Timestamp
    symbol: str
    agent: str
    action: str
    price: float
    quantity: float
    value: float
    fees: float
    pnl: float
    equity_after: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "agent": self.agent,
            "action": self.action,
            "price": self.price,
            "quantity": self.quantity,
            "value": self.value,
            "fees": self.fees,
            "pnl": self.pnl,
            "equity_after": self.equity_after,
        }


def _safe_sharpe(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    if returns.empty:
        return 0.0
    excess = returns - risk_free_rate / TRADING_DAYS_PER_YEAR
    std = excess.std(ddof=0)
    if std == 0 or np.isnan(std):
        return 0.0
    return np.sqrt(TRADING_DAYS_PER_YEAR) * excess.mean() / std


def _safe_sortino(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    if returns.empty:
        return 0.0
    excess = returns - risk_free_rate / TRADING_DAYS_PER_YEAR
    downside = excess[excess < 0]
    denom = np.sqrt((downside.pow(2).mean()))
    if denom == 0 or np.isnan(denom):
        return 0.0
    return np.sqrt(TRADING_DAYS_PER_YEAR) * excess.mean() / denom


def max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    running_max = equity_curve.cummax()
    drawdowns = (equity_curve - running_max) / running_max.replace(0, np.nan)
    return drawdowns.min()


def trade_win_rate(trades: Iterable[TradeRecord]) -> float:
    closes = [t for t in trades if t.action.lower() == "sell"]
    if not closes:
        return 0.0
    winners = sum(1 for t in closes if t.pnl > 0)
    return winners / len(closes)


def profit_factor(trades: Iterable[TradeRecord]) -> float:
    gains = 0.0
    losses = 0.0
    for t in trades:
        if t.action.lower() != "sell":
            continue
        if t.pnl >= 0:
            gains += t.pnl
        else:
            losses += t.pnl
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / abs(losses)


def aggregate_metrics(
    equity_curve: pd.Series,
    trades: List[TradeRecord],
    starting_cash: float,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """
    Compute the standard performance statistics for a backtest.
    """
    if equity_curve.empty:
        return {}

    final_equity = float(equity_curve.iloc[-1])
    total_return = (final_equity - starting_cash) / starting_cash
    returns = equity_curve.pct_change().dropna()

    drawdown_value = max_drawdown(equity_curve)

    metrics = {
        "starting_cash": starting_cash,
        "final_equity": final_equity,
        "total_return": total_return,
        "total_return_pct": total_return * 100.0,
        "sharpe_ratio": _safe_sharpe(returns, risk_free_rate),
        "sortino_ratio": _safe_sortino(returns, risk_free_rate),
        "max_drawdown": drawdown_value,
        "max_drawdown_pct": drawdown_value * 100.0,
        "num_trades": len([t for t in trades if t.action.lower() == "buy"]),
        "completed_trades": len([t for t in trades if t.action.lower() == "sell"]),
        "win_rate": trade_win_rate(trades),
        "win_rate_pct": trade_win_rate(trades) * 100.0,
        "profit_factor": profit_factor(trades),
    }

    years = len(equity_curve) / TRADING_DAYS_PER_YEAR
    if years > 0 and final_equity > 0 and starting_cash > 0:
        cagr = (final_equity / starting_cash) ** (1 / years) - 1
    else:
        cagr = 0.0
    metrics["cagr"] = cagr
    metrics["cagr_pct"] = cagr * 100.0

    return metrics

