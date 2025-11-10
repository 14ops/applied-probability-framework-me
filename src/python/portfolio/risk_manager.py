from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class DrawdownGuard:
    max_drawdown: float = 0.5  # 50%

    def __post_init__(self):
        self.peak_equity: Optional[float] = None

    def update(self, equity: float) -> None:
        if self.peak_equity is None:
            self.peak_equity = equity
        else:
            self.peak_equity = max(self.peak_equity, equity)

    def breached(self, equity: float) -> bool:
        if self.peak_equity is None or self.peak_equity <= 0:
            return False
        dd = 1.0 - (equity / self.peak_equity)
        return dd >= self.max_drawdown


def fixed_fraction_size(cash: float, price: float, fraction: float = 0.1, min_qty: int = 1) -> Tuple[int, float]:
    alloc = max(0.0, min(1.0, fraction)) * cash
    qty = int(alloc // price)
    if qty <= 0 and alloc >= price:
        qty = min_qty
    return qty, qty * price


def vol_target_size(cash: float, price: float, atr: float, target_vol: float = 0.02, lookback_vol: Optional[float] = None) -> Tuple[int, float]:
    """
    Allocate to target approximate percentage volatility using ATR as proxy.
    position value ~= (target_vol / (atr/price)) * cash
    """
    if price <= 0 or atr <= 0:
        return 0, 0.0
    vol = atr / price
    target_value = min(cash, (target_vol / max(1e-8, vol)) * cash)
    qty = int(target_value // price)
    return qty, qty * price

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class FixedFractionSizer:
    fraction: float = 0.1

    def shares(self, cash: float, price: float) -> int:
        if price <= 0:
            return 0
        return int(max(0.0, self.fraction) * cash // price)


@dataclass
class VolTargetSizer:
    target_daily_vol: float = 0.02   # 2% daily vol target
    lookback: int = 60
    min_fraction: float = 0.01
    max_fraction: float = 0.5

    def fraction(self, close: pd.Series) -> float:
        ret = close.pct_change().dropna()
        if len(ret) < max(2, self.lookback // 2):
            return self.min_fraction
        realized = ret.tail(self.lookback).std(ddof=0)
        if realized <= 0 or np.isnan(realized):
            return self.min_fraction
        frac = self.target_daily_vol / float(realized)
        return float(np.clip(frac, self.min_fraction, self.max_fraction))

    def shares(self, cash: float, price: float, close: pd.Series) -> int:
        f = self.fraction(close)
        if price <= 0:
            return 0
        return int(max(0.0, f) * cash // price)


