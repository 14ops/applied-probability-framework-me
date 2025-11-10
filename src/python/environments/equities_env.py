from dataclasses import dataclass, field
from typing import Optional, Literal

import pandas as pd
import numpy as np

Action = Literal["buy", "sell", "hold"]


@dataclass
class Risk:
    fee_bps: float = 1.0        # 1 bps = 0.01%
    slippage_bps: float = 2.0
    atr_stop: float = 2.0       # stop = entry - atr_stop * ATR
    atr_tp: float = 3.0         # take profit = entry + atr_tp * ATR


@dataclass
class EnvConfig:
    initial_cash: float = 10_000.0
    risk: Risk = field(default_factory=Risk)


class BacktestResult:
    def __init__(self, equity_curve: pd.Series, trades: pd.DataFrame):
        self.equity_curve = equity_curve
        self.trades = trades
        self.total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
        # Sharpe with daily data
        rets = equity_curve.pct_change().dropna()
        self.sharpe = (rets.mean() / rets.std(ddof=0)) * np.sqrt(252) if len(rets) > 2 else np.nan
        self.win_rate = (trades["pnl"] > 0).mean() if not trades.empty else np.nan


class EquitiesEnv:
    def __init__(self, df: pd.DataFrame, config: EnvConfig):
        self.df = df.copy()
        self.cfg = config
        self.position = 0
        self.cash = config.initial_cash
        self.entry_price: Optional[float] = None
        self.equity_curve = pd.Series(index=self.df.index, dtype=float)
        self.trades = []

    def _apply_friction(self, price: float, buy: bool) -> float:
        slip = price * (self.cfg.risk.slippage_bps / 1e4)
        fee = price * (self.cfg.risk.fee_bps / 1e4)
        return price + slip + fee if buy else price - slip - fee

    def step(self, ts, action: Action):
        row = self.df.loc[ts]
        price = float(row["Open"])  # trade at next open
        atr = float(row.get("ATR", 0.0))

        # manage stop or tp if in position
        if self.position == 1 and self.entry_price is not None and atr > 0:
            stop = self.entry_price - self.cfg.risk.atr_stop * atr
            tp = self.entry_price + self.cfg.risk.atr_tp * atr
            # mark-to-market on close for stop/tp checks
            close_price = float(row["Close"])
            if close_price <= stop or action == "sell":
                exit_px = self._apply_friction(close_price, buy=False)
                pnl = (exit_px - self.entry_price)
                self.cash += pnl
                self.trades.append({"exit": ts, "pnl": pnl})
                self.position = 0
                self.entry_price = None
            elif close_price >= tp:
                exit_px = self._apply_friction(close_price, buy=False)
                pnl = (exit_px - self.entry_price)
                self.cash += pnl
                self.trades.append({"exit": ts, "pnl": pnl})
                self.position = 0
                self.entry_price = None

        # entries
        if self.position == 0 and action == "buy":
            buy_px = self._apply_friction(price, buy=True)
            self.entry_price = buy_px
            self.position = 1
            self.trades.append({"entry": ts, "price": buy_px})

        # equity mark
        mtm = float(self.df.loc[ts, "Close"])
        if self.position == 1 and self.entry_price is not None:
            equity = self.cash + (mtm - self.entry_price)
        else:
            equity = self.cash
        self.equity_curve.loc[ts] = equity

    def run(self, policy_fn) -> BacktestResult:
        for ts in self.df.index:
            action = policy_fn(self.df, ts, self.position)
            self.step(ts, action)
        trades_df = pd.DataFrame(self.trades)
        return BacktestResult(self.equity_curve.ffill(), trades_df)


