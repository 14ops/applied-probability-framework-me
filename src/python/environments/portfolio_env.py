from dataclasses import dataclass, field
from typing import Dict, Optional, Literal, Callable

import numpy as np
import pandas as pd

from .equities_env import Risk

Action = Literal["buy", "sell", "hold"]
PolicyFn = Callable[[pd.DataFrame, pd.Timestamp, int], Action]


@dataclass
class PortfolioConfig:
    initial_cash: float = 100_000.0
    risk: Risk = field(default_factory=Risk)


class PortfolioResult:
    def __init__(self, equity_curve: pd.Series, per_symbol_equity: pd.DataFrame, trades: pd.DataFrame):
        self.equity_curve = equity_curve
        self.per_symbol_equity = per_symbol_equity
        self.trades = trades


class PortfolioEnv:
    def __init__(self, data_by_symbol: Dict[str, pd.DataFrame], config: PortfolioConfig):
        self.data = data_by_symbol
        self.cfg = config
        self.dates = sorted(set().union(*[df.index.tolist() for df in self.data.values()]))
        self.cash = config.initial_cash
        self.positions: Dict[str, Dict[str, Optional[float]]] = {
            sym: {"shares": 0, "entry_price": None} for sym in self.data
        }
        self.trades = []
        self.per_symbol_equity = pd.DataFrame(index=self.dates, columns=list(self.data.keys()), dtype=float)
        self.aggregate_equity = pd.Series(index=self.dates, dtype=float)

    def _apply_friction(self, price: float, buy: bool) -> float:
        slip = price * (self.cfg.risk.slippage_bps / 1e4)
        fee = price * (self.cfg.risk.fee_bps / 1e4)
        return price + slip + fee if buy else price - slip - fee

    def _get_row(self, sym: str, ts):
        df = self.data[sym]
        if ts in df.index:
            return df.loc[ts]
        # forward fill last known if missing
        return df.loc[df.index[df.index.get_loc(ts, method="ffill")]]

    def _mark_to_market(self, ts) -> None:
        total = self.cash
        for sym in self.data:
            row = self._get_row(sym, ts)
            close_ = float(row["Close"])
            shares = int(self.positions[sym]["shares"] or 0)
            sym_equity = shares * close_
            self.per_symbol_equity.loc[ts, sym] = sym_equity
            total += sym_equity
        self.aggregate_equity.loc[ts] = total

    def _stop_takeprofit_check(self, sym: str, ts, action: Action):
        pos = self.positions[sym]
        if pos["shares"] and pos["entry_price"] is not None:
            row = self._get_row(sym, ts)
            atr = float(row.get("ATR", 0.0))
            if atr <= 0:
                return
            stop = pos["entry_price"] - self.cfg.risk.atr_stop * atr
            tp = pos["entry_price"] + self.cfg.risk.atr_tp * atr
            close_price = float(row["Close"])
            if close_price <= stop or close_price >= tp or action == "sell":
                exit_px = self._apply_friction(close_price, buy=False)
                pnl = (exit_px - pos["entry_price"]) * int(pos["shares"])
                self.cash += pnl
                self.trades.append({"symbol": sym, "exit": ts, "pnl": pnl, "price": exit_px})
                pos["shares"] = 0
                pos["entry_price"] = None

    def step(self, ts, actions: Dict[str, Action], alloc_fn):
        for sym, action in actions.items():
            self._stop_takeprofit_check(sym, ts, action)
            if action == "buy" and self.positions[sym]["shares"] == 0:
                row = self._get_row(sym, ts)
                open_px = float(row["Open"])
                buy_px = self._apply_friction(open_px, buy=True)
                shares, alloc = alloc_fn(self.cash, buy_px, sym, ts, row)
                cost = shares * buy_px
                if shares > 0 and cost <= self.cash:
                    self.cash -= cost
                    self.positions[sym]["shares"] = shares
                    self.positions[sym]["entry_price"] = buy_px
                    self.trades.append({"symbol": sym, "entry": ts, "price": buy_px, "shares": shares})
        self._mark_to_market(ts)

    def run(self, policy_map: Dict[str, PolicyFn], alloc_fn) -> PortfolioResult:
        for ts in self.dates:
            actions = {}
            for sym, df in self.data.items():
                # if ts not in df, use ffill row for indicators
                ref_ts = ts if ts in df.index else df.index[df.index.get_loc(ts, method="ffill")]
                position = 1 if self.positions[sym]["shares"] else 0
                actions[sym] = policy_map[sym](df, ref_ts, position)
            self.step(ts, actions, alloc_fn)
        return PortfolioResult(self.aggregate_equity.ffill(), self.per_symbol_equity.ffill(), pd.DataFrame(self.trades))


