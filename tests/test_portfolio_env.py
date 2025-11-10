import pandas as pd
from datetime import datetime, timedelta

from src.python.environments.portfolio_env import PortfolioEnv, PortfolioConfig


def make_df(prices):
    dates = pd.date_range("2020-01-01", periods=len(prices), freq="D")
    df = pd.DataFrame({"Open": prices, "High": prices, "Low": prices, "Close": prices, "Volume": 1}, index=dates)
    df["ATR"] = 1.0
    return df


def test_basic_fill_and_pnl():
    # rising prices ensure profit on buy
    data = {"AAA": make_df([10, 11, 12, 13, 14])}
    env = PortfolioEnv(data, PortfolioConfig(initial_cash=1000.0))
    # buy on first day, hold
    def policy(df, ts, pos):
        return "buy" if pos == 0 else "hold"
    def alloc(cash, price, sym, ts, row):
        shares = int(cash // price)
        return shares, shares * price
    res = env.run({"AAA": policy}, alloc)
    # Equity should end above start due to rising prices
    assert res.equity_curve.iloc[-1] > res.equity_curve.iloc[0]


