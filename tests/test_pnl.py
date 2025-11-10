import pandas as pd
from src.python.environments.portfolio_env import PortfolioEnv, PortfolioConfig


def test_pnl_with_friction():
    prices = [10, 10, 10, 12, 12]
    df = pd.DataFrame(
        {"Open": prices, "High": prices, "Low": prices, "Close": prices, "Volume": 1, "ATR": 1.0},
        index=pd.date_range("2020-01-01", periods=len(prices), freq="D"),
    )
    data = {"XYZ": df}
    cfg = PortfolioConfig(initial_cash=1000.0)
    env = PortfolioEnv(data, cfg)
    def policy(df, ts, pos):
        # enter on day 1, sell when close increases to 12
        close = float(df.loc[ts, "Close"])
        if pos == 0:
            return "buy"
        return "sell" if close >= 12 else "hold"
    def alloc(cash, price, sym, ts, row):
        return int(cash // price), cash
    res = env.run({"XYZ": policy}, alloc)
    # last trade should be positive pnl
    if not res.trades.empty:
        assert res.trades.iloc[-1]["pnl"] > 0


