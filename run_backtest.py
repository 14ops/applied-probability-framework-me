import argparse
import pandas as pd

from src.python.data.equities_loader import DataSpec, load_ohlcv
from src.python.indicators.ta import sma, rsi, atr
from src.python.environments.equities_env import EnvConfig, Risk, EquitiesEnv
from src.python.strategies.equities import takeshi, aoi


def prepare(symbol: str, start: str, end: str):
    df = load_ohlcv(DataSpec(symbol=symbol, start=start, end=end))
    df["SMA_Fast"] = sma(df["Close"], 10)
    df["SMA_Slow"] = sma(df["Close"], 30)
    df["RSI14"] = rsi(df["Close"], 14)
    df["ATR"] = atr(df["High"], df["Low"], df["Close"], 14)
    return df.dropna()


def run(strategy_name: str, symbol: str, start: str, end: str):
    df = prepare(symbol, start, end)
    env = EquitiesEnv(df, EnvConfig(risk=Risk(fee_bps=1.0, slippage_bps=2.0, atr_stop=2.0, atr_tp=3.0)))
    strat = takeshi.policy if strategy_name == "takeshi" else aoi.policy
    res = env.run(strat)
    print(f"Strategy: {strategy_name}")
    print(f"Symbol: {symbol}  Period: {start} → {end}")
    print(f"Total return: {res.total_return:.2%}")
    print(f"Sharpe (daily): {res.sharpe:.2f}")
    print(f"Win rate: {res.win_rate:.2%}")
    print(f"Trades: {len(res.trades)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", default="takeshi", choices=["takeshi", "aoi"])
    p.add_argument("--symbol", default="AAPL")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2024-12-31")
    args = p.parse_args()
    run(args.strategy, args.symbol, args.start, args.end)
