import argparse
from typing import Dict, List

from src.python.data.equities_loader import load_many
from src.python.indicators.ta import sma, rsi, atr
from src.python.environments.portfolio_env import PortfolioEnv, PortfolioConfig
from src.python.portfolio.sizing import FixedFraction, CappedKelly
from src.python.strategies.equities import takeshi as s_takeshi, aoi as s_aoi
from src.python.reports.reporting import ReportConfig, save_csvs
from src.python.metrics.metrics import summarize, drawdown_table


def prepare_many(symbols: List[str], start: str, end: str) -> Dict[str, "pd.DataFrame"]:
    data = load_many(symbols, start, end)
    for sym, df in data.items():
        df["SMA_Fast"] = sma(df["Close"], 10)
        df["SMA_Slow"] = sma(df["Close"], 30)
        df["RSI14"] = rsi(df["Close"], 14)
        df["ATR"] = atr(df["High"], df["Low"], df["Close"], 14)
        data[sym] = df.dropna()
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", required=True, help="Comma-separated list, e.g., AAPL,NVDA,MSFT")
    p.add_argument("--strategy", default="takeshi", choices=["takeshi", "aoi"])
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--sizing", default="fixed", choices=["fixed", "kelly"])
    p.add_argument("--fraction", type=float, default=0.1, help="Fixed fraction of cash to allocate")
    p.add_argument("--kelly-edge", type=float, default=0.05)
    p.add_argument("--kelly-odds", type=float, default=1.0)
    p.add_argument("--initial-cash", type=float, default=100000.0)
    p.add_argument("--save", action="store_true")
    args = p.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    data = prepare_many(symbols, args.start, args.end)
    env = PortfolioEnv(data, PortfolioConfig(initial_cash=args.initial_cash))
    policy = s_takeshi.policy if args.strategy == "takeshi" else s_aoi.policy
    policy_map = {sym: policy for sym in symbols}

    if args.sizing == "fixed":
        sizing = FixedFraction(fraction=args.fraction)
        alloc_fn = lambda cash, price, sym, ts, row: sizing.allocate(cash, price)
    else:
        sizing = CappedKelly(edge=args.kelly_edge, odds=args.kelly_odds)
        alloc_fn = lambda cash, price, sym, ts, row: sizing.allocate(cash, price)

    res = env.run(policy_map, alloc_fn)
    metrics = summarize(res.equity_curve, res.trades)
    print(f"Symbols: {','.join(symbols)}  Period: {args.start} → {args.end}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    print("Top drawdowns:")
    print(drawdown_table(res.equity_curve).to_string(index=False))

    if args.save:
        save_csvs(ReportConfig(), res.equity_curve, res.trades, res.per_symbol_equity)


if __name__ == "__main__":
    main()


