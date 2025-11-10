import argparse
import pandas as pd

from src.python.data.equities_loader import DataSpec, load_ohlcv
from src.python.validation.walkforward import grid_search_sma


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--fast", default="5,10,15", help="Comma-separated")
    p.add_argument("--slow", default="20,30,50", help="Comma-separated")
    args = p.parse_args()

    df = load_ohlcv(DataSpec(symbol=args.symbol.upper(), start=args.start, end=args.end))
    fast_values = [int(x) for x in args.fast.split(",")]
    slow_values = [int(x) for x in args.slow.split(",")]
    res = grid_search_sma(df, fast_values, slow_values)
    print(res.head(10).to_string(index=False))


if __name__ == "__main__":
    main()


