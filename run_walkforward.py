import argparse
import pandas as pd

from src.python.broker.alpaca_client import get_historical_crypto_df
from src.python.validation.walkforward import purged_time_series_splits, evaluate_sma_strategy


def main(symbol: str, start: str, end: str, n_splits: int, purge_days: int, test_days: int):
    df = get_historical_crypto_df(symbol, start, end, timeframe="1Min")
    if df.empty:
        print("No data")
        return
    idx = df.index
    splits = purged_time_series_splits(idx, n_splits=n_splits, purge_days=purge_days, test_days=test_days)
    rows = []
    for (tr_s, tr_e, te_s, te_e) in splits:
        tr_df = df.loc[(df.index >= tr_s) & (df.index <= tr_e)]
        te_df = df.loc[(df.index >= te_s) & (df.index <= te_e)]
        # simple baseline: fast=10, slow=30
        m_tr = evaluate_sma_strategy(tr_df, 10, 30)
        m_te = evaluate_sma_strategy(te_df, 10, 30)
        rows.append({
            "train_start": tr_s, "train_end": tr_e, "test_start": te_s, "test_end": te_e,
            "train_sharpe": m_tr.get("sharpe", 0.0), "test_sharpe": m_te.get("sharpe", 0.0),
            "test_total_return": m_te.get("total_return", 0.0)
        })
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    print("\nAverage test Sharpe:", res["test_sharpe"].mean())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTC/USD")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--splits", type=int, default=5)
    p.add_argument("--purge", type=int, default=1)
    p.add_argument("--test_days", type=int, default=7)
    args = p.parse_args()
    main(args.symbol, args.start, args.end, args.splits, args.purge, args.test_days)


