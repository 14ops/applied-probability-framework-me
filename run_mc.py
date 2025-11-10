import argparse
import numpy as np
import pandas as pd


def bootstrap_returns(returns: pd.Series, trials: int = 1000, horizon: int = None) -> pd.DataFrame:
    r = returns.dropna().values
    n = horizon or len(r)
    sims = []
    for _ in range(trials):
        sample = np.random.choice(r, size=n, replace=True)
        equity = (1 + pd.Series(sample)).cumprod()
        sims.append({"cagr": equity.iloc[-1] ** (252 / n) - 1, "max_dd": (equity / equity.cummax() - 1).min()})
    return pd.DataFrame(sims)


def main(equity_csv: str, trials: int, horizon: int):
    eq = pd.read_csv(equity_csv, index_col=0, parse_dates=True).iloc[:, 0]
    ret = eq.pct_change().dropna()
    res = bootstrap_returns(ret, trials=trials, horizon=horizon)
    print(res.describe(percentiles=[0.05, 0.5, 0.95]).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--equity", required=True, help="Path to equity CSV (col=equity)")
    p.add_argument("--trials", type=int, default=1000)
    p.add_argument("--horizon", type=int, default=252)
    args = p.parse_args()
    main(args.equity, args.trials, args.horizon)


