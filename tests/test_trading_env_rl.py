import pandas as pd
import numpy as np

from src.python.environments.trading_env import TradingEnv, TradingEnvConfig


def make_df(n=200):
    idx = pd.date_range("2024-01-01", periods=n, freq="T")
    price = np.linspace(100, 110, n) + np.random.randn(n) * 0.1
    df = pd.DataFrame({
        "Open": price,
        "High": price + 0.2,
        "Low": price - 0.2,
        "Close": price,
        "Volume": 1
    }, index=idx)
    df.index.name = "Date"
    return df


def test_env_shapes():
    df = make_df(200)
    window = 60
    env = TradingEnv(df, TradingEnvConfig(window=window))
    obs = env.reset()
    assert len(obs) == window + 6
    for _ in range(10):
        obs, r, done, info = env.step(0)
        assert len(obs) == window + 6
        if done:
            break

import pandas as pd
from src.python.environments.trading_env import TradingEnv, TradingEnvConfig


def make_df(n=200, start=100.0):
    idx = pd.date_range("2024-01-01", periods=n, freq="T")
    price = pd.Series(start, index=idx).astype(float)
    price = price * (1.0 + 0.0005).cumprod()
    df = pd.DataFrame({"Open": price, "High": price * 1.001, "Low": price * 0.999, "Close": price, "Volume": 1}, index=idx)
    return df


def test_env_step_shapes():
    df = make_df()
    env = TradingEnv(df, TradingEnvConfig(window=60))
    s = env.reset()
    assert s.ndim == 1
    assert s.size == 60 * 6
    ns, r, done, info = env.step(0)
    assert isinstance(r, float)
    assert "equity" in info

