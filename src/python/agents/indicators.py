import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=1).mean()


def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(n, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(n, min_periods=1).mean()
    rs = gain / (loss.replace(0, 1e-9))
    return 100 - (100 / (1 + rs))


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).rolling(n, min_periods=1).mean()


def donchian_upper(df: pd.DataFrame, n: int = 20) -> pd.Series:
    return df["high"].rolling(n, min_periods=1).max()


def donchian_lower(df: pd.DataFrame, n: int = 20) -> pd.Series:
    return df["low"].rolling(n, min_periods=1).min()


def avg_volume(df: pd.DataFrame, n: int = 20) -> pd.Series:
    if "volume" in df.columns:
        return df["volume"].rolling(n, min_periods=1).mean()
    return pd.Series(index=df.index, data=np.nan)

