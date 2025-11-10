from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None


@dataclass
class DataSpec:
    symbol: str
    start: str
    end: str
    interval: str = "1d"
    cache_dir: str = "data"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure standard OHLCV columns and a DatetimeIndex named "Date"
    ren = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adj close": "Adj Close",
        "volume": "Volume",
    }
    # If dataframe has MultiIndex columns, select the level that contains OHLC names
    if isinstance(df.columns, pd.MultiIndex):
        ohlc = {"open", "high", "low", "close", "adj close", "volume"}
        chosen = 0
        for i in range(df.columns.nlevels):
            level_vals = {str(v).lower() for v in df.columns.get_level_values(i)}
            if ohlc & level_vals:
                chosen = i
                break
        df = df.copy()
        df.columns = df.columns.get_level_values(chosen)
    # Normalize column names
    cols = []
    for c in df.columns:
        name_str = str(c)
        cols.append(ren.get(name_str.lower(), name_str))
    df.columns = cols

    # Ensure Date index
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")

    if isinstance(df.index, pd.DatetimeIndex):
        df.index.name = "Date"

    needed = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in df.columns]
    # Fallbacks if some columns missing
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
        missing = [c for c in needed if c not in df.columns]
    if "Open" not in df.columns and "Close" in df.columns:
        df["Open"] = df["Close"]
        missing = [c for c in needed if c not in df.columns]
    if "High" not in df.columns and "Close" in df.columns:
        df["High"] = df["Close"]
        missing = [c for c in needed if c not in df.columns]
    if "Low" not in df.columns and "Close" in df.columns:
        df["Low"] = df["Close"]
        missing = [c for c in needed if c not in df.columns]
    if "Volume" not in df.columns:
        df["Volume"] = 0
        missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df.sort_index().dropna()


def load_ohlcv(spec: DataSpec, csv_path: Optional[str] = None) -> pd.DataFrame:
    cache = Path(spec.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    cache_file = cache / f"{spec.symbol}_{spec.interval}_{spec.start}_{spec.end}.csv"

    if csv_path:
        df = pd.read_csv(csv_path)
        return _normalize(df)

    if cache_file.exists():
        return _normalize(pd.read_csv(cache_file))

    if yf is None:
        raise RuntimeError("yfinance not available and no csv_path provided")

    df = yf.download(
        spec.symbol,
        start=spec.start,
        end=spec.end,
        interval=spec.interval,
        auto_adjust=False,
        progress=False,
    )
    df.index.name = "Date"
    df.to_csv(cache_file)
    return _normalize(df.reset_index())

def load_many(symbols: List[str], start: str, end: str, interval: str = "1d", cache_dir: str = "data") -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        df = load_ohlcv(DataSpec(symbol=symbol, start=start, end=end, interval=interval, cache_dir=cache_dir))
        out[symbol] = df
    return out


