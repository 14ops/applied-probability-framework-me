"""
Utilities for loading historical market data for backtests.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Yahoo Finance column names to lowercase OHLCV."""
    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "close",
        "Volume": "volume",
    }
    result = df.rename(columns=rename_map)
    # Ensure we keep only the expected columns
    columns = [c for c in ["open", "high", "low", "close", "volume"] if c in result]
    return result[columns].dropna(how="all")


def download_price_history(
    symbol: str,
    start: Optional[str | datetime] = None,
    end: Optional[str | datetime] = None,
    interval: str = "1d",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data via Yahoo Finance.

    Args:
        symbol: Ticker symbol (e.g. "AAPL").
        start: Start date (inclusive).
        end: End date (exclusive). Defaults to today.
        interval: Data interval supported by yfinance (e.g. "1d", "1h").
        auto_adjust: Adjust prices for splits/dividends.

    Returns:
        A DataFrame indexed by timestamp with columns open/high/low/close/volume.
    """
    raw = yf.download(
        symbol,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
    )

    if raw.empty:
        return raw

    normalized = _normalize_columns(raw)
    normalized.index.name = "timestamp"
    return normalized

