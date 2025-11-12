"""
Lightweight wrapper around the Questrade API with yfinance fallback.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Iterable, Optional

import pandas as pd

from src.python.backtesting.data import download_price_history

try:
    from questrade_api import api as questrade_api  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    questrade_api = None  # type: ignore


INTERVAL_MAP = {
    "1m": "OneMinute",
    "5m": "FiveMinutes",
    "15m": "FifteenMinutes",
    "1h": "OneHour",
    "1d": "OneDay",
}


def _to_datetime(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@dataclass
class QuestradeClient:
    token_path: Optional[str] = None
    refresh_token: Optional[str] = None
    paper: bool = True
    fallback_to_yfinance: bool = True
    _symbol_cache: Dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if questrade_api is None:
            self._api = None
        else:
            self._api = questrade_api.Questrade(
                token_yaml=self.token_path,
                refresh_token=self.refresh_token,
                environment="practice" if self.paper else "live",
            )

    def _ensure_api(self) -> None:
        if self._api is None:
            if self.fallback_to_yfinance:
                return
            raise RuntimeError(
                "questrade_api package is not installed. Install it or enable fallback."
            )

    @property
    def is_connected(self) -> bool:
        return self._api is not None

    def _lookup_symbol_id(self, symbol: str) -> Optional[int]:
        if symbol in self._symbol_cache:
            return self._symbol_cache[symbol]

        if self._api is None:
            return None

        matches = self._api.symbols_search(symbol)
        if not matches:
            raise ValueError(f"Symbol '{symbol}' not found in Questrade.")

        symbol_id = matches[0]["symbolId"]
        self._symbol_cache[symbol] = symbol_id
        return symbol_id

    def get_candles(
        self,
        symbol: str,
        start: datetime | str,
        end: datetime | str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Retrieve historical candles either from Questrade or via yfinance fallback.
        """
        qt_interval = INTERVAL_MAP.get(interval, interval)

        self._ensure_api()

        if self._api is None:
            return download_price_history(symbol, start=start, end=end, interval=interval)

        symbol_id = self._lookup_symbol_id(symbol)
        candles = self._api.markets_candles(
            symbol_id,
            start=start,
            end=end,
            interval=qt_interval,
        )

        data = candles.get("candles", [])
        df = pd.DataFrame(data)
        if df.empty:
            return df

        df["timestamp"] = df["start"].apply(_to_datetime)
        df = df.set_index("timestamp")
        df = df.rename(
            columns={
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            }
        )
        return df[["open", "high", "low", "close", "volume"]]

    def place_order(
        self,
        symbol: str,
        quantity: float,
        action: str,
        price: Optional[float] = None,
        order_type: str = "market",
    ) -> Dict[str, object]:
        """
        Submit an order via the Questrade API.
        """
        self._ensure_api()
        if self._api is None:
            raise RuntimeError("Questrade API not available; cannot place live orders.")
        return self._api.place_order(  # type: ignore[attr-defined]
            symbol=symbol,
            quantity=quantity,
            action=action,
            price=price,
            order_type=order_type,
        )

    def stream_candles(
        self,
        symbol: str,
        interval: str,
        on_batch: Callable[[pd.DataFrame], None],
        lookback: int = 500,
        poll_seconds: int = 60,
    ) -> None:
        """
        Poll candles in near-real-time and invoke ``on_batch`` with new data.
        """
        end = datetime.now(timezone.utc)
        start = end - timedelta(minutes=lookback)

        last_timestamp: Optional[datetime] = None

        while True:
            df = self.get_candles(symbol, start=start, end=end, interval=interval)
            if not df.empty:
                new_df = df if last_timestamp is None else df[df.index > last_timestamp]
                if not new_df.empty:
                    on_batch(new_df)
                    last_timestamp = new_df.index[-1]
            time.sleep(poll_seconds)
            end = datetime.now(timezone.utc)

