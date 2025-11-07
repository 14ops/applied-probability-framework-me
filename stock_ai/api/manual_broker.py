"""Simple broker that reads trade history from CSV files.

This makes the dashboard usable with brokers that do not have a first-class
integration yet.  Users can export fills from any brokerage platform and point
this adapter at the CSV to visualise their history.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .broker_clients import BrokerClient, BrokerDefinition, BrokerError, REGISTRY


@dataclass
class ManualBrokerClient(BrokerClient):
    """CSV backed broker client."""

    credentials: Dict[str, Any]

    def __init__(self, credentials: Dict[str, Any]):
        super().__init__(broker_id="manual_csv", display_name="Manual / CSV Import")
        self.credentials = credentials
        self._trade_cache: Optional[pd.DataFrame] = None

    def configure(self, credentials: Dict[str, Any]) -> None:
        self.credentials = credentials
        self._trade_cache = None

    def connect(self) -> None:
        # Nothing to connect to – just validate inputs
        trades_path = self._trade_log_path
        if trades_path and not trades_path.exists():
            raise BrokerError(f"Trades file '{trades_path}' does not exist")

    @property
    def _trade_log_path(self) -> Optional[Path]:
        path = self.credentials.get("trades_csv")
        if not path:
            return None
        return Path(path).expanduser()

    def _load_trades(self) -> pd.DataFrame:
        if self._trade_cache is not None:
            return self._trade_cache
        trades_path = self._trade_log_path
        if not trades_path:
            raise BrokerError("No trades file configured")
        try:
            df = pd.read_csv(trades_path)
        except FileNotFoundError as exc:
            raise BrokerError(f"Trades file '{trades_path}' not found") from exc
        except Exception as exc:  # pylint: disable=broad-except
            raise BrokerError(f"Could not read trades file: {exc}") from exc
        normalised = df.rename(
            columns={
                "symbol": "Symbol",
                "qty": "Quantity",
                "quantity": "Quantity",
                "side": "Side",
                "price": "Price",
                "filled_at": "Filled At",
                "timestamp": "Filled At",
                "order_id": "Order ID",
            }
        )
        if "Filled At" in normalised.columns:
            normalised["Filled At"] = pd.to_datetime(normalised["Filled At"], errors="coerce")
        self._trade_cache = normalised
        return normalised

    def get_account(self) -> Dict[str, Any]:
        return {
            "account_number": self.credentials.get("account_label", "Manual Account"),
            "cash": self.credentials.get("cash", 0.0),
            "equity": self.credentials.get("equity", self.credentials.get("cash", 0.0)),
            "buying_power": self.credentials.get("buying_power", self.credentials.get("cash", 0.0)),
        }

    def list_orders(self, status: str = "all") -> List[Dict[str, Any]]:  # pylint: disable=unused-argument
        try:
            return self._load_trades().to_dict(orient="records")
        except BrokerError:
            return []


def register() -> None:
    REGISTRY.register(
        BrokerDefinition(
            broker_id="manual_csv",
            display_name="Manual / CSV Import",
            description="Upload fills exported from any broker to view them in the dashboard.",
            required_credentials={
                "trades_csv": "Path to a CSV export of your fills",
            },
            optional_credentials={
                "account_label": "Friendly name for the account",
                "cash": "Current cash balance",
                "equity": "Latest equity value",
                "buying_power": "Buying power reported by the broker",
            },
            factory=lambda creds: ManualBrokerClient(creds),
        )
    )


register()
