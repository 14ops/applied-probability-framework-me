"""Common broker interface definitions.

Provides base classes and helper dataclasses for broker connections so that
multiple providers (Alpaca, Interactive Brokers, paper simulators, etc.) can be
used interchangeably inside the framework.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


@dataclass
class OrderResult:
    """Normalized representation of an order response."""

    broker: str
    order_id: Optional[str]
    status: str
    submitted_at: Optional[str] = None
    filled_qty: Optional[float] = None
    avg_fill_price: Optional[float] = None
    raw: Optional[Dict] = None


class BrokerInterface(ABC):
    """Minimal interface that all broker connectors must implement."""

    name: str

    @abstractmethod
    def get_account(self) -> Dict:
        """Return the latest account information."""

    @abstractmethod
    def get_positions(self) -> List[Dict]:
        """Return the list of open positions."""

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Optional helper for retrieving a single position."""
        positions = self.get_positions()
        for pos in positions:
            if pos.get("symbol") == symbol:
                return pos
        return None

    @abstractmethod
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Return the current market price for a symbol."""

    @abstractmethod
    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str = "buy",
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        extended_hours: bool = False,
    ) -> OrderResult:
        """Submit an order to the broker."""

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order if supported by the broker."""
        raise NotImplementedError("cancel_order not implemented for this broker")

    def get_clock(self) -> Optional[Dict]:
        """Return trading session clock data when available."""
        return None

    def list_orders(self, status: str = "all") -> Iterable[Dict]:
        """Return orders matching a status if supported."""
        return []

