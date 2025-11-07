"""Broker client abstractions.

Provides a simple registry-driven system that allows the dashboard to talk to
multiple broker backends.  Each broker implementation exposes a unified set of
methods so new providers can be plugged in without modifying the UI layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol


class BrokerError(RuntimeError):
    """Raised when a broker operation fails."""


class SupportsOrders(Protocol):
    """Protocol for broker clients that expose order history."""

    def list_orders(self, status: str = "all") -> List[Dict[str, Any]]:
        ...


@dataclass
class BrokerClient:
    """Abstract broker client.

    Concrete implementations only need to override the methods that their API
    supports.  The base implementation provides safe fallbacks so read-only
    brokers (for example CSV based trade importers) can still be registered.
    """

    broker_id: str
    display_name: str

    def connect(self) -> None:
        """Establish a connection with the broker if applicable."""

    # pylint: disable=unused-argument
    def configure(self, credentials: Dict[str, Any]) -> None:
        """Update internal credential state."""

    def get_account(self) -> Dict[str, Any]:
        """Return account level information when available."""
        return {}

    def list_positions(self) -> List[Dict[str, Any]]:
        """Return open positions."""
        return []

    def list_orders(self, status: str = "all") -> List[Dict[str, Any]]:
        """Return recent orders if the broker supports it."""
        return []

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str = "buy",
        order_type: str = "market",
        limit_price: Optional[float] = None,
    ) -> Optional[str]:
        """Submit an order."""
        raise BrokerError("Order placement is not supported by this broker")

    def cancel_order(self, order_id: str) -> None:
        """Cancel an existing order."""
        raise BrokerError("Cancel is not supported by this broker")


@dataclass
class BrokerDefinition:
    """Metadata describing an available broker implementation."""

    broker_id: str
    display_name: str
    description: str
    required_credentials: Dict[str, str] = field(default_factory=dict)
    optional_credentials: Dict[str, str] = field(default_factory=dict)
    factory: Callable[[Dict[str, Any]], BrokerClient] = lambda _: BrokerClient("base", "Base")


class BrokerRegistry:
    """Registry storing broker definitions."""

    def __init__(self) -> None:
        self._definitions: Dict[str, BrokerDefinition] = {}

    def register(self, definition: BrokerDefinition) -> None:
        if definition.broker_id in self._definitions:
            raise ValueError(f"Broker '{definition.broker_id}' already registered")
        self._definitions[definition.broker_id] = definition

    def list_definitions(self) -> List[BrokerDefinition]:
        return sorted(self._definitions.values(), key=lambda d: d.display_name.lower())

    def get(self, broker_id: str) -> Optional[BrokerDefinition]:
        return self._definitions.get(broker_id)

    def create(self, broker_id: str, credentials: Optional[Dict[str, Any]] = None) -> BrokerClient:
        definition = self.get(broker_id)
        if not definition:
            raise BrokerError(f"Unknown broker '{broker_id}'")
        credentials = credentials or {}
        client = definition.factory(credentials)
        client.configure(credentials)
        return client


REGISTRY = BrokerRegistry()
"""Singleton registry used across the project."""
