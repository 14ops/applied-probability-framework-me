"""Broker client that wraps :mod:`stock_ai.api.alpaca_interface`."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger

from .alpaca_interface import AlpacaInterface
from .broker_clients import BrokerClient, BrokerDefinition, BrokerError, REGISTRY


class AlpacaBrokerClient(BrokerClient):
    """Adapter around :class:`AlpacaInterface` that matches :class:`BrokerClient`."""

    def __init__(self, credentials: Dict[str, Any]):
        super().__init__(broker_id="alpaca", display_name="Alpaca Markets")
        self._credentials = credentials
        self._api: Optional[AlpacaInterface] = None

    def configure(self, credentials: Dict[str, Any]) -> None:
        self._credentials = credentials

    def connect(self) -> None:
        config_path = self._credentials.get("config_path", "config/secrets.yaml")
        try:
            self._api = AlpacaInterface(config_path=config_path)
        except Exception as exc:  # pylint: disable=broad-except
            raise BrokerError(str(exc)) from exc

    @property
    def api(self) -> AlpacaInterface:
        if self._api is None:
            self.connect()
        assert self._api is not None
        return self._api

    def get_account(self) -> Dict[str, Any]:
        return self.api.get_account()

    def list_positions(self) -> List[Dict[str, Any]]:
        return self.api.get_positions()

    def list_orders(self, status: str = "all") -> List[Dict[str, Any]]:
        try:
            return self.api.get_orders(status=status)
        except AttributeError:
            logger.warning("Installed alpaca interface does not provide get_orders")
            return []

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str = "buy",
        order_type: str = "market",
        limit_price: Optional[float] = None,
    ) -> Optional[str]:
        return self.api.submit_order(
            symbol=symbol,
            qty=int(qty),
            side=side,
            order_type=order_type,
            limit_price=limit_price,
        )

    def cancel_order(self, order_id: str) -> None:
        self.api.cancel_order(order_id)


def register() -> None:
    """Register the Alpaca broker with the global registry."""

    REGISTRY.register(
        BrokerDefinition(
            broker_id="alpaca",
            display_name="Alpaca Markets",
            description="Official Alpaca Markets trading API",
            required_credentials={
                "config_path": "Path to the secrets YAML that stores Alpaca keys",
            },
            factory=lambda creds: AlpacaBrokerClient(creds),
        )
    )


register()
