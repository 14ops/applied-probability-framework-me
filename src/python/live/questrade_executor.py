"""
Execution helper that can either paper trade against Questrade or simulate fills.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from loguru import logger

from .questrade_client import QuestradeClient


@dataclass
class ExecutionResponse:
    status: str
    filled_qty: float
    avg_price: float
    metadata: Dict[str, object]


class QuestradeExecutor:
    def __init__(
        self,
        client: Optional[QuestradeClient] = None,
        dry_run: bool = True,
        slippage_bps: float = 5.0,
    ) -> None:
        self.client = client
        self.dry_run = dry_run
        self.slippage_bps = slippage_bps

    def _simulate_fill(self, action: str, quantity: float, price: float) -> ExecutionResponse:
        slip = price * (self.slippage_bps / 10_000.0)
        fill_price = price + slip if action.lower() == "buy" else price - slip
        logger.debug(
            "Simulated Questrade fill",
            action=action,
            quantity=quantity,
            requested_price=price,
            fill_price=fill_price,
        )
        return ExecutionResponse(
            status="filled",
            filled_qty=quantity,
            avg_price=fill_price,
            metadata={"mode": "dry_run"},
        )

    def submit_order(
        self,
        symbol: str,
        action: str,
        quantity: float,
        price: Optional[float] = None,
        order_type: str = "market",
    ) -> ExecutionResponse:
        """
        Submit an order. When ``dry_run`` is True or the client is unavailable,
        simulate the fill locally. Otherwise delegates to Questrade's API.
        """
        if quantity <= 0:
            raise ValueError("Quantity must be positive")

        if self.dry_run or self.client is None or not self.client.is_connected:
            if price is None:
                raise ValueError("Price must be provided in dry_run mode")
            return self._simulate_fill(action, quantity, price)

        # Delegate to Questrade API
        qt_order_type = order_type.capitalize()
        response = self.client.place_order(
            symbol=symbol,
            quantity=quantity,
            action=action.lower(),
            price=price,
            order_type=qt_order_type,
        )
        logger.debug("Questrade order response", response=response)
        filled_qty = response.get("filledQuantity", 0.0)
        avg_price = response.get("averagePrice", price or 0.0)
        return ExecutionResponse(
            status=response.get("status", "submitted"),
            filled_qty=filled_qty,
            avg_price=avg_price,
            metadata=response,
        )

