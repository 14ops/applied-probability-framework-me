"""Order execution layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from loguru import logger

from ..api.base_broker import BrokerInterface, OrderResult
from ..api.broker_factory import BrokerFactory, BrokerNotAvailableError


@dataclass
class ExecutionResult:
    status: str
    order_id: Optional[str]
    filled_qty: Optional[float]
    avg_fill_price: Optional[float]
    metadata: Dict[str, object] = field(default_factory=dict)


class Executor:
    """Base executor interface."""

    def place(
        self,
        symbol: str,
        action: str,
        quantity: float,
        price: Optional[float] = None,
        **kwargs,
    ) -> ExecutionResult:
        raise NotImplementedError


class SimulatedExecutor(Executor):
    """
    Simple simulation executor that assumes immediate fills at the provided price.
    """

    def __init__(self, slippage_bps: float = 5.0) -> None:
        self.slippage_bps = slippage_bps

    def place(
        self,
        symbol: str,
        action: str,
        quantity: float,
        price: Optional[float] = None,
        **kwargs,
    ) -> ExecutionResult:
        if price is None:
            raise ValueError("SimulatedExecutor requires a price to fill orders")

        slip = price * (self.slippage_bps / 10_000.0)
        fill_price = price + slip if action.lower() == "buy" else price - slip

        logger.debug(
            "Simulated fill",
            symbol=symbol,
            action=action,
            quantity=quantity,
            requested_price=price,
            fill_price=fill_price,
        )

        return ExecutionResult(
            status="filled",
            order_id=None,
            filled_qty=quantity,
            avg_fill_price=fill_price,
            metadata={"executor": "simulated"},
        )


class BrokerExecutor(Executor):
    """Routes orders through a ``BrokerInterface`` implementation."""

    def __init__(self, broker: BrokerInterface) -> None:
        self.broker = broker

    def place(
        self,
        symbol: str,
        action: str,
        quantity: float,
        price: Optional[float] = None,
        **kwargs,
    ) -> ExecutionResult:
        side = action.lower()
        order_result: OrderResult = self.broker.submit_order(
            symbol=symbol,
            qty=quantity,
            side=side,
            order_type=kwargs.get("order_type", "market"),
            limit_price=price,
            time_in_force=kwargs.get("time_in_force", "day"),
        )

        return ExecutionResult(
            status=order_result.status,
            order_id=order_result.order_id,
            filled_qty=order_result.filled_qty,
            avg_fill_price=order_result.avg_fill_price,
            metadata={"executor": self.broker.name, "raw": order_result.raw},
        )


class QuestradePaperExecutor(BrokerExecutor):
    """Thin wrapper that instantiates the broker via ``BrokerFactory``."""

    def __init__(self, **broker_kwargs) -> None:
        try:
            broker = BrokerFactory.create("questrade", **broker_kwargs)
        except BrokerNotAvailableError as exc:  # pragma: no cover - depends on SDK
            raise RuntimeError(
                "Questrade broker is not available. Ensure the connector is installed."
            ) from exc
        super().__init__(broker=broker)


