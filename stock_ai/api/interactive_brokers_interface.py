"""Interactive Brokers (IBKR) API interface using ib_insync."""

from __future__ import annotations

from typing import Dict, List, Optional

from loguru import logger
import yaml

from .base_broker import BrokerInterface, OrderResult

try:  # pragma: no cover - optional dependency
    from ib_insync import IB, MarketOrder, LimitOrder, StopOrder, Stock
except ImportError:  # pragma: no cover - optional dependency
    IB = None


class InteractiveBrokersInterface(BrokerInterface):
    """Thin wrapper around :mod:`ib_insync` for Live/Paper trading."""

    name = "interactive_brokers"

    def __init__(
        self,
        config_path: str = "config/secrets.yaml",
        host: Optional[str] = None,
        port: Optional[int] = None,
        client_id: Optional[int] = None,
        account: Optional[str] = None,
    ) -> None:
        if IB is None:
            raise ImportError(
                "ib_insync not installed. Install with: pip install ib-insync"
            )

        secrets = self._load_config(config_path)
        ib_conf = secrets.get("interactive_brokers", {})

        self.host = host or ib_conf.get("host", "127.0.0.1")
        self.port = int(port or ib_conf.get("port", 7497))
        self.client_id = int(client_id or ib_conf.get("client_id", 1))
        self.account = account or ib_conf.get("account", None)

        self.ib = IB()
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            logger.info(
                "Connected to Interactive Brokers at %s:%s (client_id=%s)",
                self.host,
                self.port,
                self.client_id,
            )
        except Exception as exc:  # pragma: no cover - network interaction
            raise RuntimeError(
                f"Could not connect to Interactive Brokers TWS/Gateway: {exc}"
            ) from exc

    # -----------------
    # Helper utilities
    # -----------------
    def _load_config(self, path: str) -> Dict:
        try:
            with open(path, "r") as handle:
                return yaml.safe_load(handle) or {}
        except FileNotFoundError:
            logger.warning("Config file %s not found. Using defaults.", path)
            return {}

    def _make_contract(self, symbol: str):
        contract = Stock(symbol, "SMART", "USD")
        self.ib.qualifyContracts(contract)
        return contract

    # ---------------
    # API operations
    # ---------------
    def get_account(self) -> Dict:
        summary = self.ib.accountSummary(account=self.account)
        result: Dict[str, Optional[float]] = {}
        for item in summary:
            tag = item.tag.replace(" ", "_").lower()
            try:
                value = float(item.value)
            except (TypeError, ValueError):
                value = item.value
            result[tag] = value
        return result

    def get_positions(self) -> List[Dict]:
        positions = []
        for pos in self.ib.positions(account=self.account):
            contract = pos.contract
            positions.append(
                {
                    "symbol": contract.symbol,
                    "qty": float(pos.position),
                    "avg_entry_price": float(pos.avgCost) if pos.avgCost else None,
                    "market_value": float(pos.marketValue) if pos.marketValue else None,
                    "unrealized_pl": float(pos.unrealizedPNL) if pos.unrealizedPNL else None,
                    "account": pos.account,
                }
            )
        return positions

    def get_current_price(self, symbol: str) -> Optional[float]:
        contract = self._make_contract(symbol)
        ticker = self.ib.reqMktData(contract, "", False, False)
        self.ib.sleep(1.0)
        price = ticker.marketPrice()
        self.ib.cancelMktData(contract)
        return float(price) if price is not None else None

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
        contract = self._make_contract(symbol)
        action = side.upper()

        if order_type == "market":
            order = MarketOrder(action, qty)
        elif order_type == "limit":
            if limit_price is None:
                raise ValueError("limit_price required for limit orders")
            order = LimitOrder(action, qty, limit_price)
        elif order_type == "stop":
            if stop_price is None:
                raise ValueError("stop_price required for stop orders")
            order = StopOrder(action, qty, stop_price)
        else:
            raise ValueError(f"Unsupported order type: {order_type}")

        order.tif = time_in_force.upper()
        order.outsideRth = bool(extended_hours)

        trade = self.ib.placeOrder(contract, order)
        self.ib.sleep(0.5)
        status = trade.orderStatus.status
        return OrderResult(
            broker=self.name,
            order_id=str(trade.order.permId or trade.order.orderId),
            status=status,
            submitted_at=str(trade.log[-1].time if trade.log else ""),
            filled_qty=float(trade.orderStatus.filled) if trade.orderStatus.filled else None,
            avg_fill_price=float(trade.orderStatus.avgFillPrice)
            if trade.orderStatus.avgFillPrice
            else None,
            raw={
                "order": trade.order.__dict__,
                "status": trade.orderStatus.__dict__,
            },
        )

    def cancel_order(self, order_id: str) -> bool:
        open_orders = self.ib.openOrders()
        for order in open_orders:
            if str(order.permId or order.orderId) == str(order_id):
                self.ib.cancelOrder(order)
                return True
        return False

    def list_orders(self, status: str = "all"):
        orders = []
        for trade in self.ib.trades():
            if status != "all" and trade.orderStatus.status.lower() != status.lower():
                continue
            orders.append(
                {
                    "id": str(trade.order.permId or trade.order.orderId),
                    "symbol": trade.contract.symbol,
                    "qty": float(trade.order.totalQuantity),
                    "filled_qty": float(trade.orderStatus.filled),
                    "status": trade.orderStatus.status,
                    "type": trade.order.orderType,
                    "submitted_at": str(trade.log[-1].time if trade.log else ""),
                }
            )
        return orders

    def get_clock(self) -> Optional[Dict]:
        return None

    def disconnect(self) -> None:
        if getattr(self, "ib", None):
            try:
                self.ib.disconnect()
            except Exception:
                pass

    def __del__(self):  # pragma: no cover - defensive cleanup
        try:
            self.disconnect()
        except Exception:
            pass

