import pandas as pd
import pytest

from stock_ai.agents.base import Agent, Signal
from stock_ai.pipeline import EnsembleController
from stock_ai.risk import RiskEngine


class StubAgent(Agent):
    def __init__(self, name: str, action: str, confidence: float, metadata=None):
        super().__init__(name=name, min_history=1)
        self._action = action
        self._confidence = confidence
        self._metadata = metadata or {}

    def generate_signal(self, data, context):
        return Signal(agent=self.name, action=self._action, confidence=self._confidence, metadata=self._metadata)


def make_price_frame(price: float = 10.0, length: int = 40) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=length, freq="D")
    base = pd.DataFrame(
        {
            "open": [price] * length,
            "high": [price * 1.01] * length,
            "low": [price * 0.99] * length,
            "close": [price] * length,
            "volume": [1_000_000] * length,
        },
        index=index,
    )
    return base


def test_pipeline_executes_buy_when_consensus_and_risk_pass():
    agents = [
        StubAgent("alpha", "buy", 0.9),
        StubAgent("beta", "buy", 0.95),
        StubAgent("gamma", "buy", 0.85),
        StubAgent("yuzu", "hold", 0.0),
    ]
    controller = EnsembleController(agents=agents)
    df = make_price_frame(price=10.0)

    state = {
        "equity": 10_000.0,
        "daily_pnl": 0.0,
        "current_drawdown": 0.0,
        "exposure": 0.0,
        "price": 10.0,
    }

    result = controller.process_bar("XYZ", df, state)

    assert result.final_action == "buy"
    assert result.execution is not None
    assert result.quantity > 0


def test_pipeline_respects_risk_engine_limits():
    agents = [
        StubAgent("alpha", "buy", 1.0),
        StubAgent("beta", "buy", 1.0),
        StubAgent("gamma", "buy", 1.0),
        StubAgent("yuzu", "hold", 0.0),
    ]
    risk_engine = RiskEngine(risk_per_trade=0.01)

    def oversized_sizer(equity, price, action, engine):
        return 100.0  # Purposely over-leveraged

    controller = EnsembleController(
        agents=agents,
        risk_engine=risk_engine,
        position_sizer=oversized_sizer,
    )
    df = make_price_frame(price=50.0)
    state = {
        "equity": 5_000.0,
        "daily_pnl": 0.0,
        "current_drawdown": 0.0,
        "exposure": 0.0,
        "price": 50.0,
    }

    result = controller.process_bar("XYZ", df, state)

    assert result.final_action == "hold"
    assert not result.risk_allowed
    assert result.risk_reason == "risk_per_trade_exceeded"
    assert result.execution is None


def test_lelouch_updates_weights_based_on_metrics():
    controller = EnsembleController()
    initial_weights = controller.meta_optimizer.weights
    senku_weight_before = initial_weights["senku"]

    # Two profitable Senku trades with varying returns
    controller.record_trade_outcome(
        timestamp="2024-01-01T10:00:00Z",
        symbol="XYZ",
        action="buy",
        price=10.0,
        quantity=1.0,
        pnl=1.0,
        supporting_agents=["senku"],
    )
    controller.record_trade_outcome(
        timestamp="2024-01-02T10:00:00Z",
        symbol="XYZ",
        action="sell",
        price=10.0,
        quantity=1.0,
        pnl=0.5,
        supporting_agents=["senku"],
    )

    # One losing Takeshi trade
    controller.record_trade_outcome(
        timestamp="2024-01-03T10:00:00Z",
        symbol="XYZ",
        action="sell",
        price=10.0,
        quantity=1.0,
        pnl=-0.5,
        supporting_agents=["takeshi"],
    )

    updated_weights = controller.meta_optimizer.weights
    assert updated_weights["senku"] > senku_weight_before
    assert updated_weights["takeshi"] <= initial_weights["takeshi"]

