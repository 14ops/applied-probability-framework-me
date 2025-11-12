"""High-level controller wiring agents, consensus, risk, and execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, Optional

import pandas as pd

from ..agents import (
    AoiAgent,
    LelouchMetaOptimizer,
    SenkuAgent,
    Signal,
    TakeshiAgent,
    YuzuAgent,
)
from ..execution import ExecutionResult, Executor, SimulatedExecutor
from ..governance import ConsensusDecision, ConsensusGovernor
from ..metrics import MetricsTracker, TradeRecord
from ..risk import RiskEngine


PositionSizer = Callable[[float, float, str, RiskEngine], float]


def default_position_sizer(
    equity: float, price: float, action: str, risk_engine: RiskEngine
) -> float:
    """
    Size positions such that the notional value equals ``risk_per_trade`` of equity.

    Returns:
        Quantity (shares) rounded down to the nearest whole share.
    """
    if price <= 0 or equity <= 0:
        return 0.0
    risk_capital = equity * risk_engine.risk_per_trade
    quantity = risk_capital / price
    return max(0.0, float(int(quantity)))


@dataclass
class ProcessResult:
    symbol: str
    price: float
    signals: Dict[str, Signal]
    decision: ConsensusDecision
    final_action: str
    quantity: float
    risk_allowed: bool
    risk_reason: str
    execution: Optional[ExecutionResult]


class EnsembleController:
    """Runs the end-to-end workflow for each bar."""

    def __init__(
        self,
        agents: Optional[Iterable] = None,
        governor: Optional[ConsensusGovernor] = None,
        risk_engine: Optional[RiskEngine] = None,
        executor: Optional[Executor] = None,
        meta_optimizer: Optional[LelouchMetaOptimizer] = None,
        metrics: Optional[MetricsTracker] = None,
        position_sizer: PositionSizer = default_position_sizer,
    ) -> None:
        agents = list(agents) if agents is not None else self._default_agents()
        self.agents = {agent.name: agent for agent in agents}
        self.governor = governor or ConsensusGovernor()
        self.risk_engine = risk_engine or RiskEngine()
        self.executor = executor or SimulatedExecutor()
        self.metrics = metrics or MetricsTracker()
        self.meta_optimizer = meta_optimizer or LelouchMetaOptimizer(self.agents.keys())
        self.position_sizer = position_sizer

    def _default_agents(self):
        return [
            SenkuAgent(),
            TakeshiAgent(),
            AoiAgent(),
            YuzuAgent(),
        ]

    def process_bar(
        self,
        symbol: str,
        data: pd.DataFrame,
        market_state: Mapping[str, float],
    ) -> ProcessResult:
        """
        Run agents, consensus, risk checks, and (optionally) execution for a symbol.

        Args:
            symbol: Ticker being processed.
            data: OHLCV dataframe (latest row is the current bar).
            market_state: Dictionary containing at least ``equity``. Additional keys
                used when present: ``daily_pnl``, ``current_drawdown``, ``exposure``,
                ``price``.
        """
        equity = float(market_state.get("equity", 0.0))
        daily_pnl = float(market_state.get("daily_pnl", 0.0))
        drawdown = market_state.get("current_drawdown")
        exposure = float(market_state.get("exposure", self.risk_engine.state.current_exposure))
        price = float(market_state.get("price", data["close"].iloc[-1]))

        self.risk_engine.set_daily_realized(daily_pnl)
        self.risk_engine.set_exposure(exposure)

        context = {
            "current_drawdown": drawdown or 0.0,
            "max_drawdown": self.risk_engine.max_drawdown,
            "daily_pnl": daily_pnl,
            "daily_stop": self.risk_engine.daily_stop,
        }

        signals = {
            name: agent(data, context)
            for name, agent in self.agents.items()
        }

        decision = self.governor.decide(
            list(signals.values()),
            weights=self.meta_optimizer.weights,
        )

        quantity = 0.0
        execution: Optional[ExecutionResult] = None
        risk_allowed = True
        risk_reason = ""
        final_action = decision.action

        if decision.action in ("buy", "sell"):
            quantity = self.position_sizer(equity, price, decision.action, self.risk_engine)
            proposed_value = quantity * price
            risk_allowed, risk_reason = self.risk_engine.pre_trade(
                equity=equity,
                proposed_position_value=proposed_value,
                current_drawdown=drawdown,
            )

            if risk_allowed and quantity > 0:
                execution = self.executor.place(symbol, decision.action, quantity, price=price)
                final_action = decision.action
            else:
                final_action = "hold"

        return ProcessResult(
            symbol=symbol,
            price=price,
            signals=signals,
            decision=decision,
            final_action=final_action,
            quantity=quantity,
            risk_allowed=risk_allowed,
            risk_reason=risk_reason,
            execution=execution,
        )

    def record_trade_outcome(
        self,
        timestamp: str,
        symbol: str,
        action: str,
        price: float,
        quantity: float,
        pnl: float,
        supporting_agents: Optional[Iterable[str]] = None,
    ) -> TradeRecord:
        """
        Log trade results, refresh metrics, and update ensemble weights.
        """
        trade = self.metrics.log_trade(
            timestamp=timestamp,
            symbol=symbol,
            action=action,
            price=price,
            quantity=quantity,
            pnl=pnl,
            supporting_agents=supporting_agents,
        )

        sharpe = self.metrics.sharpe_by_agent()
        self.meta_optimizer.update(sharpe)

        return trade


