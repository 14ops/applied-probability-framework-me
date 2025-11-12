"""
Simulation engine that can evaluate individual agents or ensembles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from src.python.agents.base import Agent, PositionState
from src.python.ensemble.agent_orchestrator import collect_signals
from src.python.ensemble.governor import ConsensusGovernor

from .metrics import TradeRecord, aggregate_metrics
from .reporting import BacktestReport


StrategyFn = Callable[[pd.DataFrame, PositionState, float, Dict], Tuple[str, float, bool]]


@dataclass
class ExecutionConfig:
    symbol: str
    starting_cash: float = 100_000.0
    max_position_pct: float = 0.95
    commission_pct: float = 0.0005
    slippage_bps: float = 5.0
    risk_free_rate: float = 0.0


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    return df


def _compute_fill_price(action: str, price: float, slippage_bps: float) -> float:
    slip = price * (slippage_bps / 10_000.0)
    if action.lower() == "buy":
        return price + slip
    if action.lower() == "sell":
        return price - slip
    return price


def _update_position_state(
    pos: PositionState, action: str, quantity: float, price: float
) -> PositionState:
    if action.lower() == "buy":
        pos.in_long = True
        pos.size = quantity
        pos.entry_price = price
    elif action.lower() == "sell":
        pos.in_long = False
        pos.size = 0.0
        pos.entry_price = 0.0
    return pos


def run_strategy_backtest(
    strategy_fn: StrategyFn,
    data: pd.DataFrame,
    exec_cfg: ExecutionConfig,
    agent_name: str,
) -> BacktestReport:
    """
    Run a generic strategy callable through the simulation loop.
    """
    df = _ensure_datetime_index(data)
    pos_state = PositionState()
    cash = exec_cfg.starting_cash
    equity_curve: List[float] = []
    equity_index: List[pd.Timestamp] = []
    trades: List[TradeRecord] = []
    max_equity = exec_cfg.starting_cash
    start_of_day_equity = exec_cfg.starting_cash

    for idx, (timestamp, row) in enumerate(df.iterrows()):
        hist = df.iloc[: idx + 1]
        price = float(row["close"])

        market_value = pos_state.size * price if pos_state.in_long else 0.0
        equity = cash + market_value
        max_equity = max(max_equity, equity)
        drawdown = 0.0 if max_equity == 0 else (equity - max_equity) / max_equity
        exposure = 0.0 if equity == 0 else market_value / equity
        day_pnl = equity - start_of_day_equity

        context = {
            "drawdown": abs(drawdown),
            "day_pnl": day_pnl,
            "exposure": exposure,
        }

        action, confidence, veto = strategy_fn(hist, pos_state, equity, context)
        executed_action = action.lower()

        if veto:
            executed_action = "hold"

        fees = 0.0
        pnl = 0.0

        if executed_action == "buy" and not pos_state.in_long:
            position_value = equity * exec_cfg.max_position_pct * max(confidence, 0)
            quantity = position_value / price if price > 0 else 0.0
            quantity = round(quantity, 4)

            if quantity > 0:
                fill_price = _compute_fill_price("buy", price, exec_cfg.slippage_bps)
                cost = quantity * fill_price
                fees = cost * exec_cfg.commission_pct

                if cost + fees <= cash:
                    cash -= cost + fees
                    pos_state = _update_position_state(
                        pos_state, "buy", quantity, fill_price
                    )
                    market_value = pos_state.size * price
                    equity = cash + market_value
                    trades.append(
                        TradeRecord(
                            timestamp=timestamp,
                            symbol=exec_cfg.symbol,
                            agent=agent_name,
                            action="buy",
                            price=fill_price,
                            quantity=quantity,
                            value=cost,
                            fees=fees,
                            pnl=0.0,
                            equity_after=equity,
                        )
                    )
        elif executed_action == "sell" and pos_state.in_long:
            fill_price = _compute_fill_price("sell", price, exec_cfg.slippage_bps)
            quantity = pos_state.size
            proceeds = quantity * fill_price
            fees = proceeds * exec_cfg.commission_pct
            cash += proceeds - fees
            pnl = (fill_price - pos_state.entry_price) * quantity - fees
            pos_state = _update_position_state(pos_state, "sell", 0.0, 0.0)
            market_value = 0.0
            equity = cash
            trades.append(
                TradeRecord(
                    timestamp=timestamp,
                    symbol=exec_cfg.symbol,
                    agent=agent_name,
                    action="sell",
                    price=fill_price,
                    quantity=quantity,
                    value=proceeds,
                    fees=fees,
                    pnl=pnl,
                    equity_after=equity,
                )
            )

        # Update start-of-day equity at new sessions
        if idx + 1 < len(df):
            next_ts = df.index[idx + 1]
            if next_ts.date() != timestamp.date():
                start_of_day_equity = equity

        equity_curve.append(equity)
        equity_index.append(timestamp)

    equity_series = pd.Series(equity_curve, index=equity_index, name="equity")
    metrics = aggregate_metrics(equity_series, trades, exec_cfg.starting_cash, exec_cfg.risk_free_rate)

    return BacktestReport(
        agent=agent_name,
        symbol=exec_cfg.symbol,
        equity_curve=equity_series,
        trades=trades,
        metrics=metrics,
        details={
            "starting_cash": exec_cfg.starting_cash,
            "max_position_pct": exec_cfg.max_position_pct,
            "commission_pct": exec_cfg.commission_pct,
            "slippage_bps": exec_cfg.slippage_bps,
        },
    )


def run_agent_backtest(
    agent: Agent,
    data: pd.DataFrame,
    exec_cfg: Optional[ExecutionConfig] = None,
) -> BacktestReport:
    cfg = exec_cfg or ExecutionConfig(symbol="UNKNOWN")
    return run_strategy_backtest(agent.decide, data, cfg, agent.name)


def run_ensemble_backtest(
    data: pd.DataFrame,
    exec_cfg: ExecutionConfig,
    governor: Optional[ConsensusGovernor] = None,
    include_lelouch: bool = False,
) -> BacktestReport:
    gov = governor or ConsensusGovernor()

    def ensemble_strategy(
        hist: pd.DataFrame, pos: PositionState, equity: float, context: Dict
    ) -> Tuple[str, float, bool]:
        signals = collect_signals(hist, pos, equity, context)
        if any(sig.is_veto for sig in signals):
            return ("hold", 1.0, True)

        decision = gov.decide(signals, include_lelouch=include_lelouch)
        return decision.action, decision.confidence, False

    report = run_strategy_backtest(ensemble_strategy, data, exec_cfg, gov.name)
    report.details["strategy"] = "ensemble"
    report.details["weights"] = gov.weights
    return report

