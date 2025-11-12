"""
Live (or paper) trading pipeline wiring Questrade data through the ensemble.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.python.agents.base import PositionState
from src.python.agents.registry import build_default_agents
from src.python.backtesting.engine import ExecutionConfig
from src.python.backtesting.metrics import TradeRecord, aggregate_metrics
from src.python.backtesting.reporting import BacktestReport, save_report_bundle
from src.python.ensemble.agent_orchestrator import collect_signals
from src.python.ensemble.governor import ConsensusGovernor

from .questrade_client import QuestradeClient
from .questrade_executor import ExecutionResponse, QuestradeExecutor


@dataclass
class QuestradePaperPipeline:
    symbol: str
    starting_cash: float = 100_000.0
    max_position_pct: float = 0.5
    client: Optional[QuestradeClient] = None
    executor: Optional[QuestradeExecutor] = None
    governor: ConsensusGovernor = field(default_factory=ConsensusGovernor)
    log_dir: Path = Path("logs")

    def __post_init__(self) -> None:
        self.client = self.client or QuestradeClient()
        self.executor = self.executor or QuestradeExecutor(
            client=self.client, dry_run=True
        )
        self.agents = build_default_agents()
        self.position = PositionState()
        self.cash = self.starting_cash
        self.equity_curve: List[float] = []
        self.equity_index: List[pd.Timestamp] = []
        self.trades: List[TradeRecord] = []
        self.history = pd.DataFrame()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_equity = self.starting_cash
        self.start_of_day_equity = self.starting_cash

    def _build_context(self, equity: float, market_value: float) -> Dict[str, float]:
        exposure = 0.0 if equity == 0 else market_value / equity
        drawdown = 0.0 if self.max_equity == 0 else (self.max_equity - equity) / self.max_equity
        day_pnl = equity - self.start_of_day_equity
        return {"drawdown": drawdown, "day_pnl": day_pnl, "exposure": exposure}

    def _update_equity(self, timestamp: pd.Timestamp, price: float) -> float:
        market_value = self.position.size * price if self.position.in_long else 0.0
        equity = self.cash + market_value
        self.max_equity = max(self.max_equity, equity)
        self.equity_curve.append(equity)
        self.equity_index.append(timestamp)
        return equity

    def _record_trade(
        self,
        timestamp: pd.Timestamp,
        action: str,
        response: ExecutionResponse,
        position_before: float,
        entry_price_before: float,
    ) -> None:
        pnl = 0.0
        if action.lower() == "sell":
            pnl = (response.avg_price - entry_price_before) * position_before

        self.trades.append(
            TradeRecord(
                timestamp=timestamp,
                symbol=self.symbol,
                agent="ensemble",
                action=action,
                price=response.avg_price,
                quantity=response.filled_qty,
                value=response.filled_qty * response.avg_price,
                fees=0.0,
                pnl=pnl,
                equity_after=self.cash
                + (self.position.size * response.avg_price if self.position.in_long else 0.0),
            )
        )

    def process_candles(self, candles: pd.DataFrame) -> None:
        """
        Feed new candles into the pipeline and evaluate actions.
        """
        if candles.empty:
            return

        candles = candles.copy()
        candles.index = pd.to_datetime(candles.index)
        self.history = pd.concat([self.history, candles]).sort_index().drop_duplicates()

        for timestamp, row in candles.iterrows():
            price = float(row["close"])
            equity = self._update_equity(timestamp, price)
            market_value = self.position.size * price if self.position.in_long else 0.0
            context = self._build_context(equity, market_value)

            signals = collect_signals(
                self.history.loc[:timestamp],
                self.position,
                equity,
                context,
                agents=self.agents,
            )

            if any(sig.is_veto for sig in signals):
                continue

            decision = self.governor.decide(signals)
            action = decision.action
            if action == "hold":
                continue

            confidence = max(decision.confidence, 0.0)
            position_value = equity * self.max_position_pct * confidence
            if position_value <= 0:
                continue

            if action == "buy" and not self.position.in_long:
                quantity = position_value / price
                response = self.executor.submit_order(
                    symbol=self.symbol,
                    action="buy",
                    quantity=quantity,
                    price=price,
                )
                self.position.in_long = True
                self.position.size = response.filled_qty
                self.position.entry_price = response.avg_price
                self.cash -= response.avg_price * response.filled_qty
                self._record_trade(
                    timestamp,
                    "buy",
                    response,
                    position_before=0.0,
                    entry_price_before=0.0,
                )

            elif action == "sell" and self.position.in_long:
                quantity = self.position.size
                response = self.executor.submit_order(
                    symbol=self.symbol,
                    action="sell",
                    quantity=quantity,
                    price=price,
                )
                position_before = self.position.size
                entry_price_before = self.position.entry_price
                self.cash += response.avg_price * response.filled_qty
                self.position.in_long = False
                self.position.size = 0.0
                self.position.entry_price = 0.0
                self._record_trade(
                    timestamp,
                    "sell",
                    response,
                    position_before=position_before,
                    entry_price_before=entry_price_before,
                )

            # Reset start-of-day equity
            if self.equity_index and len(self.equity_index) >= 2:
                if self.equity_index[-1].date() != self.equity_index[-2].date():
                    self.start_of_day_equity = self.equity_curve[-1]

    def snapshot_report(self) -> BacktestReport:
        equity_series = pd.Series(self.equity_curve, index=self.equity_index, name="equity")
        cfg = ExecutionConfig(symbol=self.symbol, starting_cash=self.starting_cash)
        metrics = aggregate_metrics(equity_series, self.trades, cfg.starting_cash)
        report = BacktestReport(
            agent="ensemble",
            symbol=self.symbol,
            equity_curve=equity_series,
            trades=self.trades,
            metrics=metrics,
            details={"mode": "paper", "broker": "questrade"},
        )
        save_report_bundle(
            report,
            log_dir=self.log_dir,
            metrics_filename=f"live_{self.symbol}_metrics.json",
        )
        return report

