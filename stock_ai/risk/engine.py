"""Risk management layer ensuring capital protection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class RiskState:
    initial_equity: Optional[float] = None
    max_equity: Optional[float] = None
    daily_realized_pnl: float = 0.0
    current_exposure: float = 0.0
    last_reason: Optional[str] = None


class RiskEngine:
    """
    Applies risk constraints before a trade is executed.

    Parameters are expressed as fractions of equity (e.g. 0.01 == 1%).
    """

    def __init__(
        self,
        risk_per_trade: float = 0.01,
        daily_stop: float = 0.03,
        max_drawdown: float = 0.20,
        max_exposure: float = 0.60,
    ) -> None:
        self.risk_per_trade = risk_per_trade
        self.daily_stop = daily_stop
        self.max_drawdown = max_drawdown
        self.max_exposure = max_exposure
        self.state = RiskState()

    def reset_day(self) -> None:
        """Reset daily counters (call at start of trading session)."""
        self.state.daily_realized_pnl = 0.0

    def update_equity(self, equity: float) -> None:
        """Track equity used for drawdown calculations."""
        if self.state.initial_equity is None:
            self.state.initial_equity = equity
            self.state.max_equity = equity
        if self.state.max_equity is None or equity > self.state.max_equity:
            self.state.max_equity = equity

    def record_realized_pnl(self, pnl: float) -> None:
        """Add realized PnL (positive or negative) to daily tally."""
        self.state.daily_realized_pnl += pnl

    def set_daily_realized(self, pnl: float) -> None:
        """Overwrite the daily realized PnL (useful when fed from external accounting)."""
        self.state.daily_realized_pnl = pnl

    def set_exposure(self, exposure_value: float) -> None:
        """Set the current gross exposure (absolute position value)."""
        self.state.current_exposure = max(0.0, exposure_value)

    def pre_trade(
        self,
        equity: float,
        proposed_position_value: float,
        current_drawdown: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Run risk checks before sending an order.

        Args:
            equity: Current account equity.
            proposed_position_value: Dollar value of the trade under consideration.
            current_drawdown: Optional externally-computed drawdown fraction.
        """
        self.update_equity(equity)

        if equity <= 0:
            return False, "no_equity"

        max_allowed_trade = equity * self.risk_per_trade
        if proposed_position_value > max_allowed_trade:
            return False, "risk_per_trade_exceeded"

        if self.state.daily_realized_pnl <= -equity * self.daily_stop:
            return False, "daily_stop_triggered"

        drawdown_fraction = current_drawdown
        if drawdown_fraction is None and self.state.max_equity:
            drawdown_fraction = max(
                0.0, (self.state.max_equity - equity) / self.state.max_equity
            )

        if drawdown_fraction is not None and drawdown_fraction >= self.max_drawdown:
            return False, "max_drawdown_exceeded"

        projected_exposure = self.state.current_exposure + proposed_position_value
        if projected_exposure > equity * self.max_exposure:
            return False, "max_exposure_exceeded"

        return True, ""


