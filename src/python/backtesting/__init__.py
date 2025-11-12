"""
Backtesting package for evaluating agents and ensembles.

Exposes utilities to download historical data, run simulations, compute
performance metrics, and persist reports for downstream consumers such as
Lelouch's weighting logic or live trading monitors.
"""

from .data import download_price_history  # noqa: F401
from .engine import (  # noqa: F401
    run_strategy_backtest,
    run_agent_backtest,
    run_ensemble_backtest,
)
from .reporting import save_report_bundle  # noqa: F401

