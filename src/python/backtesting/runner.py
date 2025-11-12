"""
High-level orchestration for running backtests across agents and ensembles.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd

from src.python.agents.aoi import Aoi
from src.python.agents.senku import Senku
from src.python.agents.takeshi import Takeshi

from .data import download_price_history
from .engine import (
    ExecutionConfig,
    run_agent_backtest,
    run_ensemble_backtest,
)
from .reporting import BacktestReport, save_report_bundle


DEFAULT_TICKERS = ("AAPL", "MSFT", "SPY")
DEFAULT_AGENTS = (Senku(), Takeshi(), Aoi())


def _collect_summary(records: List[dict], output_dir: Path) -> None:
    if not records:
        return
    df = pd.DataFrame(records)
    df.to_csv(output_dir / "agent_backtest_summary.csv", index=False)
    df.to_json(output_dir / "agent_backtest_summary.json", orient="records", indent=2)


def run_individual_backtests(
    tickers: Sequence[str] = DEFAULT_TICKERS,
    start: str = "2020-01-01",
    end: str | None = None,
    agents: Sequence = DEFAULT_AGENTS,
    starting_cash: float = 100_000.0,
    log_dir: Path | str = Path("logs"),
) -> List[BacktestReport]:
    """
    Download historical data and backtest each agent individually.
    """
    results: List[BacktestReport] = []
    output_dir = Path(log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_records: List[dict] = []

    for symbol in tickers:
        df = download_price_history(symbol, start=start, end=end)
        if df.empty:
            continue

        exec_cfg = ExecutionConfig(symbol=symbol, starting_cash=starting_cash)

        for agent in agents:
            report = run_agent_backtest(agent, df, exec_cfg)
            save_report_bundle(
                report,
                log_dir=output_dir,
                metrics_filename=f"{agent.name}_{symbol}_metrics.json",
            )
            results.append(report)
            summary_records.append(
                {
                    "symbol": symbol,
                    "agent": agent.name,
                    **report.metrics,
                }
            )

    _collect_summary(summary_records, output_dir)
    return results


def run_consensus_backtests(
    tickers: Sequence[str] = DEFAULT_TICKERS,
    start: str = "2020-01-01",
    end: str | None = None,
    starting_cash: float = 100_000.0,
    log_dir: Path | str = Path("logs"),
) -> List[BacktestReport]:
    """
    Execute ensemble backtests across tickers and persist summaries.
    """
    output_dir = Path(log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: List[BacktestReport] = []
    summary_records: List[dict] = []

    for symbol in tickers:
        df = download_price_history(symbol, start=start, end=end)
        if df.empty:
            continue

        exec_cfg = ExecutionConfig(symbol=symbol, starting_cash=starting_cash)
        report = run_ensemble_backtest(df, exec_cfg)
        save_report_bundle(
            report,
            log_dir=output_dir,
            metrics_filename=f"ensemble_{symbol}_metrics.json",
        )
        reports.append(report)
        summary_records.append({"symbol": symbol, "agent": report.agent, **report.metrics})

    _collect_summary(summary_records, output_dir)
    return reports


def run_default_backtests(
    tickers: Sequence[str] = DEFAULT_TICKERS,
    start: str = "2020-01-01",
    end: str | None = None,
    starting_cash: float = 100_000.0,
    log_dir: Path | str = Path("logs"),
) -> None:
    """
    Run both individual agent and consensus ensemble backtests.
    """
    run_individual_backtests(
        tickers=tickers,
        start=start,
        end=end,
        starting_cash=starting_cash,
        log_dir=log_dir,
    )
    run_consensus_backtests(
        tickers=tickers,
        start=start,
        end=end,
        starting_cash=starting_cash,
        log_dir=log_dir,
    )


if __name__ == "__main__":
    run_default_backtests()

