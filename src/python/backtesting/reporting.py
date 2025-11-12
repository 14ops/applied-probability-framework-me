"""
Helpers for persisting backtest outputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from .metrics import TradeRecord


@dataclass
class BacktestReport:
    agent: str
    symbol: str
    equity_curve: pd.Series
    trades: List[TradeRecord]
    metrics: Dict[str, float]
    details: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "agent": self.agent,
            "symbol": self.symbol,
            "metrics": self.metrics,
            "details": self.details,
        }


def _ensure_logs_dir(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)


def append_dataframe(df: pd.DataFrame, path: Path) -> None:
    header = not path.exists()
    df.to_csv(path, mode="a", index=True, header=header)


def save_report_bundle(
    report: BacktestReport,
    log_dir: Path | str = Path("logs"),
    trades_filename: str = "trades.csv",
    equity_filename: str = "equity.csv",
    metrics_filename: str | None = None,
    daily_summary_filename: str = "daily_summary.csv",
) -> None:
    """
    Persist trade logs, equity curve and summary metrics for the given report.
    """
    log_path = Path(log_dir)
    _ensure_logs_dir(log_path)

    trade_records = pd.DataFrame([t.to_dict() for t in report.trades])
    if not trade_records.empty:
        append_dataframe(trade_records.set_index("timestamp"), log_path / trades_filename)

    equity_df = report.equity_curve.to_frame(name="equity")
    equity_df["agent"] = report.agent
    equity_df["symbol"] = report.symbol
    append_dataframe(equity_df, log_path / equity_filename)

    metrics_path = log_path / (
        metrics_filename or f"{report.agent}_{report.symbol}_metrics.json"
    )
    metrics_payload = {
        **report.to_dict(),
        "equity_path": str((log_path / equity_filename).resolve()),
        "trade_log_path": str((log_path / trades_filename).resolve()),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))

    daily_summary = pd.DataFrame(
        [
            {
                "date": report.equity_curve.index[-1].date().isoformat()
                if not report.equity_curve.empty
                else None,
                "agent": report.agent,
                "symbol": report.symbol,
                "sharpe_ratio": report.metrics.get("sharpe_ratio"),
                "sortino_ratio": report.metrics.get("sortino_ratio"),
                "max_drawdown_pct": report.metrics.get("max_drawdown_pct"),
                "total_return_pct": report.metrics.get("total_return_pct"),
                "win_rate_pct": report.metrics.get("win_rate_pct"),
                "profit_factor": report.metrics.get("profit_factor"),
            }
        ]
    )
    daily_summary = daily_summary.dropna(subset=["date"])
    if not daily_summary.empty:
        daily_summary = daily_summary.set_index("date")
        append_dataframe(daily_summary, log_path / daily_summary_filename)

