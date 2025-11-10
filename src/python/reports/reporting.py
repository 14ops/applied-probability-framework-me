from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.python.metrics.metrics import summarize, drawdown_table

try:
    import mlflow  # type: ignore
except Exception:
    mlflow = None


@dataclass
class ReportConfig:
    out_dir: str = "results"
    run_name: Optional[str] = None


def save_csvs(cfg: ReportConfig, equity: pd.Series, trades: pd.DataFrame, per_symbol_equity: Optional[pd.DataFrame] = None) -> Dict[str, str]:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    equity_path = out / "equity.csv"
    trades_path = out / "trades.csv"
    equity.to_csv(equity_path, header=["equity"])
    trades.to_csv(trades_path, index=False)
    paths = {"equity": str(equity_path), "trades": str(trades_path)}
    if per_symbol_equity is not None:
        pse_path = out / "per_symbol_equity.csv"
        per_symbol_equity.to_csv(pse_path)
        paths["per_symbol_equity"] = str(pse_path)
    return paths


def log_metrics_and_artifacts(cfg: ReportConfig, params: Dict, equity: pd.Series, trades: pd.DataFrame, per_symbol_equity: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    metrics = summarize(equity, trades)
    if mlflow is None:
        return metrics
    with mlflow.start_run(run_name=cfg.run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        paths = save_csvs(cfg, equity, trades, per_symbol_equity)
        for name, p in paths.items():
            mlflow.log_artifact(p, artifact_path=name)
        dd_tbl = drawdown_table(equity)
        dd_path = Path(cfg.out_dir) / "drawdowns.csv"
        dd_tbl.to_csv(dd_path, index=False)
        mlflow.log_artifact(str(dd_path), artifact_path="drawdowns")
    return metrics


