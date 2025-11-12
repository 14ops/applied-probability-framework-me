# Ensemble Trading Pipeline

This reference outlines the live trading flow implemented under `stock_ai`.

1. **Agents**  
   `SenkuAgent`, `TakeshiAgent`, `AoiAgent`, and `YuzuAgent` emit `(action, confidence)` signals each bar.  
   `Senku` hunts RSI(2) bounces in uptrends, `Takeshi` rides Donchian breakouts, `Aoi` confirms dual-SMA trends, and `Yuzu` monitors ATR/drawdown risk (metadata sets `veto=True` when conditions are unsafe).

2. **Consensus**  
   `ConsensusGovernor` combines signals using `LelouchMetaOptimizer` weights. It requires a minimum number of aligned agents and a weighted-confidence threshold before authorizing a trade. Any veto returns an immediate `hold`.

3. **Risk Checks**  
   `RiskEngine` enforces capital constraints (`risk_per_trade`, `daily_stop`, `max_drawdown`, `max_exposure`). `pre_trade` must pass before orders are forwarded to execution.

4. **Execution**  
   `Executor` instances (`SimulatedExecutor`, `BrokerExecutor`, `QuestradePaperExecutor`) place orders after consensus + risk approval and a valid price tick.

5. **Feedback Loop**  
   Fills are logged with `MetricsTracker`. Agent-level Sharpe ratios are recomputed over the last 100 trades and fed back into `LelouchMetaOptimizer`, nudging ensemble weights toward recent winners.

The `EnsembleController` façade in `stock_ai/pipeline/controller.py` wires all layers together. See `tests/test_ensemble_pipeline.py` for usage examples.


