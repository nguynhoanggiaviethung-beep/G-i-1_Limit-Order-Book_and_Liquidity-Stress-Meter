# Crypto volatility and tail-risk research design

The research pipeline follows `config/research.yaml` and the frozen strategy in the accompanying research documents. Binance Spot BTCUSDT and ETHUSDT 5-minute candles are the source. Research is cut off at **2026-10-01 00:00:00 UTC**, independent of the live updater. This includes the final 5-minute candle of 2026-09-30 only when its close timestamp is at or before the cutoff.

## Data and point-in-time rules

`backend/scripts/freeze_snapshot.py` copies only candles whose close time is at or before the cutoff and records SHA-256 hashes, coverage, configuration hashes, and missing intervals in `data/snapshots/<as-of>/manifest.json`. The snapshot is immutable unless explicitly rebuilt with `--force`. Live sync writes under `data/processed/` and never changes an existing research snapshot.

Daily RV is the sum of squared consecutive 5-minute log returns. The feature builder also computes BPV, jump variation, positive/negative semivariances, realized quarticity, returns, ranges, and volume features. It never fills missing bars. A UTC date is complete only when it has all 288 bars and 288 valid adjacent returns; incomplete dates remain visible with missing target measures and are excluded from training/target windows.

Forecast origins are 00:00 UTC. Direct targets are the mean RV over the next 1, 7, or 30 complete UTC dates, and cumulative return is retained as a separate loss outcome. Features are available only after the preceding UTC day closes. Training labels are admitted only when their target end is no later than the forecast origin.

## Evaluation

The primary split is expanding: initial training through 2022-12-31, validation in 2023, and OOS from 2024-01-01 through the frozen cutoff. Benchmarks include random walk, rolling 7/30-day means, and EWMA(0.94); the implemented HAR family includes HAR, log-HAR, HAR-J, HAR-CJ, SHAR, and HARQ. GARCH-t, GJR-t, EGARCH-t, skew-t GARCH, ElasticNet, random forest, LightGBM, and XGBoost implementations are provided where optional dependencies are installed. Rolling windows of 1,000/1,500 days and RV windows 5/22 are configured robustness extensions.

The report script emits RV/volatility/log-RV metrics, QLIKE, HAC Mincer-Zarnowitz calibration, Diebold-Mariano with HLN correction, and MCS outputs. Risk utilities implement parametric normal/t/skew-t, filtered historical simulation, EVT-POT, quantile loss, VaR/ES backtests, and non-overlapping sampling for horizons above one day. Thirty-day risk results are descriptive. Deep learning is staged until the classical baseline is stable, as specified by the plan.

## Reproduction commands

Run these from the repository root in PowerShell:

```powershell
$PY = ".\.venv\Scripts\python.exe"
& $PY backend\scripts\freeze_snapshot.py
& $PY backend\scripts\validate_data.py
& $PY backend\scripts\build_features.py
& $PY backend\scripts\run_walkforward.py
& $PY backend\scripts\make_report_tables.py
& $PY backend\scripts\run_risk_eval.py
```

For a quick baseline run, restrict `run_walkforward.py` with `--symbol BTCUSDT --horizon 1 --model random_walk --model har`. Full results are written below `results/forecasts`, `results/metrics`, and `results/tables`; local data and frozen Parquet snapshots are excluded from Git.

The historical file currently contains source gaps (the manifest records them). Results therefore describe only complete days and must be interpreted with the coverage manifest. This is not a claim that all possible Binance candles are present.
