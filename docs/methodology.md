# Methodology

## Frozen research data

The primary source is Binance Spot 5-minute OHLCV for BTCUSDT and ETHUSDT. The configured research cutoff is `2026-10-01T00:00:00Z`. Only candles whose close timestamp is at or before that instant enter the frozen snapshot. `freeze_snapshot.py` records candle hashes, coverage, gaps, and configuration hashes. Rebuilding an existing snapshot requires `--force`; the rolling live updater writes to a separate `data/processed/` path.

Returns are computed in timestamp order. Daily realized variance (RV) is the sum of squared adjacent intraday log returns. BPV, jump variation, semivariances, quarticity, close-to-close returns, range and volume aggregates are retained. A daily observation is complete only with 288 candles and 288 valid returns. Gaps remain missing, partial-day RV is diagnostic only, incomplete days cannot enter training or targets, and a forecast origin is skipped unless the immediately preceding UTC day is complete.

## Point-in-time forecasts

Forecast origins are 00:00 UTC after the preceding day closes. Direct targets are the mean RV over the next 1, 7 or 30 complete UTC days; cumulative close-to-close return is stored separately for tail-risk evaluation. Each training label is admitted only when `target_end <= origin`. The expanding study uses burn-in forecasts from 2019, initial training through 2022, validation in 2023, and OOS origins from 2024 through the frozen cutoff.

Burn-in forecasts are retained to form model-specific historical standardized residuals. For origin `t`, risk residuals include only forecasts whose full target window ended by `t`; each h-day return is standardized by its own ex-ante forecast and variance-to-RV calibration. Student-t degrees of freedom are estimated under unit innovation variance. For horizons above one day, FHS uses a flat daily sigma path and labels this assumption in each result.

## Models and selection

The current walk-forward set contains random-walk, rolling means and EWMA benchmarks; HAR, log-HAR, HAR-J, HAR-CJ, SHAR and HARQ; GARCH-family models; ElasticNet; and optional random forest, LightGBM and XGBoost. ElasticNet predicts log RV with a residual variance correction. Tree and linear ML inputs exclude raw unscaled volume levels. The tuning script reads `config/search_space.yaml` and runs purged expanding time-series folds with a horizon-aware embargo, retuning every 90 days in OOS.

The report script emits QLIKE and other RV metrics, HAC Mincer-Zarnowitz tests, Diebold-Mariano/HLN comparisons and a model confidence set. VaR coverage uses Kupiec and Christoffersen tests, with non-overlapping h-day samples when h>1. ES is evaluated with FZ0 and an Acerbi-Szekely diagnostic; the latter reports a reproducible Monte Carlo p-value. One-day risk includes standardized historical quantiles alongside parametric, FHS and EVT estimates. A 30-day risk result is descriptive because its effective exception count is small. `run_regime_economic_eval.py` joins forecast origins to trailing-only bull/bear/stress labels and writes conditional RV metrics plus cost-aware volatility-target and risk-budget summaries.

## Production forecasts

The API separates `mode=research` from `mode=production`. Research reads saved walk-forward output and attached risk forecasts at the frozen cutoff. Production rebuilds daily features from the rolling local Binance file, rejects a feed more than 15 minutes stale, and fits from labels available at the latest complete-day origin. It returns an explicitly labelled normal VaR/ES approximation and does not write to frozen research artifacts. Production model hyperparameters currently use code defaults rather than periodic CV tuning.

## Reproduction

From the repository root in PowerShell:

```powershell
$PY = ".\.venv\Scripts\python.exe"
& $PY backend\scripts\sync_data.py
& $PY backend\scripts\run_pipeline.py
```

For the individual stages, see the commands in `README.md`. Full historical outputs are generated locally under `results/`; older MVP artifacts are preserved in `results/archive/legacy_mvp_20261018/` and must not be combined with current v3 results.

## Not yet included

Conditional quantile-regression models, deep-learning models, external derivative/macro data, and rolling-window robustness are not part of the current baseline run. Their configuration or helper modules alone do not imply that results have been generated. Add these only with documented source availability and point-in-time rules.
