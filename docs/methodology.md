# Methodology

## Frozen research data

The primary source is Binance Spot 5-minute OHLCV for BTCUSDT and ETHUSDT. The configured research cutoff is `2026-10-01T00:00:00Z`. Only candles whose close timestamp is at or before that instant enter the frozen snapshot. `freeze_snapshot.py` records candle hashes, coverage, gaps, and configuration hashes. Rebuilding an existing snapshot requires `--force`; the rolling live updater writes to a separate `data/processed/` path.

Returns are computed in timestamp order. Daily realized variance (RV) is the sum of squared adjacent intraday log returns. BPV, jump variation, semivariances, quarticity, close-to-close returns, range and volume aggregates are retained. A daily observation is complete only with 288 candles and 288 valid returns. Gaps remain missing, partial-day RV is diagnostic only, incomplete days cannot enter training or targets, and a forecast origin is skipped unless the immediately preceding UTC day is complete.

## Point-in-time forecasts

Forecast origins are 00:00 UTC after the preceding day closes. Direct targets are the mean RV over the next 1, 7 or 30 complete UTC days; cumulative close-to-close return is stored separately for tail-risk evaluation. Each training label is admitted only when `target_end <= origin`. The expanding study uses burn-in forecasts from 2019, initial training through 2022, validation in 2023, and OOS origins from 2024 through the frozen cutoff.

Burn-in forecasts are retained to form model-specific historical standardized residuals. For origin `t`, risk residuals include only forecasts whose full target window ended by `t`; each h-day return is standardized by its own ex-ante forecast and variance-to-RV calibration. Student-t degrees of freedom are estimated under unit innovation variance. For horizons above one day, FHS uses a flat daily sigma path and labels this assumption in each result.

## Models, risk, and selection

The current walk-forward set contains random-walk, rolling means and EWMA benchmarks; HAR, log-HAR, HAR-J, HAR-CJ, SHAR and HARQ; GARCH-family models; ElasticNet; and optional random forest, LightGBM and XGBoost. ElasticNet predicts log RV with a residual variance correction. Tree and linear ML inputs exclude raw unscaled volume levels. The tuning script reads `config/search_space.yaml` and runs purged expanding time-series folds with a horizon-aware embargo, retuning every 90 days in OOS.

The report script emits QLIKE and other RV metrics, HAC Mincer-Zarnowitz tests, Diebold-Mariano/HLN comparisons with a global Holm adjustment, and a model confidence set. VaR coverage uses Kupiec, Christoffersen and dynamic-quantile tests; horizons above one day are evaluated on non-overlapping offsets. ES is evaluated with FZ0 and an Acerbi-Szekely diagnostic; the latter reports a reproducible Monte Carlo p-value. Skew-t residual fitting first centers and scales the point-in-time residual pool to unit mean and variance before maximum likelihood estimation.

One-day risk includes parametric normal/t/skew-t, FHS, EVT-POT and standardized historical residual quantiles. The optional direct conditional-risk extension runs `linear_qr`, `lightgbm_quantile` and quantile random forest at 95%, 97.5% and 99%, refits every 30 days, enforces `target_end <= origin`, and rearranges crossing quantiles. Its ES is estimated from the training-only empirical loss tail beyond the predicted VaR. Quantile settings and hashes live in `config/quantile.yaml`; these runs do not edit the frozen primary config. A 30-day risk result is descriptive because its effective exception count is small.

`run_regime_economic_eval.py` joins forecast origins to trailing-only bull/bear/stress labels and writes conditional RV metrics plus volatility-target and risk-budget summaries. It also compares ES-linked dynamic collateral with a fixed 5%-of-notional margin. Capital charge, buffer and margin floor are explicit assumptions, not an exchange liquidation rule.

`run_robustness.py` evaluates a separate HAR 5/22 feature-window variant and 1,000/1,500-day rolling training windows from `config/robustness.yaml`. `make_paper_tables.py` exports T1–T11. `make_figures.py` writes F1–F9 as portable SVG. Multiple RV model comparisons use a global Holm family; the report includes MCS outputs as a complementary comparison.

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

## Scope and deferred work

Exogenous derivative/macro factors remain out of scope because no reliable historical feed with documented publication/availability timestamps is configured. Deep learning is deferred until the classical pipeline and optional dependency environment are stable; no DL result is implied by `requirements-dl.txt`. The workspace currently contains full 2-symbol × 3-horizon classical RV, VaR/ES and HAR robustness outputs. Direct conditional-quantile forecasts have been implemented and smoke-tested but have not yet been run across the complete matrix; LightGBM/XGBoost are also absent until installed from `requirements-ml.txt`. Interpret tables as complete for the generated classical/robustness matrix, not as containing those deferred quantile/tree outputs.
