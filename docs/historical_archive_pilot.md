# Historical order-book archive pilot

This document records the retrospective archive study used for the 2026-10-26 submission. It remains explicitly labeled a historical pilot; it is not a prospective or live-predictive study. `config/settings.yaml` freezes this pilot. The dashboard and live collector use `config/settings_live.yaml`.

## Source and use

The provider publishes historical Binance Spot order-book levels and trades as hourly Parquet downloads. The downloader stores source files locally under `data_raw/cryptohftdata/binance_spot/` and records the download status and checksums in a manifest. Replay grids and features stay under `data_processed/cryptohftdata_pilot/`, separate from the dashboard's live/user-owned replay grid. The raw and derived historical order book are not exposed by the app.

Provider references:

- Dataset: <https://www.cryptohftdata.com/datasets/binance-orderbook-data>
- REST download format and rate limit: <https://www.cryptohftdata.com/docs/rest-orderbook>
- Terms: <https://www.cryptohftdata.com/terms>

The provider's standard terms allow personal/internal research but prohibit redistributing the data or embedding it in a downstream product, dashboard, or service. The project therefore isolates the archive's raw files, replay grid, and features from the app's data store. Only the trained model artifact and aggregate research metrics are produced for local use; do not share them if they could reconstruct source data, and get a special license before exposing archive-derived book data through a dashboard or service.

## Frozen pilot window

All dates below are UTC and cover full weeks:

| Role | Window |
| --- | --- |
| Initial training | 2026-09-28 through 2026-10-04 |
| Walk-forward validation | 2026-10-05 through 2026-10-11 |
| Holdout evaluation | 2026-10-12 through 2026-10-18 |

The pilot has one validation fold and one holdout week. The holdout was scored once and is not to be reopened for model selection. It is not a three-fold prospective study and is not evidence of live predictive performance.

## Completed run

The archive replay covered every requested order-book and trade hour for both symbols. The isolated validation reports passed with 1,812,849 BTC rows and 1,812,839 ETH rows across 22 partitions each. The extra partition is the single 00:00 UTC boundary sample emitted on 2026-10-19 when the replay closes the final second; it falls outside the configured holdout and is not included in test scores. Replay reported zero sequence gaps and zero missing source hours. The validation report paths are `outputs/manifests/historical_archive_pilot/lob_validation_BTCUSDT.json` and `lob_validation_ETHUSDT.json`.

The primary five-second holdout results are:

| Symbol | Direction model | Direction macro-F1 | Direction MCC | Stress model | Stress PR-AUC | Stress flag threshold | Stress positive rate |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| BTCUSDT | Random Forest | 0.554 | 0.362 | XGBoost | 0.612 | 0.41 | 4.7% |
| ETHUSDT | Logistic Regression | 0.535 | 0.316 | XGBoost | 0.624 | 0.43 | 10.3% |

The positive stress rate was about 6.3% in BTC development and 6.4% in ETH development. The holdout rates differ by symbol, showing some period drift. All three horizons (1, 5, and 10 seconds) have trained artifacts for both symbols. Full results, including calibration, class-level metrics, paired tests, and other horizons, are in `outputs/tables/historical_archive_pilot/<SYMBOL>/classification_metrics.json`; model artifacts are in `backend/artifacts/meter/<SYMBOL>/horizon_<SECONDS>.joblib`.

### DeepLOB validation-only comparison

DeepLOB was run with PyTorch on CPU for the primary five-second horizon. This comparison uses the single 2026-10-05–2026-10-11 validation week, up to 8,000 training rows per fit, and at most 2,500 evenly spaced validation origins per task. Logistic and XGBoost use the same training endpoints and validation origins. The final holdout was neither read nor scored by this run.

| Symbol | Task | n | DeepLOB | Logistic | XGBoost | Metric |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| BTCUSDT | Direction | 2,496 | 0.5681 | 0.5871 | 0.5822 | Macro-F1 |
| BTCUSDT | Stress | 2,497 | 0.2296 | 0.3705 | 0.4353 | PR-AUC |
| ETHUSDT | Direction | 2,496 | 0.5065 | 0.5571 | 0.5255 | Macro-F1 |
| ETHUSDT | Stress | 2,498 | 0.3081 | 0.4517 | 0.5337 | PR-AUC |

On this sampled validation week, DeepLOB did not outperform Logistic or XGBoost for either symbol/task. Treat this as a compute-limited development comparison, not an out-of-sample test conclusion. Pointwise validation predictions are saved in `outputs/deeplob/validation_predictions_h5.csv`; summary metrics are in `validation_metrics_h5.json`.

### Research figures

Run `backend/scripts/build_research_figures.py` to regenerate the PNGs in `outputs/research_figures/`. LOB heatmap and OBI-versus-forward-return figures use validation data. Holdout confusion matrices, calibration curves, and the two-period performance plot use the aggregate metrics already saved by the one-time classical-model holdout run; the script does not reopen source holdout partitions or rescore the holdout. The performance plot has only one validation and one holdout point, so it cannot establish long-term stability.

The source replay grid and features remain isolated from the dashboard. Restart the API after training to reload artifacts. Replay predictions still require a recent valid book in the configured replay grid; continuous live mode scores the public Binance stream separately after the live rolling features have warmed up. The pilot archive does not make its provider's historical book available through the dashboard, and live scores do not extend the archive holdout evidence.

## Download, replay, and train

From the repository root in PowerShell:

```powershell
$PY = ".\.venv\Scripts\python.exe"

# Download both BTCUSDT and ETHUSDT order-book and trade files.
& $PY backend\scripts\download_cryptohftdata_archive.py `
  --start-date 2026-09-28 --end-date 2026-10-18

# Replay into the isolated pilot grid and QA folder (not the dashboard's grid).
& $PY backend\scripts\replay_cryptohftdata_archive.py `
  --start-date 2026-09-28 --end-date 2026-10-18 `
  --output-root data_processed\cryptohftdata_pilot\grid `
  --manifest-root outputs\manifests\historical_archive_pilot

# Validate the isolated book and build its classifier features.
& $PY backend\scripts\validate_lob_data.py `
  --grid-root data_processed\cryptohftdata_pilot\grid `
  --report outputs\manifests\historical_archive_pilot\lob_validation.json
& $PY backend\scripts\build_lob_classification_dataset.py `
  --settings config\settings.yaml
```

The saved metrics and artifacts are the completed historical run. The final holdout was scored once. Do not rerun `run_lob_classification.py` to prepare the submission; its repeat guard exists to protect the holdout. The training run capped each fit at 50,000 class-stratified rows and wrote artifacts to `backend/artifacts/meter/<SYMBOL>/`. Metrics remain under `outputs/tables/historical_archive_pilot/<SYMBOL>/`.

## Data timing and interpretation

The replay clock uses the provider's `received_time` (nanoseconds converted to UTC milliseconds) as the time an event became observable. Exchange event time is retained in a separate grid field where available. This avoids assigning an update to a time before the collector received it. Provider files are replayed in published order; the book is not reconstructed by sorting sequence updates. Receive-time regressions of at most 100 ms are clamped to the preceding receive time, preserving sequence order and recording the adjustment count and maximum lag in QA; larger regressions stop replay. Missing files, sequence gaps, invalid books, coverage, and snapshot bootstraps are recorded in isolated QA outputs.

The models predict the configured one-, five-, and ten-second direction and liquidity-stress targets from one-second book features. Stress thresholds and preprocessing are learned only from the applicable training/development windows. Saved metrics describe the 2026-10-12 through 2026-10-18 historical holdout. To use model artifacts in the app, restart the API so it loads the saved files, then replay data from the app's configured grid; the dashboard must not read this provider archive under the standard license.
