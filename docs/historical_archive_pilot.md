# Historical order-book archive pilot

This document records a retrospective pilot using CryptoHFTData's hourly Binance Spot L2 and trade archive. It is separate from the prospective 2026-11 study in `config/settings.yaml` and `docs/research_design.md`.

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

The pilot has one validation fold and one holdout week. It provides an end-to-end historical training artifact, but it is not the three-fold prospective study scheduled in `config/settings.yaml` and is not evidence of live predictive performance.

## Completed run

The archive replay covered every requested order-book and trade hour for both symbols. The isolated validation reports passed with 1,812,849 BTC rows and 1,812,839 ETH rows across 22 partitions each. The extra partition is the single 00:00 UTC boundary sample emitted on 2026-10-19 when the replay closes the final second; it falls outside the configured holdout and is not included in test scores. Replay reported zero sequence gaps and zero missing source hours. The validation report paths are `outputs/manifests/historical_archive_pilot/lob_validation_BTCUSDT.json` and `lob_validation_ETHUSDT.json`.

The primary five-second holdout results are:

| Symbol | Direction model | Direction macro-F1 | Direction MCC | Stress model | Stress PR-AUC | Stress flag threshold | Stress positive rate |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| BTCUSDT | Random Forest | 0.554 | 0.362 | XGBoost | 0.612 | 0.41 | 4.7% |
| ETHUSDT | Logistic Regression | 0.535 | 0.316 | XGBoost | 0.624 | 0.43 | 10.3% |

The positive stress rate was about 6.3% in BTC development and 6.4% in ETH development. The holdout rates differ by symbol, showing some period drift. All three horizons (1, 5, and 10 seconds) have trained artifacts for both symbols. Full results, including calibration, class-level metrics, paired tests, and other horizons, are in `outputs/tables/historical_archive_pilot/<SYMBOL>/classification_metrics.json`; model artifacts are in `backend/artifacts/meter/<SYMBOL>/horizon_<SECONDS>.joblib`.

The source replay grid and features remain isolated from the app. Restart the API after training to reload the artifacts. The dashboard will only return predictions when the selected symbol/time has a recent valid book in its configured grid; these pilot artifacts do not make the provider's historical book available through the dashboard.

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
  --settings config\settings_historical_pilot.yaml

# Train the historical pilot and keep its metrics separate from the prospective test.
& $PY backend\scripts\run_lob_classification.py `
  --settings config\settings_historical_pilot.yaml `
  --metrics-dir outputs\tables\historical_archive_pilot `
  --study-label historical_archive_pilot `
  --max-train-rows 50000
```

The training command caps each individual fit at 50,000 class-stratified rows to keep this CPU pilot tractable. It writes model artifacts to `backend/artifacts/meter/<SYMBOL>/`; the app can use those models on data from its configured replay grid. Pilot metrics are kept separately under `outputs/tables/historical_archive_pilot/<SYMBOL>/` so they do not occupy the prospective study's final-test output path. The artifact metadata records the historical-pilot label, sample cap, and settings-file hash.

## Data timing and interpretation

The replay clock uses the provider's `received_time` (nanoseconds converted to UTC milliseconds) as the time an event became observable. Exchange event time is retained in a separate grid field where available. This avoids assigning an update to a time before the collector received it. Provider files are replayed in published order; the book is not reconstructed by sorting sequence updates. Receive-time regressions of at most 100 ms are clamped to the preceding receive time, preserving sequence order and recording the adjustment count and maximum lag in QA; larger regressions stop replay. Missing files, sequence gaps, invalid books, coverage, and snapshot bootstraps are recorded in isolated QA outputs.

The models predict the configured one-, five-, and ten-second direction and liquidity-stress targets from one-second book features. Stress thresholds and preprocessing are learned only from the applicable training/development windows. The model's saved `metrics` describe the 2026-10-12 through 2026-10-18 historical holdout, not the prospective November test week. To use model artifacts in the app, restart the API so it loads the new files, then replay data from the app's configured grid; the dashboard must not read this provider archive under the standard license.
