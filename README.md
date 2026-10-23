# Limit Order Book & Liquidity Stress Meter

Research and replay dashboard for short-horizon cryptocurrency market microstructure. The active project predicts (1) three-class mid-price direction and (2) liquidity stress from BTCUSDT and ETHUSDT L2 order books.

## Research scope

- Prospective Binance Spot BTCUSDT and ETHUSDT L2 depth/trade captures remain the primary study. Binance REST does not backfill historical L2 depth. An optional external CryptoHFTData archive supports a separate historical pilot; it is downloaded locally and is not bundled in Git.
- Replay into a valid one-second grid with the top 20 bid and ask levels. Sequence gaps invalidate samples until a complete snapshot restores the book.
- Primary horizon: 5 seconds. Sensitivity horizons: 1 and 10 seconds.
- Primary liquidity-stress label: train-only composite of unusually wide spread, high price impact, low depth, or high realized volatility. This avoids a discrete one-tick spread threshold producing too few positive samples.
- Baselines: one-second direction persistence, stress-state persistence, Logistic Regression, Random Forest, and XGBoost. A sequence-based DeepLOB-style CNN-LSTM is evaluated as an optional offline model.
- Expanding weekly walk-forward development through 2026-11-22; 2026-11-23 through 2026-11-29 is the frozen, one-time final test week.
- Product: a read-only Liquidity Stress Meter over normalized replay data. Live Binance collection runs as an explicit CLI command and does not place orders or connect directly to the dashboard.

The project does not place orders and does not provide investment advice.

## Current data and model availability

Market data is not stored in Git. Place approved, normalized replay partitions under:

```text
data_processed/lob/grid/
└── symbol=BTCUSDT/date=YYYY-MM-DD/book_1s.parquet
└── symbol=ETHUSDT/date=YYYY-MM-DD/book_1s.parquet
```

The pipeline separates immutable source messages under `data_raw/`, normalized replay events under `data_interim/`, and grids/features under `data_processed/`. Replay accepts chronologically ordered JSONL/JSONL.GZ with canonical `snapshot`, `depth_update`, and `trade` records or native Binance Spot events. Binance REST aggregate trades cannot reconstruct historical L2 books. See [research_design.md](docs/research_design.md) and [data_dictionary.md](docs/data_dictionary.md) for the full contract.

The live capture already in this checkout contains 750 valid BTCUSDT one-second rows from 2026-10-21 through 2026-10-22. The separate historical-archive pilot for 2026-09-28 through 2026-10-18 has been replayed, validated, and trained for BTCUSDT and ETHUSDT. It uses one train week, one validation week, and one untouched historical holdout week; it does not replace the configured prospective study (first validation week 2026-11-02, development through 2026-11-22, final test 2026-11-23 through 2026-11-29). Pilot artifacts are under `backend/artifacts/meter/<SYMBOL>/`, with aggregate evaluation under `outputs/tables/historical_archive_pilot/<SYMBOL>/`. See [historical_archive_pilot.md](docs/historical_archive_pilot.md) for results, source terms, and exact commands. The live collector itself cannot recreate earlier order-book history. Restart the API after training so it loads the artifacts. Predictions are still unavailable when the selected symbol/time has no recent valid book in the app's configured replay grid; the historical archive grid is isolated and is not served by the dashboard.

## Setup

Use Python 3.11 or newer and Node.js 20 or newer.

```powershell
py -3.11 -m venv .venv
$PY = ".\.venv\Scripts\python.exe"
& $PY -m pip install --upgrade pip
& $PY -m pip install -r requirements.txt
```

## Run the API and replay dashboard

Start the API from the repository root:

```powershell
$PY = ".\.venv\Scripts\python.exe"
& $PY -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
```

Start the dashboard in a second terminal:

```powershell
cd frontend
npm ci
npm run dev
```

Open <http://127.0.0.1:5173>. The API is documented at <http://127.0.0.1:8000/docs>.

The API reads Parquet replay files and model artifacts only. It does not start a market-data collector or download data when the server starts.

## Collect live market data

The explicit collector command uses public Binance Spot endpoints and does not require an API key. It subscribes to diff-depth and aggregate trades, synchronizes depth updates against a REST snapshot, checks update IDs, and resynchronizes after gaps or reconnects. It stores raw WebSocket messages, REST snapshots, normalized replay events, an event log of connection/gap/recovery activity, checksums, and a source manifest in their respective data layers. The public depth stream contains price-level changes, not order-ID events; displayed-size depletion is therefore only a cancellation proxy.

Collect ten minutes of BTCUSDT data for a replay/dashboard smoke check:

```powershell
& $PY backend\scripts\collect_binance_lob.py --symbol BTCUSDT --duration-seconds 600
```

This short capture does not qualify for model evaluation. For the frozen weekly study, start collection no later than 2026-10-26 and keep it running through 2026-11-29 so the dataset contains a full training week, three weekly validation folds, and the untouched test week. Omit `--duration-seconds` to collect until Ctrl+C; an interrupted run saves its partial files and manifest, but any coverage gaps remain gaps.

The command prints the raw capture folder and normalized event folder. Replay the event partitions, validate the resulting grid, and build features:

```powershell
$EVENTS = Read-Host "Paste the normalized event folder ending in \events\symbol=BTCUSDT"
& $PY backend\scripts\replay_cryptohftdata.py --input $EVENTS --symbol BTCUSDT
& $PY backend\scripts\validate_lob_data.py --symbol BTCUSDT
& $PY backend\scripts\build_lob_classification_dataset.py --symbol BTCUSDT
```

Repeat for ETHUSDT as needed. After replaying and validating the captured data, the dashboard can show its book and predictions when a compatible trained artifact exists and the selected time has a recent valid book. The historical pilot artifacts are already trained, but its provider archive remains isolated from the dashboard. Live collection begins at capture time and cannot backfill historical L2.

To download historical trade flow only, use the public aggregate-trade endpoint:

```powershell
& $PY backend\scripts\download_binance_aggtrades.py --symbol BTCUSDT --start 2025-06-28T00:00:00Z --end 2025-06-29T00:00:00Z
```

Those trade files are not valid input for historical book reconstruction. Any historical evaluation still requires an approved L2 snapshot/depth-update archive covering its study period.

## Build features and classifiers

Replay one captured event file or a capture's event directory into daily one-second partitions:

```powershell
$EVENTS = Read-Host "Paste the normalized event folder ending in \events\symbol=BTCUSDT"
& $PY backend\scripts\replay_cryptohftdata.py --input $EVENTS --symbol BTCUSDT
```

The stream must be ordered by exchange event time and include snapshots to bootstrap and recover the book. The script writes `book_1s.parquet` and QA sidecars. It does not guess proprietary source columns or download historical L2 data.

Build point-in-time features partition by partition; the builder carries the previous day's trailing 301 rows into the next day's calculation so RV, OFI, and trade-flow windows do not restart at midnight:

```powershell
& $PY backend\scripts\build_lob_classification_dataset.py
```

The script writes full-depth features, rolling OFI/trade flow, price impact, and 10/60/300-second RV to `data_processed/lob/features/` using atomic file replacement. Run the expanding weekly development and one-time final-test evaluation with:

```powershell
& $PY backend\scripts\run_lob_classification.py
```

This command uses the fixed weekly boundaries in `config/settings.yaml`, writes metrics to ignored `outputs/tables/<SYMBOL>/` and trained bundles to ignored `backend/artifacts/meter/<SYMBOL>/`, and refuses to score the 2026-11-23 through 2026-11-29 test week again when an output already exists. Pass `--allow-repeat-final-test` only for an intentional re-score. Optional filters include `--symbol BTCUSDT` and `--horizon 5`.

The run reports stress-label prevalence by month and stops if development prevalence is outside the configured 2–10% range. The primary label is the train-only composite at `stress_quantile: 0.975` and `stress_depth_quantile: 0.025`. It also reports δ and L sensitivity, paired McNemar and minute-blocked Newey–West Diebold–Mariano Brier tests against persistence, prediction latency, and model size. DM inference averages paired losses by UTC minute, uses an automatic bandwidth with a 10-minute minimum, and withholds a p-value when fewer than 30 minute blocks are available. Impact notionals and tick sizes are configured per symbol in `config/settings.yaml`.

DeepLOB requires the optional PyTorch dependency. Install it and run the sequence-model comparison with:

```powershell
& $PY -m pip install -r requirements-dl.txt
& $PY backend\scripts\run_deeplob_classification.py --symbol BTCUSDT --horizon 5
```

The DeepLOB run uses the same weekly development/final-test boundaries and compares against Logistic Regression and persistence on the same evenly spaced sample of up to 10,000 final-test origins. DeepLOB and its Logistic baseline each use at most 8,000 training rows by default (`deeplob_max_train_rows`); this is a compute-limited pilot, not a full-data DeepLOB result. Its JSON reports sampled and available row counts. It is subject to its own final-test repeat guard.

## API

- `GET /health` — process and replay-index status.
- `GET /api/v1/data-status` — available symbols, partitions, dates, and row counts.
- `GET /api/v1/replay/{symbol}?start=&end=&depth=&limit=` — historical one-second book snapshots.
- `GET /api/v1/replay/{symbol}/at?timestamp_utc=&depth=` — latest valid snapshot at or before a UTC timestamp.
- `GET /api/v1/meter/{symbol}?timestamp_utc=&horizon_seconds=` — direction probabilities, stress probability and its artifact-selected `stress_probability_threshold`, spread, imbalance, OFI, model version, and quality flags.
- `POST /api/v1/predict` — score a supplied feature row with the saved artifact.
- `GET /api/v1/models` — trained artifact availability and summary metrics.
- `GET /api/v1/evaluation/{symbol}` — out-of-sample metrics by horizon and regime.

Use ISO-8601 UTC timestamps; malformed timestamps are rejected with HTTP 422. `/replay/{symbol}/at` and timestamped `/meter/{symbol}` refuse snapshots older than the configured two-second limit. `/predict` and the meter include missing model features in `quality`; they return `unavailable` when the missing fraction exceeds 10% or the supplied book is invalid. API reads use a small partition cache. If data, labels, or trained models are missing, responses carry `status: "unavailable"` and a reason.

## Code map

- `backend/app/data/binance_client.py` and `websocket_collector.py` — public REST snapshots/metadata, depth and trade capture, and sequence recovery.
- `backend/app/data/storage.py` — append-only gzip JSONL partitions, size rotation, and checksums.
- `backend/app/data/grid_store.py` — read-only partition catalog and replay access.
- `backend/app/orderbook/` — exact-price local book state and sequence validation retained for the event replay adapter.
- `backend/app/features/` — LOB structure, event-level OFI, point-in-time features, and rolling volatility.
- `backend/app/labels/` — future mid-price direction and train-thresholded liquidity-stress labels.
- `backend/app/models/lob_classifiers.py` and `deeplob_classifier.py` — shallow baselines and the optional sequence classifier.
- `backend/app/evaluation/lob_classification.py` — imbalance-aware metrics, McNemar, and HAC Diebold–Mariano Brier tests.
- `backend/scripts/collect_binance_lob.py`, `download_binance_aggtrades.py`, `replay_cryptohftdata.py`, and `validate_lob_data.py` — collection through validated one-second grids.
- `backend/app/api/routes/replay.py` — replay, meter, model, and evaluation API.
- `frontend/src/LOBApp.tsx` — replay-first Liquidity Stress Meter.
- `config/settings.yaml` — horizons, label settings, data period, and embargo.

The previous Tail-Risk API, scripts, tests, configurations, and results are preserved under `legacy/tail_risk_backend/` and `legacy/results_tail_risk/`; they are outside the active app and CI test path.

## Research notes

- Freeze research choices before opening final-test metrics.
- Fit direction delta, stress quantiles, scaling, feature selection, and calibration using development data only.
- Keep horizons contiguous; never create labels across missing seconds or invalid book segments.
- Report macro F1, balanced accuracy, MCC, class precision/recall, Brier score, calibration, and stress PR-AUC. Accuracy alone is not a model-selection metric.
- Compare OFI against static depth imbalance using the same model and folds.
- Report the fraction of valid replay seconds and every recovery after a sequence gap.
- The repository has no historical L2 archive; the short October 2026 live capture is exploratory and does not complete the configured weekly study window. Report actual-source coverage and stress prevalence after the capture window is complete. Live data cannot replace an earlier historical holdout.

See [methodology.md](docs/methodology.md) for definitions and [research_design.md](docs/research_design.md) for frozen defaults.
