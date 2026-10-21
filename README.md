# Limit Order Book & Liquidity Stress Meter

Research and replay dashboard for short-horizon cryptocurrency market microstructure. The active project predicts (1) three-class mid-price direction and (2) liquidity stress from BTCUSDT and ETHUSDT L2 order books.

## Research scope

- Historical Binance Spot L2 events and trades from the approved CryptoHFTData archive for the frozen 2025 study period, plus a public-API collector for new BTCUSDT and ETHUSDT captures. Binance REST does not backfill historical L2 depth.
- Replay into a valid one-second grid with the top 20 bid and ask levels. Sequence gaps invalidate samples until a complete snapshot restores the book.
- Primary horizon: 5 seconds. Sensitivity horizons: 1 and 10 seconds.
- Primary liquidity-stress label: future spread above the training-period Q95; thresholds are fitted on training data only. A composite label is an optional extension.
- Baselines: one-second direction persistence, stress-state persistence, Logistic Regression, Random Forest, and XGBoost. A sequence-based DeepLOB-style CNN-LSTM is evaluated as an optional offline model.
- Expanding walk-forward development through 2025-11-30; 2025-12-01 through 2025-12-31 is the untouched final test period.
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

This checkout does not include the six-month historical source dataset or trained classification artifacts. The live collector can build new replay data from its start time onward, but it cannot recreate the frozen 2025 holdout. Until compatible data is captured or supplied and models are trained, meter probabilities return `unavailable` with a reason; the dashboard does not fill in demo probabilities.

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

Collect ten minutes of BTCUSDT data:

```powershell
& $PY backend\scripts\collect_binance_lob.py --symbol BTCUSDT --duration-seconds 600
```

The command prints the raw capture folder and normalized event folder. Replay the event partitions, validate the resulting grid, and build features:

```powershell
$EVENTS = Read-Host "Paste the normalized event folder ending in \events\symbol=BTCUSDT"
& $PY backend\scripts\replay_cryptohftdata.py --input $EVENTS --symbol BTCUSDT
& $PY backend\scripts\validate_lob_data.py --symbol BTCUSDT
& $PY backend\scripts\build_lob_classification_dataset.py --symbol BTCUSDT
```

Repeat for ETHUSDT as needed. The dashboard can replay the normalized grid; prediction probabilities remain unavailable until compatible models are trained. Live collection starts at capture time and cannot backfill the project's June–December 2025 historical book.

To download historical trade flow only, use the public aggregate-trade endpoint:

```powershell
& $PY backend\scripts\download_binance_aggtrades.py --symbol BTCUSDT --start 2025-06-28T00:00:00Z --end 2025-06-29T00:00:00Z
```

Those trade files are not valid input for historical book reconstruction. The frozen 2025 model splits still require an approved L2 snapshot/depth-update archive covering the same period.

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

The script writes full-depth features, rolling OFI/trade flow, price impact, and 10/60/300-second RV to `data_processed/lob/features/` using atomic file replacement. Run the walk-forward development and one-time December final evaluation with:

```powershell
& $PY backend\scripts\run_lob_classification.py
```

This command writes metrics to ignored `outputs/tables/<SYMBOL>/` and trained bundles to ignored `backend/artifacts/meter/<SYMBOL>/`. It refuses to score December again when an output already exists. Pass `--allow-repeat-final-test` only when an intentional re-score is approved. Optional filters include `--symbol BTCUSDT` and `--horizon 5`.

The run reports monthly stress-label prevalence and stops if development prevalence is outside the configured 2–10% range. The primary label is future spread greater than the train-only Q95. It also reports δ and L sensitivity, paired McNemar and Diebold–Mariano Brier tests against persistence, prediction latency, and model size. Impact notionals and tick sizes are configured per symbol in `config/settings.yaml`.

DeepLOB requires the optional PyTorch dependency. Install it and run the sequence-model comparison with:

```powershell
& $PY -m pip install -r requirements-dl.txt
& $PY backend\scripts\run_deeplob_classification.py --symbol BTCUSDT --horizon 5
```

The DeepLOB run uses the same development/final-test date boundaries, compares against Logistic Regression and persistence on the same evenly spaced sample of up to 10,000 final-test origins, and writes separate weights and metrics under ignored artifact/output directories. Its JSON reports sampled and available row counts. It is subject to its own final-test repeat guard.

## API

- `GET /health` — process and replay-index status.
- `GET /api/v1/data-status` — available symbols, partitions, dates, and row counts.
- `GET /api/v1/replay/{symbol}?start=&end=&depth=&limit=` — historical one-second book snapshots.
- `GET /api/v1/replay/{symbol}/at?timestamp_utc=&depth=` — latest valid snapshot at or before a UTC timestamp.
- `GET /api/v1/meter/{symbol}?timestamp_utc=&horizon_seconds=` — direction probabilities, stress probability, spread, imbalance, OFI, model version, and quality flags.
- `POST /api/v1/predict` — score a supplied feature row with the saved artifact.
- `GET /api/v1/models` — trained artifact availability and summary metrics.
- `GET /api/v1/evaluation/{symbol}` — out-of-sample metrics by horizon and regime.

Use ISO-8601 UTC timestamps; malformed timestamps are rejected with HTTP 422. `/replay/{symbol}/at` and timestamped `/meter/{symbol}` refuse snapshots older than the configured two-second limit. `/predict` and the meter include missing model features in `quality`; they return `unavailable` when the missing fraction exceeds 10%. API reads use a small partition cache. If data, labels, or trained models are missing, responses carry `status: "unavailable"` and a reason.

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
- The repository currently has no historical source archive or live capture. Generate actual-source coverage and stress prevalence after an approved archive is supplied or new captures are collected; live data does not replace the frozen 2025 holdout.

See [methodology.md](docs/methodology.md) for definitions and [research_design.md](docs/research_design.md) for frozen defaults.
