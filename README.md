# Limit Order Book & Liquidity Stress Meter

Research and replay dashboard for short-horizon cryptocurrency market microstructure. The active project predicts (1) three-class mid-price direction and (2) liquidity stress from BTCUSDT and ETHUSDT L2 order books.

## Research scope

- Historical Binance Spot L2 order-book events and trades from CryptoHFTData, 2025-06-28 through 2025-12-31 UTC.
- Replay into a valid one-second grid with the top 20 bid and ask levels. Sequence gaps invalidate samples until a complete snapshot restores the book.
- Primary horizon: 5 seconds. Sensitivity horizons: 1 and 10 seconds.
- Baselines: one-second direction persistence, stress-state persistence, Logistic Regression, Random Forest, and XGBoost. DeepLOB-style CNN-LSTM is an offline extension; a Transformer or Temporal CNN is considered only if it improves validation results.
- Expanding walk-forward development through 2025-11-30; 2025-12-01 through 2025-12-31 is the untouched final test period.
- Product: a read-only Liquidity Stress Meter over historical replay. Live Binance depth is an optional later extension.

The project does not place orders and does not provide investment advice.

## Current data and model availability

Market data is not stored in Git. Place approved, normalized replay partitions under:

```text
data_processed/cryptohftdata/grid/
└── symbol=BTCUSDT/date=YYYY-MM-DD/book_1s.parquet
└── symbol=ETHUSDT/date=YYYY-MM-DD/book_1s.parquet
```

The source archive and its checksum manifest belong under `data_raw/cryptohftdata/`. The normalized grid must contain UTC timestamps, a `book_valid` flag, 20 bid and ask price/quantity levels, event-level OFI aggregated during replay, and trade-flow aggregates. See [research_design.md](docs/research_design.md) and [data_dictionary.md](docs/data_dictionary.md) for the full contract.

This checkout does not include that six-month source dataset or trained classification artifacts. Until the data is supplied and models are trained, data-status and meter endpoints return `unavailable` with a reason; the dashboard does not fill in demo probabilities.

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

## Build features and classifiers

After the normalized one-second grids are in place, build point-in-time features partition by partition:

```powershell
& $PY backend\scripts\build_lob_classification_dataset.py
```

The script writes full-depth features, rolling OFI/trade flow, price impact, and 10/60/300-second RV to `data_processed/cryptohftdata/features/` using atomic file replacement. Run the walk-forward development and one-time December final evaluation with:

```powershell
& $PY backend\scripts\run_lob_classification.py
```

This command writes metrics to ignored `outputs/tables/<SYMBOL>/` and trained bundles to ignored `backend/artifacts/meter/<SYMBOL>/`. It refuses to score December again when an output already exists. Pass `--allow-repeat-final-test` only when an intentional re-score is approved. Optional filters include `--symbol BTCUSDT` and `--horizon 5`.

The repository contains the replay-grid contract and downstream feature/model pipeline. The CryptoHFTData source archive and its provider-specific download/normalization adapter are external inputs and must be supplied by the data owner; no source data is fabricated.

## API

- `GET /health` — process and replay-index status.
- `GET /api/v1/data-status` — available symbols, partitions, dates, and row counts.
- `GET /api/v1/replay/{symbol}?start=&end=&depth=&limit=` — historical one-second book snapshots.
- `GET /api/v1/replay/{symbol}/at?timestamp_utc=&depth=` — latest valid snapshot at or before a UTC timestamp.
- `GET /api/v1/meter/{symbol}?timestamp_utc=&horizon_seconds=` — direction probabilities, stress probability, spread, imbalance, OFI, model version, and quality flags.
- `POST /api/v1/predict` — score a supplied feature row with the saved artifact.
- `GET /api/v1/models` — trained artifact availability and summary metrics.
- `GET /api/v1/evaluation/{symbol}` — out-of-sample metrics by horizon and regime.

Use ISO-8601 UTC timestamps. If data, labels, or trained models are missing, responses carry `status: "unavailable"` and a reason.

## Code map

- `backend/app/data/grid_store.py` — read-only partition catalog and replay access.
- `backend/app/orderbook/` — exact-price local book state and sequence validation retained for the event replay adapter.
- `backend/app/features/` — LOB structure, event-level OFI, point-in-time features, and rolling volatility.
- `backend/app/labels/` — future mid-price direction and train-thresholded liquidity-stress labels.
- `backend/app/models/lob_classifiers.py` — Logistic, Random Forest, and XGBoost classifier factories.
- `backend/app/evaluation/lob_classification.py` — imbalance-aware metrics and chronological folds.
- `backend/app/api/routes/replay.py` — replay, meter, model, and evaluation API.
- `frontend/src/LOBApp.tsx` — replay-first Liquidity Stress Meter.
- `config/settings.yaml` — horizons, label settings, data period, and embargo.

Older volatility/VaR/ES source modules and research outputs remain in the repository for reference. They are not mounted by the active API or shown in the current dashboard; prior presentation files are grouped under `legacy/`.

## Research notes

- Freeze research choices before opening final-test metrics.
- Fit direction delta, stress quantiles, scaling, feature selection, and calibration using development data only.
- Keep horizons contiguous; never create labels across missing seconds or invalid book segments.
- Report macro F1, balanced accuracy, MCC, class precision/recall, Brier score, calibration, and stress PR-AUC. Accuracy alone is not a model-selection metric.
- Compare OFI against static depth imbalance using the same model and folds.
- Report the fraction of valid replay seconds and every recovery after a sequence gap.

See [methodology.md](docs/methodology.md) for definitions and [research_design.md](docs/research_design.md) for frozen defaults.
