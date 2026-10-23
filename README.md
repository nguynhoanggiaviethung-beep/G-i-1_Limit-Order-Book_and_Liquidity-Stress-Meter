# Limit Order Book & Liquidity Stress Meter

Research and replay dashboard for short-horizon cryptocurrency market microstructure. The active project predicts (1) three-class mid-price direction and (2) liquidity stress from BTCUSDT and ETHUSDT L2 order books.

## Research scope

- The research result prepared for the 2026-10-26 submission is a retrospective CryptoHFTData archive pilot. Its holdout was scored once and is not used for model selection. It is not a prospective or live-performance study. Binance REST does not backfill historical L2 depth.
- Replay into a valid one-second grid with the top 20 bid and ask levels. Sequence gaps invalidate samples until a complete snapshot restores the book.
- Primary horizon: 5 seconds. Sensitivity horizons: 1 and 10 seconds.
- Primary liquidity-stress label: train-only composite of unusually wide spread, high price impact, low depth, or high realized volatility. This avoids a discrete one-tick spread threshold producing too few positive samples.
- Baselines: one-second direction persistence, stress-state persistence, Logistic Regression, Random Forest, and XGBoost. DeepLOB CNN-LSTM has a validation-only comparison for the primary five-second horizon; it did not outperform Logistic/XGBoost on the sampled validation points and was not evaluated on the once-scored holdout.
- The frozen archive pilot uses training 2026-09-28–2026-10-04, validation 2026-10-05–2026-10-11, and holdout 2026-10-12–2026-10-18. The submission evidence audit is scheduled for 2026-10-23, report handoff for 2026-10-24–25, and submission for 2026-10-26. This is a deadline-driven retrospective pilot, not a replacement claim for a prospective study.
- Product: a read-only dashboard with continuous Binance Spot live monitoring and historical replay. Live collection starts with the API, uses public market-data endpoints only, and never places orders.

The project does not place orders and does not provide investment advice.

## Current data and model availability

Market data is not stored in Git. Research archive data and live dashboard captures use separate roots. The research configuration reads the approved archive pilot under `data_processed/cryptohftdata_pilot/`; the runtime configuration reads live or user-owned partitions under `data_processed/lob/`:

```text
data_processed/lob/grid/
└── symbol=BTCUSDT/date=YYYY-MM-DD/book_1s.parquet
└── symbol=ETHUSDT/date=YYYY-MM-DD/book_1s.parquet
```

The pipeline separates immutable source messages under `data_raw/`, normalized replay events under `data_interim/`, and grids/features under `data_processed/`. Replay accepts chronologically ordered JSONL/JSONL.GZ with canonical `snapshot`, `depth_update`, and `trade` records or native Binance Spot events. Binance REST aggregate trades cannot reconstruct historical L2 books. See [research_design.md](docs/research_design.md) and [data_dictionary.md](docs/data_dictionary.md) for the full contract.

The archive pilot has 1,812,849 BTC and 1,812,839 ETH rows across 22 daily partitions per symbol; one boundary row on 2026-10-19 is outside the holdout scores. Its validation reports and model metrics are saved locally under `outputs/manifests/historical_archive_pilot/` and `outputs/tables/historical_archive_pilot/`. Live dashboard captures remain separate: BTC has 750 replay rows over 2026-10-21–22, and ETH has 599 valid seconds on 2026-10-23 with zero sequence gaps. These short captures are dashboard smoke data only. See [historical_archive_pilot.md](docs/historical_archive_pilot.md) for pilot results, source terms, and reproducible commands. The dashboard never reads the archive grid.

## Setup

Use Python 3.11 or newer and Node.js 20 or newer.

```powershell
py -3.11 -m venv .venv
$PY = ".\.venv\Scripts\python.exe"
& $PY -m pip install --upgrade pip
& $PY -m pip install -r requirements.txt
```

## Run the API and live/replay dashboard

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

Open <http://127.0.0.1:5173>. The dashboard opens in **LIVE** mode and the API is documented at <http://127.0.0.1:8000/docs>. Switch to **replay** to inspect captured or normalized historical data.

When the API starts, it continuously collects public Binance Spot depth and aggregate-trade streams for BTCUSDT and ETHUSDT, synchronizes each local order book with a REST snapshot, and reconnects/resynchronizes after interruptions. It writes source captures under `data_raw/binance_spot/captures/` and normalized events under `data_interim/binance_spot/captures/`. Stop the API with `Ctrl+C`; the collector closes its files and writes capture manifests. The dashboard itself is read-only and does not require Binance credentials.

The live model scores only after it has a compatible artifact and enough recent continuous samples for rolling features (up to 300 seconds); after a disconnect or book resynchronization, that warm-up starts again. Live scores are operational outputs and are not evidence of out-of-sample or live predictive performance. The research study remains the isolated retrospective archive pilot described above.

## Manually collect live market data

This optional CLI flow uses the same public Binance Spot endpoints and synchronization checks as the API-managed live feed. Runtime and collector defaults load `config/settings_live.yaml`; the archive research grid remains isolated.

Collect ten minutes of BTCUSDT data for a replay/dashboard smoke check:

```powershell
& $PY backend\scripts\collect_binance_lob.py --symbol BTCUSDT --duration-seconds 600
```

This short capture does not qualify for model evaluation. Omit `--duration-seconds` to collect until Ctrl+C; an interrupted run saves its partial files and manifest, but any coverage gaps remain gaps.

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

The script writes full-depth features, rolling OFI/trade flow, price impact, and 10/60/300-second RV to `data_processed/lob/features/` using atomic file replacement. The following classifier command is an entry point for a future frozen study. The submission pilot already has saved results; do not rerun it because its holdout has been scored once:

```powershell
& $PY backend\scripts\run_lob_classification.py
```

The offline classifier uses `config/settings.yaml`; the API and live collector use `config/settings_live.yaml`. The runner refuses to score an existing final-test output again. Do not pass `--allow-repeat-final-test` for the submission pilot.

For the 2026-10-26 handoff, do not rescore the classical-model final holdout. The DeepLOB comparison is isolated to the development validation week and never loads holdout partitions. Rebuild read-only audit tables with `& $PY backend\scripts\build_submission_evidence.py`; regenerate research figures with `& $PY backend\scripts\build_research_figures.py`. See [implementation_status.md](docs/implementation_status.md) for evidence boundaries.

The run reports stress-label prevalence by month and stops if development prevalence is outside the configured 2–10% range. The primary label is the train-only composite at `stress_quantile: 0.975` and `stress_depth_quantile: 0.025`. It also reports δ and L sensitivity, paired McNemar and minute-blocked Newey–West Diebold–Mariano Brier tests against persistence, prediction latency, and model size. DM inference averages paired losses by UTC minute, uses an automatic bandwidth with a 10-minute minimum, and withholds a p-value when fewer than 30 minute blocks are available. Impact notionals and tick sizes are configured per symbol in `config/settings.yaml`.

DeepLOB requires optional PyTorch; research figures require Matplotlib. To reproduce the validation-only comparison and figures:

```powershell
& $PY -m pip install -r requirements-dl.txt --index-url https://download.pytorch.org/whl/cpu
& $PY -m pip install -r requirements-dev.txt
& $PY backend\scripts\run_deeplob_validation.py --horizon 5
& $PY backend\scripts\build_research_figures.py
```

The validation-only DeepLOB run compares DeepLOB, Logistic, and XGBoost on the same deterministically sampled validation origins, with an 8,000-row cap per training fit and at most 2,500 validation points per task. Its predictions and metrics are saved under `outputs/deeplob/`. It does not score the final holdout. Research figures are saved under `outputs/research_figures/`; holdout confusion/calibration/performance figures use previously saved aggregate metrics and do not reopen or rescore the holdout. The pilot has just one validation week and one holdout week, so its time-comparison plot cannot establish long-term stability. Follow the archive provider's redistribution restrictions for all figures and derived outputs.

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
- `backend/scripts/build_submission_evidence.py` — read-only quality, feature, label, split, metrics, and artifact inventory reports for the saved pilot.
- `backend/app/api/routes/replay.py` — replay, meter, model, and evaluation API.
- `frontend/src/LOBApp.tsx` — replay-first Liquidity Stress Meter.
- `config/settings.yaml` — isolated research archive source, frozen pilot periods, labels, and embargo.
- `config/settings_live.yaml` — runtime API and live collector paths; does not expose the archive grid.

The previous Tail-Risk API, scripts, tests, configurations, and results are preserved under `legacy/tail_risk_backend/` and `legacy/results_tail_risk/`; they are outside the active app and CI test path.

## Research notes

- Freeze research choices before opening final-test metrics.
- Fit direction delta, stress quantiles, scaling, feature selection, and calibration using development data only.
- Keep horizons contiguous; never create labels across missing seconds or invalid book segments.
- Report macro F1, balanced accuracy, MCC, class precision/recall, Brier score, calibration, and stress PR-AUC. Accuracy alone is not a model-selection metric.
- Compare OFI against static depth imbalance using the same model and folds.
- Report the fraction of valid replay seconds and every recovery after a sequence gap.
- The historical archive pilot is the research evaluation for submission; DeepLOB results are validation-only and do not extend the holdout evidence. The short October 2026 Binance captures are operational dashboard smoke data. See date-level source/coverage artifacts under `outputs/data_quality/`; do not infer coverage during periods with no observations.

See [methodology.md](docs/methodology.md) for definitions and [research_design.md](docs/research_design.md) for frozen defaults.
