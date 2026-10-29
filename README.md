# Limit Order Book & Liquidity Stress Meter

Research and replay dashboard for short-horizon cryptocurrency market microstructure. The active project predicts (1) three-class mid-price direction and (2) liquidity stress from BTCUSDT and ETHUSDT L2 order books.

## Research scope

- **Main study: six months, walk-forward** (`config/settings_6m.yaml`).
  - Development runs 2026-04-19 – 2026-08-23, with 12 expanding weekly validation folds from 2026-06-01.
  - The untouched test runs 2026-08-24 – 2026-09-27 (five weeks, exact books only). It was scored **once** per symbol on 2026-10-28, under a protocol frozen beforehand in [methodology.md](docs/methodology.md).
- **Tasks.**
  - Three-class mid-price direction at 1, 5 and 10 s (5 s is primary).
  - Composite liquidity stress: future spread, 2,000 USDT price impact or 60 s realized volatility above training Q97.5, or top-10 depth below training Q2.5.
- **Models.** Class-frequency and persistence baselines; Logistic (OBI-only, OBI+OFI, full features); random forest; XGBoost. DeepLOB CNN-LSTM was evaluated on development only and lost to the classical models.
- **Data.** CryptoHFTData Binance Spot archive, replayed into a valid one-second grid with the top 20 levels. Sequence gaps invalidate samples until a snapshot restores the book. Binance REST does not backfill historical L2 depth.
- **Archive pilot.** The earlier one-week pilot (`config/settings.yaml`, holdout 2026-10-12 – 10-18, submitted 2026-10-26) is kept for provenance and is superseded by the six-month study.
- **Product.** A read-only dashboard with continuous Binance Spot live monitoring and historical replay. It serves the six-month models, uses public market-data endpoints only, and never places orders.

The project does not place orders and does not provide investment advice.

### Six-month test results (h = 5 s, selected model vs persistence)

| Symbol | Direction MCC | Stress PR-AUC | Stress MCC |
|---|---|---|---|
| BTCUSDT | 0.367 vs 0.255 | 0.611 vs 0.306 | 0.621 vs 0.519 |
| ETHUSDT | 0.330 vs 0.206 | 0.590 vs 0.283 | 0.628 vs 0.492 |

- **RQ1, OBI vs persistence.** OBI beats persistence only at 10 s.
- **RQ2, OFI.** OFI adds a small but significant gain that fades with the horizon.
- **Full feature set.** It adds +0.13 to +0.16 MCC over OBI+OFI.
- **Stress.** Stress is predictable mainly in high-volatility regimes.

All horizons, the paired tests, the regime breakdowns and the limits are in [research_design.md](docs/research_design.md) and [change_log.md](docs/change_log.md).

The academic report (in Vietnamese) is [paper/bao_cao_nghien_cuu.md](paper/bao_cao_nghien_cuu.md), with Word and PDF versions alongside it. Its figures are copied to `paper/figures/`. To rebuild the Word and PDF files after editing the Markdown, run `& $PY paperuild_docx.py`; this needs Microsoft Word on Windows.

## Current data and model availability

Market data is not stored in Git. Research archives and live dashboard captures use separate roots:

| Use | Location |
|---|---|
| Six-month study grid and features | `data_processed/cryptohftdata_6m_unified/{grid,features}/symbol=…/date=…/` |
| Archive pilot | `data_processed/cryptohftdata_pilot/` |
| Live and replay captures read by the dashboard | `data_processed/lob/grid/` and `data_processed/lob/features/` |

The pipeline keeps immutable source messages under `data_raw/`, normalized replay events under `data_interim/`, and grids and features under `data_processed/`. Replay accepts chronologically ordered JSONL/JSONL.GZ, either canonical `snapshot`, `depth_update` and `trade` records or native Binance Spot events. See [research_design.md](docs/research_design.md) and [data_dictionary.md](docs/data_dictionary.md) for the full contract.

Saved six-month outputs (all under `outputs/`, not in Git):
- `tables/walk_forward_6m/`: development fold metrics and summaries.
- `tables/econometrics_6m/`: the econometric benchmark.
- `tables/regime_bootstrap_6m/`: development regime and bootstrap tables.
- `deeplob_6m/`: DeepLOB development results.
- `tables/final_test_6m/`: the once-scored test, with per-second predictions, `final_test_opened.json` markers and `regime_bootstrap/`.
- `tables/report_6m/`: the six report tables (class distribution, feature summary, baseline metrics, deep model metrics, performance by regime, latency and complexity) as CSV and `report_tables.md`.
- `research_figures_6m/`: figures (weekly performance, calibration, confusion matrices, paired contrasts, stress by regime, LOB heatmap, OBI vs forward return).

The served model artifacts are in `backend/artifacts/meter_6m/`. The pilot artifacts in `backend/artifacts/meter/` and the pilot outputs are unchanged. The dashboard never reads the archive grids.

## Setup

A step-by-step guide in Vietnamese for first-time users, covering both Windows and macOS, is in [docs/huong_dan_chay_du_an.md](docs/huong_dan_chay_du_an.md). The model artifacts (`backend/artifacts/meter_6m/`, about 1.7 GB) are not in Git and must be copied in separately. Without them the dashboard shows the live book but no probabilities.

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

The live model scores only after it has a compatible artifact and enough recent continuous samples for rolling features (up to 300 seconds); after a disconnect or book resynchronization, that warm-up starts again. Live scores are operational outputs and are not evidence of out-of-sample or live predictive performance.

The API serves the artifacts in `lob.artifact_directory` of `config/settings_live.yaml` (default `backend/artifacts/meter_6m`). It refuses an artifact whose training impact notional, read from the artifact's own SHA-256-checked training configuration, differs from the runtime notional; `/api/v1/models` then reports the reason. `/api/v1/evaluation/{symbol}` serves the summary from `lob.evaluation_directory` (default `outputs/tables/final_test_6m`). The first request at 1 s loads a random-forest artifact of about 0.8–0.9 GB.

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

Build point-in-time features partition by partition. The builder carries the previous day's last 301 rows into the next day's calculation, so the RV, OFI and trade-flow windows do not restart at midnight:

```powershell
& $PY backend\scripts\build_lob_classification_dataset.py
```

The builder writes features to `data_processed/lob/features/` using atomic file replacement. Price impact uses the notional in `config/settings_live.yaml`, which must match the served models (2,000 USDT).

### Reproduce the six-month study

The test has been scored once. **Do not run the final scoring again**, and never pass `--allow-repeat-final-test`: the runner refuses while `final_test_opened.json` or `classification_metrics.json` exists. Every step below either uses development data only or reads saved outputs.

```powershell
# Development walk-forward (no test partition is read), tables, econometrics, regimes, DeepLOB
& $PY backend\scripts\run_lob_classification.py --settings config\settings_6m.yaml --development-only --metrics-dir outputs\tables\walk_forward_6m --study-label six_month_development
& $PY backend\scripts\build_walk_forward_tables.py
& $PY backend\scripts\run_lob_econometrics.py
& $PY backend\scripts\run_lob_regime_bootstrap.py
& $PY backend\scripts\run_deeplob_validation.py --settings config\settings_6m.yaml --horizon 5 --out-dir outputs\deeplob_6m --study-label six_month_development

# Analyses of the saved test predictions (no refit, no rescoring)
& $PY backend\scripts\run_lob_test_regime_bootstrap.py --symbol BTCUSDT --symbol ETHUSDT
& $PY backend\scripts\build_six_month_figures.py
& $PY backend\scripts\build_six_month_report_tables.py
```

The LOB heatmap, the OBI figure and the feature summary read development feature partitions only. The scripts refuse any partition dated on or after the test start.

The one-time scoring command, kept for the record, was `run_lob_classification.py --settings config\settings_6m.yaml --symbol <SYMBOL> --metrics-dir outputs\tables\final_test_6m --artifact-dir backend\artifacts\meter_6m --study-label six_month_untouched_test`. Each symbol takes about 4 hours on a 16 GB machine.

The runners report:
- stress-label prevalence by month;
- δ and depth-level L sensitivity;
- paired McNemar and minute-blocked Newey–West Diebold–Mariano Brier tests against persistence;
- prediction latency and model size.

DM inference averages paired losses by UTC minute and uses an automatic bandwidth with a 10-minute minimum. It withholds a p-value when fewer than 30 minute blocks are available.

DeepLOB requires optional PyTorch, and the figures require Matplotlib:

```powershell
& $PY -m pip install -r requirements-dl.txt --index-url https://download.pytorch.org/whl/cpu
& $PY -m pip install -r requirements-dev.txt
```

The archive pilot's outputs are still rebuilt with `build_submission_evidence.py` and `build_research_figures.py`. Its holdout must not be rescored. Follow the archive provider's redistribution restrictions for all figures and derived outputs.

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
- `backend/scripts/build_unified_6m_grid.py`, `recompute_impact_features.py` — six-month grid assembly and impact-notional recompute with backups.
- `backend/scripts/run_lob_classification.py` — walk-forward, model selection, one-time final scoring with saved per-second predictions.
- `backend/scripts/build_walk_forward_tables.py`, `run_lob_econometrics.py`, `run_lob_regime_bootstrap.py` — development tables, econometric benchmark, regimes and bootstrap.
- `backend/scripts/run_lob_test_regime_bootstrap.py`, `build_six_month_figures.py`, `build_six_month_report_tables.py` — test analyses, figures and report tables from saved predictions and development data only.
- `backend/scripts/build_submission_evidence.py` — read-only quality, feature, label, split, metrics, and artifact inventory reports for the saved pilot.
- `backend/app/api/routes/replay.py` — replay, meter, model, and evaluation API.
- `frontend/src/LOBApp.tsx` — Liquidity Stress Meter (live/replay) and the research page with the six-month results.
- `config/settings_6m.yaml` — six-month study: grid, split, labels, 2,000 USDT impact notional, folds.
- `config/settings.yaml` — archive pilot source, frozen pilot periods, labels, and embargo.
- `config/settings_live.yaml` — runtime API and live collector paths, served artifacts and evaluation summary; does not expose the archive grid.

The previous Tail-Risk API, scripts, tests, configurations, and results are preserved under `legacy/tail_risk_backend/` and `legacy/results_tail_risk/`; they are outside the active app and CI test path.

## Research notes

- Freeze research choices before opening final-test metrics.
- Fit direction delta, stress quantiles, scaling, feature selection, and calibration using development data only.
- Keep horizons contiguous; never create labels across missing seconds or invalid book segments.
- Report macro F1, balanced accuracy, MCC, class precision/recall, Brier score, calibration, and stress PR-AUC. Accuracy alone is not a model-selection metric.
- Compare OFI against static depth imbalance using the same model and folds.
- Report the fraction of valid replay seconds and every recovery after a sequence gap.
- The six-month untouched test is the research evaluation; it was scored once, and later analyses only read its saved predictions. DeepLOB results are development-only. The October 2026 archive pilot is superseded, and short Binance captures are operational dashboard smoke data. See date-level source and coverage artifacts under `outputs/data_quality/`; do not infer coverage during periods with no observations.

See [methodology.md](docs/methodology.md) for definitions and [research_design.md](docs/research_design.md) for frozen defaults.
