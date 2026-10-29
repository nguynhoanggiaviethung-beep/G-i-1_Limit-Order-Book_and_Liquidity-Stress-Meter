# Implementation and evidence status

Statuses distinguish code presence from verification with real data.

## Six-month study (main evidence)

| Plan step | Implementation | Evidence | Status and boundary |
|---|---|---|---|
| 1. Protocol and grid | `config/settings_6m.yaml`, `build_unified_6m_grid.py` | `outputs/manifests/unified_6m/grid_sources.csv` | Frozen 2026-10-27; development is mostly warm-start books, the test uses exact books only |
| 2. Features | `build_lob_classification_dataset.py`, `recompute_impact_features.py` | impact backups in `data_processed/cryptohftdata_6m_unified/impact_backup/` | 2,000 USDT impact notional chosen by a training-only coverage rule before the test was opened |
| 3. Labels | `app/labels/` | stress prevalence by month in each `classification_metrics.json` | Composite stress inside the 2–10% guard on development and test |
| 4. Walk-forward | `run_lob_classification.py --development-only` | `outputs/tables/walk_forward_6m/` | 12 expanding weekly folds, 300 s embargo, 250k rows per fit |
| 5. RQ1/RQ2 ablations and HAC | `build_walk_forward_tables.py`, `run_lob_econometrics.py` | `walk_forward_6m/*.csv`, `outputs/tables/econometrics_6m/` | Development only |
| 6. DeepLOB and latency | `run_deeplob_validation.py` | `outputs/deeplob_6m/`, runtime profiles in the metrics JSON | DeepLOB on development only; it lost and was not scored on the test |
| 7. Regimes, bootstrap and paired tests | `run_lob_regime_bootstrap.py` | `outputs/tables/regime_bootstrap_6m/` | Development only |
| 8. Untouched test | `run_lob_classification.py` (final mode), `run_lob_test_regime_bootstrap.py` | `outputs/tables/final_test_6m/` (markers, metrics, per-second predictions, `regime_bootstrap/`) | Scored once per symbol on 2026-10-28; later analyses read saved predictions only |
| 9. Product and docs | `replay_service.py`, `config/settings_live.yaml`, `LOBApp.tsx`, `build_six_month_figures.py`, `build_six_month_report_tables.py` | `/api/v1/models`, `/api/v1/evaluation/{symbol}`, research page, `outputs/research_figures_6m/`, `outputs/tables/report_6m/` | The API serves `backend/artifacts/meter_6m` and refuses artifacts whose training impact notional differs from runtime. API predictions match the research pipeline exactly on development rows |

## Archive pilot (2026-10-26 submission)

`data-verified` in this section refers only to the completed historical archive pilot.

| Plan requirement | Implementation | Test / evidence | Status and boundary |
|---|---|---|---|
| BTCUSDT/ETHUSDT one-second LOB replay | `backend/app/orderbook/`, `backend/app/data/cryptohft_replay.py`, `replay_cryptohftdata_archive.py` | Replay and pilot validator JSON under `outputs/manifests/historical_archive_pilot/` | Implemented; archive QA verified |
| Data coverage, gaps, invalid rows, checksums | `backend/scripts/build_submission_evidence.py` | `outputs/data_quality/` | Implemented; generated from current local archive and QA sidecars |
| Deterministic replay and schema contract | `backend/app/data/`, `docs/data_dictionary.md` | replay / contract tests in `backend/tests/` | Test evidence depends on local test run |
| Point-in-time feature generation and event-level OFI | `backend/app/features/`, `build_lob_classification_dataset.py` | `outputs/features/`, project and feature tests | Implemented; feature statistics use disclosed deterministic temporal samples for quantiles/correlation |
| Direction and composite-stress targets | `backend/app/labels/` | saved thresholds and distributions in `outputs/labels/` | Implemented; thresholds/class distributions copied from saved pilot metrics; pointwise labels are not persisted |
| Purge, time-based embargo, chronological validation | `run_lob_classification.py`, `app/evaluation/lob_classification.py` | `outputs/splits/fold_manifest.csv`, `leakage_audit.json` | Implemented for the one-validation-week pilot; no multi-fold claim |
| Baselines and classical models | `app/models/lob_classifiers.py`, `run_lob_classification.py` | saved validation/holdout metrics under `outputs/tables/historical_archive_pilot/` | Historical pilot run exists; holdout scored once |
| DeepLOB CNN-LSTM fit/predict | `app/models/deeplob_classifier.py`, `run_deeplob_validation.py` | `outputs/deeplob/validation_metrics_h5.json`, `validation_predictions_h5.csv`, `status.json` | Verified on the single development validation week at primary h=5; 8,000 train cap and up to 2,500 validation origins/task; no final holdout read or score |
| Robustness and paired inference | `app/evaluation/lob_classification.py`, saved pilot metrics | `outputs/robustness/` | Available fields are exported from saved metrics; unavailable analyses are not recomputed |
| Artifact compatibility and unavailable API states | replay API/service and artifact metadata | `outputs/reproducibility/artifact_inventory.csv`, `backend/tests/test_project_contracts.py`, `/api/v1/models`, `/api/v1/meter` | Schema checks implemented; all six saved artifacts load, while the inventory identifies provenance fields absent from those older pilot artifacts |
| Read-only live/replay dashboard | `frontend/src/LOBApp.tsx`, `backend/app/services/live_market_service.py`, and `backend/app/api/routes/live.py` | frontend unit tests/build and backend API/live-service tests | Live stream is sequence-checked and starts with the API; live prediction waits for the continuous rolling-feature warm-up. It is operational monitoring, not live-performance research evidence. |
| Research figures listed in the plan | `backend/scripts/build_research_figures.py` | `outputs/research_figures/` | Generated from validation-only book/OBI samples, pointwise DeepLOB validation output, and previously saved aggregate classical holdout metrics; holdout was not re-read or rescored. Performance-by-time is limited to one validation and one holdout week |
| Pre-deadline research/report schedule | `docs/research_design.md`, `README.md` | 2026-10-23 audit; 2026-10-24–25 report handoff; 2026-10-26 submission | Feasible schedule for completed retrospective pilot; not a prospective study schedule |

## Rebuild local audit tables

From the repository root, run:

```powershell
$PY = ".\.venv\Scripts\python.exe"
& $PY backend\scripts\build_submission_evidence.py
```

The command reads existing partitions, saved metrics, and joblib metadata. It does not download data, train models, or score any holdout. Feature quantiles and correlations are calculated on a deterministic temporal sample capped at 5,000 rows per date partition; means, standard deviations, and missing rates are accumulated over all valid feature rows. The script leaves source-level duplicate-event counts null when QA does not provide them. Existing pilot artifacts predate the new dependency/manifest metadata fields, which are flagged in the artifact inventory instead of being filled retroactively.

Use `--quality-only` to refresh just the date-level quality outputs. The SHA-256 values in `data_manifest.csv` cover normalized Parquet grid partitions; provider source-file checksums remain in the source download manifests where available.

## DeepLOB and figure evidence

DeepLOB was evaluated only on development validation for horizon 5 seconds, the pilot's primary horizon. Logistic and XGBoost were trained on the same sampled sequence endpoints for a fair comparison. The validation sample is compute-limited and does not replace final holdout evaluation. The once-scored classical holdout remains closed to new model scoring.

The figure generator writes eight PNGs plus an index under `outputs/research_figures/`. Holdout confusion matrices and calibration curves are reconstructed only from the aggregate summaries already stored in `classification_metrics.json`. The LOB heatmap and OBI-forward-return plot use validation-week rows; the DeepLOB figures use the saved pointwise validation predictions. No pointwise classical holdout predictions are reconstructed.
