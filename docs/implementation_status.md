# Implementation and evidence status

Statuses distinguish code presence from verification with real data. `data-verified` refers only to the completed historical archive pilot unless stated otherwise.

| Plan requirement | Implementation | Test / evidence | Status and boundary |
|---|---|---|---|
| BTCUSDT/ETHUSDT one-second LOB replay | `backend/app/orderbook/`, `backend/app/data/cryptohft_replay.py`, `replay_cryptohftdata_archive.py` | Replay and pilot validator JSON under `outputs/manifests/historical_archive_pilot/` | Implemented; archive QA verified |
| Data coverage, gaps, invalid rows, checksums | `backend/scripts/build_submission_evidence.py` | `outputs/data_quality/` | Implemented; generated from current local archive and QA sidecars |
| Deterministic replay and schema contract | `backend/app/data/`, `docs/data_dictionary.md` | replay / contract tests in `backend/tests/` | Test evidence depends on local test run |
| Point-in-time feature generation and event-level OFI | `backend/app/features/`, `build_lob_classification_dataset.py` | `outputs/features/`, project and feature tests | Implemented; feature statistics use disclosed deterministic temporal samples for quantiles/correlation |
| Direction and composite-stress targets | `backend/app/labels/` | saved thresholds and distributions in `outputs/labels/` | Implemented; thresholds/class distributions copied from saved pilot metrics; pointwise labels are not persisted |
| Purge, time-based embargo, chronological validation | `run_lob_classification.py`, `app/evaluation/lob_classification.py` | `outputs/splits/fold_manifest.csv`, `leakage_audit.json` | Implemented for the one-validation-week pilot; no multi-fold claim |
| Baselines and classical models | `app/models/lob_classifiers.py`, `run_lob_classification.py` | saved validation/holdout metrics under `outputs/tables/historical_archive_pilot/` | Historical pilot run exists; holdout scored once |
| DeepLOB CNN-LSTM fit/predict | `app/models/deeplob_classifier.py`, `run_deeplob_classification.py` | `outputs/deeplob/status.json` | Not evaluated for the submission pilot; do not run against the once-scored holdout |
| Robustness and paired inference | `app/evaluation/lob_classification.py`, saved pilot metrics | `outputs/robustness/` | Available fields are exported from saved metrics; unavailable analyses are not recomputed |
| Artifact compatibility and unavailable API states | replay API/service and artifact metadata | `outputs/reproducibility/artifact_inventory.csv`, `backend/tests/test_project_contracts.py`, `/api/v1/models`, `/api/v1/meter` | Schema checks implemented; all six saved artifacts load, while the inventory identifies provenance fields absent from those older pilot artifacts |
| Read-only live/replay dashboard | `frontend/src/LOBApp.tsx`, `backend/app/services/live_market_service.py`, and `backend/app/api/routes/live.py` | frontend unit tests/build and backend API/live-service tests | Live stream is sequence-checked and starts with the API; live prediction waits for the continuous rolling-feature warm-up. It is operational monitoring, not live-performance research evidence. |
| Research figures listed in the plan | saved metrics and normalized archive grid | no new figure artifacts | Not generated for the deadline handoff: pointwise held-out predictions/labels were not persisted, and no synthetic plots are presented as results |
| Pre-deadline research/report schedule | `docs/research_design.md`, `README.md` | 2026-10-23 audit; 2026-10-24–25 report handoff; 2026-10-26 submission | Feasible schedule for completed retrospective pilot; not a prospective study schedule |

## Rebuild local audit tables

From the repository root, run:

```powershell
$PY = ".\.venv\Scripts\python.exe"
& $PY backend\scripts\build_submission_evidence.py
```

The command reads existing partitions, saved metrics, and joblib metadata. It does not download data, train models, or score any holdout. Feature quantiles and correlations are calculated on a deterministic temporal sample capped at 5,000 rows per date partition; means, standard deviations, and missing rates are accumulated over all valid feature rows. The script leaves source-level duplicate-event counts null when QA does not provide them. Existing pilot artifacts predate the new dependency/manifest metadata fields, which are flagged in the artifact inventory instead of being filled retroactively.

Use `--quality-only` to refresh just the date-level quality outputs. The SHA-256 values in `data_manifest.csv` cover normalized Parquet grid partitions; provider source-file checksums remain in the source download manifests where available.
