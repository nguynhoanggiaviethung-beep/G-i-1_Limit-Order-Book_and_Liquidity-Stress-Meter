# Change log

## 2026-10-23 — DeepLOB validation and research figures

- Added a validation-only DeepLOB evaluation for the primary five-second horizon. It reads only pre-holdout date partitions, compares DeepLOB/Logistic/XGBoost on common sampled origins, and saves pointwise validation predictions. The already-scored holdout is not loaded or rescored.
- Ran the CNN-LSTM on CPU with PyTorch 2.14.1; the capped pilot validation result did not outperform Logistic/XGBoost for either symbol/task. Results are labeled validation-only and do not answer final-holdout RQ3.
- Added reproducible figure generation for the LOB heatmap, OBI versus future return, confusion matrices, calibration, performance over time, and DeepLOB validation diagnostics. Existing holdout panels read saved aggregate metrics only.
- Added Matplotlib to development requirements and refreshed implementation/research documentation to state the evidence boundary.

## 2026-10-23 — Submission scope and evidence audit

- Replaced the November 2026 prospective split in the active research configuration with the already completed historical archive pilot: train 2026-09-28–10-04, validation 2026-10-05–10-11, and once-scored holdout 2026-10-12–10-18. Reason: the user must submit by 2026-10-26; the November schedule cannot be completed before that deadline.
- Kept `config/settings_live.yaml` as the runtime API/collector configuration and made API defaults resolve to it. The research classifier uses `config/settings.yaml`; the live feature builder defaults to `settings_live.yaml` so user captures continue into the dashboard data tree.
- Kept `config/settings_historical_pilot.yaml` unchanged as the exact legacy configuration referenced by the saved metrics hash; it is provenance only, not the active research or runtime default.
- Updated research design, methodology, README, and historical pilot notes to call the archive results a retrospective pilot and distinguish them from live dashboard captures. DeepLOB was not evaluated during that earlier audit; its later validation-only result is documented above, and it remains unscored on the once-scored holdout.
- Added `backend/scripts/build_submission_evidence.py`. It reads archived QA sidecars, grids, feature partitions, and already saved result JSON. It emits quality, feature, label, split, metric, and robustness tables without fitting a model or re-scoring the holdout.
- Added explicit feature/label schema checks to API artifact loading and richer dependency/manifest provenance for future training runs. All six saved BTC/ETH artifacts passed the new loader check.
- Added an artifact inventory that records checksums and flags provenance fields absent from the older saved pilot bundles instead of backfilling unknown values.
- Replaced the stale Tail-Risk E2E fixture with a controlled replay smoke fixture for the LOB dashboard; the spec can be listed, though browser execution is blocked by this environment's Playwright plugin setup timeout.
- Preserved the final-test repeat guard in `run_lob_classification.py`. Do not use `--allow-repeat-final-test` for this submission.

## Research boundary

The schedule change is explicitly deadline-driven and changes the submission claim to a retrospective pilot; it does not claim to fulfill the originally planned multi-fold prospective study. A future prospective study requires a separately frozen schedule before collecting its data.
