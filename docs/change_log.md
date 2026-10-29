# Change log

## 2026-10-27 — Six-month study protocol and unified grid

- Froze `config/settings_6m.yaml` before fitting any model on the six-month archive. Development 2026-04-19–2026-08-23 (expanding weekly walk-forward, first validation week 2026-06-01); untouched test 2026-08-24–2026-09-27 (five weeks, scored once); 2026-09-28–2026-10-18 is labeled `pilot_seen` because the archive pilot already used it for model selection and holdout scoring, so it is reported only as a secondary previously-seen check.
- Reason for this test window: it is the latest period the pilot never read, and every day in it has provider snapshots, so the test uses exact books only. Warm-start (approximate) books appear only in development; `validate_warm_start.py` showed ≥ 99.99% stress-label agreement after five minutes of warm-up.
- Relaxed the settings check from "exactly one week" to "one or more full UTC weeks" for the untouched test. The pilot settings are unchanged and still load.
- Added `backend/scripts/build_unified_6m_grid.py`. It hard-links, per symbol/day, the exact or warm-start partition with the most valid seconds (ties prefer exact) into `data_processed/cryptohftdata_6m_unified/grid/` and records each choice and its split in `outputs/manifests/unified_6m/grid_sources.csv`. It fails if any untouched-test day is not an exact replay.
- Result: 183 days per symbol. Development coverage 99.69% (warm) / 99.97% (the few exact days that won); untouched test 98.61%, lowered by a provider gap on 2026-09-22 (88.6%) and 2026-09-23 (64.5%) affecting both symbols. Invalid seconds are excluded from samples; the test window was not moved because of it.
- Built point-in-time features for all 183 days per symbol under `data_processed/cryptohftdata_6m_unified/features/`. Added depth-slope features (`bid_depth_slope_20`, `ask_depth_slope_20`, `depth_slope_imbalance_20`: resting notional per bp of distance to level 20); they are not yet in `LOB_CLASSIFICATION_FEATURES`, so saved pilot artifacts still load.
- Fixed the feature builder carrying warm-start-only columns (`book_source`, `book_age_seconds`) into later exact days as empty columns; exact days now get `book_source = "snapshot"`, and the warm-up tail keeps only the current day's columns. Added `--start-date` to resume a build; a resumed day is identical to a full run.
- Moved the 44.6 GB raw archive to `E:\CryptoHFT_raw\cryptohftdata` after verifying every file's path and size; `data_raw/cryptohftdata` is now a junction to it.
- Raised the replay's provider receive-time regression limit from 100 ms to 500 ms after a 100.594 ms regression stopped the 2026-05-26 warm replay; regressions remain clamped and recorded in QA.

## 2026-10-27 — Multi-fold walk-forward engine

- Added `--development-only` to `run_lob_classification.py`: it scores the 12 expanding weekly folds of `settings_6m.yaml` and stops. Partitions dated on or after the untouched-test start are never opened, no model is selected or saved, and the final-test repeat guard is not involved.
- Each run writes `walk_forward_metrics.json`, `fold_manifest.csv` (train/validation bounds, 300 s embargo, row counts, warm-start share) and `fold_metrics.csv` (fold × model, including the δ and depth-level L variants).
- Memory: the runner reads only the ~37 columns it uses, keeps feature-only columns in float32, samples training rows before slicing (same rows as before), and computes stress thresholds once per fold for all horizons.
- Smoke run (BTC, h = 5 s, 20k training rows per fit): 12 folds in 1 h 40 min (430–560 s per fold). It is a pipeline check, not a reported result.
- Profiling the largest fold showed that more than half of its time went to sklearn metric calls and stratified sampling re-encoding string labels. Speedups that leave results unchanged:
  - `classification_metrics` now derives accuracy, macro F1, balanced accuracy, MCC, per-class precision and recall, and the confusion matrix from one integer confusion matrix.
  - `_bounded_sample` stratifies on integer codes.
  - Validation rows are scored with one `predict_proba` pass, and the label is its argmax.
  - `backend/tests/test_metric_speedups.py` checks each fast path against the replaced implementation, kept verbatim in the test, and requires exact equality.
  - On real data (BTC, week 2026-08-17, h = 5 s, 20k rows) the new code reproduced the smoke run's metrics exactly. The one exception was the δ = 0.6 variant, and only because the timing harness ran that fold alone, which changes the first training segment that δ is fitted on.
  - Time for one fold: 569 s → 129 s at 20k training rows, and 1,015 s → 352 s at 250k. Peak working set at 250k is about 8.2 GB per symbol, so symbols run one after another.
- Binary Logistic labels now come from the argmax of `predict_proba`. These can differ from `predict` only if a probability rounds to exactly 0.5.

## 2026-10-28 — Econometric benchmark, DeepLOB on the six-month grid, walk-forward tables

- Added `backend/scripts/run_lob_econometrics.py` as the econometric benchmark for RQ1/RQ2, in the style of Chordia, Roll and Subrahmanyam. It uses development data only.
  - Target: the future h-second mid return in bps.
  - Regressors: standardized OBI (levels 1/5/10/20) and OFI (1/5/10 s), with optional spread and rv_60s controls.
  - In-sample OLS with Newey-West (Bartlett) standard errors, using 60 + h lags. `backend/tests/test_lob_econometrics.py` checks them against statsmodels HAC (`use_correction=True`) to 1e-10.
  - The same 12 expanding weekly folds give out-of-sample R², the incremental R² of OFI over OBI, and a minute-blocked Diebold-Mariano test on squared errors.
- Added `backend/scripts/build_walk_forward_tables.py`. It reads the development `fold_metrics.csv` files and writes:
  - model means and standard deviations over weeks;
  - week-paired RQ1/RQ2 and model contrasts, with weeks won, a paired t-test and Wilcoxon;
  - the L and δ sensitivity tables;
  - a Markdown summary.
- DeepLOB (step 6):
  - `run_deeplob_validation.py` now loads only the classifier columns plus the raw top-10 book. Raw levels are stored as decimal strings; they are parsed per partition to float32, the dtype `build_lob_sequences` already converted them to. This gives 4.8 GB for six months of BTC. Loading every column would need more than 13 GB.
  - New `--out-dir` and `--study-label` options keep six-month outputs apart from the pilot's `outputs/deeplob/`.
  - `build_lob_sequences` now converts only the rows inside requested windows rather than the whole frame. Output is unchanged, which a verbatim-reference test checks.
- `run_lob_classification._load_features` accepts `extra_columns`.

## 2026-10-28 — Stress impact notional revised (development data only)

- **Problem.** With a 0.01 tick, the 20 visible levels hold a median of about 90k USDT (BTC) and 62k USDT (ETH) on the thinner side. The frozen 250,000 / 100,000 USDT notionals therefore could not be filled in most seconds. `price_impact_bps` was missing in 77–98% of seconds, and because a missing component leaves the composite label undefined, stress labels existed for only a small share of validation seconds (BTC 1–22%, about 6k–134k of 600k per week; ETH 5–40%). That subset is the deep-book seconds, so it is not representative.
- **Rule.** Use the largest notional on the fixed grid 1k, 2k, 5k, 10k, 15k, 20k, 25k, 30k, 40k, 50k, 75k, 100k, 250k USDT that both sides fill in at least 99% of valid seconds before the first validation week minus the embargo. Result: 2,000 USDT for both symbols (BTC 99.54%, ETH 99.13%; at 25k only 88% / 81%). At 2k, about 8% of seconds still sweep past level 1, and impact correlates only weakly with spread (0.15) and rv_60s (0.24), so the component still carries information. The rule uses training data only. It was applied before the untouched test was opened, and the test is still closed.
- **Recompute.** `backend/scripts/recompute_impact_features.py` rewrites only the three impact columns of each feature partition. Before writing, it recomputes the stored values with the old notional and stops unless they match exactly. It also keeps a backup of the old columns under `data_processed/cryptohftdata_6m_unified/impact_backup/notional_<old>/`. All 366 partitions were rewritten (BTC 28 min, ETH 46 min). Test-window partitions were included so the feature store stays consistent; this is a row-wise transform and no metric was computed on them. `backend/tests/test_recompute_impact.py` checks agreement with the feature builder, the backup, that a rerun is a no-op, and the refusal to overwrite unexpected values.
- **Coverage after the fix.** Stress labels exist for about 98–100% of validation seconds (BTC 98.6–99.9%, ETH 97.8–99.9%) and 99.4% of BTC training seconds. Training prevalence is 6.6–7.2% (it was 6.0–6.7%).
- **Validation prevalence varies a lot by week.** BTC positive rates are 40%, 21%, 4.4% and 12% in the June weeks. From 13 July to 10 August they fall to 0.45–2%, then rise to 8.9% in the last week (17 August). ETH shows the same pattern (35%, 28%, …, 0.35%, then 7.7%). The early June weeks are a genuine stress episode: depth and RV exceedances are 13–26% of seconds there, in both warm-start and snapshot books. Weekly PR-AUC therefore partly tracks the base rate, and the pooled, regime and bootstrap analysis below is the better summary.
- **Pre-fix outputs** were moved, not deleted, to `outputs/tables/walk_forward_6m_pre_impact_fix/`, `outputs/tables/regime_bootstrap_6m_pre_impact_fix/` (BTC only) and `outputs/deeplob_6m_pre_impact_fix/`. The econometric benchmark does not use impact and was not rerun. Rerun logs are in `outputs/logs/rerun_impact_fix/`; all six steps exited 0 with empty stderr.

## 2026-10-28 — Six-month development results (walk-forward, DeepLOB)

The untouched test (2026-08-24 onward) was not opened. Every number here is a mean over the 12 weekly validation folds of development (2026-06-01 to 2026-08-17), with 250k training rows per fit. "Won" counts the weeks the first model did better. p is the Wilcoxon signed-rank test across weeks; with 12 weeks its smallest possible value is 0.0005. Tables are in `outputs/tables/walk_forward_6m/`: `model_summary.csv`, `paired_fold_comparisons.csv`, the sensitivity CSVs and `walk_forward_summary.md`. These are the post-fix numbers, with a 2,000 USDT impact notional. Run time: BTC 2 h 28 min, ETH 2 h 34 min, DeepLOB 54 min.

The impact columns are also classifier inputs, so the fix moved direction metrics of the full-feature models slightly (at most 0.022 in macro F1 and 0.014 in MCC). OBI-only, OBI+OFI and persistence results are unchanged.

Direction:

- **RQ1, OBI vs class frequency.** OBI beats class frequency in every week, on MCC 12/12 for both symbols and all three horizons (MCC 0.12–0.23).
- **RQ1, OBI vs persistence.** The result depends on the horizon:
  - At 1 s, persistence wins every week (BTC MCC 0.209 vs 0.277).
  - At 5 s they are about even (BTC MCC −0.013, p = 0.02; ETH no difference).
  - At 10 s, OBI wins 10/12 weeks for both symbols (BTC MCC +0.021, p = 0.002).
- **RQ2, adding OFI to OBI.** OFI wins 12/12 weeks in five of the six symbol × horizon cells, but the gain is small: MCC +0.0075 at BTC 1 s, falling to +0.0005 at 10 s. For ETH at 10 s there is no gain (6/12 weeks).
- **Full feature set vs OBI+OFI.** The full feature set adds much more than OFI: MCC +0.11 to +0.14, 12/12 weeks in every cell.
- **Model choice.** XGBoost vs Logistic is indistinguishable in most cells. Random forest is better at 1 s (BTC MCC +0.033, 12/12 weeks) and worse at 10 s (0/12 weeks for both symbols).
- **Depth level L.** Using a single OBI level, the top level is as good as or better than deeper levels. Performance degrades at L = 20.
- **Flat-band δ.** At 1 s, δ has no effect, because the flat band is the zero-return mass and the training quantiles coincide. At 5 and 10 s, a lower δ gives higher macro F1 and MCC. Because changing δ changes the labels, these numbers are not comparable across δ values.

Stress (composite label, about 7% positive in training; now evaluated on about 600k seconds per week instead of 6k–240k):

- **PR-AUC vs persistence.** Every model beats persistence on PR-AUC at every horizon. XGBoost wins 12/12 weeks in all six symbol × horizon cells, with +0.24 to +0.27. Logistic wins 10–12/12.
- **MCC at the 0.5 threshold.** XGBoost now also beats persistence: BTC +0.02 / +0.08 / +0.09 at 1 / 5 / 10 s (10–11/12 weeks) and ETH +0.03 / +0.11 / +0.12. Logistic still loses to persistence in every week (MCC −0.09 to −0.33), and its probabilities are poorly calibrated (BTC ECE 0.29–0.35). Stress write-ups should lead with PR-AUC and with XGBoost.
- **Weekly mean levels.** XGBoost PR-AUC at 5 s is 0.590 for BTC and 0.557 for ETH; persistence is 0.32 and 0.30.
- **Before vs after the fix.** Before the fix, these figures were 0.644 / 0.690 and 0.40 / 0.44, but they were computed on the selected deep-book subset, so the levels are not comparable. The ranking is the same, and the advantage over persistence is larger after the fix (BTC 5 s XGBoost +0.27 vs +0.24).

DeepLOB (validation only, h = 5 s, 8,000 training sequences per fit, 2,500 validation origins per week, compared on the same origins):

- DeepLOB loses to both Logistic and XGBoost on direction and on stress PR-AUC in nearly every week for both symbols.
  - BTC direction MCC: 0.293 vs 0.358 for Logistic. DeepLOB direction results are unchanged by the fix, because its input is the raw book.
  - BTC stress PR-AUC: 0.281 vs 0.476 for Logistic and 0.536 for XGBoost. Before the fix it was 0.368 vs 0.576 / 0.627.
  - ETH: direction MCC 0.211 vs 0.309; stress PR-AUC 0.297 vs 0.517 / 0.541.
  - Stress prevalence among the sampled origins is 8.8% (BTC) and 10.5% (ETH).
- At this training size the result does not support the CNN-LSTM. Per-week metrics are in `outputs/deeplob_6m/deeplob_fold_metrics_h5.csv`; this table has no Brier column, which the pre-fix table had.

Econometric benchmark (`outputs/tables/econometrics_6m/`; about 10.9 M seconds per symbol; Newey-West lags 60 + h):

- **In sample.** OBI and OFI are both strongly significant. Coefficients are in bps of future mid return per standard deviation of the regressor.
  - BTC 1 s: OBI level 1 has t = 9.9, and OFI over the last 1 s has t = 194.
  - Adding OFI to OBI raises R² from 0.050 to 0.058.
- **Out of sample** (12 expanding weekly folds; R² against the training mean):

  | | 1 s | 5 s | 10 s |
  |---|---|---|---|
  | BTC: OBI only | 0.042 | 0.052 | 0.043 |
  | BTC: OBI+OFI | 0.050 | 0.056 | 0.046 |
  | BTC: incremental R² from OFI | +0.0087 | +0.0049 | +0.0024 |
  | ETH: OBI only | 0.050 | 0.044 | 0.031 |
  | ETH: OBI+OFI | 0.056 | 0.045 | 0.032 |
  | ETH: incremental R² from OFI | +0.0056 | +0.0015 | +0.0006 |

  Adding spread and rv_60s controls changes nothing.
- **Diebold-Mariano test** (minute-blocked, squared error): OBI+OFI beats OBI alone at every horizon for both symbols (t = 12 to 39).
- **RQ2 answer.** OFI adds information that is statistically certain but economically small, and it decays with the horizon. This matches the small, mostly unanimous gains in classification MCC. OBI alone accounts for most of the explainable variance, which is about 3–6% of future-return variance.
- **Run notes.** Each symbol takes a few minutes. Pandas emits a NumPy "generic timedelta unit" `DeprecationWarning` during timestamp comparisons; it does not affect results.

## 2026-10-28 — Regimes, block bootstrap and paired tests (step 7, development only)

`backend/scripts/run_lob_regime_bootstrap.py` refits the walk-forward models on the same 12 folds. The models are persistence, OBI-only, OBI+OFI, Logistic and XGBoost for direction, and persistence, Logistic and XGBoost for stress, each with 250k training rows. The script pools all validation seconds (about 7.2 M per symbol and horizon), and stress rows are limited to those where persistence is defined. The untouched test was not opened. Outputs are in `outputs/tables/regime_bootstrap_6m/{SYMBOL}_model_metrics.csv`, `_contrasts.csv` and `_status.json`. Run time: BTC 65 min, ETH 44 min. `backend/tests/test_regime_bootstrap.py` covers the metrics, the bootstrap and Holm.

- **Regimes.**
  - Volatility terciles of rv_300s, with cutoffs fitted on each fold's training data.
  - UTC sessions: Asia 0–8 h, Europe 8–16 h, Americas 16–24 h.
- **Inference.**
  - 1-hour blocks resampled with replacement within each validation week, 2,000 replicates, seed 20261028.
  - Contrasts are paired: both models are evaluated on the same resampled blocks.
  - The p-value is two-sided and comes from the share of replicates on each side of zero. Its floor is 1/2001.
  - Holm correction is applied within symbol × horizon × regime × task × metric.
  - Stress PR-AUC is computed from 1,000 probability bins so it can be resampled. It differs from the exact pooled average precision by at most 0.008 (BTC) and 0.004 (ETH), as recorded in the status file.
- **Pooled results (Holm p < 0.05 unless noted).**
  - **RQ1, OBI vs persistence.** Persistence is better at 1 s (BTC MCC −0.085, ETH −0.088) and at 5 s (−0.033 / −0.027). At 10 s OBI is better on macro F1 (+0.026 / +0.017), while MCC is not distinguishable (−0.002, p = 0.54; −0.004, p = 0.12). The week means above show OBI ahead at 10 s on MCC; pooling all seconds into one MCC removes that small edge.
  - **RQ2, OFI.** The gain is significant but small: MCC +0.008 / +0.004 at 1 s and +0.003 / +0.0006 at 5 s. At 10 s it is +0.0007 for BTC, and zero and not significant for ETH.
  - **Full feature set vs OBI+OFI.** MCC +0.13 to +0.17.
  - **XGBoost vs Logistic, direction.** The MCC differences are within ±0.013.
  - **Stress.** XGBoost beats persistence on PR-AUC by +0.25 to +0.32 and has a lower Brier score. On MCC it is level at 1 s and better at 5 and 10 s (+0.03 to +0.07). XGBoost also beats Logistic on every stress metric. Logistic beats persistence on PR-AUC (+0.07 to +0.25) but loses on MCC and Brier.
- **By regime.** Each count below covers 6 regimes × 3 horizons × 2 symbols = 36 cells.
  - The full feature set beats OBI+OFI on MCC in 36/36 cells.
  - XGBoost beats persistence on stress PR-AUC in 36/36 cells (+0.12 to +0.32) and beats Logistic in 36/36.
  - OFI's gain is positive in every regime at 1 and 5 s and significant in all but one of those cells (ETH 5 s MCC). It disappears for ETH at 10 s (0/6 significant).
  - OBI vs persistence depends on the horizon in every regime. Persistence wins all 6 regimes at 1 s on macro F1 for both symbols; at 10 s OBI wins macro F1 in 5/6 regimes.
  - Logistic vs persistence on stress PR-AUC is mixed at 1 s, with 2/6 regimes negative for both symbols. In low and mid volatility it can be worse than persistence: BTC 5 s low-volatility PR-AUC is 0.088 vs 0.145.
- **Where stress is predictable.** BTC 5 s XGBoost stress PR-AUC is 0.78 in high volatility and about 0.31 in low and mid volatility. In those calm terciles positives are rare, but XGBoost still roughly doubles persistence (0.31 vs 0.14). Direction MCC is lower inside each volatility tercile than in the pooled sample (BTC Logistic 0.25–0.32 vs 0.40). This is consistent with the terciles removing between-regime variation that the pooled number benefits from. Across sessions, Europe is best for both tasks and Asia worst (BTC 5 s stress PR-AUC 0.76 vs 0.65).
- **Before vs after the impact fix (BTC, pooled).** Before the fix, stress was evaluated on 0.30–0.46 M rows; after it, on 7.2 M. Logistic vs persistence PR-AUC moved from −0.075 / +0.020 / +0.073 to +0.069 / +0.217 / +0.251 at 1 / 5 / 10 s. XGBoost vs persistence MCC moved from −0.018 / −0.047 / −0.024 to −0.001 / +0.030 / +0.037. XGBoost vs persistence PR-AUC kept its sign and grew from +0.17 to +0.27 to +0.25 to +0.30. The fix turned no model advantage into a loss. Two results flipped in the models' favour: Logistic now beats persistence on PR-AUC at 1 s, and XGBoost now beats it on MCC at 5 and 10 s.

## 2026-10-29 — Plain-language explanations

- **Research page.** The page is rewritten for readers new to market microstructure:
  - a 30-second summary;
  - a glossary of 10 terms;
  - RQ1–RQ5 asked and answered in everyday language, followed by a few numbers;
  - each figure captioned as the question it answers, with "how to read it", "notable numbers" and "what it means";
  - larger text.

  A frontend test checks the glossary and the explanation sections.
- **Report.** The academic text is kept. Three things are added:
  - a glossary table (13 terms) before the introduction;
  - a shaded "Cách đọc hình / Nói đơn giản" note under each of the seven figures;
  - a "Tóm lại" note after each result section for RQ1–RQ5.

  `paper/build_docx.py` turns Markdown blockquotes into shaded note paragraphs. The Word and PDF files are rebuilt; they now run to 18 pages.

## 2026-10-29 — Academic report

- `paper/bao_cao_nghien_cuu.md` (in Vietnamese) covers:
  - microstructure and LOB representation;
  - data and coverage;
  - target definitions and leakage controls;
  - model comparison and regime robustness;
  - RQ1–RQ5;
  - economic meaning and limits.
- It includes the six report tables and seven figures. The figures are copied to `paper/figures/`.
- `paper/build_docx.py` builds `bao_cao_nghien_cuu.docx` and `.pdf` from the Markdown through Microsoft Word (COM):
  - display equations are rendered with Matplotlib mathtext;
  - inline math becomes text with sub- and superscripts;
  - all pictures are embedded.

  The result is A4, 17 pages, with page numbers.
- Correction: an earlier research-page caption compared the OBI decile spread (< 1 bps) with the bid–ask spread. That comparison was wrong, because the BTC spread is about 0.0015 bps. The economically relevant comparison is with the standard Binance Spot taker fee of 0.1% (10 bps), and the caption now uses it.

## 2026-10-29 — Report tables and remaining figures for the six-month study

These cover the tables and figures that Research Blueprint 05 (section 4.1) lists and that the six-month study did not yet have.

- **Tables.** `backend/scripts/build_six_month_report_tables.py` writes the following to `outputs/tables/report_6m/`, as CSV files plus `report_tables.md`:
  1. class distribution, with stress prevalence by month;
  2. feature summary of the 27 model features on 10.94 M valid development seconds per symbol;
  3. baseline and classical model metrics on the test;
  4. DeepLOB vs Logistic and XGBoost over the 12 development weeks;
  5. selected model vs persistence by regime, with bootstrap CIs;
  6. latency and model complexity.

  The script reads saved outputs only. The feature summary reads development partitions and skips any partition dated on or after the test start.
- **What the tables show.**
  - Monthly stress prevalence moves widely in development (BTC 3.2–15.9%, ETH 1.8–16.6%, both peaking in June). The pre-registered 2–10% range applies to the pooled development and test rates, and those passed. The monthly spread matters when reading the regime results.
  - DeepLOB loses to Logistic in 12/12 weeks on direction MCC, and to Logistic and XGBoost in 12/12 weeks on stress PR-AUC, for both symbols.
  - The DeepLOB network has about 39.5 k parameters. On synthetic input it takes 0.22–0.25 ms per row on CPU, against 0.003 ms for Logistic and XGBoost and 0.14 ms for the random forest. The random forest artifact is 0.80–0.92 GB, while the others are under 1 MB.
- **Figures.** `build_six_month_figures.py` adds three figures:
  - Test confusion matrices from the saved summary.
  - An LOB heatmap of 2026-08-17 14:00–14:05 UTC.
  - OBI L10 against the 5 s forward return over the 12 validation weeks: Spearman +0.37 for BTC and +0.31 for ETH, decile means −0.6 to +0.6 bps, and a zero median in 8/10 deciles.

  The two book figures read development partitions only.
- **Shared figure code.** The heatmap, OBI and confusion functions in `build_research_figures.py` now take frames and file names as arguments, so both studies use the same code. The heatmap now drops book levels beyond ±10.5 ticks instead of piling them onto the edge rows. The published pilot heatmap was drawn before this fix, so it has that artifact.
- **Research page.** It gains the test confusion matrices. The pilot heatmap and OBI figures are replaced by the six-month development versions.
- **Tests.** `backend/tests/test_six_month_report.py` covers the guard against reading test partitions, the invalid-book filter, the DeepLOB week counts with their refusal of test results, and the Markdown output.

## 2026-10-29 — Product and documentation moved to the six-month study (step 9)

- **Served models.** `config/settings_live.yaml` now sets `artifact_directory: backend/artifacts/meter_6m` and `evaluation_directory: outputs/tables/final_test_6m`. Both are new settings, read by `ReplayService`. The pilot artifacts are left in place.
- **Impact notional.** The runtime impact notional is now 2,000 USDT for both symbols. `price_impact_bps` is a model input, so the API checks every artifact against it:
  - It reads the notional from the artifact's own training configuration, which must still match its stored SHA-256.
  - If the two notionals differ, it refuses the artifact and gives the reason in `/api/v1/meter` and `/api/v1/models`.
  - The meter always recomputes impact from the book at the runtime notional. A feature partition built under the old notional therefore cannot reach the model.
- **Parity.** On 200 development rows, the API's calibrated probabilities matched the research pipeline to within 2.2e-16 at 1 s and 5 s.
- **`/api/v1/evaluation/{symbol}`.** It previously read a path that did not exist. It now serves the once-scored six-month summary.
- **Figures.** `backend/scripts/build_six_month_figures.py` builds weekly performance (12 development weeks and 5 test weeks), test calibration, paired contrasts and stress by regime. It uses saved outputs only, and weekly test metrics are recomputed from the saved predictions. Outputs go to `outputs/research_figures_6m/` and `frontend/public/research/6m/`.
- **Research page.** RQ1–RQ5 are rewritten with six-month test evidence. The pilot holdout and pilot DeepLOB figures are removed from the page, and their files are kept. The two descriptive book figures remain, labelled as pilot.
- **Documentation.** `README.md`, `docs/research_design.md`, `docs/implementation_status.md` and `docs/huong_dan_chay_du_an.md` now describe the six-month study as the main evidence, with reproduction commands that never rescore the test.
- **Tests.** New tests cover the notional check and the live settings. Backend 119 passed; frontend 6 passed and the build is clean.

## 2026-10-29 — Untouched test scored once (step 8)

The six-month untouched test (2026-08-24 to 2026-09-27, five weeks, about 2.98 M seconds per symbol) was scored once per symbol under the protocol frozen in `docs/methodology.md` on 2026-10-28. The test was opened at 2026-10-28T17:21:44Z for BTC and 2026-10-28T21:11:25Z for ETH (`final_test_opened.json`). It has not been rescored and `--allow-repeat-final-test` was never used. Outputs are in `outputs/tables/final_test_6m/{SYMBOL}/`: `classification_metrics.json`, `fold_metrics.csv`, and the per-second predictions `test_predictions_h{h}_{task}.parquet`. Model artifacts are in `backend/artifacts/meter_6m/`; the pilot artifacts in `backend/artifacts/meter/` were left untouched.

- **Runs.**
  - Before the real run, the full path was rehearsed on development data only, with a fake test 2026-08-10 to 2026-08-23 (`outputs/rehearsal/`, label `rehearsal_not_a_result`).
  - Real run time: BTC 4 h 08 min, ETH 3 h 47 min, peak memory about 8.4 GB.
  - The walk-forward inside the final run reproduced `outputs/tables/walk_forward_6m/` exactly: 648/648 fold rows per symbol, with differences of 0 to 1e-16.
- **Selection** (development only, unchanged from the step-5 tables).
  - Direction: random forest at 1 s, Logistic at 5 and 10 s.
  - Stress: XGBoost at every horizon.
  - Stress decision thresholds: 0.40 to 0.45 on the calibrated probability.
- **Stress prevalence on the test.** BTC 3.8% (August 7.1%, September 2.8%); ETH 4.2%. Both are inside the pre-registered range of 2% to 10%.

Headline: the selected calibrated model against persistence (McNemar and minute-blocked DM on Brier; every p < 1e-80).

| Symbol | h | Direction macro F1 model / persistence | Direction MCC | Stress PR-AUC model / persistence | Stress MCC |
|---|---|---|---|---|---|
| BTC | 1 s | 0.479 / 0.522 | 0.297 / 0.289 | 0.724 / 0.453 | 0.686 / 0.650 |
| BTC | 5 s | 0.558 / 0.447 | 0.367 / 0.255 | 0.611 / 0.306 | 0.621 / 0.519 |
| BTC | 10 s | 0.526 / 0.350 | 0.318 / 0.163 | 0.567 / 0.261 | 0.591 / 0.471 |
| ETH | 1 s | 0.499 / 0.510 | 0.319 / 0.275 | 0.682 / 0.407 | 0.654 / 0.610 |
| ETH | 5 s | 0.538 / 0.410 | 0.330 / 0.206 | 0.590 / 0.283 | 0.628 / 0.492 |
| ETH | 10 s | 0.473 / 0.321 | 0.251 / 0.122 | 0.559 / 0.248 | 0.597 / 0.454 |

- **Direction.**
  - At 5 and 10 s the selected model beats persistence on every metric.
  - At 1 s it beats persistence on MCC and Brier but loses on macro F1. Isotonic calibration followed by argmax pushes predictions towards "flat". The uncalibrated random forest has macro F1 0.577 (BTC) and 0.574 (ETH), above persistence. Calibration was chosen before the test, so the calibrated number is the headline.
  - The Brier score is lower than persistence everywhere: −0.19 to −0.50.
  - Calibration error (ECE) per class is 0.005 to 0.053.
- **Stress.**
  - XGBoost beats persistence on PR-AUC by +0.27 to +0.31, and on MCC and Brier at every horizon.
  - The spread-only Q95 sensitivity is degenerate on the test, with PR-AUC 0.001 to 0.004 and a positive rate of about 0.04%, as it was on development.
- **Test vs development.** The test numbers are close to the 12-week development means, so there is no sign of overfitting to the folds. For example, BTC 5 s Logistic direction MCC is 0.387 uncalibrated on the test vs 0.365 on development, and XGBoost stress PR-AUC is 0.613 vs 0.590.
- **Latency.** Mean prediction time is 0.003 to 0.14 ms per row; the random forest is the slowest. The random forest artifact is about 0.8 to 0.9 GB per horizon.

Regimes, block bootstrap and paired tests on the saved test predictions (`backend/scripts/run_lob_test_regime_bootstrap.py`). The script only reads the saved parquet files: nothing is refitted and no feature partition is opened. The rules are the same as step 7:
- volatility terciles with cutoffs from development training rows, and UTC sessions;
- 1-hour blocks resampled within each test week, 2,000 replicates, seed 20261028, and Holm correction;
- stress scored on rows where persistence is defined.

The selected stress model is evaluated at its frozen threshold and every other model at 0.5. Binned PR-AUC differs from the exact value by at most 0.005. Outputs are in `outputs/tables/final_test_6m/regime_bootstrap/`, and `backend/tests/test_test_regime_bootstrap.py` covers the script.

- **Pooled results** (Holm p < 0.05 unless noted).
  - **RQ1, OBI vs persistence.** As on development, the answer depends on the horizon. Persistence is better at 1 s (MCC −0.089 BTC, −0.074 ETH) and at 5 s (−0.029 / −0.018; ETH macro F1 level, p = 0.51). OBI is better at 10 s (MCC +0.015 / +0.018, macro F1 +0.030 / +0.046).
  - **RQ2, OFI.** The gain is significant but small at 1 s (MCC +0.007 / +0.005) and at 5 s (+0.003 / +0.001). At 10 s it is negligible: +0.0003 for BTC and −0.0002 for ETH.
  - **Full feature set vs OBI+OFI.** MCC +0.13 to +0.16 everywhere.
  - **XGBoost vs Logistic, direction.** Within ±0.02 on MCC. It has a lower Brier score everywhere.
  - **Stress.** XGBoost beats Logistic on PR-AUC, MCC and Brier at every horizon for both symbols. Logistic beats persistence on PR-AUC (+0.09 to +0.28) but loses on MCC at 1 and 5 s.
- **By regime** (6 regimes × 3 horizons per symbol).
  - The full feature set beats OBI+OFI on MCC in 36/36 cells.
  - The selected stress model beats persistence on PR-AUC in 36/36 cells.
  - Stress MCC vs persistence depends on the regime. The model wins in high volatility and in every session. It loses for ETH in the low and mid volatility terciles at every horizon (for example 5 s low volatility, 0.099 vs 0.143); for BTC the low/mid cells are mostly small wins or ties. In calm regimes stress positives are rare, and the fixed threshold from pooled walk-forward predictions does not suit them.
  - Stress is mainly predictable in high volatility and in the European session. For BTC at 5 s, PR-AUC is 0.72 in high volatility vs 0.15 to 0.22 in low and mid, and 0.76 in Europe vs 0.32 in Asia.
  - Direction MCC inside each volatility tercile (0.20 to 0.30) is lower than the pooled value, as on development.
- **Limits.** The test is five weeks from one market period. Hour blocks within a week are treated as exchangeable, and the bootstrap does not cover a change of market regime beyond these weeks.

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
