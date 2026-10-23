# Research design: LOB direction and liquidity stress

## Submission scope and schedule

The submission deadline is 2026-10-26. A new prospective multi-week collection and a November final-test schedule cannot produce completed evidence before that deadline. The submission therefore uses the already completed **retrospective CryptoHFTData archive pilot**, and labels it as a pilot throughout. It is not presented as a prospective study or evidence of live predictive performance. The dashboard's Binance captures are operational replay data and are not mixed into the archive study.

| Stage | UTC period / deadline | Purpose and status |
|---|---|---|
| Historical training | 2026-09-28–2026-10-04 | Fit training-only thresholds and candidate models; archive data already available |
| Historical validation | 2026-10-05–2026-10-11 | Select candidate models with one chronological validation week |
| Historical holdout | 2026-10-12–2026-10-18 | Scored once; retain as holdout evidence and do not use for model selection |
| Evidence audit | 2026-10-23 | Freeze code/configuration and compile observed-data quality and result evidence |
| Report handoff | 2026-10-24–2026-10-25 | Finalize methods, limitations, and reproducibility package before submission |
| Submission | 2026-10-26 | Submit the retrospective pilot with its stated limitations |

The date windows above describe data already present in the archive; they are not a future collection schedule. The final holdout has already been scored once, so this repository must not re-run final-test evaluation as part of report preparation. If the submission later requires a prospective study, it must be a separately registered study with new dates and data, not a relabeling of this pilot.

## Objective and questions

The study examines whether point-in-time order-book state and order flow provide information about short-horizon mid-price direction and liquidity stress in Binance Spot BTCUSDT and ETHUSDT.

- **RQ1:** Do static depth-imbalance measures predict mid-price direction beyond class-frequency and persistence baselines?
- **RQ2:** Does event-level Order Flow Imbalance add information beyond static imbalance?
- **RQ3:** Do the classical baselines or DeepLOB improve mid-price/stress prediction? Classical results are reported on validation and the once-scored holdout; DeepLOB is compared with Logistic/XGBoost on validation only because the holdout is locked.
- **RQ4:** How stable are results across symbols, horizons, weeks, and volatility regimes where observations permit?
- **RQ5:** Do conclusions persist under imbalance-aware metrics, calibration, and label/feature sensitivity checks?

DeepLOB was run with PyTorch for the primary five-second horizon on the single validation week only. It uses at most 8,000 training rows and 2,500 evenly spaced validation origins per task, with Logistic and XGBoost evaluated on those same origins. In this limited comparison DeepLOB did not outperform either classical model on direction macro-F1 or stress PR-AUC. This is a validation-only pilot result, not a holdout conclusion.

## Frozen scope and defaults

| Decision | Frozen submission setting |
|---|---|
| Market | Binance Spot BTCUSDT and ETHUSDT |
| Research source | Approved CryptoHFTData public archive, isolated from dashboard data |
| Sampling | UTC one-second grid |
| Book depth | 20 levels per side; sensitivity at L=1, 5, 10, 20 |
| Horizons | 1, 5, 10 seconds; 5 seconds is primary |
| Direction classes | Up / flat / down; delta fitted on the first training week per symbol and horizon |
| Primary stress label | Composite: upper-tail spread, impact, or realized volatility and lower-tail top-10 depth; thresholds learned on training data |
| Stress sensitivity | Spread-only Q95; reported separately and never substituted after holdout inspection |
| Embargo | 300 seconds, at least the maximum of forecast horizon, 300-second feature lookback, and 100-second DeepLOB sequence |
| Study split | Train 2026-09-28–10-04; validation 2026-10-05–10-11; holdout 2026-10-12–10-18 |
| Product mode | Read-only dashboard with continuous live monitoring and replay; live scores are operational and not research evidence |

The archive replay uses provider receive time as its point-in-time replay clock. Binance public REST aggregate trades do not provide historical Level 2 depth and are not a substitute for this source. The dashboard does not read or serve the provider archive under the standard source terms.

## Data and replay contract

Replay each symbol independently. A snapshot bootstraps the local book; a depth update is accepted only when its update-ID range follows the current sequence. A sequence gap or invalid book state invalidates subsequent samples until a valid snapshot resynchronizes the book. Emit a one-second row only for a valid, non-stale state. Never interpolate across missing seconds or gaps.

Every row keeps the grid timestamp and available event/receive timestamps distinct. Event sequence—not local receive time—determines depth application and OFI accumulation. Raw inputs and normalized outputs stay outside Git. Historical and live grids use separate configured roots.

For the submission pilot, the validator and daily QA sidecars are the source of coverage and book-quality figures. The date-level output manifest must distinguish observed timestamps from expected seconds and preserve the source's timestamp basis. Missing, duplicate, invalid, stale, gap, and resynchronization counts must be reported only when measured; unknown source-level duplicate counts remain null rather than being inferred from grid rows.

## Features and targets

At origin time `t`, features use only events at or before `t`: mid-price, relative spread, spread ticks, microprice, depth and OBI at 1/5/10/20 levels, event-level OFI in trailing windows, aggressive trade imbalance, displayed-size additions/depletions, fixed-notional impact, and trailing realized variance. Public depth depletion is a proxy and does not identify order cancellations by order ID. Missing impact or volatility windows remain missing.

For horizon `h`, direction is based on `mid[t+h] - mid[t]` and a delta estimated from the first training week for each symbol/horizon. A target is unavailable if any required state in `[t, t+h]` is missing or invalid. The composite stress target uses future spread/impact/RV upper training quantiles or future top-10 depth below its training quantile. Strict threshold comparisons are retained. Incomplete stress components do not become a non-stress label.

## Validation and model comparison

Use chronological UTC-week partitions only. The pilot has one validation week and one held-out week; it cannot establish broad week-to-week generalization. Purge training origins whose label horizon overlaps validation and apply a time-based embargo of at least 300 seconds. Fit preprocessing, thresholds, and calibration using allowed training/validation data only. Candidate selection uses validation results; the holdout is opened once after that choice.

Compare class-frequency and persistence baselines with Logistic Regression, Random Forest, and XGBoost where available. Compare models on matching valid origins and target definitions. Direction metrics include macro F1, balanced accuracy, MCC, per-class precision/recall, confusion matrix, Brier score, and calibration. Stress metrics include PR-AUC, MCC, validation-selected threshold precision/recall, Brier score, calibration, and positive-class prevalence. Accuracy is contextual only.

Report delta sensitivity, OBI depth sensitivity, composite versus spread-only stress, model latency/size, and results by symbol, horizon, week, and causal trailing-volatility regime where the saved pilot outputs support them. For paired Brier-loss inference, aggregate losses by UTC minute and use the documented Newey–West rule; do not report an asymptotic p-value when there are too few minute blocks. A p-value does not substitute for effect size or calibration.

## Evidence and limits

Saved pilot metrics in `outputs/tables/historical_archive_pilot/` document the single validation week and the once-scored classical holdout. DeepLOB validation metrics and pointwise predictions are saved under `outputs/deeplob/`. Research figures are generated under `outputs/research_figures/`; holdout panels reuse saved aggregate metrics without rerunning the test. Data QA sidecars and validator outputs document actual coverage. No result may imply the November schedule was completed, that a short Binance capture is a research sample, or that the meter provides live trading predictions. Any missing requested analysis must be marked unavailable or not evaluated in the report.

The pilot is limited by its single validation week, compute-limited training sample, and reliance on an external archive's receive-time clock. DeepLOB has no final-holdout result and uses a capped validation sample. Results do not establish robust prospective performance, cross-exchange generalization, or causal effects of liquidity on price.
