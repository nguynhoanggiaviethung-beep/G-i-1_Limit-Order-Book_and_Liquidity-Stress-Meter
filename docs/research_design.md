# Research design: LOB direction and liquidity stress

The study examines whether point-in-time order-book state and order flow carry information about short-horizon mid-price direction and liquidity stress in Binance Spot BTCUSDT and ETHUSDT.

The main evidence is the **six-month walk-forward study** (`config/settings_6m.yaml`). Its untouched test was scored once on 2026-10-28. An earlier one-week **archive pilot** (`config/settings.yaml`) is kept as provenance for the 2026-10-26 submission; it is described at the end and is not mixed with the six-month results.

## Questions

- **RQ1:** Do static depth-imbalance measures (OBI) predict mid-price direction beyond the class-frequency and persistence baselines?
- **RQ2:** Does event-level Order Flow Imbalance (OFI) add information beyond OBI?
- **RQ3:** Do the classical models or DeepLOB improve direction and stress prediction? DeepLOB is compared on development only.
- **RQ4:** How stable are results across symbols, horizons, weeks and volatility or session regimes?
- **RQ5:** Do conclusions hold under imbalance-aware metrics, calibration and label sensitivity checks?

## Six-month study

| Stage | UTC period | Role |
|---|---|---|
| Development | 2026-04-19 – 2026-08-23 | Expanding weekly walk-forward. The first six weeks are training only; 12 validation weeks run from 2026-06-01 to 2026-08-23 |
| Untouched test | 2026-08-24 – 2026-09-27 | Five full weeks, exact provider-snapshot books only. Scored once per symbol on 2026-10-28 |
| Pilot seen | 2026-09-28 – 2026-10-18 | Already used by the archive pilot for selection and scoring. Never read by the six-month runs |

The protocol was frozen on 2026-10-27, and its final-test rules on 2026-10-28, before the test was opened (`docs/methodology.md`). The only design change after the freeze was the impact notional, which was fixed using training data only, before the test was opened (`docs/change_log.md`).

| Decision | Setting |
|---|---|
| Source | CryptoHFTData Binance Spot archive. The unified grid takes, per day, the exact or warm-start replay with more valid seconds (`build_unified_6m_grid.py`). Development days before 2026-08-24 mostly use warm-start books (≥ 99.99% stress-label agreement after five minutes of warm-up) |
| Sampling | UTC one-second grid, 20 levels per side; OBI sensitivity at L = 1, 5, 10, 20 |
| Horizons | 1, 5 and 10 s; 5 s is primary |
| Direction | Up / flat / down. δ is fitted on the first training segment for each symbol and horizon |
| Stress | Composite of future spread, 2,000 USDT price impact and 60 s realized volatility above training Q97.5, or top-10 depth below training Q2.5. Spread-only Q95 is a separate sensitivity. Pre-registered prevalence range: 2–10% |
| Folds | 12 expanding weekly folds, 300 s embargo, 250,000 class-stratified training rows per fit |
| Models | Class frequency, persistence, Logistic (OBI-only, OBI+OFI, full features), random forest, XGBoost. DeepLOB is evaluated on development only |
| Selection | Mean validation macro F1 for direction and mean PR-AUC for stress, with MCC as the tie-breaker |
| Final fit | Development data up to the test start minus the embargo. Isotonic calibration and the MCC-maximizing stress threshold are fitted on walk-forward out-of-sample predictions |
| Inference | McNemar and minute-blocked Diebold–Mariano tests against persistence. 1-hour block bootstrap within weeks (2,000 replicates), with Holm correction |

### Results (summary)

The full numbers are in `docs/change_log.md` (entries of 2026-10-28 and 2026-10-29) and the tables in `outputs/tables/`.

- **RQ1.** The answer depends on the horizon. On the test, persistence beats OBI at 1 and 5 s, and OBI beats persistence at 10 s (MCC +0.015 for BTC, +0.018 for ETH). This matches development.
- **RQ2.** OFI's gain is statistically significant but small, and it decays with the horizon: MCC +0.005 to +0.007 at 1 s and about 0 at 10 s. The econometric benchmark agrees: incremental out-of-sample R² falls from +0.009 to +0.002. The full feature set adds far more, +0.13 to +0.16 MCC over OBI+OFI.
- **RQ3.** DeepLOB lost to Logistic and XGBoost on development (BTC stress PR-AUC 0.281 vs 0.536 for XGBoost), so it was not scored on the test. On the test, the selected classical models beat persistence at every horizon on stress (PR-AUC +0.27 to +0.31) and on direction at 5 and 10 s. At 1 s the calibrated direction model is better on MCC and Brier but worse on macro F1.
- **RQ4.** Model rankings hold across both symbols, all 12 development weeks and all 5 test weeks, and test scores lie inside the development range. Stress is predictable mainly in high volatility (PR-AUC 0.72 at 5 s). In low volatility it is much weaker (0.22 for BTC, 0.07 for ETH), and there ETH's thresholded MCC falls below persistence.
- **RQ5.** On the test, the composite stress label had 3.8% (BTC) and 4.2% (ETH) positives, inside the pre-registered range. Calibrated stress probabilities have ECE 0.004–0.016. The spread-only label is degenerate, with about 0.04% positives.

### Limits

- The test covers five weeks of one market period. The hour-block bootstrap treats hours within a week as exchangeable and does not cover regime changes beyond these weeks.
- Development relies mainly on warm-start books, which are an approximation; the test uses exact books only.
- All results use the archive's provider receive-time clock and a single exchange.
- DeepLOB was compared at a compute-limited training size (8,000 sequences per fit).
- The meter's live scores are operational outputs, not evidence of live predictive performance. Nothing here is trading advice.

## Data and replay contract

Each symbol is replayed independently.
- A snapshot bootstraps the local book.
- A depth update is accepted only when its update-ID range follows the current sequence.
- A sequence gap or an invalid book state invalidates the samples that follow, until a valid snapshot resynchronizes the book.
- A one-second row is emitted only for a valid, non-stale state.
- Missing seconds and gaps are never interpolated.

Every row keeps the grid timestamp separate from the event and receive timestamps. Depth updates and OFI accumulation follow the event sequence, not the local receive time. Raw inputs and normalized outputs stay outside Git, and historical and live grids use separate configured roots.

## Features and targets

At origin time `t`, features use only events at or before `t`:
- mid-price, relative spread, spread in ticks and microprice;
- depth and OBI at 1, 5, 10 and 20 levels;
- event-level OFI over trailing windows;
- aggressive trade imbalance;
- displayed-size additions and depletions;
- fixed-notional price impact;
- trailing realized variance.

Depletion in public depth data is a proxy: it does not identify cancellations by order ID. Missing impact or volatility windows stay missing.

A target is unavailable if any required state in `[t, t+h]` is missing or invalid. A stress component that cannot be computed never produces a non-stress label.

## Archive pilot (2026-10-26 submission, superseded)

The pilot trained on 2026-09-28 – 10-04, validated on 2026-10-05 – 10-11 and scored a holdout once on 2026-10-12 – 10-18. It used 250,000 / 100,000 USDT impact notionals. Its outputs are in `outputs/tables/historical_archive_pilot/`, `outputs/deeplob/` and `outputs/research_figures/`. The pilot artifacts in `backend/artifacts/meter/` are kept unchanged for provenance.

The pilot has one validation week and one holdout week, so it cannot show week-to-week stability. Its holdout must not be rescored. The six-month study never reads the pilot's weeks.
