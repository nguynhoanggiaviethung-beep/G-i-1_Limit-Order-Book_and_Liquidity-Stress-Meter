# Research design: LOB direction and liquidity stress

## Objective and questions

The study tests whether order-book state and order flow provide point-in-time information about very short-horizon mid-price direction and liquidity stress in crypto spot markets.

- **RQ1:** Do static depth-imbalance measures predict mid-price direction beyond a class-frequency benchmark?
- **RQ2:** Does event-level Order Flow Imbalance add predictive information beyond static imbalance?
- **RQ3:** Do DeepLOB-style models improve on Logistic Regression and XGBoost on the untouched period?
- **RQ4:** How stable are the results across volatility regimes and calendar months?
- **RQ5:** Do conclusions persist under macro F1, MCC, Brier score, class imbalance, and calibration checks?

## Frozen scope and defaults

| Decision | Default |
|---|---|
| Market | Binance Spot BTCUSDT and ETHUSDT |
| Data | Approved CryptoHFTData L2 book events and trades for the frozen 2025 study period; public Binance Spot captures may extend the pipeline prospectively |
| Sampling | Clock-time one-second grid |
| Book depth | 20 levels per side; OBI sensitivity at L=1, 5, 10, 20 |
| Horizons | 1, 5, 10 seconds; 5 seconds is primary |
| Direction classes | Up / flat / down, with symbol- and horizon-specific delta fitted on the first training fold and then frozen |
| Stress label | Primary: future spread above train-only Q95. Optional extension: composite spread/impact/depth/RV label, reported separately. |
| Impact notional | BTCUSDT 250,000 USDT; ETHUSDT 100,000 USDT; override through settings |
| Development | Expanding walk-forward through 2025-11-30 |
| Final test | 2025-12-01 through 2025-12-31, opened once after model selection |
| Embargo | At least the maximum of the forecast horizon, 300-second feature lookback, and 100-second deep-model sequence |
| Exchange extension | Coinbase cross-exchange validation is out of scope for the primary study |
| Product mode | Read-only replay dashboard; an explicit CLI collector records live depth for later replay |

The primary source-data range may have incomplete coverage. The final report must state actual valid seconds, coverage, missing intervals, and sequence-gap recoveries; it must not describe absent periods as observed.

## Data and replay

The frozen 2025 study requires historical Level 2 order-book events and trades; public REST aggregate trades cannot replace historical depth. The acquisition pipeline can also capture new Binance Spot depth and trades prospectively. Replay each symbol independently from snapshot and update IDs. A sequence gap invalidates the book until a complete snapshot is applied. Emit a one-second row only when the state is valid and no more than two seconds old. Do not interpolate missing book states.

Every row retains exchange event time and local receive time separately. Sequence order, not local receive time, determines event application and OFI. Store raw source files and checksums outside Git. Store normalized grids in date and symbol Parquet partitions.

## Features

Features at time `t` may use only events with timestamps at or before `t`:

- mid-price, absolute and relative spread, spread in configured ticks, and microprice offset;
- quantity and notional depth, depth slope, and OBI at 1, 5, 10, and 20 levels;
- OFI calculated from every event before it is aggregated into one-, five-, and ten-second windows;
- aggressive buy/sell volume, trade count, last trade price, and trade imbalance in trailing windows;
- displayed-size addition and depletion intensity. Public diff-depth deletions mix cancellations and executions, so depletion is a proxy rather than order-ID cancellation truth;
- fixed-notional buy and sell price impact from walking the book;
- mid-price realized variance over trailing 10, 60, and 300 seconds.

No feature window may cross an invalid second or sequence gap. Any scaler, imputer, feature selector, regime threshold, or probability calibrator is fitted within the corresponding training fold.

## Targets

Direction uses the mid-price at origin `t` and the exact mid-price at `t+h`. A sample is eligible only if every one-second state from `t` through `t+h` is valid and contiguous. The flat-band threshold `delta` is estimated from absolute future mid-price changes in the first training fold for each symbol and horizon, then held fixed.

Primary liquidity stress at `t+h` is one when the configured spread metric exceeds its 95th percentile fitted on the applicable training fold. The selected metric is `spread_ticks` when available, then absolute or relative spread. The label uses a strict greater-than comparison and leaves incomplete future windows unlabeled. Report class prevalence by split and month. If the composite extension is enabled, fit each component threshold on training rows only and report it separately; never combine or switch label definitions after inspecting final-test results.

## Models and validation

1. Class-frequency and persistence baselines.
2. Logistic Regression with train-fold scaling and class balancing.
3. Random Forest and XGBoost; choose settings using chronological validation folds only.
4. DeepLOB-style spatial CNN-LSTM over 100-second windows of 40 raw LOB channels. The separate offline script evaluates it against Logistic Regression and persistence on the same evenly spaced sample of up to 10,000 held-out origins, and reports sampled and available row counts. Temporal CNN or Transformer only if DeepLOB improves validation results.

Use expanding calendar-month walk-forward folds for development. Apply an embargo of at least `max(horizon, feature lookback, sequence lookback)` between training labels and validation. Fit preprocessing and thresholds only on the fold's training data. Select models using validation results, then fit on development data and score December once. The baseline report includes δ quantiles around the frozen direction threshold and OBI depth sensitivity at L=1, 5, 10, and 20 for each horizon.

Direction metrics: macro F1, balanced accuracy, MCC, per-class precision/recall, multiclass Brier score, confusion matrix, and calibration curve. Stress metrics: PR-AUC, MCC, precision/recall at a validation-selected threshold, Brier score, calibration curve, and expected calibration error. Report accuracy for context only. Compare paired models on the same folds and include paired significance checks where sample size supports them.

Report metrics by month and by three causal volatility regimes formed from trailing mid-price RV terciles. Include exact paired McNemar tests for classification correctness and horizon-lagged Newey-West Diebold–Mariano tests on Brier loss against persistence (and Logistic Regression for DeepLOB). Add inference latency and model-size/parameter summaries. No conclusion about deep learning is supported unless its incremental performance is shown on the untouched period.

## Product boundary

The Liquidity Stress Meter is a read-only replay product. It shows spread, top-depth imbalance, OFI, `P(up)`, `P(flat)`, `P(down)`, liquidity-stress probability, model version, and book-quality status. Live capture is an explicit data-acquisition command; it is not an order-entry feature or a direct dashboard feed. If replay data or a compatible trained model is absent, the API and dashboard show an explicit unavailable state.
