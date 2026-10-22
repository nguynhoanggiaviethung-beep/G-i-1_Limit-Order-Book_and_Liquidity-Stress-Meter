# Methodology

## Local book and sampling

Replay each symbol independently. Apply a diff update only when its update-ID range follows the current book sequence. A gap invalidates the local book until a new snapshot is synchronized. Sample at one-second UTC clock boundaries and retain a sample only if the book is valid and no more than two seconds old. Never carry book state across an invalid interval.

Let `b1` and `a1` be the best bid and ask, with quantities `qb1` and `qa1`:

```text
mid_t = (a1_t + b1_t) / 2
spread_t = a1_t - b1_t
relative_spread_bps_t = 10,000 * spread_t / mid_t
spread_ticks_t = spread_t / configured_tick_size
microprice_t = (a1_t * qb1_t + b1_t * qa1_t) / (qb1_t + qa1_t)
```

If the top-level quantity sum is zero, use mid-price for microprice and set the quality flag. Tick size comes from the per-symbol configuration. For `L` levels:

```text
OBI_L(t) = (sum_bid_qty_L - sum_ask_qty_L) / (sum_bid_qty_L + sum_ask_qty_L)
```

If total depth is zero, OBI is missing. Calculate notional depth and top-20 depth slope alongside quantity depth.

## Order Flow Imbalance

For each ordered event `n`, define `P^b`, `q^b`, `P^a`, and `q^a` as the best bid/ask prices and quantities immediately before and after the event:

```text
e_n = 1(P^b_n >= P^b_(n-1)) q^b_n
    - 1(P^b_n <= P^b_(n-1)) q^b_(n-1)
    - 1(P^a_n <= P^a_(n-1)) q^a_n
    + 1(P^a_n >= P^a_(n-1)) q^a_(n-1)
```

Sum `e_n` over trailing 1, 5, and 10 second windows. OFI must be accumulated during event replay; it cannot be reconstructed correctly from one snapshot per second. Trade imbalance uses aggressive buy and sell volumes in `(t-w, t]`; Binance buyer-maker `false` denotes an aggressive buy.

Replay also reports displayed quantity additions and depletions by second. Public depth updates cannot reliably distinguish a cancellation from an execution at a price level, so depletion is a proxy. Trade count and last trade price/time are retained separately.

## Price impact and volatility features

For a fixed USDT notional, sweep asks to estimate buy VWAP and bids to estimate sell VWAP. Defaults are 250,000 USDT for BTCUSDT and 100,000 USDT for ETHUSDT. Express adverse movement from the mid-price in basis points. If either side of the visible 20 levels cannot fill the notional, record the composite impact as missing; never substitute the one-sided value.

For one-second mid-price log returns `r_t`, trailing realized variance is:

```text
RV_w(t) = sum(r_i^2), i in the trailing w seconds
```

Use windows of 10, 60, and 300 seconds. Rolling calculations reset after a sequence gap or missing grid second.

## Labels

For horizon `h`:

```text
change_(t,h) = mid_(t+h) - mid_t
up   if change_(t,h) > delta
flat if abs(change_(t,h)) <= delta
down if change_(t,h) < -delta
```

Estimate `delta` from the chosen quantile of absolute future mid-price changes in the first training fold for each symbol and horizon. Freeze it before model comparison. A label is missing if any state from `t` through `t+h` is absent or invalid.

The primary stress label is composite to avoid a degenerate binary target when most quoted spreads equal one tick. At `t+h`, stress is one when any future component crosses its training-only threshold:

```text
future spread > train Q97.5
or future price impact > train Q97.5
or future realized volatility > train Q97.5
or future top-10 depth < train Q2.5
```

Spread uses `spread_ticks` when available, then absolute or relative spread; depth uses top-10 notional when available, then quantity. Thresholds are fitted separately inside each walk-forward training fold. Incomplete future windows remain unlabeled; values tied at a threshold are not positive under strict comparisons. Report class prevalence by split and month, and check it against the configured 2–10% range before final scoring. Keep spread-only Q95 as a separately named sensitivity; do not switch definitions after inspecting final-test results.

## Walk-forward, calibration, and reporting

Use expanding full UTC-week folds. The first validation week begins 2026-11-02 and development ends 2026-11-22. The untouched final test is 2026-11-23 through 2026-11-29. The capture must include at least one full week before the first validation fold; the current 10-minute capture is only a replay/dashboard smoke sample and cannot support this evaluation. Purge training origins whose target window overlaps validation; then apply an embargo at least as long as the maximum of the target horizon, longest feature window, and deep sequence. Fit the flat threshold, stress quantiles, imputer, scaler, and calibration map only on training/validation data available at that fold.

Compare direction models against a one-second persistence baseline and a class-frequency baseline; select models by validation macro F1 and MCC. Report balanced accuracy and per-class precision/recall. For probabilities, report Brier score and calibration curves. Compare stress models against current-state persistence; select stress models with PR-AUC and MCC, and report thresholded precision/recall and calibration. Include paired McNemar tests on correctness. For Diebold–Mariano inference on paired Brier losses, first average losses by UTC minute, then use an automatic Newey–West bandwidth with a 10-minute minimum to reflect multi-minute dependence; report no asymptotic p-value with fewer than 30 minute blocks. Report δ quantile sensitivity around the frozen choice and OBI depth sensitivity at L=1, 5, 10, and 20 for h=1, 5, and 10. Keep the 2026-11-23 through 2026-11-29 final week untouched until model choice is final, and score it once.

DeepLOB is a separate optional PyTorch evaluation. It takes 100 contiguous seconds by 40 channels (bid/ask prices and quantities over ten levels), applies a convolution over the side/level structure, then an LSTM over time. Windows crossing a missing second are discarded. Compare it with Logistic Regression and persistence on the same evenly spaced sample of up to 10,000 test origins. Both DeepLOB and its Logistic baseline are capped at 8,000 training rows by default; this is a compute-limited pilot. Report the cap, sampled and available row counts, parameter count, and inference latency.

Report results by symbol, horizon, UTC week, and three volatility regimes based on trailing RV terciles. Include class distribution, monthly stress prevalence, feature summaries, confusion matrices, calibration, latency, model size, and actual data coverage. The API does not calculate replacement probabilities when a trained artifact is missing.
