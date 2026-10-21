# Methodology

## Local book and sampling

Replay each symbol independently. Apply a diff update only when its update-ID range follows the current book sequence. A gap invalidates the local book until a new snapshot is synchronized. Sample at one-second UTC clock boundaries and retain a sample only if the book is valid and no more than two seconds old. Never carry book state across an invalid interval.

Let `b1` and `a1` be the best bid and ask, with quantities `qb1` and `qa1`:

```text
mid_t = (a1_t + b1_t) / 2
spread_t = a1_t - b1_t
relative_spread_bps_t = 10,000 * spread_t / mid_t
microprice_t = (a1_t * qb1_t + b1_t * qa1_t) / (qb1_t + qa1_t)
```

If the top-level quantity sum is zero, use mid-price for microprice and set the quality flag. For `L` levels:

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

## Price impact and volatility features

For a fixed USDT notional, sweep asks to estimate buy VWAP and bids to estimate sell VWAP. Express adverse movement from the mid-price in basis points. If the visible 20 levels cannot fill the notional, record the impact as missing and flag insufficient displayed depth.

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

Fit stress thresholds from training rows only. At `t+h`, stress is one if at least one is true:

```text
relative spread >= train Q95
fixed-notional price impact >= train Q95
top-10 total depth <= train Q05
trailing 60-second RV >= train Q95
```

The source brief's spread-only `spread > Q95` label is evaluated as sensitivity. Check positive prevalence before freezing labels; BTC spot spread can be pinned to one tick, making a strict-greater-than threshold uninformative.

## Walk-forward, calibration, and reporting

Use expanding chronological month folds. Purge training origins whose target window overlaps validation; then apply an embargo at least as long as the maximum of the target horizon, longest feature window, and deep sequence. Fit the flat threshold, stress quantiles, imputer, scaler, and calibration map only on training/validation data available at that fold.

Compare direction models against a one-second persistence baseline and a class-frequency baseline; select models by validation macro F1 and MCC. Report balanced accuracy and per-class precision/recall. For probabilities, report Brier score and calibration curves. Compare stress models against current-state persistence; select stress models with PR-AUC and MCC, and report thresholded precision/recall and calibration. Keep December 2025 untouched until model choice is final, and score it once.

Report results by symbol, horizon, month, and three volatility regimes based on trailing RV terciles. Include class distribution, feature summaries, confusion matrices, calibration, latency, model size, and actual data coverage. The API does not calculate replacement probabilities when a trained artifact is missing.
