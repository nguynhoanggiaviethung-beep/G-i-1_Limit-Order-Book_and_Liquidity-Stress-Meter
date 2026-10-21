# Replay data dictionary

All time fields are UTC. Raw prices and quantities should be kept as decimal strings in source events. Numeric Parquet columns may use float64 after replay. Files are partitioned by symbol and UTC date.

## Source and normalized locations

```text
data_raw/cryptohftdata/                         # source archive and approved checksum manifest
data_processed/cryptohftdata/grid/
  symbol=<SYMBOL>/date=YYYY-MM-DD/book_1s.parquet
data_processed/cryptohftdata/features/
  symbol=<SYMBOL>/date=YYYY-MM-DD/features_1s.parquet
```

The source dataset is external and is not committed to this repository. The replay API reads only normalized `book_1s.parquet` partitions.

## One-second grid fields

| Field | Definition |
|---|---|
| `symbol` | `BTCUSDT` or `ETHUSDT`. |
| `timestamp_utc` | Clock-time sample timestamp; exactly one-second grid. |
| `event_time_utc` | Exchange event time of the latest applied update. |
| `received_at_utc` | Local receive time, retained separately from exchange time. |
| `first_update_id`, `last_update_id` | Sequence range applied by the replay at this sample. |
| `book_valid` | True only for synchronized, uncrossed, non-empty book state. |
| `sequence_gap` | True when a missing update invalidated the state. |
| `seconds_since_gap` | Elapsed valid seconds since the last recovery snapshot. |
| `bid_price_1..20`, `bid_qty_1..20` | Best-to-deeper bid levels. |
| `ask_price_1..20`, `ask_qty_1..20` | Best-to-deeper ask levels. |
| `ofi_1s` | Sum of event-level OFI contributions during the preceding second. |
| `aggressive_buy_volume_1s`, `aggressive_sell_volume_1s` | Aggressor-side trade volume in `(t-1s,t]`. |
| `trade_count_1s` | Number of trades in the same window. |
| `qa_flags` | Missing/stale/gap/recovery and displayed-depth warnings. |

Rows are omitted when no valid book is available or the last update is older than two seconds. Do not fill missing seconds.

Per-day QA sidecars are read from `outputs/manifests/symbol=<SYMBOL>/date=YYYY-MM-DD/qa.json` or an adjacent `book_1s.qa.json`. They report `valid_seconds`, `expected_seconds`, `dropped_seconds`, `sequence_gap_count`, `snapshot_recoveries`, and `coverage`. `GET /api/v1/data-status` reports these totals when sidecars are present; missing QA remains explicitly unavailable.

## Derived point-in-time features

| Field | Definition |
|---|---|
| `mid_price` | `(best_ask + best_bid) / 2`. |
| `absolute_spread`, `relative_spread_bps` | Ask minus bid; spread divided by mid and multiplied by 10,000. |
| `microprice`, `microprice_minus_mid_bps` | Opposite-side quantity-weighted quote and its offset from mid. |
| `bid_depth_qty_L`, `ask_depth_qty_L`, `depth_qty_L` | Bid, ask, and combined quantity through level L. |
| `obi_qty_L` | Signed quantity imbalance for L in 1, 5, 10, 20. |
| `ofi_1s`, `ofi_5s`, `ofi_10s` | Event-level OFI summed over trailing windows ending at the sample. |
| `trade_imbalance_5s`, `trade_imbalance_10s` | `(buy_volume - sell_volume) / total_volume` in trailing windows. |
| `impact_buy_bps`, `impact_sell_bps` | Adverse VWAP impact for a fixed USDT notional swept through visible asks/bids. |
| `price_impact_bps` | Maximum of buy and sell adverse impact. |
| `rv_10s`, `rv_60s`, `rv_300s` | Trailing sum of squared one-second mid-price log returns. |

All rolling windows reset at gaps. Every scaler, imputer, and threshold is fitted on training data only.

## Target and result fields

Direction targets are `up`, `flat`, or `down` at 1, 5, and 10 second horizons. Stress targets are binary and use train-only quantiles for spread, fixed-notional price impact, top-10 depth, and trailing RV. Store `target_end_utc`, `target_valid`, and the threshold version with every label dataset.

The meter response contains `timestamp_utc`, `horizon_seconds`, `spread`, imbalance at L=1/5/10/20, `ofi_5s`, `p_up`, `p_flat`, `p_down`, `p_stress`, `model_version`, and `quality`. Probability fields are null when the matching trained artifact is unavailable.
