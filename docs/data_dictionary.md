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

The source dataset is external and is not committed to this repository. `backend/scripts/replay_cryptohftdata.py` accepts chronologically ordered JSONL/JSONL.GZ containing canonical event records or native Binance Spot snapshot, depthUpdate, and aggTrade messages. Map other provider-specific source schemas to this event format before replay. The replay API reads only normalized `book_1s.parquet` partitions.

Canonical raw event shapes:

```json
{"kind":"snapshot","event_time_ms":1750000000000,"lastUpdateId":42,"bids":[["100.00","1.2"]],"asks":[["100.01","1.5"]]}
{"kind":"depth_update","E":1750000000100,"U":43,"u":43,"b":[["100.00","1.3"]],"a":[]}
{"kind":"trade","event_time_ms":1750000000150,"price":"100.01","quantity":"0.2","is_buyer_maker":false}
```

Records must be ordered by exchange event time. A snapshot bootstraps the book; a sequence gap invalidates it until a later complete snapshot. Public depth deletions do not distinguish cancels from executions, so the exported cancellation intensity is displayed-size depletion, not order-ID cancellation truth.

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
| `last_trade_price`, `last_trade_time_utc` | Last trade observed in the same second, if any. |
| `addition_intensity_1s`, `cancellation_intensity_1s` | Total displayed quantity added or depleted by valid depth updates; depletion is a cancellation/execution proxy. |
| `tick_size` | Configured minimum price increment for this symbol. |
| `qa_flags` | Missing/stale/gap/recovery and displayed-depth warnings. |

Rows are omitted when no valid book is available or the last update is older than two seconds. Do not fill missing seconds.

Per-day QA sidecars are read from `outputs/manifests/symbol=<SYMBOL>/date=YYYY-MM-DD/qa.json` or an adjacent `book_1s.qa.json`. They report `valid_seconds`, `expected_seconds`, `dropped_seconds`, `sequence_gap_count`, `snapshot_bootstraps`, `snapshot_recoveries`, and `coverage`. `GET /api/v1/data-status` reports these totals when sidecars are present; missing QA remains explicitly unavailable.

## Derived point-in-time features

| Field | Definition |
|---|---|
| `mid_price` | `(best_ask + best_bid) / 2`. |
| `absolute_spread`, `relative_spread_bps` | Ask minus bid; spread divided by mid and multiplied by 10,000. |
| `spread_ticks` | Absolute spread divided by the configured symbol tick size. The stress label uses this metric when available. |
| `microprice`, `microprice_minus_mid_bps` | Opposite-side quantity-weighted quote and its offset from mid. |
| `bid_depth_qty_L`, `ask_depth_qty_L`, `depth_qty_L` | Bid, ask, and combined quantity through level L. |
| `bid_depth_notional_L`, `ask_depth_notional_L`, `depth_notional_L` | Bid, ask, and combined USDT notional through level L. |
| `obi_qty_L` | Signed quantity imbalance for L in 1, 5, 10, 20. |
| `ofi_1s`, `ofi_5s`, `ofi_10s` | Event-level OFI summed over trailing windows ending at the sample. |
| `trade_imbalance_5s`, `trade_imbalance_10s` | `(buy_volume - sell_volume) / total_volume` in trailing windows. |
| `addition_intensity_1s`, `cancellation_intensity_1s`, `addition_intensity_5s`, `cancellation_intensity_5s` | Displayed quantity added/depleted from event updates. Cancellation is a proxy because public depth events do not expose order IDs. |
| `impact_buy_bps`, `impact_sell_bps` | Adverse VWAP impact for a fixed USDT notional swept through visible asks/bids. |
| `price_impact_bps` | Maximum of both buy and sell adverse impact; missing if either side cannot fill the configured notional. Defaults: 250,000 USDT BTCUSDT and 100,000 USDT ETHUSDT. |
| `rv_10s`, `rv_60s`, `rv_300s` | Trailing sum of squared one-second mid-price log returns. |

All rolling windows reset at gaps. Every scaler, imputer, and threshold is fitted on training data only.

## Target and result fields

Direction targets are `up`, `flat`, or `down` at 1, 5, and 10 second horizons. Stress targets are binary and use train-only quantiles for spread, fixed-notional price impact, top-10 depth, and trailing RV. Store `target_end_utc`, `target_valid`, and the threshold version with every label dataset.

The meter response contains `timestamp_utc`, `horizon_seconds`, `spread`, imbalance at L=1/5/10/20, `ofi_5s`, `p_up`, `p_flat`, `p_down`, `p_stress`, `model_version`, and `quality`. Probability fields are null when the matching trained artifact is unavailable.
