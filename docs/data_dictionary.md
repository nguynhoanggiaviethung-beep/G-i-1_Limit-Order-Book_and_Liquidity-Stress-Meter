# Replay data dictionary

All time fields are UTC. Raw prices and quantities should be kept as decimal strings in source events. Numeric Parquet columns may use float64 after replay. Files are partitioned by symbol and UTC date.

## Source and normalized locations

```text
data_raw/<provider>/                            # immutable source files and approved checksum manifests
data_raw/binance_spot/captures/<capture-id>/    # raw websocket, REST snapshots, collector logs, and source manifest
data_raw/cryptohftdata/binance_spot/            # hourly external Parquet/Zstandard archive
data_interim/binance_spot/captures/<capture-id>/events/
                                                # normalized event stream used for deterministic replay
data_processed/lob/grid/
  symbol=<SYMBOL>/date=YYYY-MM-DD/book_1s.parquet
data_processed/lob/features/
  symbol=<SYMBOL>/date=YYYY-MM-DD/features_1s.parquet
data_processed/cryptohftdata_pilot/grid/          # isolated third-party archive pilot; not served by the dashboard
data_processed/cryptohftdata_pilot/features/     # pilot classifier features
```

Historical L2 archives are external and are not committed to this repository. Raw WebSocket messages and REST snapshots are retained under `data_raw/`; the live collector writes normalized replay events under `data_interim/`; grids and features go under `data_processed/`. `backend/scripts/collect_binance_lob.py` captures new public Spot diff-depth and aggregate-trade messages with synchronized REST snapshots; `backend/scripts/replay_cryptohftdata.py` replays those event partitions or approved chronologically ordered JSONL/JSONL.GZ files containing canonical records or native Binance Spot snapshot, depthUpdate, and aggTrade messages. `backend/scripts/replay_cryptohftdata_archive.py` replays the provider's published hourly Parquet order-book and trade files. Binance REST aggregate trades contain no historical depth and cannot reconstruct a past book. Map other provider-specific source schemas to this event format before replay. The replay API reads only normalized `book_1s.parquet` partitions.

Canonical raw event shapes:

```json
{"kind":"snapshot","event_time_ms":1750000000000,"lastUpdateId":42,"bids":[["100.00","1.2"]],"asks":[["100.01","1.5"]]}
{"kind":"depth_update","E":1750000000100,"U":43,"u":43,"b":[["100.00","1.3"]],"a":[]}
{"kind":"trade","event_time_ms":1750000000150,"price":"100.01","quantity":"0.2","is_buyer_maker":false}
```

For Binance-native JSON input, records must be ordered by exchange event time. The historical archive replay preserves provider file order and uses provider `received_time` for the point-in-time replay clock. A snapshot bootstraps the book; a sequence gap invalidates it until a later complete snapshot. Public depth deletions do not distinguish cancels from executions, so the exported cancellation intensity is displayed-size depletion, not order-ID cancellation truth.

## One-second grid fields

| Field | Definition |
|---|---|
| `symbol` | `BTCUSDT` or `ETHUSDT`. |
| `timestamp_utc` | Clock-time sample timestamp; exactly one-second grid. |
| `event_time_utc` | Replay clock of the latest applied update. This is exchange event time for Binance-native streams and provider receive time for the historical archive. |
| `exchange_event_time_utc` | Original exchange event time when supplied by the source; otherwise null. |
| `received_at_utc` | Local receive time when captured by the live collector; historical archive receive time is reflected by `event_time_utc` and the QA timestamp basis. |
| `first_update_id`, `last_update_id` | Sequence range applied by the replay at this sample. |
| `book_valid` | True only for synchronized, uncrossed, non-empty book state. |
| `sequence_gap` | True when a missing update invalidated the state. |
| `seconds_since_gap` | Elapsed valid seconds since the last recovery snapshot. |
| `bid_price_1..20`, `bid_qty_1..20` | Best-to-deeper bid levels. |
| `ask_price_1..20`, `ask_qty_1..20` | Best-to-deeper ask levels. |
| `ofi_1s` | Sum of event-level OFI contributions during the preceding second. |
| `aggressive_buy_volume_1s`, `aggressive_sell_volume_1s` | Aggressor-side trade volume in `(t-1s,t]`. |
| `trade_count_1s` | Number of trades in the same window. |
| `last_trade_price`, `last_trade_time_utc` | Last trade observed in the same second, if any; archive timestamps use the provider receive clock. |
| `last_trade_exchange_time_utc` | Exchange trade time for the last trade when supplied by the archive. |
| `addition_intensity_1s`, `cancellation_intensity_1s` | Total displayed quantity added or depleted by valid depth updates; depletion is a cancellation/execution proxy. |
| `tick_size` | Configured minimum price increment for this symbol. |
| `qa_flags` | Missing/stale/gap/recovery and displayed-depth warnings. |

Rows are omitted when no valid book is available or the last update is older than two seconds. Do not fill missing seconds.

Per-day QA sidecars are read from `outputs/manifests/symbol=<SYMBOL>/date=YYYY-MM-DD/qa.json` or an adjacent `book_1s.qa.json`. They report `valid_seconds`, `expected_seconds`, `dropped_seconds`, `sequence_gap_count`, `invalid_book_event_count`, `snapshot_bootstraps`, `snapshot_recoveries`, provider receive-time adjustments, and `coverage`. Sequence gaps and structurally invalid book updates use separate counters because invalid updates invalidate the book but do not imply a missing sequence range. The historical archive replay clamps receive-clock regressions of at most 100 ms without changing message order and records the adjustment count and maximum regression; larger regressions stop replay. `GET /api/v1/data-status` reports these totals when sidecars are present; missing QA remains explicitly unavailable.

## Derived point-in-time features

| Field | Definition |
|---|---|
| `mid_price` | `(best_ask + best_bid) / 2`. |
| `absolute_spread`, `relative_spread_bps` | Ask minus bid; spread divided by mid and multiplied by 10,000. |
| `spread_ticks` | Absolute spread divided by the configured symbol tick size. The composite stress label uses this metric when available. |
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

Feature construction sorts timestamps stably and resolves duplicate timestamps by keeping the last input row; each resolution is logged with the duplicate count. The ordered 27-column classifier input is identified as `lob-classification-features.v1`. Trained output records the feature and label schema versions, SHA-256 of `config/settings.yaml`, observed UTC data range, study split, model version, and random seeds.

## Target and result fields

Direction targets are `up`, `flat`, or `down` at 1, 5, and 10 second horizons. The primary stress target is binary composite stress: future spread, price impact, or RV exceeds its training-fold Q97.5, or future top-10 depth falls below its training-fold Q2.5. Spread-only Q95 remains a sensitivity because discrete tick spreads may yield too few positive labels. Store `target_end_utc`, `target_valid`, the stress definition, and the threshold version with every label dataset.

The meter response contains `timestamp_utc`, `horizon_seconds`, `spread`, imbalance at L=1/5/10/20, `ofi_5s`, `p_up`, `p_flat`, `p_down`, `p_stress`, `stress_probability_threshold`, `model_version`, and `quality`. The stress threshold is read from the matching artifact and is null when unavailable; the dashboard uses this value for its stress flag. Probability fields are null when the matching trained artifact is unavailable.
