# Pilot data contract

This contract covers the available Binance Spot historical pilot for BTCUSDT and ETHUSDT on 2025-01-01. It is an integration fixture for backend and replay development. It does not represent the planned full 2025 study dataset, and it is not suitable for training or final evaluation.

## Sources and processing

- Raw data: Tardis historical `incremental_book_L2`, `trades`, and `quotes` files under `data_raw/` and `data_verify/`. Raw files are read-only.
- Processing timestamp: `local_timestamp`, integer microseconds. Exchange `timestamp` is retained for trades; all time-derived values are UTC.
- Book updates with equal `local_timestamp` are applied in file order and sampled only after the timestamp group is complete.
- Rows before the first snapshot are ignored. A new snapshot batch resets the local book. L2 `amount` is absolute size; a zero incremental amount removes the level. A remove for an absent price is recorded in `absent_level_removals.csv` and ignored for state reconstruction.
- A sampled book must have both sides and `best_bid < best_ask`. No missing seconds are interpolated.
- The pilot interval is `[t0, t0 + 300 seconds)`, where `t0` is the first snapshot's local timestamp. Sampling uses UTC second buckets from `floor(t0)` through `floor(t0)+299`; the last valid grouped state observed in each bucket wins. Consequently there are at most 300 samples; gaps remain gaps.
- Up to 20 levels per side are retained. Prices and quantities are decimal strings in CSV so exact decimal values are preserved.

## Normalized outputs

`data_processed/backend_pilot/pilot_{SYMBOL}_book_states_1s.csv` contains one observed valid book state per populated UTC second, with symbol, source/local timestamp, top-of-book, mid, microprice, spread, level counts, fixed top-20 price/quantity/cumulative-quantity columns, OBI at L=1/5/10/20, 5-second event-grouped OFI, OFI status, depth slopes, 5-second trade flow and five-interval realized volatility. Feature status columns distinguish unavailable values from real zeroes.

OBI is `(sum_bid_qty - sum_ask_qty) / (sum_bid_qty + sum_ask_qty)` over the first L present levels. OFI is computed from best-level states after each equal-local-timestamp update group, then summed over event groups in `(t-5s, t]`. A window containing a reset or invalid state yields null. It is event-grouped OFI because the source contains depth updates; the one-second sample is not used to approximate the event stream.

OFI uses the full indicator expression, including quantity changes when a best price is unchanged. Depth slope is the OLS slope with an intercept of cumulative base quantity against absolute level distance from mid in basis points. Fewer than five levels on a side yields null.

`pilot_{SYMBOL}_trades_normalized.csv` retains the source `side` as `source_side`, exact decimal price/quantity, exchange and local timestamps, and trade ID. Tardis defines this side as the liquidity taker/aggressor side. The five-second trade imbalance is `(buy quantity - sell quantity) / (buy quantity + sell quantity)` over `(t-5s,t]`; unknown-side rows are retained and excluded from numerator and denominator. The normalizer includes up to five seconds before the first book sample so its first feature window can be complete; if the source date does not cover the whole interval, trade-window fields are null with `insufficient_trade_coverage`. A full window with only unknown-side trades has null imbalance and `no_known_trades_in_window`. Realized volatility is the square root of summed squared one-second log mid-price returns over five contiguous past intervals; it is null until five such intervals exist, with an explicit status.

`pilot_{SYMBOL}_quotes_comparison.csv` compares each reconstructed sampled best bid/ask to the latest quote with `local_timestamp <= sample timestamp`. This validates top-of-book only, not deeper quantities.

## Quality and limitations

`outputs/manifests/pilot_data_qa.json` records the processing window, counts, absent-level removal details, quote match rates, outputs, and limitations. `outputs/manifests/source_log.json` records source-file sizes and SHA-256 hashes. Absent-level deletes are classified against initial snapshot bounds and prior level presence; unseen deletes within those bounds remain unresolved. Never infer deeper-book accuracy from a best-bid/ask match.

This pilot does not produce direction or stress labels, training splits, fitted thresholds, model metrics, or probabilities. These require a real chronological data span, training-only thresholds, shared UTC boundaries for both symbols, and a purge at split edges. `split_manifest_pilot.json` records why no split was made.

## Replay API field conventions

- Every response uses `schema_version: "1.0"`, `data_mode: "historical_replay"`, and `source: "tardis"`.
- API timestamps use ISO-8601 UTC with `Z`; price and quantity fields use decimal strings.
- Missing values are `null` with a status. They are never replaced by zero.
- Unavailable models return null predictions and an unavailable status. The pilot has no model artifacts.
- Snapshot depth is explicit and bounded by 20 levels. Backend serves only the requested snapshot/series slice, not the entire dataset.
