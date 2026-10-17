# Data dictionary

## Source and scope

Binance Spot public klines for `BTCUSDT` and `ETHUSDT`, on 5-minute intervals by default. Public market data requires no API key. Timestamps are stored in UTC. The research start date is configurable; the current default is 2019-01-01.

## Normalized candle fields

| Field | Meaning | Unit / rule |
| --- | --- | --- |
| `symbol` | Trading pair | `BTCUSDT` or `ETHUSDT` |
| `interval` | Kline interval | `5m` by default |
| `open_time` | Candle opening time | UTC, unique with symbol and interval |
| `close_time` | Candle closing time | UTC |
| `open`, `high`, `low`, `close` | OHLC prices | Numeric; close must be positive; high/low must bound open and close |
| `volume` | Base asset traded | Numeric and non-negative |
| `quote_volume` | Quote asset traded | Numeric and non-negative |
| `number_of_trades` | Trades in candle | Integer-valued numeric field |
| `taker_buy_base_volume` | Aggressive buy volume in base asset | Numeric |
| `taker_buy_quote_volume` | Aggressive buy volume in quote asset | Numeric |

Raw API fields are retained in `data/raw/`; normalized closed candles are written to `data/processed/`. Features are kept separately under `data/features/`. Missing intervals are reported and never filled or backfilled.

## Initial features

`log_return` is `100 * ln(close_t / close_(t-1))`, in percentage points. Daily realized variance is the sum of squared 5-minute log returns; daily realized volatility is its square root. The first return in each symbol series is undefined and excluded from RV.
