# Methodology

## Data processing

Use one Binance client for both initial history and later updates. The data manager resumes from the latest saved candle and re-fetches that boundary candle to make synchronization idempotent. Data are deduplicated by `(symbol, interval, open_time)`; only fully closed candles are retained. Writes use a temporary Parquet file and an atomic replacement.

Validation reports schema issues, invalid OHLCV rows, duplicate keys, incomplete intervals, and open candles. Source gaps remain missing; no future values are used to fill them.

## Features

Returns are computed from close prices in timestamp order. Daily realized variance is the sum of squared intraday log returns; realized volatility is the square root of that sum. Additional lagged and microstructure features should be added only with a documented point-in-time availability rule.

## Models and evaluation

The repository provides a rolling-mean benchmark and a GARCH adapter as initial baselines. The train/test split must preserve time order and use the same out-of-sample dates for every model. Forecast horizon, VaR/ES convention, and test dates remain research-design decisions and must be fixed before reporting model performance.


For the one-day MVP, model inputs use 5-minute returns observed through the same UTC time on each origin day. Incomplete cutoff windows are excluded. The target is full realized variance on the following UTC day. The GARCH risk forecast uses the fitted Student-t innovation distribution; VaR and ES are reported as positive daily loss percentages.

