# Research design

## Objective

Estimate whether econometric, machine learning, and deep learning methods improve forecasts of realized volatility and tail risk for major crypto assets. The project focuses on out-of-sample evidence for volatility, Value at Risk (VaR), and Expected Shortfall (ES), not next-price prediction.

## Initial data configuration

- Assets: BTCUSDT and ETHUSDT on Binance Spot.
- Sampling: 5-minute OHLCV candles; the default history starts 2019-01-01.
- Pipeline: raw klines → validated UTC candles → returns and realized volatility → point-in-time model features.
- Default storage: Parquet in repository `data/` folders; raw and processed market files are excluded from Git.

## Decisions to freeze before evaluation

- Forecast horizon (for example 60 minutes or one day).
- Realized volatility measurement window and target timestamp.
- VaR/ES loss sign convention, confidence levels, and estimation method.
- Expanding or rolling window dates and untouched test period.
- Baseline and advanced model set and the metrics used for comparison.

Do not choose these settings after examining test results. Avoid shuffling time series and prevent any feature from using information unavailable at its forecast origin.
