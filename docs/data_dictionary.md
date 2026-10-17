# Research data dictionary

| Field | Meaning | Availability / use |
|---|---|---|
| `open_time`, `close_time` | Binance Spot candle interval timestamps (UTC) | Raw 5-minute source |
| `utc_date` | UTC calendar date of candle | Daily aggregation key |
| `n_bars`, `n_obs` | Candle count and valid adjacent close-to-close returns | Complete day requires both to equal 288 |
| `is_complete_day` | All expected bars and returns are present | Only complete days enter RV targets |
| `rv`, `realized_variance` | Sum of squared 5-minute log returns | Daily target and predictors |
| `rv_partial` | Sum of observed squared returns on an incomplete day | Diagnostic only; never modeled |
| `bpv` | Bipower variation estimate | Jump-robust variation feature |
| `jump` | Positive part of RV minus BPV | Jump variation component |
| `rs_pos`, `rs_neg` | Positive/negative realized semivariances | Asymmetric variation predictors |
| `rq` | Realized quarticity estimate | HARQ adjustment input |
| `close_ret` | Sum of daily intraday log returns | Return target and EWMA input |
| `available_at_utc` | UTC midnight after the date's close | Enforces point-in-time eligibility |
| `rv_mean_7`, `rv_mean_30` | Trailing mean RV windows | HAR predictors, known at origin |
| `target_start`, `target_end` | Direct forecast interval, end exclusive | Next h full UTC days |
| `actual_rv_mean` | Mean daily RV across target interval | Missing if target is incomplete/unavailable |
| `y_cum_return` | Sum of daily log returns over target interval | Separate economic downside outcome |
| `rv_forecast` | Predicted mean RV for direct horizon h | Main forecast output |
| `sigma_h_approx` | Approximate cumulative return sigma from estimated kappa and h-day RV | Risk scaling input; not a full tail distribution |
| `data_version` | Hash binding snapshot data and frozen configs | Reproducibility identifier |
| `config_hash` | SHA-256 of research design config | Reproducibility identifier |

All daily features are computed from the immutable snapshot under `data/snapshots/<as-of>/`. Forecast outputs are Parquet under `results/forecasts/` and include model status; rows with a non-`ok` status must not be treated as predictions.
