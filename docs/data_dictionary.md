# Research data dictionary

| Field | Meaning | Availability / use |
|---|---|---|
| `open_time`, `close_time` | Binance Spot candle interval timestamps (UTC) | Raw 5-minute source |
| `utc_date` | UTC calendar date of candle | Daily aggregation key |
| `n_bars`, `n_obs` | Candle count and valid adjacent close-to-close returns | Complete day requires both to equal 288 |
| `is_complete_day` | All expected bars and returns are present | Only complete days enter RV targets |
| `rv`, `realized_variance` | Sum of squared 5-minute log returns; returns are scaled by 100 | Daily target and predictors, in percent squared |
| `rv_partial` | Sum of observed squared returns on an incomplete day | Diagnostic only; never modeled |
| `bpv` | Bipower variation estimate | Jump-robust variation feature |
| `jump` | Positive part of RV minus BPV | Jump variation component |
| `rs_pos`, `rs_neg` | Positive/negative realized semivariances | Asymmetric variation predictors |
| `rq` | Realized quarticity estimate | HARQ adjustment input |
| `close_ret` | Sum of daily intraday log returns scaled by 100 | Return target and EWMA input, in percent |
| `available_at_utc` | UTC midnight after the date's close | Enforces point-in-time eligibility |
| `rv_mean_7`, `rv_mean_30` | Trailing mean RV windows | HAR predictors, known at origin |
| `target_start`, `target_end` | Direct forecast interval, end exclusive | Next h full UTC days |
| `actual_rv_mean` | Mean daily RV across target interval | Missing if target is incomplete/unavailable |
| `y_cum_return` | Sum of daily log returns over target interval | Separate economic downside outcome |
| `rv_forecast` | Predicted mean RV for direct horizon h | Main forecast output |
| `sigma_h_approx` | Approximate cumulative return sigma from estimated kappa and h-day RV | In percent; risk scaling input, not a full tail distribution |
| `var_loss_percent`, `expected_shortfall_loss_percent` | Positive-loss VaR and ES | In percent, at the requested confidence level |
| `confidence`, `method` | VaR/ES confidence and estimator name | `quantile_model` identifies direct conditional risk forecasts |
| `actual_loss_percent` | Negative cumulative return, so losses are positive | Risk forecast target; missing until target interval completes |
| `n_train`, `refit_at` | Training sample count and most recent model refit origin | Point-in-time model audit fields for quantile forecasts |
| `reliability`, `evaluation_scope` | Interpretive label and OOS/pilot scope | 30-day risk is marked descriptive; pilot is not a full backtest |
| `data_version` | Research: snapshot-manifest hash; production: synchronized-candle-frame hash | Identifies the exact market data used |
| `config_hash` | SHA-256 identifier for the relevant model/config contract | Primary RV, direct quantile, and robustness runs may use distinct config hashes |
| `feature_name`, `observation_time_utc`, `available_at_utc`, `value`, `source` | Normalized external source event | Funding/OI/basis/FRED values are only joined after their recorded availability; raw events live in `data/external/events.parquet` |
| `external_funding_rate_last` | Latest Binance USDⓈ-M funding settlement | Requires an event within one day and coverage of at least 95% to be emitted |
| `external_open_interest_value_last`, `external_basis_rate_last` | Latest open-interest notional and perpetual basis rate | Require a recent event and at least 95% coverage; sparse 30-day history is excluded |
| `external_vix_last`, `external_usd_broad_index_last` | FRED VIXCLS and DTWEXBGS macro observations | First-release vintage and conservative one-day availability lag; DTWEXBGS is a broad USD proxy, not ICE DXY |

All daily features are computed from the immutable snapshot under `data/snapshots/<as-of>/`. Forecast outputs are Parquet under `results/forecasts/`. Direct quantile forecasts are written first to `quantile_forecasts.parquet` and incorporated into `risk_forecasts.parquet` by risk evaluation. Rows with a non-`ok` status must not be treated as predictions. Confidence levels are probabilities such as 0.95, while VaR and ES are positive loss magnitudes in percentage points.

External observations and features are stored separately under `data/external/` and `data/features/external/`; they do not alter the immutable baseline research features or forecasts.
