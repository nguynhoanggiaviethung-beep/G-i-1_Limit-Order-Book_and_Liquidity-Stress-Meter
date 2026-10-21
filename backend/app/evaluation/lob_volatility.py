"""Walk-forward LOB-derived RV and tail-risk evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.datasets.leakage_checks import assert_no_target_leakage, feature_columns
from app.evaluation.es_backtests import fz0_loss
from app.evaluation.var_es_backtest import kupiec_uc, quantile_loss
from app.models.lob_benchmarks import empirical_var_es, persistence_rv
from app.models.lob_ml import make_har_rv_regressor, make_lob_quantile_regressor, make_lob_rv_regressors


def _vol_metrics(actual: np.ndarray, forecast: np.ndarray) -> dict:
    valid = np.isfinite(actual) & np.isfinite(forecast) & (actual >= 0) & (forecast > 0)
    actual, forecast = actual[valid], forecast[valid]
    if not len(actual):
        return {"n": 0, "mae": None, "rmse": None, "qlike": None}
    qlike = np.log(forecast) + actual / forecast
    return {
        "n": int(len(actual)),
        "mae": float(np.mean(np.abs(actual - forecast))),
        "rmse": float(np.sqrt(np.mean(np.square(actual - forecast)))),
        "qlike": float(np.mean(qlike)),
    }


def _risk_metrics(frame: pd.DataFrame, confidence: float, suffix: int, var_column: str, es_column: str) -> dict:
    actual_loss = -frame["actual_return"].to_numpy(dtype=float)
    var = pd.to_numeric(frame[var_column], errors="coerce").to_numpy(dtype=float)
    es = pd.to_numeric(frame[es_column], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(actual_loss) & np.isfinite(var) & np.isfinite(es) & (var > 0) & (es > var)
    if valid.sum() < 3:
        return {"status": "insufficient_oos_tail_forecasts", "n": int(valid.sum())}
    hits = actual_loss[valid] > var[valid]
    return {
        "status": "ok",
        "n": int(valid.sum()),
        "var_pinball_loss": quantile_loss(actual_loss[valid], var[valid], confidence),
        "kupiec": kupiec_uc(hits, 1 - confidence),
        "fz0_mean": float(np.mean(fz0_loss(actual_loss[valid], var[valid], es[valid], confidence))),
        "evaluation_sample": "non-overlapping origins",
    }


def walk_forward_lob(
    dataset: pd.DataFrame,
    horizon_samples: int,
    test_fraction: float = 0.2,
    minimum_train_rows: int = 1000,
    refit_interval_samples: int = 3600,
    confidence_levels: tuple[float, ...] = (0.95, 0.99),
) -> tuple[pd.DataFrame, dict]:
    """Fit only on labels whose future horizon ended before each forecast origin."""
    if not 0 < test_fraction < 1 or minimum_train_rows < 30 or refit_interval_samples < 1 or horizon_samples < 1:
        raise ValueError("Invalid walk-forward settings")
    assert_no_target_leakage(dataset)
    frame = dataset.sort_values("sample_time_utc").reset_index(drop=True).copy()
    frame["sample_time_utc"] = pd.to_datetime(frame["sample_time_utc"], utc=True)
    frame["target_end_utc"] = pd.to_datetime(frame["target_end_utc"], utc=True)
    frame["target_rv"] = pd.to_numeric(frame["target_rv"], errors="coerce")
    frame["target_return"] = pd.to_numeric(frame["target_return"], errors="coerce")
    feature_names = [
        name for name in feature_columns(frame)
        if name not in {"event_age_ms", "sampling_interval_seconds", "bid_level_count", "ask_level_count"}
    ]
    har_features = [name for name in ("past_rv_60", "past_rv_300", "past_rv_3600") if name in feature_names]
    numeric = frame[feature_names].apply(pd.to_numeric, errors="coerce")
    eligible = frame["target_valid"].fillna(False).astype(bool) & frame["target_rv"].notna()
    eligible_indices = np.flatnonzero(eligible.to_numpy())
    test_start = int(len(eligible_indices) * (1 - test_fraction))
    test_indices = eligible_indices[test_start:]

    rows: list[dict] = []
    fitted_rv: dict = {}
    fitted_har = None
    fitted_quantile: dict[float, object] = {}
    last_fit_index = -10**12
    cached_risk = {confidence: (float("nan"), float("nan")) for confidence in confidence_levels}
    fit_failures: dict[str, str] = {}
    for step_index, idx in enumerate(test_indices):
        origin = frame.loc[idx, "sample_time_utc"]
        train_mask = eligible & (frame["target_end_utc"] <= origin) & (frame["sample_time_utc"] < origin)
        train_indices = np.flatnonzero(train_mask.to_numpy())
        past_window = train_indices[-max(30000, minimum_train_rows):]
        should_refit = len(train_indices) >= minimum_train_rows and (
            not fitted_rv or step_index - last_fit_index >= refit_interval_samples
        )
        if should_refit:
            fit_x = numeric.iloc[past_window]
            fit_y = frame.loc[past_window, "target_rv"].to_numpy(dtype=float)
            valid_rv = np.isfinite(fit_y) & (fit_y > 0)
            if valid_rv.sum() >= minimum_train_rows:
                if har_features:
                    try:
                        fitted_har = make_har_rv_regressor()
                        fitted_har.fit(fit_x.loc[:, har_features].iloc[np.flatnonzero(valid_rv)], np.log(fit_y[valid_rv]))
                        fit_failures.pop("har_rv", None)
                    except Exception as exc:
                        fitted_har = None
                        fit_failures["har_rv"] = f"{type(exc).__name__}: {exc}"
                for name, estimator in make_lob_rv_regressors().items():
                    try:
                        estimator.fit(fit_x.iloc[np.flatnonzero(valid_rv)], np.log(fit_y[valid_rv]))
                        fitted_rv[name] = estimator
                        fit_failures.pop(name, None)
                    except Exception as exc:  # A failed optional fit must not suppress valid baselines.
                        fit_failures[name] = f"{type(exc).__name__}: {exc}"
                last_fit_index = step_index

            train_returns = frame.loc[past_window, "target_return"].to_numpy(dtype=float)
            valid_returns = np.isfinite(train_returns)
            cached_returns = train_returns[valid_returns]
            cached_risk = {
                confidence: empirical_var_es(cached_returns, confidence)
                for confidence in confidence_levels
            }
            fitted_quantile = {}
            if valid_returns.sum() >= minimum_train_rows:
                return_x = fit_x.iloc[np.flatnonzero(valid_returns)]
                return_y = train_returns[valid_returns]
                for confidence in confidence_levels:
                    try:
                        quantile_model = make_lob_quantile_regressor(confidence)
                        quantile_model.fit(return_x, return_y)
                        fitted_quantile[confidence] = quantile_model
                        fit_failures.pop(f"quantile_{confidence}", None)
                    except Exception as exc:
                        fit_failures[f"quantile_{confidence}"] = f"{type(exc).__name__}: {exc}"

        actual_rv = float(frame.loc[idx, "target_rv"])
        actual_return = float(frame.loc[idx, "target_return"])
        persistence_window = frame.loc[idx].get(f"past_rv_{horizon_samples}")
        persistence_reference = horizon_samples
        if persistence_window is None or pd.isna(persistence_window):
            persistence_window = frame.loc[idx].get("past_rv_60")
            persistence_reference = 60
        row = {
            "symbol": frame.loc[idx].get("symbol"),
            "origin_time_utc": origin,
            "target_end_utc": frame.loc[idx, "target_end_utc"],
            "horizon_samples": horizon_samples,
            "actual_rv": actual_rv,
            "actual_return": actual_return,
            "persistence_rv_forecast": persistence_rv(
                persistence_window, horizon_samples, reference_window=persistence_reference
            ),
            "har_rv_forecast": None,
            "ridge_rv_forecast": None,
            "hist_gradient_boosting_rv_forecast": None,
            "n_train": int(len(train_indices)),
            "status": "ok" if len(train_indices) >= minimum_train_rows else "insufficient_past_labels",
        }
        if fitted_har is not None and har_features:
            prediction = float(np.exp(np.clip(fitted_har.predict(numeric.loc[[idx], har_features])[0], -700, 700)))
            row["har_rv_forecast"] = max(prediction, np.finfo(float).tiny)
        for name, estimator in fitted_rv.items():
            prediction = float(np.exp(np.clip(estimator.predict(numeric.iloc[[idx]])[0], -700, 700)))
            row[f"{name}_rv_forecast"] = max(prediction, np.finfo(float).tiny)
        for confidence in confidence_levels:
            suffix = int(round(confidence * 1000))
            empirical_var, empirical_es = cached_risk[confidence]
            row[f"var_loss_{suffix}"] = empirical_var if np.isfinite(empirical_var) else None
            row[f"es_loss_{suffix}"] = empirical_es if np.isfinite(empirical_es) else None
            quantile_model = fitted_quantile.get(confidence)
            qvar = qes = None
            if quantile_model is not None:
                predicted_return_quantile = float(quantile_model.predict(numeric.iloc[[idx]])[0])
                qvar = max(-predicted_return_quantile, np.finfo(float).tiny)
                historical_losses = -cached_returns
                tail = historical_losses[historical_losses >= qvar]
                qes = max(float(tail.mean()) if tail.size else qvar, qvar + np.finfo(float).eps)
            row[f"quantile_var_loss_{suffix}"] = qvar
            row[f"quantile_es_loss_{suffix}"] = qes
        rows.append(row)

    forecasts = pd.DataFrame(rows)
    common = {
        "dataset_rows": int(len(frame)),
        "eligible_targets": int(eligible.sum()),
        "minimum_train_rows": minimum_train_rows,
        "refit_interval_samples": refit_interval_samples,
        "feature_columns": feature_names,
        "input_source": "Binance Spot LOB only; mid-price derived from synchronized best bid and best ask",
    }
    if forecasts.empty:
        return forecasts, {"status": "insufficient_oos_targets", **common}

    metrics = {
        "status": "ok",
        **common,
        "oos_rows": int(len(forecasts)),
        "fit_failures": fit_failures,
        "volatility": {
            name: _vol_metrics(
                forecasts["actual_rv"].to_numpy(dtype=float),
                pd.to_numeric(forecasts[column], errors="coerce").to_numpy(dtype=float),
            )
            for name, column in {
                "persistence": "persistence_rv_forecast",
                "har_rv": "har_rv_forecast",
                "ridge": "ridge_rv_forecast",
                "hist_gradient_boosting": "hist_gradient_boosting_rv_forecast",
            }.items()
        },
        "risk": {},
    }
    risk_rows = forecasts.iloc[::horizon_samples] if horizon_samples > 1 else forecasts
    for confidence in confidence_levels:
        suffix = int(round(confidence * 1000))
        metrics["risk"][str(confidence)] = {
            "historical": _risk_metrics(
                risk_rows, confidence, suffix, f"var_loss_{suffix}", f"es_loss_{suffix}"
            ),
            "linear_quantile": _risk_metrics(
                risk_rows, confidence, suffix,
                f"quantile_var_loss_{suffix}", f"quantile_es_loss_{suffix}"
            ),
        }
    return forecasts, metrics
