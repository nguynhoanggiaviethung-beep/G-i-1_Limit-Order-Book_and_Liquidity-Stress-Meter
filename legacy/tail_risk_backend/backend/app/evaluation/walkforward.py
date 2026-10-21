"""Expanding, direct-horizon forecast generation with point-in-time training."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Iterable

import numpy as np
import pandas as pd

from app.core.research_config import load_research_config
from app.evaluation.tuning import tune_hyperparameters
from app.models.benchmark import ewma_forecast, random_walk_forecast, rolling_mean_forecast
from app.models.deep_learning import DEEP_LEARNING_MODELS, DeepSequenceRegressor
from app.models.garch import fit_garch_family, forecast_with_fixed_garch_parameters
from app.models.har import HAR_FEATURE_SETS, make_har_model
from app.models.ml_linear import make_elasticnet
from app.models.ml_trees import make_lightgbm, make_random_forest, make_xgboost


ELASTICNET_FEATURES = (
    "ln_rv_lag_1", "ln_rv_mean_7", "ln_rv_mean_30", "bpv_share", "jump_share",
    "close_ret", "hl_range", "ln_volume", "ln_turnover", "ln_trade_count",
    "taker_buy_ratio", "day_of_week", "is_weekend",
)
ML_FEATURES = ELASTICNET_FEATURES


DEFAULT_MODELS = (
    "random_walk", "rolling_mean_7", "rolling_mean_30", "ewma_094",
    "har", "har_log", "har_j", "har_cj", "shar", "harq", "elasticnet", "random_forest",
    "garch_t", "gjr_t", "egarch_t", "garch_skewt",
    "lightgbm_l2", "lightgbm_qlike", "xgboost", *DEEP_LEARNING_MODELS,
)
BENCHMARK_MODELS = {"random_walk", "rolling_mean_7", "rolling_mean_30", "ewma_094"}
GARCH_MODELS = {"garch_t", "gjr_t", "egarch_t", "garch_skewt"}
OPTIONAL_MODELS = {"lightgbm_l2", "lightgbm_qlike", "xgboost", *DEEP_LEARNING_MODELS}
LOG_TARGET_MODELS = {"elasticnet", "lightgbm_l2", "xgboost"}


@dataclass
class _FitState:
    estimator: object
    fitted_at: pd.Timestamp
    feature_columns: tuple[str, ...]
    n_train: int
    kappa: float | None = None
    log_target: bool = False
    residual_variance: float = 0.0
    variance_floor: float = np.finfo(float).tiny
    hyperparameters: dict | None = None


def _model_features(frame: pd.DataFrame) -> list[str]:
    return [name for name in ML_FEATURES if name in frame and frame[name].notna().any()]


def _make_estimator(model: str, seed: int, hyperparameters: dict | None = None):
    hyperparameters = hyperparameters or {}
    if model in HAR_FEATURE_SETS:
        return make_har_model(log_scale=model == "har_log"), tuple(HAR_FEATURE_SETS[model])
    if model == "elasticnet":
        return make_elasticnet(random_state=seed, **hyperparameters), None
    if model == "random_forest":
        return make_random_forest(random_state=seed, **hyperparameters), None
    if model == "lightgbm_l2":
        return make_lightgbm(random_state=seed, **hyperparameters), None
    if model == "lightgbm_qlike":
        return make_lightgbm(qlike=True, random_state=seed, **hyperparameters), None
    if model == "xgboost":
        return make_xgboost(random_state=seed, **hyperparameters), None
    if model in DEEP_LEARNING_MODELS:
        return DeepSequenceRegressor(model_name=model, random_state=seed, **hyperparameters), None
    raise ValueError(f"Unknown fitted model: {model}")


def _fit_model(
    model: str,
    x: pd.DataFrame,
    y: pd.Series,
    seed: int,
    fit_time: pd.Timestamp,
    hyperparameters: dict | None = None,
) -> _FitState:
    estimator, required = _make_estimator(model, seed, hyperparameters)
    columns = list(required) if required else _model_features(x)
    if model in HAR_FEATURE_SETS:
        columns = list(required)
    valid = y.notna() & x[columns].notna().all(axis=1)
    if int(valid.sum()) < 100:
        raise ValueError(f"Too few complete expanding-window samples for {model}: {int(valid.sum())}")
    if model in DEEP_LEARNING_MODELS:
        valid_x = x.loc[valid, columns].astype(float)
        valid_y = y.loc[valid].to_numpy(dtype=float)
        timestamps = pd.DatetimeIndex(pd.to_datetime(valid_x.index, utc=True))
        lookback = estimator.lookback
        sequences, sequence_targets = [], []
        for end in range(lookback - 1, len(valid_x)):
            start = end - lookback + 1
            window_times = timestamps[start : end + 1]
            if len(window_times) != lookback or not np.all(np.diff(window_times.asi8) == pd.Timedelta(days=1).value):
                continue
            sequences.append(valid_x.iloc[start : end + 1].to_numpy(dtype=np.float32))
            sequence_targets.append(max(float(valid_y[end]), np.finfo(float).tiny))
        if len(sequence_targets) < 30:
            raise ValueError(f"Too few contiguous sequences for {model}: {len(sequence_targets)}")
        estimator.fit(np.stack(sequences), np.asarray(sequence_targets, dtype=float))
        n_train = len(sequence_targets)
    else:
        fit_y = np.log(np.maximum(y.loc[valid].to_numpy(dtype=float), np.finfo(float).tiny)) if model in LOG_TARGET_MODELS else y.loc[valid]
        estimator.fit(x.loc[valid, columns], fit_y)
        n_train = int(valid.sum())
    residual_variance = 0.0
    if model in LOG_TARGET_MODELS:
        residuals = fit_y - np.asarray(estimator.predict(x.loc[valid, columns]), dtype=float)
        residual_variance = float(np.var(residuals, ddof=1)) if len(residuals) > 1 else 0.0
    variance_floor = np.finfo(float).tiny
    if model in LOG_TARGET_MODELS | {"lightgbm_qlike"}:
        variance_floor = max(float(y.loc[valid].quantile(0.01)), np.finfo(float).tiny)
    return _FitState(
        estimator, fit_time, tuple(columns), n_train,
        log_target=model in LOG_TARGET_MODELS, residual_variance=residual_variance,
        variance_floor=variance_floor, hyperparameters=hyperparameters or {},
    )


def _kappa_for_returns(daily: pd.DataFrame, origin: pd.Timestamp) -> float:
    history = daily.loc[
        (pd.to_datetime(daily["available_at_utc"], utc=True) <= origin)
        & daily["is_complete_day"].fillna(False)
    ].tail(750)
    rv = pd.to_numeric(history["rv"], errors="coerce").to_numpy(dtype=float)
    returns = pd.to_numeric(history["close_ret"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(rv) & (rv > 0) & np.isfinite(returns)
    if valid.sum() < 30:
        return 1.0
    ratio = float(np.mean(np.square(returns[valid])) / np.mean(rv[valid]))
    return max(ratio, 1e-8)


def _forecast_benchmark(model: str, daily: pd.DataFrame, origin: pd.Timestamp, horizon: int) -> tuple[float, float]:
    history = daily.loc[
        (pd.to_datetime(daily["available_at_utc"], utc=True) <= origin)
        & daily["is_complete_day"].fillna(False)
    ]
    rv = pd.to_numeric(history["rv"], errors="coerce").dropna()
    if model == "random_walk":
        return random_walk_forecast(rv, horizon), 1.0
    if model == "rolling_mean_7":
        return rolling_mean_forecast(rv, 7, horizon), 1.0
    if model == "rolling_mean_30":
        return rolling_mean_forecast(rv, 30, horizon), 1.0
    if model == "ewma_094":
        kappa = _kappa_for_returns(daily, origin)
        returns = pd.to_numeric(history["close_ret"], errors="coerce").dropna()
        return ewma_forecast(returns, 0.94, horizon) / kappa, kappa
    raise ValueError(f"Unknown benchmark: {model}")


def _forecast_garch(state: _FitState, model: str, returns, horizon: int, kappa: float) -> float:
    path = forecast_with_fixed_garch_parameters(returns, model, state.estimator.params, horizon)
    return float(np.mean(path) / kappa)


def walk_forward(
    daily: pd.DataFrame,
    dataset,
    symbol: str,
    horizon: int,
    models: Iterable[str] | None = None,
    start_after: str | pd.Timestamp | None = None,
    data_version: str | None = None,
    config_hash: str | None = None,
    progress_every_origins: int | None = None,
    training_window_days: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Generate OOS rows while enforcing target_end <= origin for each refit."""
    research = load_research_config()
    names = tuple(models or DEFAULT_MODELS)
    oos_start = pd.Timestamp(research["windows"]["oos_start"], tz="UTC")
    burn_in_start = pd.Timestamp(research["windows"].get("burn_in_start", "2019-01-01"), tz="UTC")
    as_of = pd.Timestamp(research["as_of_utc"], tz="UTC")
    first_train_start = pd.Timestamp(research["windows"]["first_train_start"], tz="UTC")
    start_after_ts = pd.Timestamp(start_after) if start_after is not None else None
    if start_after_ts is not None and start_after_ts.tzinfo is None:
        start_after_ts = start_after_ts.tz_localize("UTC")
    daily = daily.sort_values("utc_date").copy()
    daily["utc_date"] = pd.to_datetime(daily["utc_date"], utc=True)
    daily["available_at_utc"] = pd.to_datetime(daily["available_at_utc"], utc=True)
    complete_origin_times = set(
        (daily.loc[daily["is_complete_day"].fillna(False), "utc_date"] + pd.Timedelta(days=1)).tolist()
    )
    configured = research["refit_every_days"]
    refit_days = {
        **{name: int(configured["har"]) for name in HAR_FEATURE_SETS},
        "elasticnet": int(configured["linear_ml"]),
        "random_forest": int(configured["trees"]),
        "lightgbm_l2": int(configured["trees"]),
        "lightgbm_qlike": int(configured["trees"]),
        "xgboost": int(configured["trees"]),
        **{name: int(configured.get("dl", 30)) for name in DEEP_LEARNING_MODELS},
        **{name: int(configured["garch"]) for name in GARCH_MODELS},
    }
    states: dict[str, _FitState] = {}
    tuning_params: dict[str, dict] = {}
    tuning_scores: dict[str, dict] = {}
    tuned_at: dict[str, pd.Timestamp] = {}
    rows = []
    skipped: dict[str, str] = {}
    all_x = dataset.x
    all_y = dataset.y
    meta = dataset.meta
    for position, origin in enumerate(all_x.index, start=1):
        if progress_every_origins and position % progress_every_origins == 0:
            print(
                f"{symbol} h={horizon} {','.join(names)}: "
                f"processed {position:,}/{len(all_x):,} daily origins through {origin}",
                flush=True,
            )
        origin = pd.Timestamp(origin)
        if origin < burn_in_start or origin > as_of:
            continue
        if origin not in complete_origin_times:
            continue
        should_emit = start_after_ts is None or origin > start_after_ts
        current_x = all_x.loc[[origin]]
        actual = float(all_y.loc[origin]) if pd.notna(all_y.loc[origin]) else np.nan
        info = meta.loc[origin]
        train_mask = (
            all_y.notna()
            & (meta["target_end"] <= origin)
            & (meta["origin_time"] >= first_train_start)
        )
        if training_window_days is not None:
            if training_window_days < 1:
                raise ValueError("training_window_days must be positive")
            train_mask &= meta["origin_time"] >= origin - pd.Timedelta(days=training_window_days)
        train_y = all_y.loc[train_mask]
        if train_y.empty:
            continue
        cached_train_x = None
        positive_training = train_y[train_y > 0]
        training_floor = float(positive_training.quantile(0.01)) if not positive_training.empty else np.finfo(float).tiny
        kappa = _kappa_for_returns(daily, origin)
        record_base = {
            "symbol": symbol,
            "horizon_days": horizon,
            "origin_time": origin,
            "target_start": info["target_start"],
            "target_end": info["target_end"],
            "actual_rv_mean": actual,
            "y_cum_return": info["y_cum_return"],
            "n_dropped_gaps": int(info["n_dropped_gaps"]),
            "n_train_available": int(len(train_y)),
            "train_forecast_floor": training_floor,
            "data_version": data_version,
            "config_hash": config_hash,
            "fold": "oos" if origin >= oos_start else "burn_in",
            "kappa_return_variance_to_rv": kappa,
        }
        for model in names:
            try:
                if model in BENCHMARK_MODELS:
                    forecast, model_kappa = _forecast_benchmark(model, daily, origin, horizon)
                    fitted_at = origin
                    n_train = len(train_y)
                elif model in GARCH_MODELS:
                    history = daily.loc[
                        (daily["available_at_utc"] <= origin)
                        & daily["is_complete_day"].fillna(False)
                    ]
                    returns = pd.to_numeric(history["close_ret"], errors="coerce").dropna()
                    state = states.get(model)
                    refit = state is None or (origin - state.fitted_at).days >= refit_days[model]
                    if refit:
                        if len(returns) < 500:
                            raise ValueError("GARCH requires at least 500 complete daily returns")
                        state = _FitState(
                            fit_garch_family(returns, model),
                            origin,
                            (),
                            len(returns),
                            kappa,
                        )
                        states[model] = state
                    forecast = _forecast_garch(state, model, returns, horizon, kappa)
                    fitted_at, n_train, model_kappa = state.fitted_at, state.n_train, kappa
                else:
                    state = states.get(model)
                    refit = state is None or (origin - state.fitted_at).days >= refit_days[model]
                    if refit:
                        if cached_train_x is None:
                            cached_train_x = all_x.loc[train_mask]
                        if model in {"elasticnet", "random_forest", "lightgbm_l2", "lightgbm_qlike", "xgboost"} and origin >= oos_start:
                            last_tuned = tuned_at.get(model)
                            retune_days = int(research["tuning"]["retune_every_days"])
                            if last_tuned is None or (origin - last_tuned).days >= retune_days:
                                try:
                                    tuning_params[model], tuning_scores[model] = tune_hyperparameters(
                                        model,
                                        cached_train_x,
                                        train_y,
                                        meta.loc[train_mask, "target_end"],
                                        folds=int(research["tuning"]["folds"]),
                                        embargo_days=int(research["tuning"]["embargo_days"]),
                                        max_candidates=int(research["tuning"].get("max_candidates_per_model", 6)),
                                        seed=int(research["seed"]),
                                    )
                                except Exception as exc:
                                    tuning_params[model] = {}
                                    tuning_scores[model] = {"status": "fallback_defaults", "error": f"{type(exc).__name__}: {exc}"}
                                tuned_at[model] = origin
                        state = _fit_model(
                            model,
                            cached_train_x,
                            train_y,
                            int(research["seed"]),
                            origin,
                            tuning_params.get(model),
                        )
                        states[model] = state
                    columns = list(state.feature_columns)
                    if current_x[columns].isna().any(axis=None):
                        raise ValueError("Current point-in-time features are incomplete")
                    if model in DEEP_LEARNING_MODELS:
                        sequence = all_x.loc[all_x.index <= origin, columns].tail(state.estimator.lookback)
                        if len(sequence) != state.estimator.lookback:
                            raise ValueError("Insufficient lookback history for deep-learning forecast")
                        prediction_input = sequence.to_numpy(dtype=np.float32)[None, :, :]
                    else:
                        prediction_input = current_x[columns]
                    prediction = float(np.asarray(state.estimator.predict(prediction_input)).reshape(-1)[0])
                    if model == "lightgbm_qlike":
                        prediction = float(np.exp(np.clip(prediction, -30, 30)))
                    elif state.log_target:
                        prediction = float(np.exp(np.clip(prediction + 0.5 * state.residual_variance, -30, 30)))
                    forecast = max(prediction, state.variance_floor)
                    fitted_at, n_train, model_kappa = state.fitted_at, state.n_train, kappa
                if should_emit:
                    rows.append({
                    **record_base,
                    "model": model,
                    "rv_forecast": max(float(forecast), np.finfo(float).tiny),
                    "sigma_h_approx": float(np.sqrt(max(kappa * horizon * max(float(forecast), 0), 0))),
                    "refit_at": fitted_at,
                    "n_train": int(n_train),
                    "hyperparameters": json.dumps(state.hyperparameters or {}, sort_keys=True) if model not in BENCHMARK_MODELS | GARCH_MODELS else "{}",
                    "tuned_at": tuned_at.get(model),
                    "tuning_score": json.dumps(tuning_scores.get(model, {}), sort_keys=True),
                    "status": "ok",
                    })
            except (ImportError, RuntimeError) as exc:
                skipped[model] = str(exc)
            except Exception as exc:
                if should_emit:
                    rows.append({
                    **record_base,
                    "model": model,
                    "rv_forecast": np.nan,
                    "sigma_h_approx": np.nan,
                    "refit_at": states[model].fitted_at if model in states else pd.NaT,
                    "n_train": int(len(train_y)),
                    "status": f"error: {type(exc).__name__}: {exc}",
                    })
    output = pd.DataFrame(rows)
    if not output.empty:
        output = output.sort_values(["origin_time", "model"]).reset_index(drop=True)
    return output, {"skipped_models": skipped, "forecast_rows": int(len(output))}
