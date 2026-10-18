"""Serve immutable historical research forecasts and current-data production forecasts."""

from datetime import datetime, timezone
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from app.core.config import REPOSITORY_ROOT, load_settings
from app.core.research_config import load_research_config
from app.core.seeds import set_global_seed
from app.core.versioning import sha256_file
from app.evaluation.walkforward import (
    BENCHMARK_MODELS,
    GARCH_MODELS,
    _fit_model,
    _forecast_benchmark,
    _kappa_for_returns,
)
from app.features.dataset import DatasetSpec, build_dataset
from app.features.realized_volatility import daily_realized_volatility
from app.models.garch import fit_garch_family, forecast_with_fixed_garch_parameters
from app.services.data_service import load_market_data


def _forecast_path(symbol: str, horizon_days: int) -> Path:
    return REPOSITORY_ROOT / "results" / "forecasts" / f"{symbol.lower()}_h{horizon_days}_walkforward.parquet"


def _risk_for_forecast(
    symbol: str,
    model: str,
    horizon_days: int,
    origin: pd.Timestamp,
    confidence_levels: list[float],
    data_version: str | None = None,
    config_hash: str | None = None,
) -> list[dict] | None:
    path = REPOSITORY_ROOT / "results" / "forecasts" / "risk_forecasts.parquet"
    if not path.exists():
        return None
    risks = pd.read_parquet(path)
    selected = risks.loc[
        risks["symbol"].eq(symbol)
        & risks["model"].eq(model)
        & risks["horizon_days"].eq(horizon_days)
        & (pd.to_datetime(risks["origin_time"], utc=True) == origin)
        & risks["confidence"].isin(confidence_levels)
    ]
    if data_version is not None and "data_version" in selected:
        selected = selected.loc[selected["data_version"].eq(data_version)]
    if config_hash is not None and "config_hash" in selected:
        selected = selected.loc[selected["config_hash"].eq(config_hash)]
    if selected.empty:
        return None
    columns = ["confidence", "method", "var_loss_percent", "expected_shortfall_loss_percent", "reliability"]
    return selected[[column for column in columns if column in selected]].to_dict("records")


def create_forecast(symbol: str, model: str, horizon_days: int, confidence_levels: list[float]) -> dict:
    symbol = symbol.upper()
    settings = load_settings()
    if symbol not in settings.symbols:
        raise ValueError(f"Unsupported symbol: {symbol}")
    invalid = set(confidence_levels) - set(settings.confidence_levels)
    if invalid:
        raise ValueError(f"Unsupported confidence level(s): {sorted(invalid)}")
    path = _forecast_path(symbol, horizon_days)
    if not path.exists():
        raise FileNotFoundError(f"No saved {horizon_days}-day forecast for {symbol}; run backend/scripts/run_walkforward.py")
    forecasts = pd.read_parquet(path)
    selected = forecasts.loc[(forecasts["model"] == model) & (forecasts["status"] == "ok")].copy()
    if selected.empty:
        available = sorted(forecasts.loc[forecasts["status"] == "ok", "model"].unique().tolist())
        raise FileNotFoundError(f"No successful forecast for model {model}; available models: {available}")
    row = selected.sort_values("origin_time").iloc[-1]
    origin = pd.Timestamp(row["origin_time"])
    return {
        "forecast_id": f"{symbol.lower()}-h{horizon_days}-{model}-{origin.strftime('%Y%m%d')}",
        "mode": "research",
        "symbol": symbol,
        "model": model,
        "horizon_days": horizon_days,
        "origin_time": origin.isoformat(),
        "target_start": pd.Timestamp(row["target_start"]).isoformat(),
        "target_end": pd.Timestamp(row["target_end"]).isoformat(),
        "data_version": row.get("data_version"),
        "config_hash": row.get("config_hash"),
        "actual_rv_mean": None if pd.isna(row["actual_rv_mean"]) else float(row["actual_rv_mean"]),
        "rv_forecast": float(row["rv_forecast"]),
        "sigma_h_approx": float(row["sigma_h_approx"]),
        "status": row["status"],
        "var_es": _risk_for_forecast(
            symbol, model, horizon_days, origin, confidence_levels,
            row.get("data_version"), row.get("config_hash"),
        ),
        "note": "Forecast loaded from immutable research snapshot; actual value is blank when unavailable at the frozen as-of date.",
    }


def create_production_forecast(
    symbol: str,
    model: str,
    horizon_days: int,
    confidence_levels: list[float],
) -> dict:
    """Fit from the rolling local Binance feed without changing frozen research files.

    Production requests use only the last complete UTC day's features and require
    the synchronized 5-minute candle feed to be no more than 15 minutes old.
    VaR/ES are returned as an explicitly labelled normal approximation.
    """
    symbol = symbol.upper()
    settings = load_settings()
    research = load_research_config()
    if symbol not in settings.symbols:
        raise ValueError(f"Unsupported symbol: {symbol}")
    if horizon_days not in (1, 7, 30):
        raise ValueError("Production horizon_days must be 1, 7, or 30")
    invalid = set(confidence_levels) - set(settings.confidence_levels)
    if invalid:
        raise ValueError(f"Unsupported confidence level(s): {sorted(invalid)}")

    now = pd.Timestamp(datetime.now(timezone.utc))
    candles = load_market_data(symbol, settings.interval)
    if candles.empty:
        raise FileNotFoundError(f"No synchronized production candles for {symbol}")
    candles = candles.sort_values("open_time").drop_duplicates("open_time", keep="last")
    latest_open = pd.Timestamp(pd.to_datetime(candles["open_time"], utc=True).max())
    latest_close = latest_open + pd.Timedelta(minutes=5)
    age_seconds = max((now - latest_close).total_seconds(), 0.0)
    if age_seconds > 15 * 60:
        raise ValueError(
            f"Production feed is stale: last closed candle {latest_close.isoformat()}, "
            f"age {int(age_seconds)} seconds (maximum 900)"
        )

    set_global_seed(int(research["seed"]))
    daily = daily_realized_volatility(candles, bars_per_day=int(research["sampling"]["bars_per_day"]))
    complete = daily.loc[daily["is_complete_day"].fillna(False)]
    if complete.empty:
        raise ValueError(f"No complete UTC daily feature row is available for {symbol}")
    latest_complete_day = pd.Timestamp(complete["utc_date"].max())
    origin = latest_complete_day + pd.Timedelta(days=1)
    if origin > now:
        raise ValueError("The latest complete UTC daily feature is not yet available")
    if origin < now.floor("D"):
        raise ValueError(
            f"Latest complete UTC day only supports origin {origin.isoformat()}; "
            "production requires the current UTC day's origin"
        )

    dataset = build_dataset(daily, DatasetSpec(horizon=horizon_days), as_of=now)
    if origin not in dataset.x.index:
        raise ValueError(f"No feature row is available at production origin {origin.isoformat()}")
    current_x = dataset.x.loc[[origin]]
    if current_x.isna().all(axis=1).iloc[0]:
        raise ValueError("All current production features are missing")
    available_labels = (
        dataset.y.notna()
        & (dataset.meta["target_end"] <= origin)
        & (dataset.meta["origin_time"] >= pd.Timestamp(research["windows"]["first_train_start"], tz="UTC"))
    )
    train_y = dataset.y.loc[available_labels]
    train_x = dataset.x.loc[available_labels]
    if train_y.empty and model not in BENCHMARK_MODELS:
        raise ValueError("No point-in-time training labels are available for this production model")

    kappa = _kappa_for_returns(daily, origin)
    if model in BENCHMARK_MODELS:
        rv_forecast, model_kappa = _forecast_benchmark(model, daily, origin, horizon_days)
    elif model in GARCH_MODELS:
        history = daily.loc[
            pd.to_datetime(daily["available_at_utc"], utc=True).le(origin)
            & daily["is_complete_day"].fillna(False)
        ]
        returns = pd.to_numeric(history["close_ret"], errors="coerce").dropna()
        if len(returns) < 500:
            raise ValueError("GARCH production forecast requires at least 500 complete daily returns")
        state = fit_garch_family(returns, model)
        variance_path = forecast_with_fixed_garch_parameters(returns, model, state.params, horizon_days)
        rv_forecast = float(np.mean(variance_path) / kappa)
        model_kappa = kappa
    else:
        state = _fit_model(
            model,
            train_x,
            train_y,
            int(research["seed"]),
            origin,
        )
        columns = list(state.feature_columns)
        if current_x[columns].isna().any(axis=None):
            raise ValueError("Current production features are incomplete for this model")
        prediction = float(np.asarray(state.estimator.predict(current_x[columns])).reshape(-1)[0])
        if model == "lightgbm_qlike":
            prediction = float(np.exp(np.clip(prediction, -30, 30)))
        elif state.log_target:
            prediction = float(np.exp(np.clip(prediction + 0.5 * state.residual_variance, -30, 30)))
        rv_forecast = max(prediction, np.finfo(float).tiny)
        model_kappa = kappa

    rv_forecast = max(float(rv_forecast), np.finfo(float).tiny)
    sigma_h = float(np.sqrt(max(kappa * horizon_days * rv_forecast, np.finfo(float).tiny)))
    var_es = []
    for confidence in confidence_levels:
        z = float(norm.ppf(confidence))
        var_es.append({
            "confidence": float(confidence),
            "method": "param_normal",
            "var_loss_percent": z * sigma_h,
            "expected_shortfall_loss_percent": float(norm.pdf(z) / (1 - confidence) * sigma_h),
            "risk_unit": f"loss-positive, % for {horizon_days} day(s)",
            "reliability": "production_normal_approximation",
        })

    generated_at = now.isoformat()
    return {
        "forecast_id": f"production-{symbol.lower()}-h{horizon_days}-{model}-{now.strftime('%Y%m%dT%H%M%SZ')}",
        "mode": "production",
        "symbol": symbol,
        "model": model,
        "horizon_days": horizon_days,
        "origin_time": origin.isoformat(),
        "target_start": origin.isoformat(),
        "target_end": (origin + pd.Timedelta(days=horizon_days)).isoformat(),
        "generated_at_utc": generated_at,
        "source_data_as_of_utc": latest_close.isoformat(),
        "source_data_age_seconds": round(age_seconds, 3),
        "data_version": hashlib.sha256(
            pd.util.hash_pandas_object(candles, index=False).to_numpy().tobytes()
        ).hexdigest(),
        "config_hash": sha256_file(REPOSITORY_ROOT / "config" / "research.yaml"),
        "actual_rv_mean": None,
        "rv_forecast": rv_forecast,
        "sigma_h_approx": sigma_h,
        "risk_unit": f"returns are percent; RV is percent squared; VaR/ES are loss-positive % for {horizon_days} day(s)",
        "kappa_return_variance_to_rv": float(kappa),
        "model_kappa": float(model_kappa),
        "status": "ok",
        "var_es": var_es,
        "note": (
            "Current rolling Binance data; data_version fingerprints the synchronized candle frame. "
            "This request refits from point-in-time complete days, "
            "does not write to the frozen research snapshot, and labels VaR/ES as a normal approximation."
        ),
    }


def get_forecast(forecast_id: str) -> dict:
    for symbol in load_settings().symbols:
        for horizon in (1, 7, 30):
            path = _forecast_path(symbol, horizon)
            if not path.exists():
                continue
            data = pd.read_parquet(path)
            rows = data.loc[
                (data["symbol"] == symbol)
                & (data["horizon_days"] == horizon)
                & data["status"].eq("ok")
            ]
            for _, row in rows.iterrows():
                candidate = f"{symbol.lower()}-h{horizon}-{row['model']}-{pd.Timestamp(row['origin_time']).strftime('%Y%m%d')}"
                if candidate == forecast_id:
                    origin = pd.Timestamp(row["origin_time"])
                    return {
                        "forecast_id": candidate,
                        "mode": "research",
                        "symbol": symbol,
                        "model": row["model"],
                        "horizon_days": horizon,
                        "origin_time": origin.isoformat(),
                        "target_start": pd.Timestamp(row["target_start"]).isoformat(),
                        "target_end": pd.Timestamp(row["target_end"]).isoformat(),
                        "data_version": row.get("data_version"),
                        "config_hash": row.get("config_hash"),
                        "actual_rv_mean": None if pd.isna(row["actual_rv_mean"]) else float(row["actual_rv_mean"]),
                        "rv_forecast": float(row["rv_forecast"]),
                        "sigma_h_approx": float(row["sigma_h_approx"]),
                        "status": row["status"],
                        "var_es": _risk_for_forecast(
                            symbol, row["model"], horizon,
                            pd.Timestamp(row["origin_time"]), list(load_settings().confidence_levels),
                            row.get("data_version"), row.get("config_hash"),
                        ),
                    }
    raise FileNotFoundError(f"Forecast not found: {forecast_id}")
