"""Generate loss-positive VaR/ES forecasts and horizon-aware backtests."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import load_settings  # noqa: E402
from app.core.research_config import load_research_config  # noqa: E402
from app.evaluation.es_backtests import es_backtest  # noqa: E402
from app.evaluation.var_es_backtest import var_backtest  # noqa: E402
from app.risk.evt import evt_pot_var_es  # noqa: E402
from app.risk.fhs import fhs_var_es  # noqa: E402
from app.risk.var_es import fit_skewt_params, param_var_es  # noqa: E402


def _seed(*parts) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _residual_pool(daily: pd.DataFrame, origin: pd.Timestamp, kappa: float, window: int) -> np.ndarray:
    history = daily.loc[
        (pd.to_datetime(daily["available_at_utc"], utc=True) <= origin)
        & daily["is_complete_day"].fillna(False)
    ].tail(window)
    returns = pd.to_numeric(history["close_ret"], errors="coerce").to_numpy(dtype=float)
    rv = pd.to_numeric(history["rv"], errors="coerce").to_numpy(dtype=float)
    scale = np.sqrt(np.maximum(kappa * rv, np.finfo(float).tiny))
    residuals = returns / scale
    return residuals[np.isfinite(residuals)]


def _risk_record(
    method: str,
    sigma: float,
    residuals: np.ndarray,
    confidence: float,
    horizon: int,
    paths: int,
    seed: int,
    skew_params: dict | None = None,
) -> dict:
    if method == "fhs":
        return fhs_var_es(np.full(horizon, sigma), residuals, confidence, paths, seed)
    if method == "param_normal":
        var, es = param_var_es(sigma, "normal", confidence)
    elif method == "param_t":
        from scipy.stats import t as student_t

        try:
            degrees, _, _ = student_t.fit(residuals, floc=0, fscale=1)
        except Exception:
            degrees = 8.0
        degrees = float(np.clip(degrees, 2.05, 200))
        var, es = param_var_es(sigma, "t", confidence, {"nu": degrees})
    elif method == "param_skewt":
        skew_params = skew_params or fit_skewt_params(residuals)
        var, es = param_var_es(sigma, "skewt", confidence, skew_params)
    elif method == "evt":
        return evt_pot_var_es(
            sigma,
            residuals,
            confidence,
            threshold_quantile=0.90,
            min_exceedances=30,
        )
    else:
        raise ValueError(f"Unsupported risk method: {method}")
    return {
        "var_loss_percent": var,
        "expected_shortfall_loss_percent": es,
        "method": method,
        "horizon_days": 1,
        "risk_unit": "loss-positive, % for 1 day",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--horizon", type=int, action="append")
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--paths", type=int, help="Override FHS simulation paths")
    parser.add_argument("--last-origins", type=int, help="Limit each series to its most recent N forecast origins (for a quick pilot)")
    args = parser.parse_args()
    settings = load_settings()
    research = load_research_config()
    symbols = [value.upper() for value in (args.symbol or research["symbols"])]
    horizons = args.horizon or research["horizons_days"]
    model_filter = set(args.models or [])
    paths = int(args.paths or research["risk"]["fhs_paths"])
    forecast_dir = ROOT / "results" / "forecasts"
    metric_dir = ROOT / "results" / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)
    risk_rows: list[dict] = []
    skipped: dict[str, str] = {}

    for symbol in symbols:
        daily_path = (
            settings.data_dir / "features" / "snapshots"
            / pd.Timestamp(research["as_of_utc"]).strftime("%Y-%m-%dT%H-%M-%SZ")
            / f"symbol={symbol}" / "daily_rv.parquet"
        )
        if not daily_path.exists():
            raise FileNotFoundError(f"Missing frozen daily features: {daily_path}")
        daily = pd.read_parquet(daily_path)
        for horizon in horizons:
            forecast_path = forecast_dir / f"{symbol.lower()}_h{horizon}_walkforward.parquet"
            if not forecast_path.exists():
                raise FileNotFoundError(f"Run walk-forward forecasts first: {forecast_path}")
            forecasts = pd.read_parquet(forecast_path)
            forecasts = forecasts.loc[forecasts["status"].eq("ok")]
            if model_filter:
                forecasts = forecasts.loc[forecasts["model"].isin(model_filter)]
            if args.last_origins is not None:
                if args.last_origins < 1:
                    raise ValueError("--last-origins must be positive")
                selected_origins = (
                    pd.to_datetime(forecasts["origin_time"], utc=True)
                    .drop_duplicates().sort_values().tail(args.last_origins)
                )
                forecasts = forecasts.loc[
                    pd.to_datetime(forecasts["origin_time"], utc=True).isin(selected_origins)
                ]
            residual_cache: dict[pd.Timestamp, np.ndarray] = {}
            skew_cache: dict[pd.Timestamp, dict | None] = {}
            for row in forecasts.itertuples(index=False):
                origin = pd.Timestamp(row.origin_time)
                kappa = max(float(row.kappa_return_variance_to_rv), 1e-8)
                residuals = residual_cache.get(origin)
                if residuals is None:
                    residuals = _residual_pool(
                        daily,
                        origin,
                        kappa,
                        int(research["risk"]["fhs_window"]),
                    )
                    residual_cache[origin] = residuals
                if len(residuals) < 30:
                    skipped[f"{symbol}:h{horizon}:{row.model}:{origin}"] = "fewer than 30 standardized residuals"
                    continue
                sigma = float(np.sqrt(max(kappa * float(row.rv_forecast), np.finfo(float).tiny)))
                if horizon == 1:
                    methods = ["param_normal", "param_t", "param_skewt", "fhs", "evt"]
                    confidences = research["risk"]["confidence"]
                elif horizon == 7:
                    methods = ["fhs"]
                    confidences = research["risk"]["confidence"]
                else:
                    methods = ["fhs"]
                    confidences = [0.95]
                for method in methods:
                    for confidence in confidences:
                        try:
                            skew_params = None
                            if method == "param_skewt":
                                if origin not in skew_cache:
                                    try:
                                        skew_cache[origin] = fit_skewt_params(residuals)
                                    except Exception:
                                        skew_cache[origin] = None
                                skew_params = skew_cache[origin]
                                if skew_params is None:
                                    raise ValueError("skew-t innovation fit failed")
                            result = _risk_record(
                                method,
                                sigma,
                                residuals,
                                float(confidence),
                                int(horizon),
                                paths,
                                _seed(symbol, horizon, row.model, origin, method, confidence),
                                skew_params,
                            )
                            risk_rows.append({
                                "symbol": symbol,
                                "model": row.model,
                                "horizon_days": int(horizon),
                                "origin_time": origin,
                                "target_start": row.target_start,
                                "target_end": row.target_end,
                                "actual_loss_percent": -float(row.y_cum_return) if pd.notna(row.y_cum_return) else np.nan,
                                "confidence": float(confidence),
                                "method": method,
                                "sigma_daily_percent": sigma,
                                **result,
                                "reliability": "descriptive_only" if horizon == 30 else result.get("reliability", "estimated"),
                            })
                        except Exception as exc:
                            skipped[f"{symbol}:h{horizon}:{row.model}:{origin}:{method}:{confidence}"] = (
                                f"{type(exc).__name__}: {exc}"
                            )

    risk_frame = pd.DataFrame(risk_rows)
    if not risk_frame.empty:
        output = forecast_dir / "risk_forecasts.parquet"
        temporary = output.with_suffix(".parquet.tmp")
        risk_frame.to_parquet(temporary, index=False)
        temporary.replace(output)
    backtests = []
    if not risk_frame.empty:
        for keys, group in risk_frame.groupby(["symbol", "model", "horizon_days", "method", "confidence"], sort=True):
            group = group.dropna(subset=["actual_loss_percent"]).sort_values("origin_time")
            if group.empty:
                continue
            symbol, model, horizon, method, confidence = keys
            risk_summary = {
                "symbol": symbol, "model": model, "horizon_days": int(horizon),
                "method": method, "confidence": float(confidence),
                "risk_unit": f"loss-positive, % for {horizon} day(s)",
                "n_obs": int(len(group)),
                "n_independent_approx": int(len(group) // int(horizon)),
                "expected_exceedances": float(len(group) * (1 - float(confidence))),
            }
            if int(horizon) == 30:
                hits = group["actual_loss_percent"] > group["var_loss_percent"]
                risk_summary.update({
                    "exceptions": int(hits.sum()),
                    "exception_rate": float(hits.mean()),
                    "reliability": "descriptive_only",
                    "warning": "Small effective sample; no Kupiec or Christoffersen conclusion",
                })
            else:
                risk_summary["offset_backtests"] = [
                    var_backtest(
                        group["actual_loss_percent"].to_numpy(),
                        group["var_loss_percent"].to_numpy(),
                        float(confidence),
                        horizon=int(horizon),
                        offset=offset,
                    )
                    for offset in range(int(horizon))
                    if int(horizon) == 1 or float(confidence) < 0.99
                ]
            try:
                risk_summary["es_evaluation"] = es_backtest(
                    group["actual_loss_percent"],
                    group["var_loss_percent"],
                    group["expected_shortfall_loss_percent"],
                    float(confidence),
                )
            except ValueError as exc:
                risk_summary["es_evaluation_error"] = str(exc)
            backtests.append(risk_summary)

    payload = {
        "as_of_utc": research["as_of_utc"],
        "loss_convention": "positive percentage loss, L=-cumulative close-to-close return",
        "horizons_days": horizons,
        "fhs_paths": paths,
        "backtests": backtests,
        "skipped": skipped,
    }
    target = metric_dir / "var_es_backtests.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(target)
    print(f"Wrote risk forecasts to {forecast_dir / 'risk_forecasts.parquet'}")
    print(f"Wrote VaR/ES evaluation to {target}")


if __name__ == "__main__":
    main()
