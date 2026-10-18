"""Generate point-in-time direct VaR/ES forecasts from conditional quantiles."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import load_settings  # noqa: E402
from app.core.research_config import load_research_config  # noqa: E402
from app.core.seeds import set_global_seed  # noqa: E402
from app.core.versioning import sha256_bytes, sha256_file  # noqa: E402
from app.features.dataset import DatasetSpec, build_dataset  # noqa: E402
from app.models.quantile import (  # noqa: E402
    QUANTILE_MODELS,
    empirical_expected_shortfall,
    fit_conditional_quantiles,
    predict_conditional_quantiles,
)


def _config_hash(research_path: Path, quantile_path: Path) -> str:
    packages = {}
    for package in ("scikit-learn", "lightgbm"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "missing"
    payload = {
        "research_config_sha256": sha256_file(research_path),
        "quantile_config_sha256": sha256_file(quantile_path),
        "packages": packages,
    }
    return sha256_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))


def _atomic_write(frame: pd.DataFrame, destination: Path) -> None:
    temporary = destination.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--horizon", type=int, action="append")
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    settings = load_settings()
    research = load_research_config()
    quantile_path = ROOT / "config" / "quantile.yaml"
    quantile_config = yaml.safe_load(quantile_path.read_text(encoding="utf-8"))
    research_path = ROOT / "config" / "research.yaml"
    config_hash = _config_hash(research_path, quantile_path)
    set_global_seed(int(research["seed"]))

    symbols = [value.upper() for value in (args.symbol or research["symbols"])]
    horizons = args.horizon or research["horizons_days"]
    models = args.models or list(QUANTILE_MODELS)
    unknown = set(models) - set(QUANTILE_MODELS)
    if unknown:
        parser.error(f"unsupported quantile model(s): {sorted(unknown)}")
    available_models = []
    for model in models:
        if model == "lightgbm_quantile" and importlib.util.find_spec("lightgbm") is None:
            print("SKIP lightgbm_quantile: install backend/requirements-optional.txt to enable it", flush=True)
        else:
            available_models.append(model)
    models = available_models

    cutoff = pd.Timestamp(research["as_of_utc"])
    label = cutoff.strftime("%Y-%m-%dT%H-%M-%SZ")
    manifest_path = settings.data_dir / "snapshots" / label / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Freeze the research snapshot first: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_dir = settings.data_dir / "features" / "snapshots" / label
    output_path = ROOT / "results" / "forecasts" / "quantile_forecasts.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    previous = pd.read_parquet(output_path) if output_path.exists() else pd.DataFrame()
    if not previous.empty and {"data_version", "config_hash"}.issubset(previous.columns):
        stale = (
            previous["data_version"].ne(manifest["data_version"]).any()
            or previous["config_hash"].ne(config_hash).any()
        )
        if stale:
            print("Quantile data/config version changed; rebuilding quantile forecasts", flush=True)
            previous = pd.DataFrame()
    elif not previous.empty:
        previous = pd.DataFrame()

    if args.overwrite and not previous.empty:
        mask = previous["symbol"].isin(symbols) & previous["horizon_days"].isin(horizons)
        if args.models:
            mask &= previous["model"].isin(models)
        previous = previous.loc[~mask].copy()

    quantiles = tuple(float(value) for value in quantile_config["quantiles"])
    refit_days = int(quantile_config["refit_every_days"])
    min_train_rows = int(quantile_config["min_train_rows"])
    as_of = pd.Timestamp(research["as_of_utc"], tz="UTC")
    burn_in = pd.Timestamp(research["windows"].get("burn_in_start", "2019-01-01"), tz="UTC")
    oos_start = pd.Timestamp(research["windows"]["oos_start"], tz="UTC")
    first_train_start = pd.Timestamp(research["windows"]["first_train_start"], tz="UTC")

    for symbol in symbols:
        daily_path = feature_dir / f"symbol={symbol}" / "daily_rv.parquet"
        if not daily_path.exists():
            raise FileNotFoundError(f"Build frozen features first: {daily_path}")
        daily = pd.read_parquet(daily_path).sort_values("utc_date").copy()
        complete_origins = set(
            (pd.to_datetime(daily.loc[daily["is_complete_day"].fillna(False), "utc_date"], utc=True)
             + pd.Timedelta(days=1)).tolist()
        )
        for horizon in horizons:
            dataset = build_dataset(daily, DatasetSpec(horizon=int(horizon)), as_of=as_of)
            feature_frame = dataset.x.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
            feature_frame = feature_frame.loc[:, feature_frame.notna().any(axis=0)]
            target_loss = -pd.to_numeric(dataset.meta["y_cum_return"], errors="coerce")
            target_end = pd.to_datetime(dataset.meta["target_end"], utc=True)
            origins = pd.to_datetime(dataset.meta["origin_time"], utc=True)
            output_rows = previous.loc[
                previous["symbol"].eq(symbol) & previous["horizon_days"].eq(int(horizon))
            ] if not previous.empty else pd.DataFrame()

            for model in models:
                model_previous = output_rows.loc[output_rows["model"].eq(model)] if not output_rows.empty else pd.DataFrame()
                start_after = (
                    pd.to_datetime(model_previous.loc[model_previous["status"].eq("ok"), "origin_time"], utc=True).max()
                    if not model_previous.empty and model_previous["status"].eq("ok").any()
                    else None
                )
                model_state = None
                last_refit = None
                new_rows: list[dict] = []
                model_error = None

                for position, origin in enumerate(feature_frame.index, start=1):
                    origin = pd.Timestamp(origin)
                    if position % 90 == 0:
                        print(
                            f"{symbol} h={horizon} {model}: processed {position:,}/{len(feature_frame):,} origins",
                            flush=True,
                        )
                    if origin < burn_in or origin > as_of or origin not in complete_origins:
                        continue
                    if start_after is not None and origin <= start_after:
                        # The model state still has to be rebuilt through this point for exact resume.
                        pass
                    train_mask = (
                        target_loss.notna()
                        & target_end.le(origin)
                        & origins.ge(first_train_start)
                    )
                    x_train = feature_frame.loc[train_mask]
                    valid_train = x_train.notna().all(axis=1)
                    x_train = x_train.loc[valid_train]
                    y_train = target_loss.loc[x_train.index]
                    current_x = feature_frame.loc[origin]
                    if len(y_train) < max(min_train_rows, int(quantile_config["models"].get(model, {}).get("min_train_rows", min_train_rows))):
                        continue
                    if current_x.isna().any():
                        continue
                    if model_state is None or last_refit is None or (origin - last_refit).days >= refit_days:
                        try:
                            model_state = fit_conditional_quantiles(
                                model,
                                x_train.to_numpy(dtype=float),
                                y_train.to_numpy(dtype=float),
                                quantiles=quantiles,
                                random_state=int(research["seed"]),
                                config={**quantile_config["models"].get(model, {}), "min_train_rows": min_train_rows},
                            )
                            last_refit = origin
                            model_error = None
                        except Exception as exc:
                            model_state = None
                            model_error = f"{type(exc).__name__}: {exc}"
                            print(f"{symbol} h={horizon} {model}: fit failed at {origin}: {model_error}", flush=True)
                            continue
                    if origin < oos_start:
                        continue
                    if start_after is not None and origin <= start_after:
                        continue
                    try:
                        predictions = predict_conditional_quantiles(
                            model,
                            model_state,
                            current_x.to_numpy(dtype=float),
                            quantiles,
                        )
                        actual = float(target_loss.loc[origin]) if pd.notna(target_loss.loc[origin]) else np.nan
                        for confidence in quantiles:
                            var = max(float(predictions[confidence]), 0.0)
                            es = empirical_expected_shortfall(y_train.to_numpy(dtype=float), var)
                            new_rows.append({
                                "symbol": symbol,
                                "model": model,
                                "horizon_days": int(horizon),
                                "origin_time": origin,
                                "target_start": dataset.meta.loc[origin, "target_start"],
                                "target_end": dataset.meta.loc[origin, "target_end"],
                                "actual_loss_percent": actual,
                                "confidence": float(confidence),
                                "method": "quantile_model",
                                "var_loss_percent": var,
                                "expected_shortfall_loss_percent": max(es, np.nextafter(var, np.inf)),
                                "n_train": int(len(y_train)),
                                "refit_at": last_refit,
                                "data_version": manifest["data_version"],
                                "config_hash": config_hash,
                                "status": "ok",
                                "reliability": "descriptive_only" if horizon == 30 else "estimated",
                                "volatility_path_assumption": "direct_conditional_quantile",
                                "risk_unit": f"loss-positive, % for {horizon} day(s)",
                            })
                    except Exception as exc:
                        print(f"{symbol} h={horizon} {model}: prediction failed at {origin}: {type(exc).__name__}: {exc}", flush=True)

                if model_error and not new_rows:
                    print(f"{symbol} h={horizon} {model}: no forecasts; last error: {model_error}", flush=True)
                if new_rows:
                    current = pd.DataFrame(new_rows)
                    previous = pd.concat([previous, current], ignore_index=True) if not previous.empty else current
                    previous = previous.drop_duplicates(
                        ["symbol", "horizon_days", "origin_time", "model", "confidence"], keep="last"
                    ).sort_values(["symbol", "horizon_days", "origin_time", "model", "confidence"])
                    _atomic_write(previous.reset_index(drop=True), output_path)
                print(
                    f"{symbol} h={horizon} {model}: {len(new_rows):,} new quantile rows; "
                    f"file={output_path}",
                    flush=True,
                )

    if not output_path.exists() and previous.empty:
        print("No quantile rows produced. Install optional packages or inspect fit errors above.", flush=True)


if __name__ == "__main__":
    main()
