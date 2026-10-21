"""Generate resumable, direct-horizon out-of-sample forecasts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import load_settings  # noqa: E402
from app.core.research_config import load_research_config  # noqa: E402
from app.core.seeds import set_global_seed  # noqa: E402
from app.core.versioning import sha256_file  # noqa: E402
from app.features.dataset import DatasetSpec, build_dataset  # noqa: E402
from app.evaluation.walkforward import DEEP_LEARNING_MODELS, DEFAULT_MODELS, OPTIONAL_MODELS, walk_forward  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--horizon", type=int, action="append")
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild outputs from OOS start")
    parser.add_argument(
        "--rebuild-models", action="store_true",
        help="Recompute only the selected models from OOS start while preserving other models",
    )
    args = parser.parse_args()
    settings = load_settings()
    research = load_research_config()
    set_global_seed(int(research["seed"]))
    symbols = [value.upper() for value in (args.symbol or research["symbols"])]
    horizons = args.horizon or research["horizons_days"]
    models = args.models or DEFAULT_MODELS
    available_models = []
    for model in models:
        package = (
            "lightgbm" if model.startswith("lightgbm")
            else "xgboost" if model == "xgboost"
            else "torch" if model in DEEP_LEARNING_MODELS
            else None
        )
        if model in OPTIONAL_MODELS and package and importlib.util.find_spec(package) is None:
            requirements = "requirements-dl.txt" if package == "torch" else "requirements-ml.txt"
            print(f"SKIP {model}: install {requirements} to enable it", flush=True)
            continue
        available_models.append(model)
    models = available_models
    cutoff = pd.Timestamp(research["as_of_utc"])
    snapshot_dir = settings.data_dir / "snapshots" / cutoff.strftime("%Y-%m-%dT%H-%M-%SZ")
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Freeze the research snapshot first: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_dir = settings.data_dir / "features" / "snapshots" / cutoff.strftime("%Y-%m-%dT%H-%M-%SZ")
    output_dir = ROOT / "results" / "forecasts"
    output_dir.mkdir(parents=True, exist_ok=True)
    config_hash = sha256_file(ROOT / "config" / "research.yaml")

    for symbol in symbols:
        daily_path = feature_dir / f"symbol={symbol}" / "daily_rv.parquet"
        if not daily_path.exists():
            raise FileNotFoundError(f"Build frozen features first: {daily_path}")
        daily = pd.read_parquet(daily_path)
        for horizon in horizons:
            dataset = build_dataset(
                daily,
                DatasetSpec(horizon=int(horizon)),
                as_of=research["as_of_utc"],
            )
            path = output_dir / f"{symbol.lower()}_h{horizon}_walkforward.parquet"
            previous = pd.DataFrame()
            if path.exists() and not args.overwrite:
                previous = pd.read_parquet(path)
                version_invalid = not {"data_version", "config_hash"}.issubset(previous.columns)
                if not previous.empty and not version_invalid:
                    version_invalid = (
                        previous[["data_version", "config_hash"]].isna().any().any()
                        or previous["data_version"].ne(manifest["data_version"]).any()
                        or previous["config_hash"].ne(config_hash).any()
                    )
                if not previous.empty and version_invalid:
                    print(f"{symbol} h={horizon}: data/config version changed; rebuilding stored forecasts")
                    previous = pd.DataFrame()
            if args.overwrite:
                previous = pd.DataFrame()
            current_frames = []
            run_summaries = []
            for model in models:
                model_previous = (
                    previous.loc[previous["model"].eq(model)]
                    if not previous.empty and not args.rebuild_models
                    else pd.DataFrame()
                )
                start_after = None
                if not model_previous.empty:
                    successful = model_previous.loc[model_previous["status"].eq("ok")]
                    if not successful.empty:
                        start_after = pd.to_datetime(successful["origin_time"], utc=True).max()
                current, run_info = walk_forward(
                    daily=daily,
                    dataset=dataset,
                    symbol=symbol,
                    horizon=int(horizon),
                    models=[model],
                    start_after=start_after,
                    data_version=manifest["data_version"],
                    config_hash=config_hash,
                    progress_every_origins=90,
                )
                if not current.empty:
                    current_frames.append(current)
                run_summaries.append((model, run_info))
            current_all = pd.concat(current_frames, ignore_index=True) if current_frames else pd.DataFrame()
            # --overwrite clears prior rows; otherwise resume per model and preserve other models.
            combined = pd.concat([previous, current_all], ignore_index=True)
            if not combined.empty:
                combined = combined.drop_duplicates(["symbol", "horizon_days", "origin_time", "model"], keep="last")
                combined = combined.sort_values(["origin_time", "model"]).reset_index(drop=True)
            temporary = path.with_suffix(".parquet.tmp")
            combined.to_parquet(temporary, index=False)
            temporary.replace(path)
            print(f"{symbol} h={horizon}: {sum(info['forecast_rows'] for _, info in run_summaries)} new rows; {len(combined)} stored; file={path}")
            for model, info in run_summaries:
                if info["skipped_models"]:
                    print(f"  {model}: skipped {info['skipped_models']}")


if __name__ == "__main__":
    from _legacy_ohlcv import refuse_legacy_ohlcv
    refuse_legacy_ohlcv("run_walkforward.py")
