"""Generate resumable, direct-horizon out-of-sample forecasts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import load_settings  # noqa: E402
from app.core.research_config import load_research_config  # noqa: E402
from app.core.versioning import sha256_file  # noqa: E402
from app.features.dataset import DatasetSpec, build_dataset  # noqa: E402
from app.evaluation.walkforward import DEFAULT_MODELS, walk_forward  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--horizon", type=int, action="append")
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild outputs from OOS start")
    args = parser.parse_args()
    settings = load_settings()
    research = load_research_config()
    symbols = [value.upper() for value in (args.symbol or research["symbols"])]
    horizons = args.horizon or research["horizons_days"]
    models = args.models or DEFAULT_MODELS
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
            start_after = None
            if path.exists() and not args.overwrite:
                previous = pd.read_parquet(path)
                if not previous.empty:
                    start_after = pd.to_datetime(previous["origin_time"], utc=True).max()
            current, run_info = walk_forward(
                daily=daily,
                dataset=dataset,
                symbol=symbol,
                horizon=int(horizon),
                models=models,
                start_after=start_after,
                data_version=manifest["data_version"],
                config_hash=config_hash,
            )
            combined = pd.concat([previous, current], ignore_index=True)
            if not combined.empty:
                combined = combined.drop_duplicates(["symbol", "horizon_days", "origin_time", "model"], keep="last")
                combined = combined.sort_values(["origin_time", "model"]).reset_index(drop=True)
            temporary = path.with_suffix(".parquet.tmp")
            combined.to_parquet(temporary, index=False)
            temporary.replace(path)
            print(
                f"{symbol} h={horizon}: {run_info['forecast_rows']} new rows; "
                f"{len(combined)} stored; skipped={run_info['skipped_models']}; file={path}"
            )


if __name__ == "__main__":
    main()
