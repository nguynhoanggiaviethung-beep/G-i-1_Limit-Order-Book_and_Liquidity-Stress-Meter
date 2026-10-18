"""Run isolated HAR window and rolling-training robustness variants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import load_settings  # noqa: E402
from app.core.research_config import load_research_config  # noqa: E402
from app.core.versioning import sha256_bytes, sha256_file  # noqa: E402
from app.evaluation.volatility_metrics import volatility_metrics  # noqa: E402
from app.features.dataset import DatasetSpec, build_dataset  # noqa: E402
from app.evaluation.walkforward import walk_forward  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--horizon", type=int, action="append")
    parser.add_argument("--variant", action="append")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    settings = load_settings()
    research = load_research_config()
    config_path = ROOT / "config" / "robustness.yaml"
    robustness = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    variants = robustness["variants"]
    selected_names = args.variant or list(variants)
    unknown = set(selected_names) - set(variants)
    if unknown:
        parser.error(f"unknown robustness variant(s): {sorted(unknown)}")

    symbols = [value.upper() for value in (args.symbol or research["symbols"])]
    horizons = args.horizon or research["horizons_days"]
    cutoff = pd.Timestamp(research["as_of_utc"])
    snapshot_label = cutoff.strftime("%Y-%m-%dT%H-%M-%SZ")
    manifest_path = settings.data_dir / "snapshots" / snapshot_label / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Freeze the research snapshot first: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_dir = settings.data_dir / "features" / "snapshots" / snapshot_label
    results_root = ROOT / "results" / "robustness"
    results_root.mkdir(parents=True, exist_ok=True)
    table_path = ROOT / "results" / "tables" / "robustness_summary.csv"
    summary_rows = pd.read_csv(table_path).to_dict("records") if table_path.exists() else []

    for variant_name in selected_names:
        variant = variants[variant_name]
        feature_windows = tuple(int(value) for value in variant["feature_windows"])
        training_window_days = variant.get("training_window_days")
        model = str(variant["model"])
        variant_hash = sha256_bytes(
            f"{sha256_file(ROOT / 'config' / 'research.yaml')}|{sha256_file(config_path)}|{variant_name}".encode()
        )
        variant_dir = results_root / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)
        for symbol in symbols:
            daily_path = feature_dir / f"symbol={symbol}" / "daily_rv.parquet"
            if not daily_path.exists():
                raise FileNotFoundError(f"Missing frozen features: {daily_path}")
            daily = pd.read_parquet(daily_path)
            for horizon in horizons:
                output_path = variant_dir / f"{symbol.lower()}_h{horizon}.parquet"
                if output_path.exists() and not args.overwrite:
                    existing = pd.read_parquet(output_path)
                    valid_version = (
                        not existing.empty
                        and existing.get("data_version", pd.Series(dtype=str)).eq(manifest["data_version"]).all()
                        and existing.get("config_hash", pd.Series(dtype=str)).eq(variant_hash).all()
                    )
                    if valid_version:
                        print(f"Reuse {output_path}; variant and snapshot hashes match", flush=True)
                        frame = existing
                    else:
                        frame = pd.DataFrame()
                else:
                    frame = pd.DataFrame()

                if frame.empty:
                    dataset = build_dataset(
                        daily,
                        DatasetSpec(horizon=int(horizon), windows=feature_windows),
                        as_of=research["as_of_utc"],
                    )
                    frame, info = walk_forward(
                        daily=daily,
                        dataset=dataset,
                        symbol=symbol,
                        horizon=int(horizon),
                        models=[model],
                        data_version=manifest["data_version"],
                        config_hash=variant_hash,
                        progress_every_origins=90,
                        training_window_days=(int(training_window_days) if training_window_days else None),
                    )
                    frame.to_parquet(output_path, index=False)
                    print(
                        f"{variant_name} {symbol} h={horizon}: {info['forecast_rows']:,} forecast rows; {output_path}",
                        flush=True,
                    )
                evaluated = frame.loc[
                    frame["status"].eq("ok")
                    & frame["fold"].eq("oos")
                    & frame["actual_rv_mean"].notna()
                ].copy()
                if evaluated.empty:
                    continue
                benchmark_path = ROOT / "results" / "forecasts" / f"{symbol.lower()}_h{horizon}_walkforward.parquet"
                benchmark = None
                if benchmark_path.exists():
                    base = pd.read_parquet(benchmark_path)
                    base = base.loc[
                        base["model"].eq("random_walk")
                        & base["status"].eq("ok")
                        & base["fold"].eq("oos")
                    ].drop_duplicates("origin_time").set_index("origin_time")["rv_forecast"]
                    benchmark = evaluated["origin_time"].map(base).to_numpy(dtype=float)
                result = volatility_metrics(
                    evaluated["actual_rv_mean"].to_numpy(dtype=float),
                    evaluated["rv_forecast"].to_numpy(dtype=float),
                    benchmark=benchmark,
                    train_floor=evaluated["train_forecast_floor"].to_numpy(dtype=float),
                    hac_lags=int(research["evaluation"]["hac_lags"][int(horizon)]),
                )
                summary_rows = [row for row in summary_rows if not (
                    row.get("variant") == variant_name
                    and row.get("symbol") == symbol
                    and int(row.get("horizon_days", -1)) == int(horizon)
                )]
                summary_rows.append({
                    "variant": variant_name,
                    "symbol": symbol,
                    "horizon_days": int(horizon),
                    "model": model,
                    **result,
                })
                table_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(summary_rows).to_csv(table_path, index=False)


if __name__ == "__main__":
    main()
