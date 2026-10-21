"""Run walk-forward LOB-derived RV, historical VaR, and ES baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import REPOSITORY_ROOT, load_settings
from app.datasets.lob_builder import load_dataset
from app.evaluation.lob_volatility import walk_forward_lob
from app.storage.manifest import sha256_file


def main() -> None:
    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", choices=settings.symbols, action="append")
    parser.add_argument("--horizon-seconds", type=int, action="append")
    parser.add_argument("--minimum-train-rows", type=int, default=settings.lob_minimum_train_rows)
    args = parser.parse_args()
    symbols = args.symbol or list(settings.symbols)
    horizons = args.horizon_seconds or list(settings.lob_target_horizons_seconds)
    for symbol in symbols:
        for horizon_seconds in horizons:
            horizon_samples = horizon_seconds // settings.lob_sampling_interval_seconds
            try:
                dataset = load_dataset(settings.data_dir, symbol, horizon_samples)
            except (FileNotFoundError, ValueError) as exc:
                print(f"Skip {symbol} H={horizon_seconds}s: {exc}")
                continue
            forecasts, metrics = walk_forward_lob(
                dataset,
                horizon_samples,
                settings.lob_test_fraction,
                args.minimum_train_rows,
                settings.lob_refit_interval_samples,
            )
            forecast_dir = REPOSITORY_ROOT / "results" / "forecasts" / "lob"
            metric_dir = REPOSITORY_ROOT / "results" / "metrics" / "lob"
            forecast_dir.mkdir(parents=True, exist_ok=True)
            metric_dir.mkdir(parents=True, exist_ok=True)
            stem = f"{symbol.lower()}_h{horizon_seconds}s"
            forecasts.to_parquet(forecast_dir / f"{stem}_walkforward.parquet", index=False)
            metrics.update({
                "symbol": symbol,
                "horizon_seconds": horizon_seconds,
                "sampling_interval_seconds": settings.lob_sampling_interval_seconds,
                "data_version": sha256_file(
                    settings.data_dir / "features" / "lob_datasets" / f"symbol={symbol}" / f"horizon_samples={horizon_samples}.parquet"
                ),
                "config_sha256": sha256_file(REPOSITORY_ROOT / "config" / "settings.yaml"),
            })
            (metric_dir / f"{stem}_evaluation.json").write_text(
                json.dumps(metrics, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
            )
            print(f"{symbol} H={horizon_seconds}s: {metrics['status']} ({metrics.get('oos_rows', 0)} OOS rows)")


if __name__ == "__main__":
    main()
