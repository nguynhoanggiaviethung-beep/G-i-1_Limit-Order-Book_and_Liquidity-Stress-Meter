"""Create LOB-mid-price RV/return targets using fixed clock-time horizons."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import load_settings
from app.datasets.lob_builder import build_dataset, load_lob_features, save_dataset


def main() -> None:
    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", choices=settings.symbols, action="append")
    parser.add_argument("--horizon-seconds", type=int, action="append")
    args = parser.parse_args()
    symbols = args.symbol or list(settings.symbols)
    horizons = args.horizon_seconds or list(settings.lob_target_horizons_seconds)
    for symbol in symbols:
        samples = load_lob_features(settings.data_dir, symbol)
        for horizon_seconds in horizons:
            if horizon_seconds % settings.lob_sampling_interval_seconds:
                raise ValueError("Each horizon must be divisible by the LOB sampling interval")
            horizon_samples = horizon_seconds // settings.lob_sampling_interval_seconds
            dataset = build_dataset(samples, horizon_samples, settings.lob_sampling_interval_seconds)
            path = save_dataset(dataset, settings.data_dir, symbol, horizon_samples)
            valid = int(dataset["target_valid"].sum())
            print(f"{symbol} H={horizon_seconds}s: {valid:,}/{len(dataset):,} valid labels -> {path}")


if __name__ == "__main__":
    main()
