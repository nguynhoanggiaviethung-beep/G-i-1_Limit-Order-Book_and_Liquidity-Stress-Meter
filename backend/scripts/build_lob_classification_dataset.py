"""Build point-in-time classification features from normalized replay partitions."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import REPOSITORY_ROOT, load_settings
from app.data.grid_store import GridStore
from app.features.classification_features import build_classification_features


def build_partition_features(raw_frame: pd.DataFrame, previous_tail: pd.DataFrame, date: str, symbol: str, settings, impact_override: float | None = None):
    """Build daily output with enough preceding rows to warm every rolling feature."""
    warmup_rows = settings.lob_feature_lookback_seconds + 1
    frame = pd.concat([previous_tail, raw_frame], ignore_index=True) if not previous_tail.empty else raw_frame.copy()
    impact_notional = impact_override if impact_override is not None else settings.impact_notional_for(symbol)
    features = build_classification_features(frame, impact_notional, settings.tick_size_for(symbol))
    if date != "unknown":
        day_start = pd.Timestamp(f"{date}T00:00:00Z")
        day_end = day_start + pd.DateOffset(days=1)
        features = features.loc[(features["timestamp_utc"] >= day_start) & (features["timestamp_utc"] < day_end)].reset_index(drop=True)
    return features, frame.tail(warmup_rows).copy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, default=REPOSITORY_ROOT / "config" / "settings.yaml", help="Settings file that selects the source grid and feature policy")
    parser.add_argument("--symbol", choices=("BTCUSDT", "ETHUSDT"), help="Build only one configured symbol")
    parser.add_argument("--impact-notional-usdt", type=float)
    args = parser.parse_args()
    settings_path = args.settings if args.settings.is_absolute() else REPOSITORY_ROOT / args.settings
    settings = load_settings(settings_path)
    store = GridStore(settings.data_dir / "grid")
    output_root = store.root.parent / "features"
    symbols = (args.symbol,) if args.symbol else settings.symbols
    found = 0
    for symbol in symbols:
        previous_tail = pd.DataFrame()
        for source in store._files(symbol):
            raw_frame = pd.read_parquet(source)
            date = store._date_from_path(source) or "unknown"
            features, previous_tail = build_partition_features(raw_frame, previous_tail, date, symbol, settings, args.impact_notional_usdt)
            output = output_root / f"symbol={symbol}" / f"date={date}" / "features_1s.parquet"
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(".parquet.tmp")
            features.to_parquet(temporary, index=False)
            os.replace(temporary, output)
            valid_count = int(features.get("book_valid", pd.Series(True, index=features.index)).fillna(False).astype(bool).sum())
            print(f"{symbol} {date}: {len(features):,} rows, {valid_count:,} valid -> {output.relative_to(REPOSITORY_ROOT)}")
            found += 1
    if not found:
        print(f"No normalized replay partitions found under {store.root}. Replay a captured or approved historical L2 event stream first.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

