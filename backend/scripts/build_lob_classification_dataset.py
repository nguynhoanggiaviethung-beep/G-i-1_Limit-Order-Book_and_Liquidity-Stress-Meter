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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", choices=("BTCUSDT", "ETHUSDT"), help="Build only one configured symbol")
    parser.add_argument("--impact-notional-usdt", type=float)
    args = parser.parse_args()
    settings = load_settings()
    impact_notional = args.impact_notional_usdt if args.impact_notional_usdt is not None else settings.lob_stress_impact_notional_usdt
    store = GridStore()
    output_root = store.root.parent / "features"
    symbols = (args.symbol,) if args.symbol else settings.symbols
    found = 0
    for symbol in symbols:
        for source in store._files(symbol):
            frame = pd.read_parquet(source)
            features = build_classification_features(frame, impact_notional)
            date = store._date_from_path(source) or "unknown"
            output = output_root / f"symbol={symbol}" / f"date={date}" / "features_1s.parquet"
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(".parquet.tmp")
            features.to_parquet(temporary, index=False)
            os.replace(temporary, output)
            valid_count = int(features.get("book_valid", pd.Series(True, index=features.index)).fillna(False).astype(bool).sum())
            print(f"{symbol} {date}: {len(features):,} rows, {valid_count:,} valid -> {output.relative_to(REPOSITORY_ROOT)}")
            found += 1
    if not found:
        print(f"No normalized replay partitions found under {store.root}. Add approved CryptoHFTData book_1s.parquet files first.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

