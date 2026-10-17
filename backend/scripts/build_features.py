"""Build daily realized volatility features from normalized 5-minute candles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import pandas as pd  # noqa: E402
from app.core.config import load_settings  # noqa: E402
from app.features.realized_volatility import daily_realized_volatility  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", action="append")
    args = parser.parse_args()
    settings = load_settings()
    for symbol in args.symbol or settings.symbols:
        source = settings.data_dir / "processed" / f"symbol={symbol}" / f"interval={settings.interval}" / "klines.parquet"
        if not source.exists():
            raise FileNotFoundError(f"Run sync_data.py first; missing {source}")
        output = daily_realized_volatility(pd.read_parquet(source))
        destination = settings.data_dir / "features" / f"symbol={symbol}" / "daily_realized_volatility.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".parquet.tmp")
        output.to_parquet(temporary, index=False)
        temporary.replace(destination)
        print(f"Wrote {len(output)} daily rows to {destination}")


if __name__ == "__main__":
    main()
