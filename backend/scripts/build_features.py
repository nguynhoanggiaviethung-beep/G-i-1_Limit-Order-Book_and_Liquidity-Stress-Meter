"""Build research RV features from a frozen, hashed candle snapshot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import load_settings  # noqa: E402
from app.core.research_config import load_research_config  # noqa: E402
from app.data.snapshot import load_snapshot  # noqa: E402
from app.features.realized_volatility import daily_realized_volatility  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", help="UTC timestamp; defaults to frozen research config")
    parser.add_argument("--symbol", action="append", help="Limit work to a configured symbol")
    args = parser.parse_args()
    settings = load_settings()
    research = load_research_config()
    candles_by_symbol, manifest = load_snapshot(args.as_of)
    selected = [value.upper() for value in (args.symbol or research["symbols"])]
    unknown = sorted(set(selected) - set(candles_by_symbol))
    if unknown:
        raise ValueError(f"Symbols absent from the frozen snapshot: {unknown}")
    label = pd.Timestamp(manifest["as_of_utc"]).strftime("%Y-%m-%dT%H-%M-%SZ")
    base = settings.data_dir / "features" / "snapshots" / label
    for symbol in selected:
        daily = daily_realized_volatility(
            candles_by_symbol[symbol],
            bars_per_day=int(research["sampling"]["bars_per_day"]),
        )
        destination = base / f"symbol={symbol}" / "daily_rv.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".parquet.tmp")
        daily.to_parquet(temporary, index=False)
        temporary.replace(destination)
        print(
            f"{symbol}: {len(daily)} UTC dates, "
            f"{int(daily['is_complete_day'].sum())} complete; wrote {destination}"
        )


if __name__ == "__main__":
    main()
