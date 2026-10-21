"""Service access to locally synchronized market data."""

from pathlib import Path

import pandas as pd

from app.core.config import load_settings


def load_market_data(symbol: str, interval: str = "5m") -> pd.DataFrame:
    settings = load_settings()
    path = Path(settings.data_dir) / "processed" / f"symbol={symbol.upper()}" / f"interval={interval}" / "klines.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No synchronized data for {symbol} {interval}")
    return pd.read_parquet(path)
