"""Parquet repository for normalized market data."""

from pathlib import Path

import pandas as pd


class ParquetRepository:
    def __init__(self, root: Path):
        self.root = Path(root)

    def market_data_path(self, symbol: str, interval: str) -> Path:
        return self.root / "processed" / f"symbol={symbol.upper()}" / f"interval={interval}" / "klines.parquet"

    def read_market_data(self, symbol: str, interval: str = "5m") -> pd.DataFrame:
        return pd.read_parquet(self.market_data_path(symbol, interval))
