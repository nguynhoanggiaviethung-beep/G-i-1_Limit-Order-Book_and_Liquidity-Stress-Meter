"""Availability-time checks for leakage-resistant feature frames."""

from __future__ import annotations

import pandas as pd


class PITFrame:
    """Pair feature values with their timestamps of availability."""

    def __init__(self, values: pd.DataFrame, available_at_utc: pd.DataFrame | pd.Series):
        self.values = values
        self.available_at_utc = available_at_utc

    def assert_available(self, origins: pd.Series | pd.DatetimeIndex) -> None:
        origin = pd.Series(pd.to_datetime(origins, utc=True)).reset_index(drop=True)
        if isinstance(self.available_at_utc, pd.Series):
            latest = pd.Series(pd.to_datetime(self.available_at_utc, utc=True)).reset_index(drop=True)
            if (latest > origin).any():
                bad = int((latest > origin).to_numpy().nonzero()[0][0])
                raise ValueError(f"Point-in-time leakage at row {bad}: feature available after origin")
            return
        available = self.available_at_utc.apply(lambda column: pd.to_datetime(column, utc=True))
        for column in available:
            values = pd.Series(available[column]).reset_index(drop=True)
            if (values > origin).any():
                bad = int((values > origin).to_numpy().nonzero()[0][0])
                raise ValueError(f"Point-in-time leakage at row {bad}: feature available after origin")
