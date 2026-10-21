"""Past-only lags and rolling features for sampled LOB states."""

from __future__ import annotations

from collections import deque
from math import log

import numpy as np


class TemporalFeatureEngine:
    """Add sample-to-sample changes and trailing moments without look-ahead."""

    def __init__(self, max_history: int = 3600) -> None:
        self._history: deque[dict] = deque(maxlen=max_history)

    def reset(self) -> None:
        """Start a fresh contiguous segment after a stale or invalid sample gap."""
        self._history.clear()

    def transform(self, row: dict) -> dict:
        current = dict(row)
        previous = self._history[-1] if self._history else None
        current_mid = float(current["mid_price"])
        current_return = None if previous is None else log(current_mid / float(previous["mid_price"]))
        current["mid_return"] = current_return
        for name in ("relative_spread_bps", "obi_qty_5", "obi_qty_10", "obi_qty_20", "microprice_minus_mid_bps"):
            current[f"{name}_lag_1"] = previous.get(name) if previous else None
            current[f"{name}_change"] = (
                float(current[name]) - float(previous[name]) if previous and current.get(name) is not None else None
            )
        prior = list(self._history)
        for name in ("relative_spread_bps", "obi_qty_5", "obi_qty_10", "obi_qty_20", "microprice_minus_mid_bps"):
            values = [float(item[name]) for item in prior if item.get(name) is not None]
            for window in (60, 300, 3600):
                sample = values[-window:]
                current[f"{name}_mean_{window}"] = sum(sample) / len(sample) if sample else None
                current[f"{name}_q50_{window}"] = float(np.quantile(sample, 0.50)) if sample else None
                current[f"{name}_q95_{window}"] = float(np.quantile(sample, 0.95)) if sample else None
                if len(sample) > 1:
                    mean = current[f"{name}_mean_{window}"]
                    current[f"{name}_std_{window}"] = (
                        sum((value - mean) ** 2 for value in sample) / (len(sample) - 1)
                    ) ** 0.5
                else:
                    current[f"{name}_std_{window}"] = None
        self._history.append(current)
        returns = [float(item["mid_return"]) for item in list(self._history)[-3600:] if item.get("mid_return") is not None]
        for window in (60, 300, 3600):
            selected = returns[-window:]
            current[f"past_rv_{window}"] = sum(value * value for value in selected) if selected else None
        return current
