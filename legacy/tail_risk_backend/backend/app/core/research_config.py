"""Frozen study configuration and reproducibility metadata."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from app.core.config import REPOSITORY_ROOT


@lru_cache(maxsize=1)
def load_research_config() -> dict[str, Any]:
    path = REPOSITORY_ROOT / "config" / "research.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a mapping in {path}")
    horizons = [int(value) for value in raw.get("horizons_days", [])]
    if not horizons or horizons != sorted(set(horizons)) or any(value < 1 for value in horizons):
        raise ValueError("horizons_days must be a sorted list of positive integers")
    embargo = int(raw.get("tuning", {}).get("embargo_days", 0))
    if embargo < max(horizons) - 1:
        raise ValueError("embargo_days must be at least max(horizons_days) - 1")
    if raw.get("windows", {}).get("scheme") != "expanding":
        raise ValueError("The primary research scheme is frozen as expanding")
    if raw.get("primary_horizon") not in horizons:
        raise ValueError("primary_horizon must be included in horizons_days")
    return raw
