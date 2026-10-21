"""Read RV comparison metrics generated from saved walk-forward forecasts."""

import json
from pathlib import Path

from app.core.config import REPOSITORY_ROOT


def get_evaluation(symbol: str, horizon_days: int | None = None) -> dict:
    path = REPOSITORY_ROOT / "results" / "metrics" / "rv_model_comparison.json"
    if not path.exists():
        raise FileNotFoundError("No research evaluation output; run backend/scripts/make_report_tables.py first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    key = str(symbol).upper()
    if key not in payload.get("symbols", {}):
        raise FileNotFoundError(f"No evaluation results for {key}")
    by_horizon = payload["symbols"][key]
    if horizon_days is not None:
        if str(horizon_days) not in by_horizon:
            raise FileNotFoundError(f"No {horizon_days}-day evaluation results for {key}")
        return {"symbol": key, "as_of_utc": payload.get("as_of_utc"), **by_horizon[str(horizon_days)]}
    return {"symbol": key, "as_of_utc": payload.get("as_of_utc"), "horizons": by_horizon}
