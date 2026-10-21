"""Read ex-ante regime labels from frozen or rolling daily features."""

import hashlib
import json
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
import pandas as pd

from app.core.config import load_settings
from app.core.research_config import load_research_config
from app.evaluation.regimes import label_regimes
from app.features.realized_volatility import daily_realized_volatility
from app.services.data_service import load_market_data

router = APIRouter(prefix="/api/v1/regimes", tags=["research regimes"])


@router.get("/{symbol}")
def get_regime(symbol: str, mode: Literal["research", "production"] = Query(default="research")) -> dict:
    symbol = symbol.upper()
    settings = load_settings()
    research = load_research_config()
    if symbol not in research["symbols"]:
        raise HTTPException(status_code=404, detail=f"Unsupported symbol: {symbol}")
    data_version = None
    if mode == "research":
        label = pd.Timestamp(research["as_of_utc"])
        folder = label.strftime("%Y-%m-%dT%H-%M-%SZ")
        path = settings.data_dir / "features" / "snapshots" / folder / f"symbol={symbol}" / "daily_rv.parquet"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Frozen daily features are missing; run backend/scripts/build_features.py")
        daily = pd.read_parquet(path)
        manifest_path = settings.data_dir / "snapshots" / folder / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                data_version = manifest.get("data_version")
            except (OSError, json.JSONDecodeError):
                data_version = None
        cutoff = pd.Timestamp(research["as_of_utc"])
        source = "frozen_snapshot"
    else:
        try:
            candles = load_market_data(symbol, settings.interval)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if candles.empty:
            raise HTTPException(status_code=404, detail=f"No synchronized candles are available for {symbol}")
        daily = daily_realized_volatility(
            candles,
            bars_per_day=int(research["sampling"]["bars_per_day"]),
        )
        cutoff = pd.Timestamp(datetime.now(timezone.utc))
        daily = daily.loc[pd.to_datetime(daily["available_at_utc"], utc=True) <= cutoff].copy()
        candle_hash = pd.util.hash_pandas_object(candles, index=False).to_numpy().tobytes()
        data_version = hashlib.sha256(candle_hash).hexdigest()
        source = "rolling_local_market_data"
    daily["available_at_utc"] = pd.to_datetime(daily["available_at_utc"], utc=True)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    daily = daily.loc[daily["available_at_utc"] <= cutoff].copy()
    if daily.empty:
        raise HTTPException(status_code=404, detail=f"No point-in-time regime history for {symbol}")
    rules = research.get("regimes", {})
    regimes = label_regimes(daily, rules)
    latest = daily.iloc[-1]
    return {
        "symbol": symbol,
        "mode": mode,
        "as_of_utc": cutoff.isoformat(),
        "latest_available_date_utc": pd.Timestamp(latest["utc_date"]).isoformat(),
        "regime": str(regimes.iloc[-1]),
        "data_version": data_version,
        "source": source,
        "rules": rules,
        "counts": regimes.value_counts().to_dict(),
    }
