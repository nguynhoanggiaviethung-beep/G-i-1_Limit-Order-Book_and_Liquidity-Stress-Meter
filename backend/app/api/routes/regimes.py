"""Read ex-ante regime labels from the frozen daily feature snapshot."""

from fastapi import APIRouter, HTTPException
import pandas as pd

from app.core.config import load_settings
from app.core.research_config import load_research_config
from app.evaluation.regimes import label_regimes

router = APIRouter(prefix="/api/v1/regimes", tags=["research regimes"])


@router.get("/{symbol}")
def get_regime(symbol: str) -> dict:
    symbol = symbol.upper()
    settings = load_settings()
    research = load_research_config()
    if symbol not in research["symbols"]:
        raise HTTPException(status_code=404, detail=f"Unsupported symbol: {symbol}")
    label = pd.Timestamp(research["as_of_utc"])
    folder = label.strftime("%Y-%m-%dT%H-%M-%SZ")
    path = settings.data_dir / "features" / "snapshots" / folder / f"symbol={symbol}" / "daily_rv.parquet"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Frozen daily features are missing; run backend/scripts/build_features.py")
    daily = pd.read_parquet(path)
    daily["available_at_utc"] = pd.to_datetime(daily["available_at_utc"], utc=True)
    cutoff = pd.Timestamp(research["as_of_utc"])
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    daily = daily.loc[daily["available_at_utc"] <= cutoff].copy()
    rules = research.get("regimes", {})
    regimes = label_regimes(daily, rules)
    if daily.empty:
        raise HTTPException(status_code=404, detail=f"No point-in-time regime history for {symbol}")
    latest = daily.iloc[-1]
    return {
        "symbol": symbol,
        "as_of_utc": cutoff.isoformat(),
        "latest_available_date_utc": pd.Timestamp(latest["utc_date"]).isoformat(),
        "regime": str(regimes.iloc[-1]),
        "rules": rules,
        "counts": regimes.value_counts().to_dict(),
    }
