"""Read out-of-sample evaluation results produced by train_models.py."""

import json
from pathlib import Path


def get_evaluation(symbol: str) -> dict:
    path = Path(__file__).resolve().parents[3] / "results" / "metrics" / "volatility_backtest.json"
    if not path.exists():
        raise FileNotFoundError("No evaluation output; run backend/scripts/train_models.py first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    symbol = symbol.upper()
    try:
        return {"symbol": symbol, **payload["symbols"][symbol]}
    except KeyError as exc:
        raise FileNotFoundError(f"No evaluation results for {symbol}") from exc
