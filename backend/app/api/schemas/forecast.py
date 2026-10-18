"""Research forecast request schema, using the frozen day horizons."""

from typing import Literal

from pydantic import BaseModel, Field


ForecastModel = Literal[
    "random_walk", "rolling_mean_7", "rolling_mean_30", "ewma_094",
    "har", "har_log", "har_j", "har_cj", "shar", "harq",
    "elasticnet", "random_forest", "lightgbm_l2", "lightgbm_qlike", "xgboost",
    "garch_t", "gjr_t", "egarch_t", "garch_skewt",
]


class ForecastRequest(BaseModel):
    symbol: Literal["BTCUSDT", "ETHUSDT"] = Field(
        description="Binance Spot symbol included in the configured study."
    )
    horizon_days: Literal[1, 7, 30] = Field(
        default=1,
        description="Direct forecast horizon in UTC days; research design supports 1, 7, or 30.",
    )
    model: ForecastModel = Field(
        default="random_walk",
        description="Volatility model name. Optional models require backend/requirements-optional.txt.",
    )
    mode: Literal["research", "production"] = Field(
        default="research",
        description=(
            "research loads the frozen walk-forward result; production fits a point-in-time model "
            "from the latest synchronized local Binance data."
        ),
    )
    confidence_levels: list[Literal[0.95, 0.975, 0.99]] = Field(
        default_factory=lambda: [0.95, 0.975, 0.99],
        min_length=1,
        description="VaR/ES confidence levels supported by the current risk configuration.",
    )
