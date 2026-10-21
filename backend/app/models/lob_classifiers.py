"""Leakage-safe classifier factories for direction and liquidity stress."""

from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


class EncodedClassifier:
    """Preserve original string labels while fitting numeric-only boosters."""

    def __init__(self, estimator, class_order: tuple[object, ...]) -> None:
        self.estimator = estimator
        self.classes_ = list(class_order)

    def fit(self, x, y):
        mapping = {label: index for index, label in enumerate(self.classes_)}
        encoded = [mapping[label] for label in y]
        self.estimator.fit(x, encoded)
        return self

    def predict_proba(self, x):
        return self.estimator.predict_proba(x)

    def predict(self, x):
        encoded = self.estimator.predict(x)
        return [self.classes_[int(value)] for value in encoded]


LOB_CLASSIFICATION_FEATURES = (
    "spread_ticks", "relative_spread_bps", "microprice_minus_mid_bps",
    "obi_qty_1", "obi_qty_5", "obi_qty_10", "obi_qty_20",
    "bid_depth_qty_5", "ask_depth_qty_5", "bid_depth_qty_10", "ask_depth_qty_10",
    "bid_depth_qty_20", "ask_depth_qty_20", "depth_notional_10",
    "ofi_1s", "ofi_5s", "ofi_10s",
    "trade_imbalance_5s", "trade_imbalance_10s", "price_impact_bps",
    "addition_intensity_1s", "cancellation_intensity_1s",
    "addition_intensity_5s", "cancellation_intensity_5s",
    "rv_10s", "rv_60s", "rv_300s",
)


def make_classifiers(task: str, random_state: int = 42) -> dict[str, object]:
    """Return the required Logistic/RF baselines and optional XGBoost model."""
    if task not in {"direction", "stress"}:
        raise ValueError("task must be 'direction' or 'stress'")
    models: dict[str, object] = {
        "logistic": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=random_state),
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=random_state,
            ),
        ),
    }
    try:
        from xgboost import XGBClassifier

        estimator = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.04,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softprob" if task == "direction" else "binary:logistic",
                eval_metric="mlogloss" if task == "direction" else "logloss",
                num_class=3 if task == "direction" else None,
                random_state=random_state,
                n_jobs=-1,
            ),
        )
        models["xgboost"] = EncodedClassifier(estimator, ("down", "flat", "up") if task == "direction" else (0, 1))
    except ImportError:
        pass
    return models

