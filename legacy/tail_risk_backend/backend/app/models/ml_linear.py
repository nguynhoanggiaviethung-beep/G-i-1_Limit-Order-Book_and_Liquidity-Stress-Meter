"""Regularized linear volatility models."""

from __future__ import annotations

from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_elasticnet(alpha: float = 0.01, l1_ratio: float = 0.5, random_state: int = 20260930):
    return make_pipeline(
        StandardScaler(),
        ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20000, random_state=random_state),
    )
