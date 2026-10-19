"""Leakage-safe periodic hyperparameter selection using purged expanding folds."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from app.evaluation.purged_cv import purged_time_series_splits
from app.evaluation.volatility_metrics import qlike_loss
from app.models.ml_linear import make_elasticnet
from app.models.ml_trees import make_lightgbm, make_random_forest, make_xgboost


ROOT = Path(__file__).resolve().parents[3]
LOG_TARGET_MODELS = {"elasticnet", "lightgbm_l2", "xgboost"}


def _candidates(model: str, seed: int, cap: int) -> list[dict]:
    search = yaml.safe_load((ROOT / "config" / "search_space.yaml").read_text(encoding="utf-8"))
    group = "elasticnet" if model == "elasticnet" else "random_forest" if model == "random_forest" else "lightgbm" if model.startswith("lightgbm") else "xgboost"
    space = search[group]
    keys = list(space)
    candidates = [dict(zip(keys, values)) for values in product(*(space[key] for key in keys))]
    if len(candidates) > cap:
        indices = np.random.default_rng(seed).choice(len(candidates), size=cap, replace=False)
        candidates = [candidates[index] for index in sorted(indices)]
    return candidates


def _estimator(model: str, params: dict, seed: int):
    if model == "elasticnet":
        return make_elasticnet(random_state=seed, **params)
    if model == "random_forest":
        return make_random_forest(random_state=seed, **params)
    if model == "lightgbm_l2":
        return make_lightgbm(random_state=seed, **params)
    if model == "lightgbm_qlike":
        return make_lightgbm(qlike=True, random_state=seed, **params)
    if model == "xgboost":
        return make_xgboost(random_state=seed, **params)
    raise ValueError(f"No tuning search space for {model}")


def tune_hyperparameters(
    model: str,
    x: pd.DataFrame,
    y: pd.Series,
    target_end: pd.Series,
    folds: int = 5,
    embargo_days: int = 29,
    max_candidates: int = 6,
    seed: int = 20260930,
) -> tuple[dict, dict]:
    """Choose QLIKE-minimizing parameters using only labels available at this origin."""
    origins = pd.DatetimeIndex(pd.to_datetime(x.index, utc=True))
    ends = pd.Series(pd.to_datetime(target_end, utc=True).to_numpy(), index=x.index)
    validation_start = origins.max() - pd.Timedelta(days=365)
    validation_mask = origins >= validation_start
    splits = list(purged_time_series_splits(
        origins, ends, validation_mask=validation_mask,
        folds=folds, embargo_days=embargo_days,
    ))
    if len(splits) < 2:
        raise ValueError("Purged tuning needs at least two non-empty validation folds")
    candidates = _candidates(model, seed, max_candidates)
    scored = []
    for candidate_index, params in enumerate(candidates):
        fold_scores = []
        for train_positions, valid_positions in splits:
            x_train, x_valid = x.iloc[train_positions], x.iloc[valid_positions]
            y_train, y_valid = y.iloc[train_positions], y.iloc[valid_positions]
            usable_train = y_train.notna() & x_train.notna().all(axis=1)
            usable_valid = y_valid.notna() & x_valid.notna().all(axis=1)
            if usable_train.sum() < 100 or usable_valid.sum() == 0:
                continue
            estimator = _estimator(model, params, seed + candidate_index)
            target_train = np.log(np.maximum(y_train.loc[usable_train], np.finfo(float).tiny)) if model in LOG_TARGET_MODELS else y_train.loc[usable_train]
            estimator.fit(x_train.loc[usable_train], target_train)
            prediction = np.asarray(estimator.predict(x_valid.loc[usable_valid]), dtype=float)
            if model in LOG_TARGET_MODELS:
                train_residual = target_train.to_numpy() - np.asarray(estimator.predict(x_train.loc[usable_train]), dtype=float)
                residual_variance = float(np.var(train_residual, ddof=1)) if len(train_residual) > 1 else 0.0
                prediction = np.exp(np.clip(prediction + 0.5 * residual_variance, -30, 30))
            elif model == "lightgbm_qlike":
                prediction = np.exp(np.clip(prediction, -30, 30))
            prediction = np.maximum(prediction, np.finfo(float).tiny)
            actual = y_valid.loc[usable_valid].to_numpy(dtype=float)
            floor = float(y_train.loc[usable_train].quantile(0.01))
            fold_scores.append(float(np.mean(qlike_loss(actual, prediction, floor))))
        if fold_scores:
            scored.append((float(np.mean(fold_scores)), params, len(fold_scores)))
    if not scored:
        raise ValueError(f"No usable purged CV score for {model}")
    score, best, n_folds = min(scored, key=lambda result: result[0])
    return best, {"qlike_cv": score, "folds_used": n_folds, "candidates_scored": len(scored), "validation_start": validation_start.isoformat()}
