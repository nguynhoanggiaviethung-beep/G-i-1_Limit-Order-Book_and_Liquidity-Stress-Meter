"""Regime breakdown, hour-block bootstrap and paired tests on the development walk-forward (step 7).

Refits the models behind the RQ1/RQ2 and stress contrasts on the same 12 expanding weekly folds,
with the same training samples as ``run_lob_classification.py --development-only``. Instead of
saving 7 M pointwise predictions per horizon, it keeps sufficient statistics per (UTC hour block,
regime):

* direction: the 3x3 confusion matrix and the Brier sum;
* stress: the 2x2 confusion matrix at 0.5, the Brier sum, and a histogram of the stress
  probability by true class (``PR_BINS`` bins), from which PR-AUC is computed.

Regimes use only information available at the origin. Volatility terciles come from ``rv_300s``,
with cut-offs from each fold's own training rows. Sessions are UTC Asia 00-08, Europe 08-16 and
Americas 16-24.

The bootstrap resamples hour blocks with replacement within each validation week, so every
replicate keeps the 12-week composition. Contrasts are paired: both models are scored on the same
resampled blocks. Stress contrasts are scored on the rows where the persistence baseline is defined.
p-values are two-sided bootstrap p-values, with Holm correction across the contrasts of one
symbol x horizon x regime x task x metric.

Development data only: no partition on or after the untouched-test start is opened.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import REPOSITORY_ROOT, load_settings
from app.evaluation.lob_classification import expanding_week_folds
from app.labels.direction import fit_direction_delta, make_direction_labels
from app.labels.stress import fit_primary_stress_thresholds, make_stress_labels
from app.models.lob_classifiers import LOB_CLASSIFICATION_FEATURES, make_classifiers
from scripts.run_lob_classification import (
    DIRECTION_CLASSES,
    STRESS_CLASSES,
    _bounded_sample,
    _current_stress_state,
    _direction_persistence_predictions,
    _load_features,
    _predict_with_probabilities,
    _sampled_training,
    _stress_threshold_rows,
    _threshold_frame,
)

PR_BINS = 1000
REGIMES = ("all", "vol_low", "vol_mid", "vol_high", "session_asia", "session_europe", "session_americas")
OBI_ONLY = ("obi_qty_1", "obi_qty_5", "obi_qty_10", "obi_qty_20")
OBI_PLUS_OFI = (*OBI_ONLY, "ofi_1s", "ofi_5s", "ofi_10s")
DIRECTION_MODELS = ("direction_persistence", "logistic_obi_only", "logistic_obi_plus_ofi", "logistic", "xgboost")
STRESS_MODELS = ("stress_persistence", "logistic", "xgboost")
# (task, challenger, reference, label)
CONTRASTS = (
    ("direction", "logistic_obi_only", "direction_persistence", "RQ1: OBI vs persistence"),
    ("direction", "logistic_obi_plus_ofi", "logistic_obi_only", "RQ2: adding OFI to OBI"),
    ("direction", "logistic", "logistic_obi_plus_ofi", "Full feature set vs OBI+OFI"),
    ("direction", "xgboost", "logistic", "XGBoost vs Logistic"),
    ("stress", "logistic", "stress_persistence", "Stress: Logistic vs persistence"),
    ("stress", "xgboost", "stress_persistence", "Stress: XGBoost vs persistence"),
    ("stress", "xgboost", "logistic", "Stress: XGBoost vs Logistic"),
)
METRICS = {"direction": ("mcc", "macro_f1", "brier"), "stress": ("pr_auc", "mcc", "brier")}
LOWER_IS_BETTER = {"brier"}


# ---------------------------------------------------------------------------
# Metrics from sufficient statistics. Every function takes a leading batch axis.

def mcc_from_confusion(confusion: np.ndarray) -> np.ndarray:
    """Multiclass MCC (Gorodkin) for ``confusion[..., true, predicted]``; 0 where undefined, as sklearn."""
    confusion = confusion.astype(float)
    total = confusion.sum(axis=(-1, -2))
    correct = np.trace(confusion, axis1=-2, axis2=-1)
    true_sum, pred_sum = confusion.sum(axis=-1), confusion.sum(axis=-2)
    numerator = correct * total - (true_sum * pred_sum).sum(axis=-1)
    denominator = np.sqrt((total**2 - (pred_sum**2).sum(axis=-1)) * (total**2 - (true_sum**2).sum(axis=-1)))
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def macro_f1_from_confusion(confusion: np.ndarray) -> np.ndarray:
    confusion = confusion.astype(float)
    tp = np.diagonal(confusion, axis1=-2, axis2=-1)
    denominator = confusion.sum(axis=-1) + confusion.sum(axis=-2)
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    return f1.mean(axis=-1)


def average_precision_from_histogram(histogram: np.ndarray) -> np.ndarray:
    """PR-AUC as sklearn's step-wise average precision, treating scores in one bin as tied.

    ``histogram[..., bin, class]`` counts rows by probability bin (ascending) and true class (0/1).
    """
    histogram = histogram.astype(float)[..., ::-1, :]  # descending score
    tp = np.cumsum(histogram[..., 1], axis=-1)
    fp = np.cumsum(histogram[..., 0], axis=-1)
    positives = tp[..., -1:]
    predicted = tp + fp
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall_step = np.divide(histogram[..., 1], positives, out=np.zeros_like(tp), where=positives > 0)
    return (recall_step * precision).sum(axis=-1)


def probability_bins(probability: np.ndarray) -> np.ndarray:
    return np.clip((probability * PR_BINS).astype(int), 0, PR_BINS - 1)


# ---------------------------------------------------------------------------
# Accumulation

def regime_codes(times: pd.Series, rv_300s: np.ndarray, cutoffs: tuple[float, float] | None) -> np.ndarray:
    """Boolean matrix rows x REGIMES."""
    hours = times.dt.hour.to_numpy()
    codes = np.zeros((len(times), len(REGIMES)), dtype=bool)
    codes[:, 0] = True
    if cutoffs is not None:
        finite = np.isfinite(rv_300s)
        codes[:, 1] = finite & (rv_300s <= cutoffs[0])
        codes[:, 2] = finite & (rv_300s > cutoffs[0]) & (rv_300s <= cutoffs[1])
        codes[:, 3] = finite & (rv_300s > cutoffs[1])
    codes[:, 4] = hours < 8
    codes[:, 5] = (hours >= 8) & (hours < 16)
    codes[:, 6] = hours >= 16
    return codes


class Accumulator:
    """Per (block, regime) sufficient statistics for one task and model."""

    def __init__(self, task: str, n_blocks: int):
        k = len(DIRECTION_CLASSES) if task == "direction" else 2
        self.task = task
        self.confusion = np.zeros((n_blocks, len(REGIMES), k, k), dtype=np.int64)
        self.brier = np.zeros((n_blocks, len(REGIMES)), dtype=float)
        self.histogram = np.zeros((n_blocks, len(REGIMES), PR_BINS, 2), dtype=np.int32) if task == "stress" else None
        self.scores: list[np.ndarray] = []  # pooled stress scores, for an exact PR-AUC check
        self.truth: list[np.ndarray] = []

    def add(self, blocks: np.ndarray, regimes: np.ndarray, truth_codes: np.ndarray, probabilities: np.ndarray,
            predicted: np.ndarray | None = None) -> None:
        """``predicted`` overrides the argmax class codes, e.g. for a tuned stress threshold."""
        if predicted is None:
            predicted = np.argmax(probabilities, axis=1)
        target = np.zeros_like(probabilities)
        target[np.arange(len(truth_codes)), truth_codes] = 1.0
        if self.task == "direction":
            brier = np.sum((probabilities - target) ** 2, axis=1)
        else:
            brier = (probabilities[:, 1] - truth_codes) ** 2
            bins = probability_bins(probabilities[:, 1])
        for regime in range(len(REGIMES)):
            rows = regimes[:, regime]
            if not rows.any():
                continue
            b = blocks[rows]
            np.add.at(self.confusion[:, regime], (b, truth_codes[rows], predicted[rows]), 1)
            self.brier[:, regime] += np.bincount(b, weights=brier[rows], minlength=len(self.brier))
            if self.histogram is not None:
                np.add.at(self.histogram[:, regime], (b, bins[rows], truth_codes[rows]), 1)
        if self.task == "stress":
            self.scores.append(probabilities[:, 1].astype(np.float32))
            self.truth.append(truth_codes.astype(np.int8))

    def metrics(self, weights: np.ndarray, regime: int) -> dict[str, np.ndarray]:
        """Metric arrays for bootstrap weights ``(replicates, blocks)``."""
        confusion = np.tensordot(weights, self.confusion[:, regime].reshape(len(self.confusion), -1), axes=1)
        k = self.confusion.shape[-1]
        confusion = confusion.reshape(len(weights), k, k)
        n = confusion.sum(axis=(1, 2))
        brier_sum = weights @ self.brier[:, regime]
        result = {
            "mcc": mcc_from_confusion(confusion),
            "brier": np.divide(brier_sum, n, out=np.full(len(n), np.nan), where=n > 0),
            "n": n,
        }
        if self.task == "direction":
            result["macro_f1"] = macro_f1_from_confusion(confusion)
        else:
            histogram = np.tensordot(weights, self.histogram[:, regime].reshape(len(self.histogram), -1).astype(np.float64), axes=1)
            result["pr_auc"] = average_precision_from_histogram(histogram.reshape(len(weights), PR_BINS, 2))
        return result


def bootstrap_weights(block_weeks: np.ndarray, replicates: int, seed: int) -> np.ndarray:
    """Counts of each hour block in each replicate; blocks are drawn with replacement within their week."""
    rng = np.random.default_rng(seed)
    weights = np.zeros((replicates, len(block_weeks)), dtype=np.float64)
    for week in np.unique(block_weeks):
        members = np.flatnonzero(block_weeks == week)
        draws = rng.integers(0, len(members), size=(replicates, len(members)))
        for replicate in range(replicates):
            weights[replicate] += np.bincount(members[draws[replicate]], minlength=len(block_weeks))
    return weights


def bootstrap_p_value(differences: np.ndarray) -> float:
    """Two-sided percentile p-value for H0: difference = 0, floored at 1 / (replicates + 1)."""
    differences = differences[np.isfinite(differences)]
    if not len(differences):
        return float("nan")
    tail = min(np.mean(differences <= 0), np.mean(differences >= 0))
    return float(min(1.0, max(2 * tail, 1.0 / (len(differences) + 1))))


def holm(p_values: pd.Series) -> pd.Series:
    order = p_values.sort_values().index
    m = len(order)
    adjusted, running = pd.Series(np.nan, index=p_values.index), 0.0
    for rank, index in enumerate(order):
        if np.isnan(p_values[index]):
            continue
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


# ---------------------------------------------------------------------------
# Walk-forward refit

def _fit_predict(task: str, name: str, x_train: pd.DataFrame, y_train: pd.Series, x_valid: pd.DataFrame, max_train_rows: int) -> np.ndarray:
    class_order = DIRECTION_CLASSES if task == "direction" else STRESS_CLASSES
    estimator = make_classifiers(task)[name]
    x_fit, y_fit = _bounded_sample(x_train, y_train, max_train_rows)
    estimator.fit(x_fit, y_fit.to_numpy())
    return _predict_with_probabilities(estimator, x_valid, class_order)[1]


def run_symbol(symbol: str, settings, horizons: tuple[int, ...], embargo_seconds: int, max_train_rows: int):
    """Yield one state per horizon so its histograms can be summarized and freed before the next."""
    test_start = pd.Timestamp(settings.lob_untouched_test_start_utc)
    frame = _load_features(symbol, settings, end_exclusive=test_start)
    frame = frame.loc[pd.to_datetime(frame["timestamp_utc"], utc=True) < test_start].reset_index(drop=True)
    times = pd.to_datetime(frame["timestamp_utc"], utc=True)
    folds = expanding_week_folds(settings.lob_first_validation_week_utc, settings.lob_development_end_utc)
    first_cutoff = pd.Timestamp(f"{folds[0]}T00:00:00Z") - pd.Timedelta(seconds=embargo_seconds)
    first_train = frame.loc[times < first_cutoff, ["timestamp_utc", "book_valid", "mid_price"]].copy()
    rv_300s = pd.to_numeric(frame["rv_300s"], errors="coerce").to_numpy(dtype=float)
    # Hour blocks across the validation weeks.
    first_start = pd.Timestamp(f"{folds[0]}T00:00:00Z")
    hour_index = ((times - first_start) // pd.Timedelta(hours=1)).to_numpy()
    n_blocks = len(folds) * 7 * 24
    block_weeks = np.repeat(np.arange(len(folds)), 7 * 24)

    regime_cutoffs: dict[str, list[float] | None] = {}
    threshold_cache: dict[str, object] = {}
    for horizon in horizons:
        accumulators: dict[tuple[int, str, str], Accumulator] = {}
        delta = fit_direction_delta(first_train, horizon, quantile=settings.lob_direction_delta_quantile)
        direction_labels = make_direction_labels(frame, horizon, delta)
        for name in DIRECTION_MODELS:
            accumulators[(horizon, "direction", name)] = Accumulator("direction", n_blocks)
        for name in STRESS_MODELS:
            accumulators[(horizon, "stress", name)] = Accumulator("stress", n_blocks)
        for week in folds:
            started = time.perf_counter()
            start = pd.Timestamp(f"{week}T00:00:00Z")
            end = start + pd.Timedelta(days=7)
            train_end = start - pd.Timedelta(seconds=embargo_seconds)
            if week not in regime_cutoffs:
                train_rv = rv_300s[(times < train_end).to_numpy()]
                train_rv = train_rv[np.isfinite(train_rv)]
                low, high = (np.quantile(train_rv, [1 / 3, 2 / 3]) if len(train_rv) else (np.nan, np.nan))
                regime_cutoffs[week] = [float(low), float(high)] if np.isfinite(low) and low < high else None

            # Direction
            train_mask = (times < train_end) & direction_labels.notna()
            valid_mask = (times >= start) & (times < end - pd.Timedelta(seconds=horizon)) & direction_labels.notna()
            x_train, y_train = _sampled_training(frame, direction_labels, train_mask, LOB_CLASSIFICATION_FEATURES, max_train_rows)
            valid_rows = np.flatnonzero(valid_mask.to_numpy())
            x_valid = frame.loc[valid_mask, LOB_CLASSIFICATION_FEATURES]
            truth = pd.Index(DIRECTION_CLASSES).get_indexer(direction_labels.loc[valid_mask].to_numpy())
            blocks = hour_index[valid_rows]
            cutoffs = regime_cutoffs[week]
            regimes = regime_codes(times.iloc[valid_rows], rv_300s[valid_rows], tuple(cutoffs) if cutoffs else None)
            persistence = _direction_persistence_predictions(frame.loc[valid_mask], delta)
            probabilities = {
                "direction_persistence": np.column_stack([persistence == label for label in DIRECTION_CLASSES]).astype(float),
                "logistic_obi_only": _fit_predict("direction", "logistic", x_train.loc[:, OBI_ONLY], y_train, x_valid.loc[:, OBI_ONLY], max_train_rows),
                "logistic_obi_plus_ofi": _fit_predict("direction", "logistic", x_train.loc[:, OBI_PLUS_OFI], y_train, x_valid.loc[:, OBI_PLUS_OFI], max_train_rows),
                "logistic": _fit_predict("direction", "logistic", x_train, y_train, x_valid, max_train_rows),
                "xgboost": _fit_predict("direction", "xgboost", x_train, y_train, x_valid, max_train_rows),
            }
            for name, probability in probabilities.items():
                accumulators[(horizon, "direction", name)].add(blocks, regimes, truth, probability)
            del x_train, x_valid, probabilities

            # Stress, on rows where the persistence baseline is defined.
            if week not in threshold_cache:
                threshold_cache[week] = fit_primary_stress_thresholds(
                    _stress_threshold_rows(_threshold_frame(frame), times < train_end, settings.lob_stress_definition),
                    settings.lob_stress_definition, settings.lob_stress_quantile, settings.lob_stress_depth_quantile,
                )
            thresholds = threshold_cache[week]
            stress_labels = make_stress_labels(frame, horizon, thresholds)
            stress_train = (times < train_end) & stress_labels.notna()
            stress_valid = (times >= start) & (times < end - pd.Timedelta(seconds=horizon)) & stress_labels.notna()
            state = _current_stress_state(frame.loc[stress_valid], thresholds)
            stress_valid.loc[state.index[state.isna().to_numpy()]] = False
            state = state.dropna().astype(int).to_numpy()
            xs_train, ys_train = _sampled_training(frame, stress_labels, stress_train, LOB_CLASSIFICATION_FEATURES, max_train_rows)
            ys_train = ys_train.astype(int)
            stress_rows = np.flatnonzero(stress_valid.to_numpy())
            xs_valid = frame.loc[stress_valid, LOB_CLASSIFICATION_FEATURES]
            stress_truth = stress_labels.loc[stress_valid].astype(int).to_numpy()
            stress_blocks = hour_index[stress_rows]
            stress_regimes = regime_codes(times.iloc[stress_rows], rv_300s[stress_rows], tuple(cutoffs) if cutoffs else None)
            stress_probabilities = {
                "stress_persistence": np.column_stack([1 - state, state]).astype(float),
                "logistic": _fit_predict("stress", "logistic", xs_train, ys_train, xs_valid, max_train_rows),
                "xgboost": _fit_predict("stress", "xgboost", xs_train, ys_train, xs_valid, max_train_rows),
            }
            for name, probability in stress_probabilities.items():
                accumulators[(horizon, "stress", name)].add(stress_blocks, stress_regimes, stress_truth, probability)
            del xs_train, xs_valid, stress_probabilities
            print(f"{symbol} h={horizon} week {week}: direction {len(valid_rows):,} rows, stress {len(stress_rows):,} rows, {time.perf_counter() - started:.1f}s", flush=True)
        yield {"accumulators": accumulators, "block_weeks": block_weeks, "regime_cutoffs": regime_cutoffs, "folds": folds}


def summarize(symbol: str, state: dict, replicates: int, seed: int, contrasts=CONTRASTS) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    accumulators = state["accumulators"]
    weights = bootstrap_weights(state["block_weeks"], replicates, seed)
    point = np.ones((1, len(state["block_weeks"])))
    model_rows, contrast_rows, pr_checks = [], [], []
    horizons = sorted({key[0] for key in accumulators})
    for horizon in horizons:
        for regime_index, regime in enumerate(REGIMES):
            computed: dict[tuple[str, str], tuple[dict, dict]] = {}
            for (h, task, name), accumulator in accumulators.items():
                if h != horizon:
                    continue
                estimate, replicate = accumulator.metrics(point, regime_index), accumulator.metrics(weights, regime_index)
                computed[(task, name)] = (estimate, replicate)
                for metric in METRICS[task]:
                    values = replicate[metric]
                    model_rows.append({
                        "symbol": symbol, "horizon_seconds": horizon, "task": task, "model": name, "regime": regime,
                        "metric": metric, "n": int(estimate["n"][0]), "estimate": float(estimate[metric][0]),
                        "ci_low": float(np.nanquantile(values, 0.025)), "ci_high": float(np.nanquantile(values, 0.975)),
                    })
                if task == "stress" and regime == "all" and accumulator.scores:
                    scores, truth = np.concatenate(accumulator.scores), np.concatenate(accumulator.truth)
                    pr_checks.append({"symbol": symbol, "horizon_seconds": horizon, "model": name,
                                      "pr_auc_binned": float(estimate["pr_auc"][0]), "pr_auc_exact": float(average_precision_score(truth, scores))})
            for task, challenger, reference, question in contrasts:
                if (task, challenger) not in computed or (task, reference) not in computed:
                    continue
                (estimate_c, replicate_c), (estimate_r, replicate_r) = computed[(task, challenger)], computed[(task, reference)]
                for metric in METRICS[task]:
                    difference = replicate_c[metric] - replicate_r[metric]
                    sign = -1.0 if metric in LOWER_IS_BETTER else 1.0
                    contrast_rows.append({
                        "symbol": symbol, "horizon_seconds": horizon, "regime": regime, "question": question, "task": task,
                        "challenger": challenger, "reference": reference, "metric": metric, "n": int(estimate_c["n"][0]),
                        "challenger_estimate": float(estimate_c[metric][0]), "reference_estimate": float(estimate_r[metric][0]),
                        "difference": float(estimate_c[metric][0] - estimate_r[metric][0]),
                        "ci_low": float(np.nanquantile(difference, 0.025)), "ci_high": float(np.nanquantile(difference, 0.975)),
                        "share_replicates_challenger_better": float(np.nanmean(sign * difference > 0)),
                        "bootstrap_p": bootstrap_p_value(difference),
                    })
    contrasts = pd.DataFrame(contrast_rows)
    if not contrasts.empty:
        contrasts["bootstrap_p_holm"] = np.nan
        for _, group in contrasts.groupby(["horizon_seconds", "regime", "task", "metric"]):
            contrasts.loc[group.index, "bootstrap_p_holm"] = holm(group["bootstrap_p"])
    return pd.DataFrame(model_rows), contrasts, pr_checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--settings", type=Path, default=REPOSITORY_ROOT / "config" / "settings_6m.yaml")
    parser.add_argument("--symbol", action="append", help="Defaults to every configured symbol")
    parser.add_argument("--horizons", type=int, nargs="+", default=None)
    parser.add_argument("--max-train-rows", type=int, default=250_000)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--out-dir", type=Path, default=REPOSITORY_ROOT / "outputs" / "tables" / "regime_bootstrap_6m")
    args = parser.parse_args()
    settings_path = args.settings if args.settings.is_absolute() else REPOSITORY_ROOT / args.settings
    settings = load_settings(settings_path)
    out_dir = args.out_dir if args.out_dir.is_absolute() else REPOSITORY_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    horizons = tuple(args.horizons or settings.lob_classification_horizons_seconds)
    embargo = int(settings.lob_embargo_seconds)
    for symbol in args.symbol or list(settings.symbols):
        started = time.perf_counter()
        model_parts, contrast_parts, pr_checks = [], [], []
        for state in run_symbol(symbol, settings, horizons, embargo, args.max_train_rows):
            models, contrasts, checks = summarize(symbol, state, args.replicates, args.seed)
            model_parts.append(models)
            contrast_parts.append(contrasts)
            pr_checks.extend(checks)
            state["accumulators"].clear()
        models, contrasts = pd.concat(model_parts, ignore_index=True), pd.concat(contrast_parts, ignore_index=True)
        models.to_csv(out_dir / f"{symbol}_model_metrics.csv", index=False)
        contrasts.to_csv(out_dir / f"{symbol}_contrasts.csv", index=False)
        (out_dir / f"{symbol}_status.json").write_text(json.dumps({
            "symbol": symbol, "split": "development_validation_only", "untouched_test_opened": False,
            "configuration_file": str(settings_path.relative_to(REPOSITORY_ROOT)) if settings_path.is_relative_to(REPOSITORY_ROOT) else str(settings_path),
            "validation_weeks_utc": list(state["folds"]), "horizons_seconds": list(horizons), "max_train_rows_per_fit": args.max_train_rows,
            "bootstrap": {"scheme": "hour blocks resampled with replacement within each validation week", "replicates": args.replicates, "seed": args.seed},
            "regimes": list(REGIMES), "regime_cutoffs_rv_300s_by_fold": state["regime_cutoffs"],
            "pr_auc_bins": PR_BINS, "pr_auc_binned_vs_exact": pr_checks,
            "seconds": round(time.perf_counter() - started, 1),
        }, indent=2), encoding="utf-8")
        print(f"{symbol}: wrote regime/bootstrap tables to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
