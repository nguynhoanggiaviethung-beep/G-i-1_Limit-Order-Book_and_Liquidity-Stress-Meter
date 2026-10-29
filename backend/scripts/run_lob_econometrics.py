"""Econometric benchmark for RQ1/RQ2: order-imbalance regressions in the spirit of Chordia, Roll and Subrahmanyam (2002).

For each horizon h the target is the future mid-price return over h seconds, in basis points.
Regressors are standardized static depth imbalance (OBI, levels 1/5/10/20) and event-level order
flow imbalance (OFI, 1/5/10 s), measured at the origin second.

Two outputs, both from development data only (no partition on or after the untouched-test start
is opened):

* In-sample OLS on all development origins, with Newey-West (Bartlett) standard errors.
* Expanding weekly walk-forward on the same folds as the classifiers. Each spec is fitted on the
  rows before the fold's embargo cutoff and scored on the validation week. Scores are the
  out-of-sample R² against the training-mean forecast, and the incremental R² of OBI+OFI over
  OBI alone. A minute-blocked Diebold-Mariano test compares their squared errors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import REPOSITORY_ROOT, load_settings
from app.evaluation.lob_classification import expanding_week_folds
from app.labels.direction import valid_future_origins

OBI_COLUMNS = ("obi_qty_1", "obi_qty_5", "obi_qty_10", "obi_qty_20")
OFI_COLUMNS = ("ofi_1s", "ofi_5s", "ofi_10s")
CONTROL_COLUMNS = ("spread_ticks", "rv_60s")
SPECS = {
    "obi": OBI_COLUMNS,
    "ofi": OFI_COLUMNS,
    "obi_ofi": (*OBI_COLUMNS, *OFI_COLUMNS),
    "obi_ofi_controls": (*OBI_COLUMNS, *OFI_COLUMNS, *CONTROL_COLUMNS),
}
LOAD_COLUMNS = ("timestamp_utc", "book_valid", "mid_price", *OBI_COLUMNS, *OFI_COLUMNS, *CONTROL_COLUMNS)


def newey_west_ols(x: np.ndarray, y: np.ndarray, lags: int) -> dict[str, np.ndarray | float]:
    """OLS with Bartlett-kernel HAC covariance, as statsmodels ``cov_type="HAC"`` with ``use_correction=True`` (n/(n-k)).

    ``x`` must already contain the intercept column. Rows are treated as one ordered series.
    """
    n, k = x.shape
    xtx_inverse = np.linalg.inv(x.T @ x)
    beta = xtx_inverse @ (x.T @ y)
    residual = y - x @ beta
    scores = x * residual[:, None]
    meat = scores.T @ scores
    for lag in range(1, min(lags, n - 1) + 1):
        gamma = scores[lag:].T @ scores[:-lag]
        meat += (1.0 - lag / (lags + 1.0)) * (gamma + gamma.T)
    covariance = xtx_inverse @ meat @ xtx_inverse * (n / (n - k))
    standard_error = np.sqrt(np.diag(covariance))
    total = y - y.mean()
    return {
        "beta": beta,
        "se": standard_error,
        "t": beta / standard_error,
        "p": 2 * norm.sf(np.abs(beta / standard_error)),
        "r2": float(1.0 - residual @ residual / (total @ total)),
        "n": int(n),
    }


def blocked_dm(loss_difference: np.ndarray, minute_keys: np.ndarray, lags: int = 10) -> dict[str, float | int | None]:
    """Diebold-Mariano on minute means of a loss differential, Newey-West over consecutive minutes."""
    minutes, inverse = np.unique(minute_keys, return_inverse=True)
    block = np.bincount(inverse, weights=loss_difference) / np.bincount(inverse)
    n_blocks = len(block)
    if n_blocks < 30:
        return {"n_blocks": n_blocks, "mean_difference": float(np.mean(block)) if n_blocks else None, "statistic": None, "p_value": None}
    centered = block - block.mean()
    variance = float(centered @ centered / n_blocks)
    for lag in range(1, min(lags, n_blocks - 1) + 1):
        variance += 2.0 * (1.0 - lag / (lags + 1.0)) * float(centered[lag:] @ centered[:-lag] / n_blocks)
    if variance <= 0:
        return {"n_blocks": n_blocks, "mean_difference": float(block.mean()), "statistic": None, "p_value": None}
    statistic = float(block.mean() / np.sqrt(variance / n_blocks))
    return {"n_blocks": n_blocks, "mean_difference": float(block.mean()), "statistic": statistic, "p_value": float(2 * norm.sf(abs(statistic)))}


def load_development(symbol: str, settings, test_start: pd.Timestamp) -> pd.DataFrame:
    root = settings.data_dir / "features" / f"symbol={symbol}"
    last_date = test_start.tz_convert("UTC").date().isoformat()
    parts = []
    for path in sorted(root.glob("date=*/features_1s.parquet")):
        if path.parent.name.removeprefix("date=") >= last_date:
            continue
        available = set(pq.ParquetFile(path).schema_arrow.names)
        parts.append(pd.read_parquet(path, columns=[name for name in LOAD_COLUMNS if name in available]))
    if not parts:
        raise FileNotFoundError(f"No development feature partitions for {symbol} under {root}")
    frame = pd.concat(parts, ignore_index=True)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    frame = frame.loc[frame["timestamp_utc"] < test_start]
    return frame.sort_values("timestamp_utc", kind="mergesort").drop_duplicates("timestamp_utc", keep="last").reset_index(drop=True)


def future_return_bps(frame: pd.DataFrame, horizon: int) -> np.ndarray:
    """Return over exactly ``horizon`` contiguous valid seconds; NaN where that path is incomplete."""
    mid = frame["mid_price"].to_numpy(dtype=float)
    output = np.full(len(frame), np.nan)
    valid = valid_future_origins(frame, horizon)[:-horizon]
    origins = np.flatnonzero(valid)
    output[origins] = (mid[origins + horizon] / mid[origins] - 1.0) * 10000.0
    return output


def _design(frame: pd.DataFrame, columns, mean: np.ndarray, scale: np.ndarray, rows: np.ndarray) -> np.ndarray:
    values = frame.loc[rows, list(columns)].to_numpy(dtype=float)
    return np.column_stack([np.ones(len(values)), (values - mean) / scale])


def run_symbol(symbol: str, settings, horizons: tuple[int, ...], embargo_seconds: int) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    test_start = pd.Timestamp(settings.lob_untouched_test_start_utc)
    frame = load_development(symbol, settings, test_start)
    times = frame["timestamp_utc"]
    minute_keys = (times.astype("int64").to_numpy() // 60_000_000_000)
    folds = expanding_week_folds(settings.lob_first_validation_week_utc, settings.lob_development_end_utc)
    all_columns = list(SPECS["obi_ofi_controls"])
    finite = np.isfinite(frame[all_columns].to_numpy(dtype=float)).all(axis=1)
    coefficient_rows, fold_rows, tests = [], [], []
    for horizon in horizons:
        target = future_return_bps(frame, horizon)
        usable = finite & np.isfinite(target)
        dev_rows = usable & (times < test_start - pd.Timedelta(seconds=embargo_seconds)).to_numpy()
        lags = 60 + horizon  # overlapping h-second returns plus a minute of microstructure dependence
        for spec, columns in SPECS.items():
            sample = frame.loc[dev_rows, list(columns)].to_numpy(dtype=float)
            mean, scale = sample.mean(axis=0), sample.std(axis=0)
            del sample
            x = _design(frame, columns, mean, scale, dev_rows)
            fit = newey_west_ols(x, target[dev_rows], lags)
            del x
            for name, beta, se, t, p in zip(("intercept", *columns), fit["beta"], fit["se"], fit["t"], fit["p"]):
                coefficient_rows.append({
                    "symbol": symbol, "horizon_seconds": horizon, "spec": spec, "term": name,
                    "coef_bps_per_sd": float(beta), "nw_se": float(se), "nw_t": float(t), "p_value": float(p),
                    "r2": fit["r2"], "n": fit["n"], "nw_lags_seconds": lags,
                })
            print(f"{symbol} h={horizon} {spec}: in-sample R2={fit['r2']:.5f} n={fit['n']:,}", flush=True)

        pooled = {"obi": [], "obi_ofi": [], "minute": []}
        for week in folds:
            start = pd.Timestamp(f"{week}T00:00:00Z")
            cutoff = start - pd.Timedelta(seconds=embargo_seconds)
            train = usable & (times < cutoff).to_numpy()
            valid = usable & ((times >= start) & (times < start + pd.Timedelta(days=7) - pd.Timedelta(seconds=horizon))).to_numpy()
            y_train, y_valid = target[train], target[valid]
            benchmark_sse = float(np.sum((y_valid - y_train.mean()) ** 2))
            errors, week_rows = {}, {}
            for spec, columns in SPECS.items():
                sample = frame.loc[train, list(columns)].to_numpy(dtype=float)
                mean, scale = sample.mean(axis=0), sample.std(axis=0)
                del sample
                x_train = _design(frame, columns, mean, scale, train)
                beta = np.linalg.solve(x_train.T @ x_train, x_train.T @ y_train)
                del x_train
                errors[spec] = y_valid - _design(frame, columns, mean, scale, valid) @ beta
                week_rows[spec] = {
                    "symbol": symbol, "horizon_seconds": horizon, "validation_week_utc": week, "spec": spec,
                    "train_rows": int(train.sum()), "validation_rows": int(valid.sum()),
                    "oos_r2_vs_train_mean": float(1.0 - np.sum(errors[spec] ** 2) / benchmark_sse),
                }
            week_rows["obi_ofi"]["incremental_oos_r2_over_obi"] = float(1.0 - np.sum(errors["obi_ofi"] ** 2) / np.sum(errors["obi"] ** 2))
            fold_rows.extend(week_rows.values())
            pooled["obi"].append(errors["obi"] ** 2)
            pooled["obi_ofi"].append(errors["obi_ofi"] ** 2)
            pooled["minute"].append(minute_keys[valid])
        difference = np.concatenate(pooled["obi"]) - np.concatenate(pooled["obi_ofi"])
        tests.append({
            "symbol": symbol, "horizon_seconds": horizon,
            "comparison": "squared error, OBI-only minus OBI+OFI (positive favours adding OFI)",
            "folds": list(folds),
            **blocked_dm(difference, np.concatenate(pooled["minute"])),
        })
        print(f"{symbol} h={horizon}: walk-forward DM (OBI vs OBI+OFI) {tests[-1]}", flush=True)
    return pd.DataFrame(coefficient_rows), pd.DataFrame(fold_rows), tests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--settings", type=Path, default=REPOSITORY_ROOT / "config" / "settings_6m.yaml")
    parser.add_argument("--symbol", choices=("BTCUSDT", "ETHUSDT"))
    parser.add_argument("--out-dir", type=Path, default=REPOSITORY_ROOT / "outputs" / "tables" / "econometrics_6m")
    args = parser.parse_args()
    settings_path = args.settings if args.settings.is_absolute() else REPOSITORY_ROOT / args.settings
    settings = load_settings(settings_path)
    out_dir = args.out_dir if args.out_dir.is_absolute() else REPOSITORY_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for symbol in ((args.symbol,) if args.symbol else settings.symbols):
        coefficients, folds, tests = run_symbol(symbol, settings, settings.lob_classification_horizons_seconds, settings.lob_embargo_seconds)
        coefficients.to_csv(out_dir / f"{symbol}_in_sample_nw.csv", index=False)
        folds.to_csv(out_dir / f"{symbol}_walk_forward_oos.csv", index=False)
        (out_dir / f"{symbol}_tests.json").write_text(json.dumps({
            "configuration_file": settings_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "split": "development_only",
            "untouched_test_accessed": False,
            "tests": tests,
        }, indent=2), encoding="utf-8")
        print(f"{symbol}: wrote econometric benchmark to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
