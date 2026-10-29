"""Regime breakdown, hour-block bootstrap and paired tests on the once-scored untouched test (step 8).

Reads the per-second test predictions that ``run_lob_classification.py`` saved when it scored the
test (``test_predictions_h{h}_{task}.parquet``) and applies the step-7 analysis to them. Nothing is
refitted and no feature partition is opened, so this does not score the test a second time.

* Volatility terciles use the ``rv_300s`` cut-offs fitted on development training rows, stored as
  ``regime_cutoffs_rv_300s_train_terciles`` in ``classification_metrics.json``. Sessions are UTC
  Asia 00-08, Europe 08-16 and Americas 16-24.
* The bootstrap resamples 1-hour blocks with replacement within each test week.
* Stress rows are limited to those where the persistence baseline is defined, as in step 7.
* The selected direction model is scored by argmax of its calibrated probabilities. The selected
  stress model uses the frozen MCC-maximizing threshold; every other model uses 0.5, as in the
  pre-specified secondary comparisons.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import REPOSITORY_ROOT
from scripts.run_lob_classification import DIRECTION_CLASSES, FINAL_TEST_OPENED_MARKER
from scripts.run_lob_regime_bootstrap import CONTRASTS, PR_BINS, REGIMES, Accumulator, regime_codes, summarize

SELECTED = "selected_calibrated"
TEST_CONTRASTS = (
    ("direction", SELECTED, "direction_persistence", "Headline: selected model vs persistence"),
    ("stress", SELECTED, "stress_persistence", "Headline: selected model vs persistence"),
    *CONTRASTS,
)


def _models(columns: pd.Index, suffix: str) -> list[str]:
    return [column[: -len(suffix)] for column in columns if column.endswith(suffix)]


def horizon_state(symbol_dir: Path, horizon: int, result: dict, test_start: pd.Timestamp, test_end: pd.Timestamp) -> dict:
    n_weeks = int(np.ceil((test_end - test_start) / pd.Timedelta(days=7)))
    n_blocks = n_weeks * 7 * 24
    block_weeks = np.repeat(np.arange(n_weeks), 7 * 24)
    cutoffs = result.get("regime_cutoffs_rv_300s_train_terciles")
    cutoffs = tuple(cutoffs) if cutoffs else None
    accumulators: dict[tuple[int, str, str], Accumulator] = {}

    def blocks_and_regimes(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        times = pd.to_datetime(frame["timestamp_utc"], utc=True)
        blocks = ((times - test_start) // pd.Timedelta(hours=1)).to_numpy()
        if blocks.min() < 0 or blocks.max() >= n_blocks:
            raise ValueError("saved predictions fall outside the test window")
        return blocks, regime_codes(times, frame["rv_300s"].to_numpy(dtype=float), cutoffs)

    files = result["test_prediction_files"]
    direction = pd.read_parquet(symbol_dir / files["direction"])
    blocks, regimes = blocks_and_regimes(direction)
    truth = pd.Index(DIRECTION_CLASSES).get_indexer(direction["y_true"].to_numpy())
    persistence = direction["direction_persistence"].to_numpy()
    probabilities = {"direction_persistence": np.column_stack([persistence == label for label in DIRECTION_CLASSES]).astype(float)}
    for name in _models(direction.columns, f"_p_{DIRECTION_CLASSES[0]}"):
        probabilities[name] = direction[[f"{name}_p_{label}" for label in DIRECTION_CLASSES]].to_numpy(dtype=float)
    for name, probability in probabilities.items():
        accumulators[(horizon, "direction", name)] = accumulator = Accumulator("direction", n_blocks)
        accumulator.add(blocks, regimes, truth, probability)
    del direction, probabilities

    stress = pd.read_parquet(symbol_dir / files["stress"])
    stress = stress.loc[stress["stress_persistence"] >= 0].reset_index(drop=True)
    blocks, regimes = blocks_and_regimes(stress)
    truth = stress["y_true"].to_numpy().astype(int)
    state = stress["stress_persistence"].to_numpy().astype(int)
    probabilities = {"stress_persistence": np.column_stack([1 - state, state]).astype(float)}
    for name in _models(stress.columns, "_p_stress"):
        p = stress[f"{name}_p_stress"].to_numpy(dtype=float)
        probabilities[name] = np.column_stack([1 - p, p])
    threshold = float(result["stress_decision_threshold"])
    for name, probability in probabilities.items():
        predicted = (probability[:, 1] >= threshold).astype(int) if name == SELECTED else None
        accumulators[(horizon, "stress", name)] = accumulator = Accumulator("stress", n_blocks)
        accumulator.add(blocks, regimes, truth, probability, predicted=predicted)
    return {"accumulators": accumulators, "block_weeks": block_weeks, "regime_cutoffs": cutoffs}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics-dir", type=Path, default=REPOSITORY_ROOT / "outputs" / "tables" / "final_test_6m")
    parser.add_argument("--symbol", action="append", required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--out-dir", type=Path, default=None, help="Defaults to <metrics-dir>/regime_bootstrap")
    args = parser.parse_args()
    metrics_dir = args.metrics_dir if args.metrics_dir.is_absolute() else REPOSITORY_ROOT / args.metrics_dir
    out_dir = args.out_dir or metrics_dir / "regime_bootstrap"
    out_dir = out_dir if out_dir.is_absolute() else REPOSITORY_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for symbol in args.symbol:
        started = time.perf_counter()
        symbol_dir = metrics_dir / symbol
        if not (symbol_dir / FINAL_TEST_OPENED_MARKER).exists():
            raise SystemExit(f"{symbol}: no {FINAL_TEST_OPENED_MARKER} in {symbol_dir}; these are not final-test predictions")
        metrics = json.loads((symbol_dir / "classification_metrics.json").read_text(encoding="utf-8"))
        test_start = pd.Timestamp(metrics["reproducibility"]["final_test_start_utc"])
        test_end = pd.Timestamp(metrics["reproducibility"]["final_test_end_utc"])
        model_parts, contrast_parts, pr_checks, cutoffs = [], [], [], {}
        for horizon, result in sorted(metrics["horizons"].items(), key=lambda item: int(item[0])):
            state = horizon_state(symbol_dir, int(horizon), result, test_start, test_end)
            models, contrasts, checks = summarize(symbol, state, args.replicates, args.seed, contrasts=TEST_CONTRASTS)
            model_parts.append(models)
            contrast_parts.append(contrasts)
            pr_checks.extend(checks)
            cutoffs[horizon] = state["regime_cutoffs"]
            print(f"{symbol} h={horizon}: summarized in {time.perf_counter() - started:.1f}s", flush=True)
        pd.concat(model_parts, ignore_index=True).to_csv(out_dir / f"{symbol}_model_metrics.csv", index=False)
        pd.concat(contrast_parts, ignore_index=True).to_csv(out_dir / f"{symbol}_contrasts.csv", index=False)
        (out_dir / f"{symbol}_status.json").write_text(json.dumps({
            "symbol": symbol, "split": "untouched_test_saved_predictions", "refitted_or_rescored": False,
            "test_window_utc": [str(test_start), str(test_end)],
            "bootstrap": {"scheme": "hour blocks resampled with replacement within each test week", "replicates": args.replicates, "seed": args.seed},
            "regimes": list(REGIMES), "regime_cutoffs_rv_300s_development_training": cutoffs,
            "selected_stress_threshold": {h: r["stress_decision_threshold"] for h, r in metrics["horizons"].items()},
            "pr_auc_bins": PR_BINS, "pr_auc_binned_vs_exact": pr_checks,
            "seconds": round(time.perf_counter() - started, 1),
        }, indent=2), encoding="utf-8")
        print(f"{symbol}: wrote test regime/bootstrap tables to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
