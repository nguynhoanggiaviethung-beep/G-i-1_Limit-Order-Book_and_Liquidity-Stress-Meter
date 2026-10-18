"""Run the frozen research workflow from snapshot through forecasts and reports."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable


def run(script: str, *arguments: str) -> None:
    command = [PYTHON, str(ROOT / "backend" / "scripts" / script), *arguments]
    print("\n>", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sync", action="store_true", help="Incrementally update Binance candles before freezing")
    parser.add_argument("--force-snapshot", action="store_true", help="Rebuild the configured frozen snapshot")
    parser.add_argument("--symbol", action="append", dest="symbols")
    parser.add_argument("--horizon", action="append", type=int, dest="horizons")
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--overwrite-forecasts", action="store_true", help="Discard and rebuild walk-forward output files")
    parser.add_argument("--skip-mcs", action="store_true")
    parser.add_argument("--skip-risk", action="store_true")
    parser.add_argument("--skip-quantile", action="store_true", help="Skip direct conditional-quantile VaR/ES models")
    parser.add_argument("--skip-robustness", action="store_true", help="Skip the separate HAR robustness runs")
    args = parser.parse_args()

    symbols = [item for symbol in (args.symbols or []) for item in ("--symbol", symbol)]
    horizons = [item for horizon in (args.horizons or []) for item in ("--horizon", str(horizon))]
    models = [item for model in (args.models or []) for item in ("--model", model)]

    if args.sync:
        run("sync_data.py")
    freeze_args = ["--force"] if args.force_snapshot else []
    run("freeze_snapshot.py", *freeze_args)
    run("validate_data.py")
    run("build_features.py", *symbols)
    walkforward_args = ["--overwrite"] if args.overwrite_forecasts else []
    run("run_walkforward.py", *symbols, *horizons, *models, *walkforward_args)
    if not args.skip_quantile:
        quantile_args = ["--overwrite"] if args.overwrite_forecasts else []
        run("run_quantile_walkforward.py", *symbols, *horizons, *quantile_args)
    table_args = ["--skip-mcs"] if args.skip_mcs else []
    run("make_report_tables.py", *symbols, *horizons, *table_args)
    if not args.skip_risk:
        run("run_risk_eval.py", *symbols, *horizons, *models)
    run("run_regime_economic_eval.py", *symbols, *horizons)
    if not args.skip_robustness:
        robustness_args = ["--overwrite"] if args.overwrite_forecasts else []
        run("run_robustness.py", *symbols, *horizons, *robustness_args)
    run("make_paper_tables.py")
    run("make_figures.py")


if __name__ == "__main__":
    main()
