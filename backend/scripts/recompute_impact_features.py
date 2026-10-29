"""Recompute the fixed-notional impact columns of built feature partitions for a new notional.

Only ``impact_buy_bps``, ``impact_sell_bps`` and ``price_impact_bps`` change; they are computed
from the 20 stored book levels exactly as ``build_classification_features`` does. Every partition
is checked first: recomputing with ``--old-notional`` must reproduce the stored columns
bit-for-bit, or the partition is left untouched and the run stops.

The previous columns are saved under ``<data>/impact_backup/notional_<old>/`` before each file is
replaced (written to a temporary file, then ``os.replace``). A partition whose stored columns
already equal the new values is skipped, so an interrupted run can be resumed.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import REPOSITORY_ROOT, load_settings
from app.features.classification_features import _impact

IMPACT_COLUMNS = ("impact_buy_bps", "impact_sell_bps", "price_impact_bps")
LEVELS = range(1, 21)
BOOK_COLUMNS = tuple(f"{side}_{kind}_{level}" for side in ("bid", "ask") for kind in ("price", "qty") for level in LEVELS)


def impact_columns(book: pd.DataFrame, notional: float) -> dict[str, np.ndarray]:
    """Same arithmetic as ``build_classification_features``: missing quantities are 0, prices NaN."""
    def levels(side: str, kind: str) -> np.ndarray:
        values = book[[f"{side}_{kind}_{level}" for level in LEVELS]].apply(pd.to_numeric, errors="coerce")
        return (values.fillna(0) if kind == "qty" else values).to_numpy(dtype=float)

    bp, bq, ap, aq = levels("bid", "price"), levels("bid", "qty"), levels("ask", "price"), levels("ask", "qty")
    mid = (bp[:, 0] + ap[:, 0]) / 2.0
    buy = _impact(ap, aq, mid, notional, buy=True)
    sell = _impact(bp, bq, mid, notional, buy=False)
    both = ~np.isnan(buy) & ~np.isnan(sell)
    composite = np.where(both, pd.DataFrame({"b": buy, "s": sell}).max(axis=1).to_numpy(), np.nan)
    return {"impact_buy_bps": buy, "impact_sell_bps": sell, "price_impact_bps": composite}


def _same(stored: np.ndarray, computed: np.ndarray) -> bool:
    return bool(np.array_equal(stored, computed, equal_nan=True))


def process_partition(path: Path, old_notional: float, new_notional: float, backup_path: Path) -> str:
    # ParquetFile reads the file as stored; read_table would add a hive "date" column from the path.
    parquet_file = pq.ParquetFile(path)
    table = parquet_file.read()
    compression = parquet_file.metadata.row_group(0).column(0).compression.lower()
    book = table.select(list(BOOK_COLUMNS)).to_pandas()
    stored = {name: table.column(name).to_numpy(zero_copy_only=False).astype(float) for name in IMPACT_COLUMNS}
    new = impact_columns(book, new_notional)
    if all(_same(stored[name], new[name]) for name in IMPACT_COLUMNS):
        return "already_new"
    old = impact_columns(book, old_notional)
    mismatched = [name for name in IMPACT_COLUMNS if not _same(stored[name], old[name])]
    if mismatched:
        raise RuntimeError(f"{path}: stored {mismatched} do not reproduce with notional {old_notional:g}; refusing to overwrite")
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"timestamp_utc": table.column("timestamp_utc"), **{name: table.column(name) for name in IMPACT_COLUMNS}}), backup_path)
    for name in IMPACT_COLUMNS:
        index = table.schema.get_field_index(name)
        table = table.set_column(index, table.schema.field(index), pa.array(new[name], type=table.schema.field(index).type))
    del parquet_file
    temporary = path.with_suffix(".parquet.tmp")
    pq.write_table(table, temporary, compression=compression)
    os.replace(temporary, path)
    return "recomputed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--settings", type=Path, default=REPOSITORY_ROOT / "config" / "settings_6m.yaml")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--old-notional", type=float, required=True)
    parser.add_argument("--new-notional", type=float, help="Defaults to the notional configured in --settings")
    args = parser.parse_args()
    settings_path = args.settings if args.settings.is_absolute() else REPOSITORY_ROOT / args.settings
    settings = load_settings(settings_path)
    new_notional = args.new_notional or settings.impact_notional_for(args.symbol)
    if new_notional == args.old_notional:
        raise SystemExit("New and old notional are equal; nothing to do")
    root = settings.data_dir / "features" / f"symbol={args.symbol}"
    backup_root = settings.data_dir / "impact_backup" / f"notional_{args.old_notional:g}" / f"symbol={args.symbol}"
    counts: dict[str, int] = {}
    for path in sorted(root.glob("date=*/features_1s.parquet")):
        status = process_partition(path, args.old_notional, new_notional, backup_root / path.parent.name / "impact.parquet")
        counts[status] = counts.get(status, 0) + 1
        print(f"{args.symbol} {path.parent.name}: {status}", flush=True)
    print(f"{args.symbol}: notional {args.old_notional:g} -> {new_notional:g} USDT; {counts}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
