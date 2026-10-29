from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from app.features.classification_features import build_classification_features
from scripts.recompute_impact_features import IMPACT_COLUMNS, impact_columns, process_partition


def _grid(rows: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.date_range("2026-06-01T00:00:00Z", periods=rows, freq="s")
    data: dict[str, object] = {"timestamp_utc": times, "symbol": "BTCUSDT", "book_valid": True, "tick_size": 0.01}
    for level in range(1, 21):
        data[f"bid_price_{level}"] = 100.0 - 0.01 * level
        data[f"ask_price_{level}"] = 100.0 + 0.01 * level
        data[f"bid_qty_{level}"] = rng.uniform(0.1, 5.0, rows)
        data[f"ask_qty_{level}"] = rng.uniform(0.1, 5.0, rows)
    data["ask_qty_20"] = np.where(np.arange(rows) % 7 == 0, np.nan, data["ask_qty_20"])  # missing quantity counts as 0
    return pd.DataFrame(data)


@pytest.mark.parametrize("notional", [50.0, 2_000.0, 6_000.0])
def test_impact_columns_match_the_feature_builder(notional):
    grid = _grid()
    built = build_classification_features(grid, impact_notional_usdt=notional, tick_size=0.01)
    recomputed = impact_columns(grid, notional)
    for name in IMPACT_COLUMNS:
        np.testing.assert_array_equal(recomputed[name], built[name].to_numpy(dtype=float))
    if notional == 6_000.0:
        assert np.isnan(recomputed["price_impact_bps"]).any()  # some books cannot fill


def _string_grid() -> pd.DataFrame:
    # As in the real pipeline, the grid carries book levels as decimal strings and the builder
    # passes them through, so the builder and the recompute parse the same strings.
    grid = _grid()
    for column in grid.columns:
        if column.startswith(("bid_price_", "ask_price_", "bid_qty_", "ask_qty_")):
            grid[column] = grid[column].map(lambda value: "" if pd.isna(value) else f"{value:.8f}")
    return grid


def _write_partition(tmp_path, notional):
    built = build_classification_features(_string_grid(), impact_notional_usdt=notional, tick_size=0.01)
    path = tmp_path / "date=2026-06-01" / "features_1s.parquet"
    path.parent.mkdir(parents=True)
    built.to_parquet(path, index=False)
    return path, built


def test_partition_is_rewritten_backed_up_and_resumable(tmp_path):
    path, before = _write_partition(tmp_path, 6_000.0)
    backup = tmp_path / "backup" / "impact.parquet"
    assert process_partition(path, 6_000.0, 2_000.0, backup) == "recomputed"
    after = pq.ParquetFile(path).read().to_pandas()
    expected = build_classification_features(_string_grid(), impact_notional_usdt=2_000.0, tick_size=0.01)
    for name in IMPACT_COLUMNS:
        np.testing.assert_array_equal(after[name].to_numpy(dtype=float), expected[name].to_numpy(dtype=float))
        np.testing.assert_array_equal(pd.read_parquet(backup)[name].to_numpy(dtype=float), before[name].to_numpy(dtype=float))
    untouched = [name for name in before.columns if name not in IMPACT_COLUMNS]
    pd.testing.assert_frame_equal(after[untouched], before[untouched])
    assert pq.read_schema(path).names == list(before.columns)
    assert process_partition(path, 6_000.0, 2_000.0, backup) == "already_new"


def test_partition_with_unexpected_stored_impact_is_not_overwritten(tmp_path):
    path, _ = _write_partition(tmp_path, 6_000.0)
    original = path.read_bytes()
    with pytest.raises(RuntimeError, match="do not reproduce"):
        process_partition(path, 5_000.0, 2_000.0, tmp_path / "backup" / "impact.parquet")
    assert path.read_bytes() == original
