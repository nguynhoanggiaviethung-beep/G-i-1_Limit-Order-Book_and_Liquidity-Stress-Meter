from __future__ import annotations

import pandas as pd

from scripts.build_submission_evidence import _confusion_distribution, _missing_intervals


def test_missing_intervals_report_only_unobserved_grid_seconds():
    timestamps = pd.Series(pd.to_datetime([
        "2026-10-01T00:00:00Z",
        "2026-10-01T00:00:01Z",
        "2026-10-01T00:00:04Z",
    ], utc=True))

    missing = _missing_intervals("BTCUSDT", "2026-10-01", 5, timestamps)

    assert missing == [{
        "symbol": "BTCUSDT",
        "date_utc": "2026-10-01",
        "missing_start_utc": "2026-10-01T00:00:02Z",
        "missing_end_utc_exclusive": "2026-10-01T00:00:04Z",
        "duration_seconds": 2,
        "reason": "no grid row observed; QA does not attribute cause per second",
    }]


def test_confusion_distribution_uses_true_class_rows():
    metric = {"labels": ["down", "flat", "up"], "confusion_matrix": [[3, 0, 0], [0, 4, 0], [0, 0, 3]]}

    assert _confusion_distribution(metric) == {"down": 0.3, "flat": 0.4, "up": 0.3}
