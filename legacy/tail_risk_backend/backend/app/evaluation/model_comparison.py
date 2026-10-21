"""Paired forecast comparison helpers."""

import numpy as np


def loss_differential(loss_a, loss_b) -> np.ndarray:
    a, b = np.asarray(loss_a, dtype=float), np.asarray(loss_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("forecast loss series must be aligned")
    return a - b
