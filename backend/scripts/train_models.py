"""Compatibility entry point for the multi-horizon walk-forward runner."""

from run_walkforward import main


if __name__ == "__main__":
    from _legacy_ohlcv import refuse_legacy_ohlcv
    refuse_legacy_ohlcv("train_models.py")
