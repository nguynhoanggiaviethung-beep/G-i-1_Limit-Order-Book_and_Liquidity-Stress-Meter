"""Compatibility entry point for VaR and ES backtests."""

from run_risk_eval import main


if __name__ == "__main__":
    from _legacy_ohlcv import refuse_legacy_ohlcv
    refuse_legacy_ohlcv("run_backtest.py")
