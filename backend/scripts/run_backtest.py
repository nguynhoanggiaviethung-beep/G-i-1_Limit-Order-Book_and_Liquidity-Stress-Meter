"""Backtesting entry point; implemented after the horizon and risk definitions are frozen."""

import argparse


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    raise NotImplementedError("Define VaR/ES horizon and out-of-sample dates before backtesting")


if __name__ == "__main__":
    main()
