"""HAR-RV model fitted to daily, weekly, and monthly realized variance."""

from sklearn.linear_model import LinearRegression


HAR_FEATURES = ["rv_lag_1", "rv_mean_5", "rv_mean_22"]


def make_har_model() -> LinearRegression:
    return LinearRegression()
