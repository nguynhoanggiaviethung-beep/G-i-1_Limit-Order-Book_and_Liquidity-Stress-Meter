"""Time ordered machine learning model helpers."""

from sklearn.ensemble import RandomForestRegressor


def make_random_forest(random_state: int = 42) -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=random_state, n_jobs=-1)
