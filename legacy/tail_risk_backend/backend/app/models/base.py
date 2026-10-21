"""Shared model interface."""

from abc import ABC, abstractmethod
from typing import Any


class ForecastModel(ABC):
    @abstractmethod
    def fit(self, values: Any) -> "ForecastModel": ...

    @abstractmethod
    def forecast(self, horizon: int = 1) -> Any: ...
