from __future__ import annotations

"""
MarketTrendModel
================

Lightweight time-series forecaster for market trend signals used by the
Researcher Agent. To avoid heavy dependencies, this uses a simple
regression with lag features (works well for short-horizon forecasts).

Persisted with joblib at models/market_trend_model.joblib
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import os
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib


MODEL_PATH = os.path.join("models", "market_trend_model.joblib")


@dataclass
class ForecastResult:
    next_period_forecast: float
    growth_rate_percent: float


class MarketTrendModel:
    def __init__(self, model_path: str = MODEL_PATH, max_lag: int = 6):
        self.model_path = model_path
        self.max_lag = max_lag
        self.model: Optional[LinearRegression] = None

    def load(self) -> bool:
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            return True
        return False

    def save(self) -> None:
        if self.model is not None:
            joblib.dump(self.model, self.model_path)

    def _build_design(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Create supervised dataset with lag features
        X, y = [], []
        for t in range(self.max_lag, len(series)):
            X.append(series[t - self.max_lag:t])
            y.append(series[t])
        return np.asarray(X), np.asarray(y)

    def fit(self, series: np.ndarray) -> None:
        series = np.asarray(series, dtype=float).ravel()
        X, y = self._build_design(series)
        if len(y) == 0:
            raise ValueError("Insufficient data to train MarketTrendModel")
        self.model = LinearRegression()
        self.model.fit(X, y)

    def forecast_next(self, recent_series: np.ndarray) -> ForecastResult:
        if self.model is None:
            raise RuntimeError("MarketTrendModel not trained/loaded")
        window = np.asarray(recent_series[-self.max_lag:], dtype=float).ravel()
        if window.shape[0] < self.max_lag:
            # pad with last value
            if window.shape[0] == 0:
                window = np.zeros(self.max_lag)
            else:
                window = np.pad(window, (self.max_lag - window.shape[0], 0), mode='edge')
        pred = float(self.model.predict(window.reshape(1, -1))[0])
        last = float(window[-1]) if window[-1] != 0 else 1e-6
        growth = (pred - last) / abs(last) * 100.0
        return ForecastResult(next_period_forecast=round(pred, 2), growth_rate_percent=round(growth, 2))

    # Convenience for synthetic training for demos
    def fit_synthetic(self, length: int = 60) -> None:
        rng = np.random.RandomState(42)
        t = np.arange(length)
        # trend + seasonality + noise
        series = 100 + 0.8 * t + 5 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 2, size=length)
        self.fit(series)


