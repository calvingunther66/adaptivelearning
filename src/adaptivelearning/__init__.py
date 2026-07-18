"""adaptivelearning: next-value prediction for any numeric time series.

An ensemble of analytic forecasters is trained online: after every observed
value, models that predicted well are rewarded (their voting weight grows)
and models that missed are punished (their weight shrinks), via a
multiplicative-weights update.
"""

from .data import DataError, load_series
from .ensemble import AdaptiveEnsemble, ModelState, RoundResult
from .predictors import (
    AutoRegressive,
    Drift,
    ExponentialSmoothing,
    Holt,
    LastValue,
    LinearTrend,
    MovingAverage,
    Predictor,
    SeasonalNaive,
    default_predictors,
)

__version__ = "0.1.0"

__all__ = [
    "AdaptiveEnsemble",
    "AutoRegressive",
    "DataError",
    "Drift",
    "ExponentialSmoothing",
    "Holt",
    "LastValue",
    "LinearTrend",
    "ModelState",
    "MovingAverage",
    "Predictor",
    "RoundResult",
    "SeasonalNaive",
    "default_predictors",
    "load_series",
]
