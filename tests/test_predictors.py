import math

import pytest

from adaptivelearning.predictors import (
    AutoRegressive,
    Drift,
    ExponentialSmoothing,
    Holt,
    LastValue,
    LinearTrend,
    MovingAverage,
    SeasonalNaive,
    default_predictors,
)


LINEAR = [float(2 * t + 1) for t in range(20)]  # 1, 3, 5, ... next is 41
SEASONAL = [10.0, 20.0, 30.0] * 5  # period 3, next is 10


def test_last_value():
    assert LastValue().predict([1.0, 2.0, 7.5]) == 7.5


def test_drift_exact_on_linear_series():
    assert Drift().predict(LINEAR) == pytest.approx(41.0)


def test_moving_average():
    assert MovingAverage(3).predict([1.0, 2.0, 3.0, 4.0]) == pytest.approx(3.0)
    # Window longer than history uses everything available.
    assert MovingAverage(10).predict([2.0, 4.0]) == pytest.approx(3.0)


def test_exponential_smoothing_alpha_one_is_last_value():
    assert ExponentialSmoothing(1.0).predict([5.0, 9.0, 2.0]) == pytest.approx(2.0)


def test_linear_trend_exact_on_linear_series():
    assert LinearTrend(10).predict(LINEAR) == pytest.approx(41.0)


def test_holt_tracks_linear_trend():
    assert Holt().predict(LINEAR) == pytest.approx(41.0, abs=0.5)


def test_seasonal_naive():
    assert SeasonalNaive(3).predict(SEASONAL) == pytest.approx(10.0)


def test_autoregressive_learns_linear_recurrence():
    # x_t = 2*x_{t-1} - x_{t-2} generates any linear series exactly.
    assert AutoRegressive(2).predict(LINEAR) == pytest.approx(41.0, abs=1e-6)


def test_autoregressive_constant_series_is_stable():
    assert AutoRegressive(4).predict([5.0] * 30) == pytest.approx(5.0, abs=1e-3)


@pytest.mark.parametrize("history", [[], [3.0], [3.0, 4.0]])
def test_all_predictors_handle_short_history(history):
    for pred in default_predictors():
        value = pred.predict(history)
        assert math.isfinite(value)
        if history:
            # Degrades to something in the range of the data, not garbage.
            assert min(history) - 5 <= value <= max(history) + 5


def test_predictor_names_are_unique():
    names = [p.name for p in default_predictors()]
    assert len(names) == len(set(names))
