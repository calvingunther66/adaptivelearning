"""Analytic next-value predictors.

Each predictor is stateless: it looks at the full history it is given and
returns a single prediction for the next value. Keeping predictors stateless
makes them trivially safe to backtest (no leakage between walk-forward steps)
and lets the ensemble own all of the adaptive state.

Every predictor must tolerate arbitrarily short histories; the convention is
to degrade toward the last observed value (or 0.0 for an empty history).
"""

from __future__ import annotations

from typing import List, Sequence


class Predictor:
    """Base class. Subclasses implement predict()."""

    name: str = "predictor"

    def predict(self, history: Sequence[float]) -> float:
        raise NotImplementedError

    def _fallback(self, history: Sequence[float]) -> float:
        return history[-1] if history else 0.0


class LastValue(Predictor):
    """Naive forecast: tomorrow looks like today."""

    name = "last-value"

    def predict(self, history: Sequence[float]) -> float:
        return self._fallback(history)


class Drift(Predictor):
    """Last value plus the average historical step."""

    name = "drift"

    def predict(self, history: Sequence[float]) -> float:
        if len(history) < 2:
            return self._fallback(history)
        step = (history[-1] - history[0]) / (len(history) - 1)
        return history[-1] + step


class MovingAverage(Predictor):
    """Mean of the last `window` values."""

    def __init__(self, window: int = 5):
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self.name = f"moving-avg-{window}"

    def predict(self, history: Sequence[float]) -> float:
        if not history:
            return 0.0
        tail = history[-self.window:]
        return sum(tail) / len(tail)


class ExponentialSmoothing(Predictor):
    """Single exponential smoothing with fixed alpha."""

    def __init__(self, alpha: float = 0.5):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.name = f"exp-smooth-{alpha:g}"

    def predict(self, history: Sequence[float]) -> float:
        if not history:
            return 0.0
        level = history[0]
        for x in history[1:]:
            level = self.alpha * x + (1.0 - self.alpha) * level
        return level


class Holt(Predictor):
    """Double exponential smoothing: tracks a level and a trend."""

    def __init__(self, alpha: float = 0.5, beta: float = 0.3):
        if not 0.0 < alpha <= 1.0 or not 0.0 < beta <= 1.0:
            raise ValueError("alpha and beta must be in (0, 1]")
        self.alpha = alpha
        self.beta = beta
        self.name = f"holt-{alpha:g}-{beta:g}"

    def predict(self, history: Sequence[float]) -> float:
        if len(history) < 2:
            return self._fallback(history)
        level = history[0]
        trend = history[1] - history[0]
        for x in history[1:]:
            prev_level = level
            level = self.alpha * x + (1.0 - self.alpha) * (level + trend)
            trend = self.beta * (level - prev_level) + (1.0 - self.beta) * trend
        return level + trend


class LinearTrend(Predictor):
    """Ordinary least squares line through the last `window` points."""

    def __init__(self, window: int = 10):
        if window < 2:
            raise ValueError("window must be >= 2")
        self.window = window
        self.name = f"linear-trend-{window}"

    def predict(self, history: Sequence[float]) -> float:
        tail = list(history[-self.window:])
        n = len(tail)
        if n < 2:
            return self._fallback(history)
        xs = range(n)
        mean_x = (n - 1) / 2.0
        mean_y = sum(tail) / n
        sxx = sum((x - mean_x) ** 2 for x in xs)
        sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, tail))
        slope = sxy / sxx if sxx else 0.0
        intercept = mean_y - slope * mean_x
        return intercept + slope * n


class SeasonalNaive(Predictor):
    """Repeat the value from one season ago."""

    def __init__(self, period: int = 7):
        if period < 1:
            raise ValueError("period must be >= 1")
        self.period = period
        self.name = f"seasonal-naive-{period}"

    def predict(self, history: Sequence[float]) -> float:
        if len(history) < self.period:
            return self._fallback(history)
        return history[-self.period]


class AutoRegressive(Predictor):
    """AR(p) model fit by ridge-regularized least squares on each call.

    Solves (X'X + lambda*I) b = X'y with Gaussian elimination, where each row
    of X holds the p lagged values (plus a bias term) preceding a target y.
    The small ridge term keeps the solve stable on near-constant series.
    """

    def __init__(self, order: int = 4, ridge: float = 1e-6):
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        self.ridge = ridge
        self.name = f"autoreg-{order}"

    def predict(self, history: Sequence[float]) -> float:
        p = self.order
        # Need enough rows to make the fit meaningful.
        if len(history) < p + 3:
            return self._fallback(history)
        rows: List[List[float]] = []
        targets: List[float] = []
        for t in range(p, len(history)):
            rows.append([1.0] + [history[t - k] for k in range(1, p + 1)])
            targets.append(history[t])
        coeffs = self._solve_normal_equations(rows, targets)
        if coeffs is None:
            return self._fallback(history)
        features = [1.0] + [history[-k] for k in range(1, p + 1)]
        return sum(c * f for c, f in zip(coeffs, features))

    def _solve_normal_equations(self, rows, targets):
        dim = len(rows[0])
        # Build A = X'X + ridge*I and b = X'y.
        a = [[self.ridge if i == j else 0.0 for j in range(dim)] for i in range(dim)]
        b = [0.0] * dim
        for row, y in zip(rows, targets):
            for i in range(dim):
                b[i] += row[i] * y
                for j in range(dim):
                    a[i][j] += row[i] * row[j]
        # Gaussian elimination with partial pivoting.
        for col in range(dim):
            pivot = max(range(col, dim), key=lambda r: abs(a[r][col]))
            if abs(a[pivot][col]) < 1e-12:
                return None
            a[col], a[pivot] = a[pivot], a[col]
            b[col], b[pivot] = b[pivot], b[col]
            for r in range(col + 1, dim):
                factor = a[r][col] / a[col][col]
                b[r] -= factor * b[col]
                for c in range(col, dim):
                    a[r][c] -= factor * a[col][c]
        coeffs = [0.0] * dim
        for i in range(dim - 1, -1, -1):
            acc = b[i] - sum(a[i][j] * coeffs[j] for j in range(i + 1, dim))
            coeffs[i] = acc / a[i][i]
        return coeffs


def default_predictors() -> List[Predictor]:
    """The stock lineup used by the CLI: a spread of behaviors so the
    ensemble has meaningfully different opinions to arbitrate between."""
    return [
        LastValue(),
        Drift(),
        MovingAverage(3),
        MovingAverage(10),
        ExponentialSmoothing(0.3),
        ExponentialSmoothing(0.7),
        Holt(),
        LinearTrend(10),
        SeasonalNaive(7),
        AutoRegressive(4),
    ]
