"""Adaptive ensemble with reward/punishment weight updates.

The ensemble holds a weight for every predictor. Each round:

1. ``predict()`` asks every predictor for its next-value forecast and blends
   them by weight.
2. ``update(actual)`` scores every predictor against the observed value and
   applies a multiplicative-weights (Hedge) update: each weight is multiplied
   by ``exp(-eta * loss)``. A small loss (good prediction) barely shrinks the
   weight while everyone else shrinks more — that is the reward. A large loss
   collapses the weight — that is the punishment. Weights are then
   renormalized so they always sum to 1.

Losses are normalized by the typical one-step move of the series (a running
mean of recent absolute changes), which makes the learning rate ``eta``
scale-free: the same settings work on a $500 stock and on microvolt sensor
readings.

A weight floor mixes a little uniform mass back in each round so a predictor
that was wrong for a long stretch can still recover when the regime changes.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .predictors import Predictor, default_predictors


@dataclass
class ModelState:
    """Bookkeeping for one predictor inside the ensemble."""

    predictor: Predictor
    weight: float
    last_prediction: Optional[float] = None
    total_abs_error: float = 0.0
    rounds: int = 0
    last_reward: Optional[float] = None  # multiplicative factor from last update

    @property
    def name(self) -> str:
        return self.predictor.name

    @property
    def mae(self) -> Optional[float]:
        return self.total_abs_error / self.rounds if self.rounds else None


@dataclass
class RoundResult:
    """What happened when one actual value was scored."""

    actual: float
    ensemble_prediction: float
    ensemble_error: float
    scale: float
    models: List[dict] = field(default_factory=list)


class AdaptiveEnsemble:
    def __init__(
        self,
        predictors: Optional[Sequence[Predictor]] = None,
        eta: float = 1.0,
        weight_floor: float = 0.01,
        scale_window: int = 30,
        max_loss: float = 5.0,
    ):
        """
        eta: learning rate for the multiplicative update. Higher means faster
            reward/punishment swings.
        weight_floor: fraction of uniform weight mixed back in every round so
            no model's weight can hit zero permanently.
        scale_window: how many recent one-step changes to average when
            normalizing losses.
        max_loss: cap on normalized loss so one wild outlier cannot wipe a
            model out in a single round.
        """
        preds = list(predictors) if predictors is not None else default_predictors()
        if not preds:
            raise ValueError("need at least one predictor")
        if eta <= 0:
            raise ValueError("eta must be > 0")
        if not 0.0 <= weight_floor < 1.0:
            raise ValueError("weight_floor must be in [0, 1)")
        uniform = 1.0 / len(preds)
        self.models: List[ModelState] = [ModelState(p, uniform) for p in preds]
        self.eta = eta
        self.weight_floor = weight_floor
        self.max_loss = max_loss
        self.history: List[float] = []
        self._recent_moves: deque = deque(maxlen=scale_window)
        self.rounds = 0
        self.total_abs_error = 0.0
        self._pending = False  # True between predict() and update()

    # ------------------------------------------------------------------ core

    def observe(self, value: float) -> None:
        """Append a value to history without scoring anyone (warm-up data)."""
        self._track_move(value)
        self.history.append(value)
        self._pending = False

    def predict(self) -> float:
        """Blend every predictor's forecast by its current weight."""
        total = 0.0
        for m in self.models:
            m.last_prediction = m.predictor.predict(self.history)
            total += m.weight * m.last_prediction
        self._pending = True
        return total

    def update(self, actual: float) -> RoundResult:
        """Score the outstanding predictions against `actual`, reward the
        close models, punish the far ones, and fold `actual` into history."""
        if not self._pending:
            self.predict()  # ensure every model has a live prediction

        scale = self._loss_scale()
        ensemble_pred = sum(m.weight * m.last_prediction for m in self.models)
        result = RoundResult(
            actual=actual,
            ensemble_prediction=ensemble_pred,
            ensemble_error=abs(ensemble_pred - actual),
            scale=scale,
        )

        for m in self.models:
            error = abs(m.last_prediction - actual)
            loss = min(error / scale, self.max_loss)
            reward = math.exp(-self.eta * loss)
            m.weight *= reward
            m.total_abs_error += error
            m.rounds += 1
            m.last_reward = reward
            result.models.append(
                {
                    "name": m.name,
                    "prediction": m.last_prediction,
                    "error": error,
                    "reward": reward,
                }
            )

        self._normalize_weights()
        for entry, m in zip(result.models, self.models):
            entry["weight"] = m.weight

        self.rounds += 1
        self.total_abs_error += result.ensemble_error
        self._track_move(actual)
        self.history.append(actual)
        self._pending = False
        return result

    def run(self, values: Sequence[float], warmup: int = 3) -> List[RoundResult]:
        """Feed a whole series through: warm up on the first few points, then
        predict-and-score every remaining point. Returns one result per
        scored point."""
        values = list(values)
        results: List[RoundResult] = []
        for i, v in enumerate(values):
            if i < warmup:
                self.observe(v)
            else:
                self.predict()
                results.append(self.update(v))
        return results

    # ------------------------------------------------------------- reporting

    @property
    def mae(self) -> Optional[float]:
        return self.total_abs_error / self.rounds if self.rounds else None

    def weights(self) -> Dict[str, float]:
        return {m.name: m.weight for m in self.models}

    def leaderboard(self) -> List[ModelState]:
        return sorted(self.models, key=lambda m: m.weight, reverse=True)

    # -------------------------------------------------------------- internal

    def _track_move(self, value: float) -> None:
        if self.history:
            self._recent_moves.append(abs(value - self.history[-1]))

    def _loss_scale(self) -> float:
        """Typical one-step move; the yardstick for what counts as a bad miss."""
        if self._recent_moves:
            mean_move = sum(self._recent_moves) / len(self._recent_moves)
            if mean_move > 0:
                return mean_move
        # Constant-so-far series: fall back to the magnitude of the data.
        if self.history:
            magnitude = abs(self.history[-1])
            if magnitude > 0:
                return magnitude * 0.01
        return 1.0

    def _normalize_weights(self) -> None:
        total = sum(m.weight for m in self.models)
        n = len(self.models)
        if total <= 0 or not math.isfinite(total):
            for m in self.models:
                m.weight = 1.0 / n
            return
        floor_share = self.weight_floor / n
        for m in self.models:
            m.weight = (1.0 - self.weight_floor) * (m.weight / total) + floor_share
