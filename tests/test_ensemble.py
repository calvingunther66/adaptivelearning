import math
import random

import pytest

from adaptivelearning.ensemble import AdaptiveEnsemble
from adaptivelearning.predictors import Drift, LastValue, LinearTrend, MovingAverage


def test_weights_start_uniform_and_stay_normalized():
    ens = AdaptiveEnsemble(predictors=[LastValue(), Drift(), MovingAverage(3)])
    assert all(w == pytest.approx(1 / 3) for w in ens.weights().values())
    ens.run([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], warmup=3)
    assert sum(ens.weights().values()) == pytest.approx(1.0)


def test_good_model_is_rewarded_bad_model_is_punished():
    # On a clean linear ramp, Drift is exact and LastValue always lags by 2.
    ens = AdaptiveEnsemble(predictors=[LastValue(), Drift()])
    series = [float(2 * t) for t in range(30)]
    ens.run(series, warmup=3)
    weights = ens.weights()
    assert weights["drift"] > 0.8
    assert weights["last-value"] < 0.2


def test_reward_factors_reported_per_round():
    ens = AdaptiveEnsemble(predictors=[LastValue(), Drift()])
    for v in [1.0, 2.0, 3.0]:
        ens.observe(v)
    ens.predict()
    result = ens.update(4.0)
    by_name = {m["name"]: m for m in result.models}
    # Drift predicted exactly 4.0 -> no punishment; LastValue said 3.0.
    assert by_name["drift"]["error"] == pytest.approx(0.0)
    assert by_name["drift"]["reward"] == pytest.approx(1.0)
    assert by_name["last-value"]["reward"] < 1.0


def test_weight_floor_lets_models_recover():
    ens = AdaptiveEnsemble(predictors=[LastValue(), Drift()], weight_floor=0.05)
    ens.run([float(2 * t) for t in range(50)], warmup=3)
    # Even after 40+ losing rounds, the floor keeps LastValue alive.
    assert ens.weights()["last-value"] >= 0.025


def test_regime_change_shifts_weights():
    ens = AdaptiveEnsemble(predictors=[LastValue(), Drift(), LinearTrend(5)])
    ramp = [float(t) for t in range(40)]
    flat = [39.0] * 40
    ens.run(ramp + flat, warmup=3)
    # After the long flat stretch, drift (still extrapolating the old ramp
    # from full history) should be punished into irrelevance, while models
    # that adapt to the flat regime hold nearly all the weight.
    weights = ens.weights()
    assert weights["drift"] < 0.05
    assert weights["last-value"] + weights["linear-trend-5"] > 0.9


def test_ensemble_close_to_best_model_on_noisy_trend():
    rng = random.Random(7)
    series = [0.5 * t + rng.gauss(0, 1.0) for t in range(150)]
    warmup = 5

    ens = AdaptiveEnsemble()
    results = ens.run(series, warmup=warmup)
    ens_mae = sum(r.ensemble_error for r in results) / len(results)

    from adaptivelearning.predictors import default_predictors

    best_solo = min(
        sum(abs(p.predict(series[:t]) - series[t]) for t in range(warmup, len(series)))
        / (len(series) - warmup)
        for p in default_predictors()
    )
    # Adaptive ensemble should land within 15% of the best model it contains,
    # without knowing in advance which one that is.
    assert ens_mae <= best_solo * 1.15


def test_update_without_predict_still_works():
    ens = AdaptiveEnsemble(predictors=[LastValue(), Drift()])
    ens.observe(1.0)
    ens.observe(2.0)
    result = ens.update(3.0)  # implicit predict
    assert math.isfinite(result.ensemble_prediction)
    assert ens.rounds == 1


def test_scale_invariance_of_weights():
    """The same shaped series at wildly different magnitudes should produce
    essentially the same weight allocation."""
    small = [math.sin(t / 3) + 0.1 * t for t in range(60)]
    big = [x * 1e6 for x in small]
    w_small = AdaptiveEnsemble()
    w_small.run(small, warmup=3)
    w_big = AdaptiveEnsemble()
    w_big.run(big, warmup=3)
    for name in w_small.weights():
        assert w_small.weights()[name] == pytest.approx(w_big.weights()[name], abs=1e-6)


def test_constant_series_does_not_blow_up():
    ens = AdaptiveEnsemble()
    ens.run([5.0] * 20, warmup=3)
    assert ens.predict() == pytest.approx(5.0, abs=0.1)
    assert all(math.isfinite(w) for w in ens.weights().values())


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        AdaptiveEnsemble(predictors=[])
    with pytest.raises(ValueError):
        AdaptiveEnsemble(eta=0)
    with pytest.raises(ValueError):
        AdaptiveEnsemble(weight_floor=1.0)
