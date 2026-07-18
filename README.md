# adaptivelearning

Predict the next number in any time series — stock prices, sensor readings,
experiment measurements — and **learn which prediction strategy to trust by
rewarding models that get it right and punishing models that miss**.

No dependencies. Pure Python 3.9+ standard library.

```
$ adaptivelearning predict examples/sample_stock.csv

Trained on 120 points (117 scored rounds, ensemble MAE 1.0211)

Predicted next value: 139.4172

Current model weights (after rewards/punishments):
model                  weight   share          MAE
last-value             0.6547  65.5%       0.9850  ##########################
drift                  0.2398  24.0%       0.9961  ##########
autoreg-4              0.0433   4.3%       1.2536  ##
exp-smooth-0.7         0.0284   2.8%       1.0466  #
...
```

## How it works

There is no single best forecasting model: a noisy stock is best predicted by
"tomorrow ≈ today", a growth curve by a trend model, a weekly pattern by a
seasonal model. Instead of guessing which applies to your data, this app runs
**ten analytic predictors in parallel** and lets an online learning loop decide
who to listen to:

1. **Predict.** Every model forecasts the next value. The app's prediction is
   the weighted average of their forecasts.
2. **Score.** When the actual value arrives, each model's absolute error is
   measured and normalized by the series' typical one-step move (so the same
   settings work on a $500 stock and on microvolt sensor data).
3. **Reward / punish.** Each model's weight is multiplied by
   `exp(-eta * loss)` — the multiplicative-weights (Hedge) update from online
   learning theory. A model that nailed the prediction keeps its weight while
   everyone else shrinks; a model that missed badly is cut down hard. Weights
   are renormalized to sum to 1.
4. **Repeat.** Influence flows continuously toward whichever models are
   currently working. A small *weight floor* (1% of mass redistributed
   uniformly each round) means a punished model is never eliminated — if the
   data regime changes, it can earn its way back.

Multiplicative weights comes with a classic guarantee: the ensemble's
cumulative loss approaches that of the best single model in hindsight, without
ever knowing in advance which model that is. In practice you can see it in the
backtests: on a synthetic trend+seasonality series the autoregressive model
ends up with ~72% of the weight; on a noisy near-random-walk stock the naive
and drift models take over instead.

### The model lineup

| model | what it assumes |
|---|---|
| `last-value` | tomorrow looks like today (random walk) |
| `drift` | today plus the average historical step |
| `moving-avg-3`, `moving-avg-10` | mean-reversion to a short/long average |
| `exp-smooth-0.3`, `exp-smooth-0.7` | smoothed level, slow/fast reacting |
| `holt-0.5-0.3` | smoothed level **and** trend |
| `linear-trend-10` | straight line through the last 10 points |
| `seasonal-naive-7` | repeats the value from 7 steps ago |
| `autoreg-4` | AR(4): a learned linear function of the last 4 values |

All predictors are stateless (they recompute from history on every call), so
backtests are leak-free by construction, and the ensemble owns all adaptive
state.

## Installation

```bash
pip install .        # installs the `adaptivelearning` command
# or run straight from the repo with no install:
PYTHONPATH=src python3 -m adaptivelearning demo
```

## Usage

**Predict the next value** after a series (trains on the whole file first):

```bash
adaptivelearning predict data.csv
adaptivelearning predict prices.csv --column close --verbose
cat readings.txt | adaptivelearning predict -
```

Input can be a plain list of numbers (one per line) or a CSV. With a CSV, pick
a column by name or 0-based index with `--column`; otherwise the last numeric
column is used. Dollar signs and thousands separators are handled.

**Backtest** — walk-forward evaluation of the ensemble against every solo
model, so you can see whether adaptivity paid off on *your* data:

```bash
adaptivelearning backtest data.csv
```

**Interactive** — enter values as they happen (lab readings, daily closes) and
watch the reward/punishment loop run live:

```bash
adaptivelearning interactive              # start from scratch
adaptivelearning interactive -f data.csv  # seed with existing history
```

```
My prediction for the next value: 104.8810
actual value: 106.5
Actual 106.5000 -> I was off by 1.6190 (0.81x the typical move).
  rewarded  drift                (error 0.5000, weight -> 0.279)
  punished  exp-smooth-0.3       (error 4.8800, weight -> 0.007)
```

Type `weights` at the prompt to see the leaderboard, `quit` to exit.

**Demo** — no data needed:

```bash
adaptivelearning demo
```

### Tuning

| flag | default | effect |
|---|---|---|
| `--eta` | 1.0 | learning rate: how hard each round rewards/punishes. Higher adapts faster but is twitchier on noise. |
| `--floor` | 0.01 | uniform weight mixed back each round; raise it if your data switches regimes often. |
| `--warmup` | 3 | initial points observed before scoring starts. |

## Library API

```python
from adaptivelearning import AdaptiveEnsemble

ens = AdaptiveEnsemble()          # default 10-model lineup
ens.run([101.2, 102.5, 101.9, 103.4], warmup=3)

guess = ens.predict()             # my forecast for the next value
result = ens.update(104.1)        # actual arrives: reward/punish, learn
print(result.ensemble_error)      # how far off the blend was
print(ens.weights())              # current trust per model
```

Bring your own models by subclassing `Predictor` (implement
`predict(history) -> float`) and passing a list to
`AdaptiveEnsemble(predictors=[...])`.

## Development

```bash
pip install pytest
PYTHONPATH=src pytest tests/     # 38 tests
```

Layout: `src/adaptivelearning/predictors.py` (the analytic models),
`ensemble.py` (the reward/punishment loop), `data.py` (CSV/text loading),
`cli.py` (commands).
