"""Command-line interface.

Commands:
  predict      train on a file, print the predicted next value
  backtest     walk-forward evaluation: ensemble vs. every solo model
  interactive  feed values one at a time and watch rewards/punishments
  demo         run the whole loop on a built-in synthetic series
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from typing import List, Optional, Sequence

from .data import DataError, load_series
from .ensemble import AdaptiveEnsemble
from .predictors import default_predictors


# ------------------------------------------------------------------ helpers

def _fmt(x: Optional[float], width: int = 12) -> str:
    if x is None:
        return " " * (width - 3) + "n/a"
    return f"{x:>{width}.4f}"


def _build_ensemble(args: argparse.Namespace) -> AdaptiveEnsemble:
    return AdaptiveEnsemble(eta=args.eta, weight_floor=args.floor)


def _print_leaderboard(ensemble: AdaptiveEnsemble) -> None:
    print(f"{'model':<20} {'weight':>8} {'share':>7} {'MAE':>12}")
    for m in ensemble.leaderboard():
        bar = "#" * max(1, round(m.weight * 40))
        print(f"{m.name:<20} {m.weight:>8.4f} {m.weight:>6.1%} {_fmt(m.mae)}  {bar}")


def _feed(ensemble: AdaptiveEnsemble, series: Sequence[float], warmup: int) -> None:
    ensemble.run(series, warmup=warmup)


# ----------------------------------------------------------------- commands

def cmd_predict(args: argparse.Namespace) -> int:
    series = load_series(args.file, args.column)
    if len(series) < 3:
        print("need at least 3 data points", file=sys.stderr)
        return 1
    ensemble = _build_ensemble(args)
    _feed(ensemble, series, args.warmup)
    prediction = ensemble.predict()

    print(f"Trained on {len(series)} points "
          f"({ensemble.rounds} scored rounds, ensemble MAE {_fmt(ensemble.mae, 0).strip()})")
    print()
    print(f"Predicted next value: {prediction:.4f}")
    print()
    print("Current model weights (after rewards/punishments):")
    _print_leaderboard(ensemble)
    if args.verbose:
        print()
        print("Individual next-value opinions:")
        for m in ensemble.leaderboard():
            print(f"  {m.name:<20} {m.last_prediction:>12.4f}")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    series = load_series(args.file, args.column)
    warmup = max(args.warmup, 3)
    if len(series) <= warmup + 1:
        print(f"need more than {warmup + 1} data points to backtest", file=sys.stderr)
        return 1

    ensemble = _build_ensemble(args)
    results = ensemble.run(series, warmup=warmup)
    ens_errors = [r.ensemble_error for r in results]

    # Score each predictor solo over the same walk-forward split.
    solo_rows = []
    for pred in default_predictors():
        errors = []
        for t in range(warmup, len(series)):
            errors.append(abs(pred.predict(series[:t]) - series[t]))
        solo_rows.append((pred.name, _mae(errors), _rmse(errors)))
    solo_rows.sort(key=lambda r: r[1])

    print(f"Walk-forward backtest on {args.file}: "
          f"{len(series)} points, {len(results)} scored predictions "
          f"(warmup {warmup})")
    print()
    print(f"{'model':<20} {'MAE':>12} {'RMSE':>12}")
    print(f"{'ENSEMBLE':<20} {_fmt(_mae(ens_errors))} {_fmt(_rmse(ens_errors))}")
    for name, mae, rmse in solo_rows:
        print(f"{name:<20} {_fmt(mae)} {_fmt(rmse)}")
    print()
    print("Final weights after the run:")
    _print_leaderboard(ensemble)

    best_solo = solo_rows[0]
    ens_mae = _mae(ens_errors)
    print()
    if ens_mae <= best_solo[1]:
        print(f"The adaptive ensemble beat every individual model "
              f"(best solo: {best_solo[0]} at MAE {best_solo[1]:.4f}).")
    else:
        pct = (ens_mae - best_solo[1]) / best_solo[1] * 100
        print(f"Best solo model was {best_solo[0]} (MAE {best_solo[1]:.4f}); "
              f"the ensemble came within {pct:.1f}% of it without knowing "
              f"in advance which model to trust.")
    return 0


def cmd_interactive(args: argparse.Namespace) -> int:
    ensemble = _build_ensemble(args)
    seeded = 0
    if args.file:
        series = load_series(args.file, args.column)
        _feed(ensemble, series, args.warmup)
        seeded = len(series)
        print(f"Seeded with {seeded} points from {args.file}.")

    print("Interactive mode. Enter one number per line; I predict first, then")
    print("you tell me the real value and I reward/punish my models.")
    print("Commands: 'weights' shows the leaderboard, 'quit' exits.")
    print()

    while len(ensemble.history) < 2:
        value = _read_value(f"seed value {len(ensemble.history) + 1} (need 2 to start): ")
        if value is None:
            return 0
        ensemble.observe(value)

    while True:
        prediction = ensemble.predict()
        print(f"\nMy prediction for the next value: {prediction:.4f}")
        value = _read_value("actual value: ", ensemble)
        if value is None:
            break
        result = ensemble.update(value)
        miss = result.ensemble_error
        print(f"Actual {value:.4f} -> I was off by {miss:.4f} "
              f"({miss / result.scale:.2f}x the typical move).")
        rewarded = max(result.models, key=lambda m: m["reward"])
        punished = min(result.models, key=lambda m: m["reward"])
        print(f"  rewarded  {rewarded['name']:<20} "
              f"(error {rewarded['error']:.4f}, weight -> {rewarded['weight']:.3f})")
        print(f"  punished  {punished['name']:<20} "
              f"(error {punished['error']:.4f}, weight -> {punished['weight']:.3f})")

    print(f"\nSession over: {ensemble.rounds} scored rounds, "
          f"ensemble MAE {_fmt(ensemble.mae, 0).strip()}.")
    _print_leaderboard(ensemble)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    series = []
    for t in range(200):
        trend = 100 + 0.4 * t
        season = 6.0 * math.sin(2 * math.pi * t / 7)
        noise = rng.gauss(0, 1.5)
        series.append(trend + season + noise)
    print("Demo: 200 synthetic points = upward trend + weekly cycle + noise")
    print()

    ensemble = _build_ensemble(args)
    results = ensemble.run(series, warmup=5)
    checkpoints = {10, 50, 100, len(results)}
    seen = []
    for i, r in enumerate(results, start=1):
        seen.append(r.ensemble_error)
        if i in checkpoints:
            recent = seen[-25:]
            print(f"after {i:>3} rounds: recent MAE {sum(recent) / len(recent):8.4f}   "
                  f"top model: {ensemble.leaderboard()[0].name}")
    print()
    print("Final leaderboard — weight is the accumulated reward/punishment:")
    _print_leaderboard(ensemble)
    print()
    prediction = ensemble.predict()
    print(f"Predicted next value: {prediction:.4f}")
    return 0


def _read_value(prompt: str, ensemble: Optional[AdaptiveEnsemble] = None) -> Optional[float]:
    while True:
        try:
            raw = input(prompt).strip()
        except EOFError:
            return None
        if raw.lower() in {"quit", "exit", "q"}:
            return None
        if raw.lower() == "weights" and ensemble is not None:
            _print_leaderboard(ensemble)
            continue
        try:
            return float(raw.replace(",", ""))
        except ValueError:
            print(f"  couldn't parse {raw!r} as a number (or 'weights'/'quit')")


def _mae(errors: List[float]) -> float:
    return sum(errors) / len(errors)


def _rmse(errors: List[float]) -> float:
    return math.sqrt(sum(e * e for e in errors) / len(errors))


# --------------------------------------------------------------------- main

def _add_common(p: argparse.ArgumentParser, needs_file: bool = True) -> None:
    if needs_file:
        p.add_argument("file", help="CSV or plain-text file of numbers ('-' for stdin)")
        p.add_argument("--column", "-c", help="column name or 0-based index to use")
    p.add_argument("--eta", type=float, default=1.0,
                   help="learning rate: how hard to reward/punish (default 1.0)")
    p.add_argument("--floor", type=float, default=0.01,
                   help="uniform weight mixed back each round so models can recover (default 0.01)")
    p.add_argument("--warmup", type=int, default=3,
                   help="initial points to observe before scoring begins (default 3)")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="adaptivelearning",
        description="Predict the next value in any numeric series with a "
                    "reward/punishment-trained ensemble of analytic models.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("predict", help="predict the next value after a series")
    _add_common(p)
    p.add_argument("--verbose", "-v", action="store_true",
                   help="also show each model's individual prediction")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("backtest", help="walk-forward evaluation of ensemble vs. solo models")
    _add_common(p)
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("interactive", help="enter values live and watch the weights adapt")
    p.add_argument("--file", "-f", help="optional file to seed history from")
    p.add_argument("--column", "-c", help="column name or 0-based index to use")
    _add_common(p, needs_file=False)
    p.set_defaults(func=cmd_interactive)

    p = sub.add_parser("demo", help="run on a built-in synthetic series")
    p.add_argument("--seed", type=int, default=42)
    _add_common(p, needs_file=False)
    p.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except DataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print()
        return 130


if __name__ == "__main__":
    sys.exit(main())
