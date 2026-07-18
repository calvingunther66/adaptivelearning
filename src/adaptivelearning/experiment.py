"""Walk-forward market experiment.

Protocol (as opposed to conventional train/test fitting):

1. Feed the ensemble the first N trading days as warm-up context — no
   scoring, it just observes.
2. From then on, for every tick: predict the next close, see the real one,
   reward/punish every model by how close it came, fold the tick into
   history, repeat until the end of the data (i.e. "modern day").

The report splits the scored span into segments so you can see whether the
reward/punishment loop actually improved anything as it went. Because raw
price levels (and volatility) drift over long spans, errors are also
reported relative to the naive last-value baseline — beating that baseline
is the meaningful bar for a market series.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .ensemble import AdaptiveEnsemble


@dataclass
class SegmentStats:
    label: str
    ticks: int
    mae: float
    naive_mae: float          # last-value baseline over the same ticks
    win_rate: float           # fraction of ticks ensemble beat the baseline
    direction_hits: float     # fraction of moves whose sign we called right
    top_model: str

    @property
    def edge_vs_naive(self) -> float:
        """Positive = ensemble error smaller than baseline (improvement)."""
        if self.naive_mae == 0:
            return 0.0
        return 1.0 - self.mae / self.naive_mae


@dataclass
class ExperimentReport:
    warmup_ticks: int
    warmup_days: int
    scored_ticks: int
    segments: List[SegmentStats]
    final_weights: List[Tuple[str, float]]  # sorted desc
    overall_mae: float
    overall_naive_mae: float
    overall_win_rate: float
    overall_direction_hits: float

    @property
    def overall_edge(self) -> float:
        if self.overall_naive_mae == 0:
            return 0.0
        return 1.0 - self.overall_mae / self.overall_naive_mae


def warmup_ticks_for_days(dates: Sequence[str], days: int) -> int:
    """Number of leading ticks covering the first `days` distinct dates.

    `dates` holds one date-ish string per tick (only the date part, before
    any space, is compared), assumed chronological.
    """
    seen: set = set()
    for i, stamp in enumerate(dates):
        day = stamp.split(" ")[0]
        if day not in seen:
            if len(seen) == days:
                return i
            seen.add(day)
    return len(dates)  # fewer distinct days than requested


def run_experiment(
    rows: Sequence[Tuple[str, float]],
    warmup_days: int = 4,
    n_segments: int = 4,
    ensemble: Optional[AdaptiveEnsemble] = None,
) -> ExperimentReport:
    dates = [r[0] for r in rows]
    series = [r[1] for r in rows]
    warmup = warmup_ticks_for_days(dates, warmup_days)
    if len(series) - warmup < n_segments * 2:
        raise ValueError(
            f"only {len(series) - warmup} ticks left after a {warmup_days}-day "
            f"warm-up; need more data (or fewer warmup days)"
        )

    ens = ensemble if ensemble is not None else AdaptiveEnsemble()
    results = ens.run(series, warmup=warmup)

    # Per-tick stats. For scored tick i: prev close is series[warmup+i-1],
    # actual is series[warmup+i].
    errors, naive_errors, wins, dir_hits, dir_total = [], [], 0, 0, 0
    for i, r in enumerate(results):
        prev = series[warmup + i - 1]
        naive_err = abs(prev - r.actual)
        errors.append(r.ensemble_error)
        naive_errors.append(naive_err)
        if r.ensemble_error < naive_err:
            wins += 1
        actual_move = r.actual - prev
        predicted_move = r.ensemble_prediction - prev
        if actual_move != 0 and predicted_move != 0:
            dir_total += 1
            if (actual_move > 0) == (predicted_move > 0):
                dir_hits += 1

    # Split the scored span into segments. Each RoundResult already carries
    # the post-update weight of every model, so the segment-end leaderboard
    # comes straight from the recorded rounds.
    n = len(results)
    bounds = [round(n * k / n_segments) for k in range(n_segments + 1)]

    segments = []
    for k in range(n_segments):
        lo, hi = bounds[k], bounds[k + 1]
        seg_err = errors[lo:hi]
        seg_naive = naive_errors[lo:hi]
        seg_dir_hits = seg_dir_total = 0
        for i in range(lo, hi):
            prev = series[warmup + i - 1]
            actual_move = results[i].actual - prev
            predicted_move = results[i].ensemble_prediction - prev
            if actual_move != 0 and predicted_move != 0:
                seg_dir_total += 1
                if (actual_move > 0) == (predicted_move > 0):
                    seg_dir_hits += 1
        end_weights = results[hi - 1].models
        segments.append(
            SegmentStats(
                label=f"{k + 1}/{n_segments}",
                ticks=hi - lo,
                mae=sum(seg_err) / len(seg_err),
                naive_mae=sum(seg_naive) / len(seg_naive),
                win_rate=sum(1 for e, ne in zip(seg_err, seg_naive) if e < ne) / len(seg_err),
                direction_hits=seg_dir_hits / seg_dir_total if seg_dir_total else 0.0,
                top_model=max(end_weights, key=lambda m: m["weight"])["name"],
            )
        )

    return ExperimentReport(
        warmup_ticks=warmup,
        warmup_days=warmup_days,
        scored_ticks=n,
        segments=segments,
        final_weights=[(m.name, m.weight) for m in ens.leaderboard()],
        overall_mae=sum(errors) / n,
        overall_naive_mae=sum(naive_errors) / n,
        overall_win_rate=wins / n,
        overall_direction_hits=dir_hits / dir_total if dir_total else 0.0,
    )
