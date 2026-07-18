import math

import pytest

from adaptivelearning.experiment import run_experiment, warmup_ticks_for_days
from adaptivelearning.fetch import FetchError, load_csv, save_csv


def make_rows(n_days=10, ticks_per_day=20):
    """Synthetic intraday rows: gentle upward ramp with a within-day wiggle."""
    rows = []
    value = 100.0
    for d in range(n_days):
        for t in range(ticks_per_day):
            value += 0.05 + 0.3 * math.sin(t / 3.0)
            rows.append((f"2026-01-{d + 1:02d} 09:{t:02d}:00", value))
    return rows


def test_warmup_ticks_counts_distinct_days():
    rows = make_rows(n_days=5, ticks_per_day=10)
    dates = [r[0] for r in rows]
    assert warmup_ticks_for_days(dates, 3) == 30
    assert warmup_ticks_for_days(dates, 5) == 50  # all of it
    assert warmup_ticks_for_days(dates, 99) == 50  # fewer days than asked


def test_run_experiment_basic():
    rows = make_rows()
    report = run_experiment(rows, warmup_days=3, n_segments=4)
    assert report.warmup_ticks == 60
    assert report.scored_ticks == len(rows) - 60
    assert sum(s.ticks for s in report.segments) == report.scored_ticks
    assert all(math.isfinite(s.mae) for s in report.segments)
    assert 0.0 <= report.overall_win_rate <= 1.0
    assert report.final_weights[0][1] >= report.final_weights[-1][1]
    # The synthetic series is predictable, so the ensemble should beat naive.
    assert report.overall_edge > 0


def test_run_experiment_rejects_tiny_data():
    rows = make_rows(n_days=4, ticks_per_day=2)
    with pytest.raises(ValueError, match="warm-up"):
        run_experiment(rows, warmup_days=4, n_segments=4)


def test_csv_roundtrip(tmp_path):
    rows = [("2026-01-01 09:30:00", 101.5), ("2026-01-01 09:31:00", 102.0)]
    path = str(tmp_path / "x.csv")
    save_csv(path, rows)
    assert load_csv(path) == rows


def test_load_csv_skips_header_and_junk(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text("datetime,close\n2026-01-01 09:30:00,101.5\nbad,line\n")
    assert load_csv(str(path)) == [("2026-01-01 09:30:00", 101.5)]


def test_load_csv_empty_raises(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text("datetime,close\n")
    with pytest.raises(FetchError):
        load_csv(str(path))
