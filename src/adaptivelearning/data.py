"""Loading numeric series from CSV or plain-text files.

Accepts:
- one number per line (plain text)
- CSV with or without a header row
- CSV with several columns: pick one with --column (name or 0-based index),
  otherwise the last column that parses as numeric is used (dates and labels
  usually come first, the measurement last).
"""

from __future__ import annotations

import csv
import io
import sys
from typing import List, Optional


class DataError(Exception):
    pass


def _to_float(text: str) -> Optional[float]:
    text = text.strip().replace(",", "")
    if not text:
        return None
    if text.startswith("$"):
        text = text[1:]
    try:
        return float(text)
    except ValueError:
        return None


def load_series(path: str, column: Optional[str] = None) -> List[float]:
    """Read a numeric series from `path` ('-' for stdin)."""
    if path == "-":
        content = sys.stdin.read()
    else:
        with open(path, "r", encoding="utf-8-sig") as fh:
            content = fh.read()
    rows = [row for row in csv.reader(io.StringIO(content)) if any(c.strip() for c in row)]
    if not rows:
        raise DataError(f"{path}: no data found")

    header: Optional[List[str]] = None
    first_numeric = [_to_float(c) for c in rows[0]]
    if not any(v is not None for v in first_numeric):
        header = [c.strip() for c in rows[0]]
        rows = rows[1:]
        if not rows:
            raise DataError(f"{path}: header only, no data rows")

    col_idx = _resolve_column(column, header, rows)
    series: List[float] = []
    for lineno, row in enumerate(rows, start=2 if header else 1):
        if col_idx >= len(row):
            continue
        value = _to_float(row[col_idx])
        if value is None:
            raise DataError(
                f"{path}: non-numeric value {row[col_idx]!r} in column {col_idx} at line {lineno}"
            )
        series.append(value)
    if not series:
        raise DataError(f"{path}: no numeric values found in column {col_idx}")
    return series


def _resolve_column(column: Optional[str], header: Optional[List[str]], rows) -> int:
    width = max(len(r) for r in rows)
    if column is not None:
        if header is not None:
            lowered = [h.lower() for h in header]
            if column.lower() in lowered:
                return lowered.index(column.lower())
        if column.isdigit() and int(column) < width:
            return int(column)
        available = f" (available: {', '.join(header)})" if header else ""
        raise DataError(f"column {column!r} not found{available}")
    # Default: last column that is numeric in the first data row.
    for idx in range(len(rows[0]) - 1, -1, -1):
        if _to_float(rows[0][idx]) is not None:
            return idx
    raise DataError("no numeric column found in first data row")
