"""Fetching historical price series from Yahoo Finance's chart API.

No API key needed. Finest free resolution is 1-minute bars for the last
~7 days ("1m"/"7d"); daily bars ("1d"/"max") go back decades. True
second-by-second history only exists on paid tick feeds.

Saved CSVs have two columns: exchange-local datetime and close price, so
downstream code can group ticks into trading days.
"""

from __future__ import annotations

import csv
import json
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

YAHOO_URL = ("https://query1.finance.yahoo.com/v8/finance/chart/"
             "{symbol}?interval={interval}&range={range}")

VALID_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo"}


class FetchError(Exception):
    pass


def fetch_series(symbol: str, interval: str = "1m", range_: str = "7d") -> List[Tuple[str, float]]:
    """Return [(iso_local_datetime, close), ...] for `symbol`."""
    if interval not in VALID_INTERVALS:
        raise FetchError(f"unsupported interval {interval!r} (try 1m, 5m, 1h, 1d)")
    url = YAHOO_URL.format(symbol=urllib.parse.quote(symbol), interval=interval, range=range_)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as exc:
        raise FetchError(f"{symbol}: HTTP {exc.code} from Yahoo ({exc.reason})") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise FetchError(f"{symbol}: network error ({exc})") from exc

    chart = payload.get("chart", {})
    if chart.get("error"):
        raise FetchError(f"{symbol}: {chart['error'].get('description', 'unknown Yahoo error')}")
    results = chart.get("result") or []
    if not results:
        raise FetchError(f"{symbol}: empty response")
    result = results[0]
    timestamps = result.get("timestamp") or []
    closes = (result.get("indicators", {}).get("quote") or [{}])[0].get("close") or []
    offset = timedelta(seconds=result.get("meta", {}).get("gmtoffset", 0))

    rows: List[Tuple[str, float]] = []
    for ts, close in zip(timestamps, closes):
        if close is None:
            continue  # halted/missing bar
        local = datetime.fromtimestamp(ts, tz=timezone.utc) + offset
        rows.append((local.strftime("%Y-%m-%d %H:%M:%S"), float(close)))
    if len(rows) < 2:
        raise FetchError(f"{symbol}: got {len(rows)} usable bars")
    return rows


def save_csv(path: str, rows: List[Tuple[str, float]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["datetime", "close"])
        writer.writerows(rows)


def load_csv(path: str) -> List[Tuple[str, float]]:
    """Read a fetch-format CSV back into [(datetime, close), ...]."""
    rows: List[Tuple[str, float]] = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                rows.append((row[0], float(row[1])))
            except ValueError:
                continue  # header or junk line
    if not rows:
        raise FetchError(f"{path}: no datetime,close rows found")
    return rows
