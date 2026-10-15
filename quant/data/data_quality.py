"""Data-quality validation for raw daily prices (section 10 of the proposal).

Rows that fail validation are returned separately so the caller can quarantine
them; they must never silently enter ``daily_prices``.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import pandas as pd


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _to_py_date(value: Any) -> date | None:
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, date):
        return value
    return None


def validate_prices(
    symbol: str, prices: pd.DataFrame
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split provider price bars into (good_rows, bad_rows).

    ``good_rows`` are ready to upsert into ``daily_prices``. ``bad_rows`` each
    carry a ``reason`` and a ``raw`` snapshot for the quarantine table.

    Expected columns: Open, High, Low, Close, Volume, indexed by date.
    """
    good: list[dict[str, Any]] = []
    bad: list[dict[str, Any]] = []
    seen_dates: set[date] = set()

    for index_value, row in prices.iterrows():
        raw = {str(k): (None if _is_missing(v) else v) for k, v in row.items()}
        bar_date = _to_py_date(index_value)

        def reject(reason: str) -> None:
            bad.append(
                {"symbol": symbol, "date": bar_date, "reason": reason, "raw": raw}
            )

        if bar_date is None:
            reject("missing or invalid date")
            continue
        if bar_date in seen_dates:
            reject("duplicate date")
            continue

        o, h, l, c = (row.get("Open"), row.get("High"), row.get("Low"), row.get("Close"))
        if any(_is_missing(v) for v in (o, h, l, c)):
            reject("missing OHLC value")
            continue
        o, h, l, c = float(o), float(h), float(l), float(c)
        if any(v <= 0 for v in (o, h, l, c)):
            reject("non-positive price")
            continue
        if h < max(o, c, l):
            reject("invalid OHLC: high is not the maximum")
            continue
        if l > min(o, c, h):
            reject("invalid OHLC: low is not the minimum")
            continue

        volume = row.get("Volume")
        if _is_missing(volume):
            volume_value: int | None = None
        else:
            volume_value = int(volume)
            if volume_value < 0:
                reject("negative volume")
                continue

        seen_dates.add(bar_date)
        good.append(
            {
                "symbol": symbol,
                "date": bar_date,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": volume_value,
            }
        )

    return good, bad
