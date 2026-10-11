"""Current holdings (proposal section 6.8): read the positions CSV, value it at the close of T.

The file lists what you hold now, i.e. after the close of T and before the open of T+1:

    symbol,quantity,cost_basis
    AAPL,10,175.20
    CASH,25000,

- ``quantity`` is shares (fractions allowed); no short positions.
- ``cost_basis`` (per share) is optional and only stored.
- Exactly one ``CASH`` row; its quantity is the cash amount.
- Blank lines and lines starting with ``#`` are ignored.

A stock outside the configured pool is reported as not managed: it gets no signal and is
left out of the portfolio value the target weights refer to. Without a file the portfolio
is all cash with a notional value (``backtest.initial_cash``), so signals can still be read.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

CASH = "CASH"
MANUAL = "MANUAL"
REQUIRED_COLUMNS = ("symbol", "quantity")
OPTIONAL_COLUMNS = ("cost_basis",)


class PositionsError(ValueError):
    """The positions file cannot be used as it is; the message says which line and why."""


@dataclass(frozen=True)
class Holdings:
    quantities: Mapping[str, float]  # managed stocks with quantity > 0
    cost_basis: Mapping[str, float | None]
    cash: float
    unmanaged: Mapping[str, float]  # held, but outside the configured stock pool
    source: str  # MANUAL; PAPER arrives with paper trading
    path: Path | None  # None: no file, notional all-cash portfolio
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class Valuation:
    date: pd.Timestamp  # the session T the closes are from
    closes: pd.Series  # as-traded close used for each managed holding
    values: pd.Series  # quantity x close
    cash: float
    warnings: tuple[str, ...] = ()

    @property
    def total(self) -> float:
        """Portfolio value the weights refer to: managed holdings plus cash."""
        return float(self.values.sum()) + self.cash

    @property
    def weights(self) -> pd.Series:
        """Current weight per managed holding."""
        total = self.total
        return self.values / total if total > 0 else self.values * 0.0

    @property
    def cash_weight(self) -> float:
        return self.cash / self.total if self.total > 0 else 1.0


def _number(text: str, what: str, line: int) -> float:
    try:
        value = float(text)
    except ValueError:
        raise PositionsError(f"line {line}: {what} {text!r} is not a number") from None
    if not math.isfinite(value):
        raise PositionsError(f"line {line}: {what} must be a finite number")
    return value


def parse_positions(lines: Iterable[str], stocks: Iterable[str], path: Path | None = None) -> Holdings:
    """Validate the CSV rows; every problem is an error except a stock outside the pool."""
    pool = set(stocks)
    numbered = [(n, text) for n, text in enumerate(lines, start=1) if text.strip() and not text.lstrip().startswith("#")]
    if not numbered:
        raise PositionsError("the positions file is empty; it needs a header and a CASH row")
    reader = csv.reader([text for _, text in numbered], skipinitialspace=True)
    rows = list(reader)
    header = [cell.strip().lower() for cell in rows[0]]
    missing = [c for c in REQUIRED_COLUMNS if c not in header]
    if missing:
        raise PositionsError(f"header must have the columns {', '.join(REQUIRED_COLUMNS + OPTIONAL_COLUMNS)}; "
                             f"missing {', '.join(missing)}")
    unknown = [c for c in header if c not in REQUIRED_COLUMNS + OPTIONAL_COLUMNS]
    if unknown:
        raise PositionsError(f"unknown column(s) {', '.join(unknown)}; use {', '.join(REQUIRED_COLUMNS + OPTIONAL_COLUMNS)}")

    quantities: dict[str, float] = {}
    cost_basis: dict[str, float | None] = {}
    unmanaged: dict[str, float] = {}
    cash: float | None = None
    seen: dict[str, int] = {}
    warnings: list[str] = []
    for (line, _), cells in zip(numbered[1:], rows[1:]):
        if len(cells) > len(header):
            raise PositionsError(f"line {line}: {len(cells)} values for {len(header)} columns")
        record = dict(zip(header, [cell.strip() for cell in cells]))
        symbol = record.get("symbol", "").upper()
        if not symbol:
            raise PositionsError(f"line {line}: symbol is empty")
        if symbol in seen:
            raise PositionsError(f"line {line}: {symbol} is listed twice (also on line {seen[symbol]})")
        seen[symbol] = line
        if not record.get("quantity"):
            raise PositionsError(f"line {line}: {symbol} has no quantity")
        quantity = _number(record["quantity"], "quantity", line)
        if quantity < 0:
            raise PositionsError(f"line {line}: {symbol} quantity {quantity:g} is negative; short positions are not supported")
        basis_text = record.get("cost_basis", "")
        if symbol == CASH:
            if basis_text:
                raise PositionsError(f"line {line}: the CASH row has no cost_basis")
            cash = quantity
            continue
        basis = _number(basis_text, "cost_basis", line) if basis_text else None
        if basis is not None and basis < 0:
            raise PositionsError(f"line {line}: {symbol} cost_basis must not be negative")
        if quantity == 0:
            continue  # listed but not held
        if symbol not in pool:
            unmanaged[symbol] = quantity
            warnings.append(f"{symbol} is not in the stock pool: not managed, no signal, left out of the portfolio value")
            continue
        quantities[symbol] = quantity
        cost_basis[symbol] = basis
    if cash is None:
        raise PositionsError("no CASH row; add one (CASH,0, if you hold no cash)")
    return Holdings(quantities, cost_basis, cash, unmanaged, MANUAL, path, tuple(warnings))


def read_positions(path: str | Path, stocks: Iterable[str], notional_cash: float) -> Holdings:
    """Read the positions file; a missing file is an all-cash notional portfolio (with a warning)."""
    path = Path(path)
    if not path.exists():
        return Holdings({}, {}, float(notional_cash), {}, MANUAL, None, (
            f"no positions file at {path}: assuming all cash, notional value {notional_cash:,.0f} "
            "(create the file to get signals for your holdings)",
        ))
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise PositionsError(f"cannot read {path}: {exc}") from exc
    try:
        return parse_positions(text.splitlines(), stocks, path)
    except PositionsError as exc:
        raise PositionsError(f"{path}: {exc}") from None


def value_holdings(holdings: Holdings, raw_closes: pd.DataFrame, session: pd.Timestamp) -> Valuation:
    """Value each managed holding at its as-traded close of ``session`` (T).

    ``raw_closes`` is dates x symbols of stored (unadjusted) closes: positions are counted in
    real shares, so they are valued at the price they trade at. A holding without a close on
    T uses its last close before T, with a warning; one with no close at all is an error.
    """
    session = pd.Timestamp(session)
    closes, warnings = {}, []
    for symbol in holdings.quantities:
        if symbol not in raw_closes.columns:
            raise PositionsError(f"{symbol}: no stored prices; run update-market-data first")
        history = raw_closes[symbol].loc[:session].dropna()
        if history.empty:
            raise PositionsError(f"{symbol}: no close on or before {session.date()}")
        if history.index[-1] != session:
            warnings.append(f"{symbol}: no close on {session.date()}; valued at the {history.index[-1].date()} close")
        closes[symbol] = float(history.iloc[-1])
    closes = pd.Series(closes, dtype=float)
    quantities = pd.Series(dict(holdings.quantities), dtype=float).reindex(closes.index)
    return Valuation(session, closes, quantities * closes, holdings.cash, tuple(warnings))


def snapshot_rows(holdings: Holdings, as_of: date) -> list[dict]:
    """Rows for the positions table: every managed and unmanaged holding plus the CASH row."""
    rows = [
        {"symbol": s, "as_of_date": as_of, "quantity": q, "cost_basis": holdings.cost_basis.get(s), "managed": 1,
         "source": holdings.source}
        for s, q in holdings.quantities.items()
    ]
    rows += [{"symbol": s, "as_of_date": as_of, "quantity": q, "cost_basis": None, "managed": 0,
              "source": holdings.source} for s, q in holdings.unmanaged.items()]
    rows.append({"symbol": CASH, "as_of_date": as_of, "quantity": holdings.cash, "cost_basis": None, "managed": 1,
                 "source": holdings.source})
    return rows
