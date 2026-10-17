"""The real account in the forward test (Phase 10): start positions plus a trade log.

``positions.csv`` is read once, at the first freeze, and stored as the start. After that the
holdings at the close of any session are the start plus every trade dated on or before it, from
``positions.trades_path``::

    date,symbol,side,quantity,price,fees,amount
    2026-10-13,BRK-B,SELL,9,514.80,0.50,
    2026-10-13,AMZN,BUY,2,263.10,,
    2026-11-02,AAPL,DIVIDEND,,,,2.60

- ``BUY`` / ``SELL``: shares and the fill price per share; ``fees`` (total, optional).
- ``DIVIDEND``: ``amount`` of cash received (part of the return; deposits and withdrawals are not
  tracked, by the owner's choice).
- Trades must be dated after the start session and not after the latest stored session.
- Cash follows from the trades; it may go negative (a warning, e.g. a missing deposit).

A trade added later with its real date corrects the real history; frozen targets never change.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd
from sqlalchemy import Engine

from quant.config.settings import Settings
from quant.persistence import queries
from quant.signals.positions import CASH, MANUAL, Holdings

BUY, SELL, DIVIDEND = "BUY", "SELL", "DIVIDEND"
COLUMNS = ("date", "symbol", "side", "quantity", "price", "fees", "amount")
REQUIRED = ("date", "symbol", "side")


class TradesError(ValueError):
    """The trade log cannot be used as it is; the message names the line."""


@dataclass(frozen=True)
class Trade:
    line: int
    date: pd.Timestamp
    symbol: str
    side: str
    quantity: float = 0.0
    price: float = 0.0
    fees: float = 0.0
    amount: float = 0.0

    @property
    def cash_change(self) -> float:
        if self.side == BUY:
            return -self.quantity * self.price - self.fees
        if self.side == SELL:
            return self.quantity * self.price - self.fees
        return self.amount


@dataclass(frozen=True)
class Start:
    started_on: pd.Timestamp
    quantities: Mapping[str, float]  # every stock held at the start (in the pool or not)
    cost_basis: Mapping[str, float | None]
    cash: float


def _number(text: str, what: str, line: int, positive: bool = True) -> float:
    try:
        value = float(text)
    except ValueError:
        raise TradesError(f"line {line}: {what} {text!r} is not a number") from None
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise TradesError(f"line {line}: {what} must be {'positive' if positive else 'zero or more'}")
    return value


def parse_trades(lines: Iterable[str]) -> list[Trade]:
    numbered = [(n, t) for n, t in enumerate(lines, start=1) if t.strip() and not t.lstrip().startswith("#")]
    if not numbered:
        return []
    rows = list(csv.reader([t for _, t in numbered], skipinitialspace=True))
    header = [c.strip().lower() for c in rows[0]]
    missing = [c for c in REQUIRED if c not in header]
    unknown = [c for c in header if c not in COLUMNS]
    if missing or unknown:
        raise TradesError(f"header must be {','.join(COLUMNS)} (fees and amount optional); "
                          f"missing {', '.join(missing) or '-'}, unknown {', '.join(unknown) or '-'}")
    trades = []
    for (line, _), cells in zip(numbered[1:], rows[1:]):
        if len(cells) > len(header):
            raise TradesError(f"line {line}: {len(cells)} values for {len(header)} columns")
        r = {k: v.strip() for k, v in zip(header, cells)}
        try:
            date = pd.Timestamp(r.get("date", ""))
        except ValueError:
            raise TradesError(f"line {line}: date {r.get('date')!r} is not YYYY-MM-DD") from None
        if pd.isna(date):
            raise TradesError(f"line {line}: date is empty")
        symbol, side = r.get("symbol", "").upper(), r.get("side", "").upper()
        if not symbol or symbol == CASH:
            raise TradesError(f"line {line}: symbol must be a stock (cash follows from the trades)")
        fees = _number(r["fees"], "fees", line, positive=False) if r.get("fees") else 0.0
        if side in (BUY, SELL):
            if r.get("amount"):
                raise TradesError(f"line {line}: amount is only for DIVIDEND lines")
            if not r.get("quantity") or not r.get("price"):
                raise TradesError(f"line {line}: {side} needs quantity and price")
            trades.append(Trade(line, date.normalize(), symbol, side, _number(r["quantity"], "quantity", line),
                                _number(r["price"], "price", line), fees))
        elif side == DIVIDEND:
            if r.get("quantity") or r.get("price"):
                raise TradesError(f"line {line}: DIVIDEND takes only amount (the cash received)")
            if not r.get("amount"):
                raise TradesError(f"line {line}: DIVIDEND needs amount")
            trades.append(Trade(line, date.normalize(), symbol, DIVIDEND, fees=fees,
                                amount=_number(r["amount"], "amount", line) - fees))
        else:
            raise TradesError(f"line {line}: side must be BUY, SELL or DIVIDEND, got {side!r}")
    return trades


def read_trades(path: str | Path) -> list[Trade]:
    """The trade log; a missing file means no trades yet."""
    path = Path(path)
    if not path.exists():
        return []
    try:
        return parse_trades(path.read_text(encoding="utf-8-sig").splitlines())
    except TradesError as exc:
        raise TradesError(f"{path}: {exc}") from None


def start_rows(holdings: Holdings) -> list[dict]:
    """The positions file as the stored start: every holding (in the pool or not) plus CASH."""
    rows = [{"symbol": s, "quantity": q, "cost_basis": holdings.cost_basis.get(s)}
            for s, q in {**holdings.quantities, **holdings.unmanaged}.items()]
    return rows + [{"symbol": CASH, "quantity": holdings.cash, "cost_basis": None}]


def load_start(engine: Engine, run_id: int | None = None) -> Start | None:
    """A run's start (default: the open run), or None."""
    row = queries.forward_run(engine, run_id)
    if row is None:
        return None
    frame = queries.forward_start_positions(engine, row["run_id"]).set_index("symbol")
    cash = float(frame.loc[CASH, "quantity"]) if CASH in frame.index else 0.0
    stocks = frame.drop(index=CASH, errors="ignore")
    return Start(pd.Timestamp(row["started_on"]), stocks["quantity"].astype(float).to_dict(),
                 {s: (None if pd.isna(v) else float(v)) for s, v in stocks["cost_basis"].items()}, cash)


@dataclass(frozen=True)
class Ledger:
    quantities: pd.DataFrame  # sessions x symbols, shares held at each close
    cash: pd.Series  # cash at each close
    trades: list[Trade]
    warnings: tuple[str, ...]


def replay(start: Start, trades: list[Trade], sessions: pd.DatetimeIndex) -> Ledger:
    """Holdings and cash at the close of every session from the start (pure)."""
    last = sessions[-1]
    for trade in trades:
        if trade.date <= start.started_on:
            raise TradesError(f"line {trade.line}: {trade.date.date()} is not after the start "
                              f"({start.started_on.date()}); the start positions already include it")
        if trade.date > last:
            raise TradesError(f"line {trade.line}: {trade.date.date()} is after the latest stored session "
                              f"({last.date()}); run `quant run-daily` first")
    ordered = sorted(trades, key=lambda t: (t.date, t.line))
    symbols = sorted(set(start.quantities) | {t.symbol for t in trades if t.side != DIVIDEND})
    held = {s: float(start.quantities.get(s, 0.0)) for s in symbols}
    cash, warnings, position = start.cash, [], 0
    rows, cash_rows = [], []
    for session in sessions:
        while position < len(ordered) and ordered[position].date <= session:
            trade = ordered[position]
            if trade.side == SELL and trade.quantity > held.get(trade.symbol, 0.0) + 1e-9:
                raise TradesError(f"line {trade.line}: sells {trade.quantity:g} {trade.symbol} but only "
                                  f"{held.get(trade.symbol, 0.0):g} are held on {trade.date.date()}")
            if trade.side == BUY:
                held[trade.symbol] += trade.quantity
            elif trade.side == SELL:
                held[trade.symbol] -= trade.quantity
            cash += trade.cash_change
            position += 1
        if cash < -0.005 and not any("negative" in w for w in warnings):
            warnings.append(f"cash is negative ({cash:,.2f}) on {session.date()}: a missing deposit or trade?")
        rows.append(dict(held))
        cash_rows.append(cash)
    quantities = pd.DataFrame(rows, index=sessions, columns=symbols).fillna(0.0)
    return Ledger(quantities, pd.Series(cash_rows, index=sessions), ordered, tuple(warnings))


def value_ledger(ledger: Ledger, closes: pd.DataFrame, start: Start) -> tuple[pd.DataFrame, list[str]]:
    """Per session: value, invested and each stock's market value, at as-traded closes. A stock
    without a stored close keeps its last trade price (or cost basis), with a warning."""
    warnings = []
    prices = closes.reindex(index=ledger.quantities.index, columns=ledger.quantities.columns).ffill()
    for symbol in prices.columns:
        if prices[symbol].isna().any():
            fills = [t.price for t in ledger.trades if t.symbol == symbol and t.side != DIVIDEND]
            fallback = fills[-1] if fills else start.cost_basis.get(symbol)
            warnings.append(f"{symbol}: no stored close; valued at {fallback if fallback else 0:g} "
                            "(its last trade price or cost basis)")
            prices[symbol] = prices[symbol].fillna(fallback or 0.0)
    holdings = ledger.quantities * prices
    stock_value = holdings.sum(axis=1)
    value = stock_value + ledger.cash
    frame = pd.DataFrame({"value": value, "invested": stock_value / value.where(value != 0)})
    return pd.concat([frame, holdings.add_prefix("value:")], axis=1), warnings


def real_portfolio(engine: Engine, settings: Settings, sessions: pd.DatetimeIndex, run_id: int | None = None):
    """(start, ledger, valued frame, warnings) from a run's start to the last of ``sessions``; None
    without a run. Only trades after the run's start count (earlier runs' trades stay in the log)."""
    start = load_start(engine, run_id)
    if start is None:
        return None
    sessions = sessions[sessions >= start.started_on]
    logged = read_trades(settings.positions.trades_path)
    trades = [t for t in logged if t.date > start.started_on]
    early = len(logged) - len(trades)
    if start_is_ended(engine, run_id):
        trades = [t for t in trades if t.date <= sessions[-1]]  # an ended run stops at its last session
    ledger = replay(start, trades, sessions)
    symbols = list(ledger.quantities.columns)
    closes = queries.load_raw_closes(engine, symbols, since=start.started_on.date()) if symbols else pd.DataFrame()
    valued, warnings = value_ledger(ledger, closes, start)
    if early:
        warnings.insert(0, f"{early} trade(s) dated on or before the start ({start.started_on.date()}) are left out: "
                           "the start positions (or an earlier run) already include them")
    return start, ledger, valued, list(ledger.warnings) + warnings


def start_is_ended(engine: Engine, run_id: int | None) -> bool:
    run = queries.forward_run(engine, run_id)
    return run is not None and run["ended_at"] is not None


def holdings_at(engine: Engine, settings: Settings, session: pd.Timestamp) -> Holdings | None:
    """The account at the close of ``session`` as the signals need it; None before the start."""
    from quant.forward.broker_real import broker_holdings, is_broker_run

    if is_broker_run(queries.forward_run(engine)):  # run 2 design: read from the broker snapshots
        return broker_holdings(engine, settings, session)
    start = load_start(engine)
    if start is None:
        return None
    sessions = pd.DatetimeIndex([d for d in pd.bdate_range(start.started_on, session)])
    trades = read_trades(settings.positions.trades_path)
    trades = [t for t in trades if start.started_on < t.date <= session]  # earlier runs' trades stay out
    ledger = replay(start, trades, sessions)
    pool = set(settings.market.stocks)
    last = ledger.quantities.iloc[-1]
    held = {s: float(q) for s, q in last.items() if q > 1e-9}
    managed = {s: q for s, q in held.items() if s in pool}
    unmanaged = {s: q for s, q in held.items() if s not in pool}
    warnings = [f"{s} is not in the stock pool: not managed, no signal, left out of the portfolio value"
                for s in unmanaged] + list(ledger.warnings)
    return Holdings(managed, {s: start.cost_basis.get(s) for s in managed}, float(ledger.cash.iloc[-1]),
                    unmanaged, MANUAL, Path(settings.positions.trades_path), tuple(warnings))


def followed_share(actual_weights: pd.DataFrame, live_targets: pd.DataFrame) -> pd.Series:
    """Each session: sum over stocks of min(actual weight at the close, the live target frozen the
    session before, which that day's trades were meant to reach)."""
    targets = live_targets.shift(1).reindex(actual_weights.index).dropna(how="all")
    actual = actual_weights.reindex(targets.index).reindex(columns=targets.columns.union(actual_weights.columns),
                                                           fill_value=0.0).fillna(0.0)
    targets = targets.reindex(columns=actual.columns, fill_value=0.0).fillna(0.0)
    return pd.concat([actual, targets], keys=["a", "t"]).groupby(level=1).min().sum(axis=1)


def execution_costs(trades: list[Trade], opens: pd.DataFrame) -> pd.DataFrame:
    """Each BUY / SELL against the as-traded open of its date (the fill the backtest assumes),
    in basis points (positive = worse than the open), fees included."""
    rows = []
    for t in trades:
        if t.side == DIVIDEND or t.symbol not in opens.columns or t.date not in opens.index:
            continue
        reference = float(opens.loc[t.date, t.symbol])
        if not reference or math.isnan(reference):
            continue
        slip = (t.price / reference - 1) if t.side == BUY else (reference / t.price - 1)
        notional = t.quantity * t.price
        rows.append({"date": t.date, "symbol": t.symbol, "side": t.side, "notional": notional,
                     "cost_bps": (slip + t.fees / notional) * 10_000})
    return pd.DataFrame(rows, columns=["date", "symbol", "side", "notional", "cost_bps"])
