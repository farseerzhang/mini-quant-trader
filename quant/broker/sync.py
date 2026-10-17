"""`quant broker sync-accounts`: store today's read-only snapshot of each broker account.

Rules (forward-test run 2 design, proposal Phase 10):

- **Session.** IBKR's Flex statement names its own date (the session it values). Trading 212
  answers with live prices, so its snapshot belongs to the latest completed NYSE session and is
  stored only between that session's close and 07:00 UTC of the next weekday, before European
  exchanges open and move the prices of its Xetra listings.
- **USD.** Each position is valued in USD: Trading 212's price times the rate of the instrument's
  currency (`<CCY>USD=X`, stored by update-market-data, on that session; USD is 1); IBKR's position
  value times its `fxRateToBase` (the base currency is USD). The account value is the broker's own
  total, converted the same way.
- **Cash flows.** Deposits and withdrawals are stored once each (by the broker's reference) with
  the first NYSE session on or after their date; they never count as returns.
- A rerun for the same session replaces that session's snapshot.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, and_, delete, select

from quant.broker.tables import broker_account_snapshots, broker_cash_flows, broker_position_snapshots
from quant.config.settings import Settings
from quant.data.calendar import last_completed_session, sessions_between
from quant.persistence import tables

TRADING212, IBKR = "trading212", "ibkr"
INSTRUMENTS_CACHE = Path("data/broker/trading212_instruments.json")
INSTRUMENTS_MAX_AGE_DAYS = 30
EUROPE_OPENS_UTC = time(7, 0)
FLOW_KINDS = {"DEPOSIT": "DEPOSIT", "WITHDRAW": "WITHDRAWAL", "WITHDRAWAL": "WITHDRAWAL"}


class SyncError(RuntimeError):
    pass


@dataclass
class SyncResult:
    stored: list[str] = field(default_factory=list)  # "trading212 2026-10-16: 8 positions, $12,900"
    flows_added: int = 0
    failed: dict[str, str] = field(default_factory=dict)


def fx_to_usd(engine: Engine, currency: str, session: date) -> float:
    """USD per unit of ``currency`` at the close of ``session`` (the last stored close on or before it)."""
    if currency == "USD":
        return 1.0
    symbol = f"{currency}USD=X"
    p = tables.daily_prices
    with engine.connect() as conn:
        row = conn.execute(select(p.c.date, p.c.close).where(and_(p.c.symbol == symbol, p.c.date <= session))
                           .order_by(p.c.date.desc()).limit(1)).first()
    if row is None:
        raise SyncError(f"no stored {symbol} rate; add {symbol} to market.market_symbols and run update-market-data")
    if (session - row.date).days > 5:
        raise SyncError(f"the latest stored {symbol} rate is from {row.date}, too old for {session}; "
                        "run update-market-data")
    return float(row.close)


def trading212_session(now: datetime, calendar: str) -> date:
    """The session a Trading 212 snapshot taken at ``now`` belongs to, or SyncError outside the window."""
    session = last_completed_session(now, calendar)
    following = session + timedelta(days=1)
    while following.weekday() >= 5:
        following += timedelta(days=1)
    deadline = datetime.combine(following, EUROPE_OPENS_UTC, tzinfo=timezone.utc)
    if now >= deadline:
        raise SyncError(f"Trading 212 is read with live prices, so its snapshot of {session} must be taken between "
                        f"that session's close and {deadline:%Y-%m-%d %H:%M} UTC (before European exchanges open); "
                        "run sync-accounts after the next US close")
    return session


def first_session_on_or_after(day: date, calendar: str) -> date:
    sessions = sessions_between(day, day + timedelta(days=14), calendar)
    if not sessions:
        raise SyncError(f"no NYSE session within two weeks after {day}")
    return sessions[0]


def _store_snapshot(engine: Engine, account: dict, positions: list[dict]) -> None:
    a, p = broker_account_snapshots, broker_position_snapshots
    key_a = and_(a.c.session == account["session"], a.c.broker == account["broker"])
    key_p = and_(p.c.session == account["session"], p.c.broker == account["broker"])
    with engine.begin() as conn:
        conn.execute(delete(p).where(key_p))
        conn.execute(delete(a).where(key_a))
        conn.execute(a.insert(), [account])
        if positions:
            conn.execute(p.insert(), positions)


def _store_flows(engine: Engine, flows: list[dict]) -> int:
    f = broker_cash_flows
    with engine.begin() as conn:
        known = {(r.broker, r.reference) for r in conn.execute(select(f.c.broker, f.c.reference))}
        new = [row for row in flows if (row["broker"], row["reference"]) not in known]
        if new:
            conn.execute(f.insert(), new)
    return len(new)


def _known_references(engine: Engine, broker: str) -> set[str]:
    f = broker_cash_flows
    with engine.connect() as conn:
        return {r.reference for r in conn.execute(select(f.c.reference).where(f.c.broker == broker))}


def project_symbol_trading212(ticker: str, instrument: dict | None) -> tuple[str, bool]:
    """(the project's symbol, whether it is a US listing). US listings take Trading 212's current
    short name (FB_US_EQ -> META); others keep their short name (BMWd_EQ -> BMW)."""
    from quant.broker.trading212 import trading212_symbol

    us_listing = ticker.endswith("_US_EQ")
    short = (instrument or {}).get("shortName")
    if us_listing:
        return (short.replace(".", "-") if short else trading212_symbol(ticker)), True
    return (short or ticker), False


def _instruments(client, now: datetime) -> dict[str, dict]:
    path = INSTRUMENTS_CACHE
    if path.exists() and (now.timestamp() - path.stat().st_mtime) < INSTRUMENTS_MAX_AGE_DAYS * 86400:
        return json.loads(path.read_text(encoding="utf-8"))
    instruments = client.instruments()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(instruments), encoding="utf-8")
    return instruments


def sync_trading212(engine: Engine, settings: Settings, client, now: datetime, instruments: dict | None = None
                    ) -> tuple[str, int]:
    session = trading212_session(now, settings.market.calendar)
    instruments = instruments if instruments is not None else _instruments(client, now)
    account = client.account()
    currency = account.currency or "EUR"
    account_fx = fx_to_usd(engine, currency, session)
    pool = set(settings.market.stocks)
    rows = []
    for position in account.positions:
        instrument = instruments.get(position.ticker)
        symbol, us_listing = project_symbol_trading212(position.ticker, instrument)
        price_currency = (instrument or {}).get("currencyCode") or ("USD" if us_listing else currency)
        price = position.current_price
        value_usd = position.quantity * (price or 0.0) * fx_to_usd(engine, price_currency, session)
        rows.append({"session": session, "broker": TRADING212, "broker_ticker": position.ticker, "symbol": symbol,
                     "name": (instrument or {}).get("name"), "asset_type": (instrument or {}).get("type"),
                     "in_pool": int(us_listing and symbol in pool), "quantity": position.quantity, "price": price,
                     "price_currency": price_currency, "value_usd": value_usd})
    total = account.cash_total
    if total is None:
        raise SyncError("Trading 212 returned no account total")
    _store_snapshot(engine, {
        "session": session, "broker": TRADING212, "account_id": account.account_id, "account_currency": currency,
        "fx_to_usd": account_fx, "value_account_currency": total, "cash_account_currency": account.cash_free,
        "value_usd": total * account_fx,
        "cash_usd": None if account.cash_free is None else account.cash_free * account_fx,
        "statement_date": now.date().isoformat(), "fetched_at": now.replace(tzinfo=None)}, rows)
    flows = []
    for item in client.transactions(_known_references(engine, TRADING212)):
        kind = FLOW_KINDS.get(str(item.get("type", "")).upper())
        if kind is None or item.get("amount") is None:
            continue  # interest, fees, transfers into pies: part of the return or internal
        happened = pd.Timestamp(item["dateTime"]).tz_convert("UTC").tz_localize(None)
        amount = float(item["amount"])
        amount = -abs(amount) if kind == "WITHDRAWAL" else abs(amount)
        flow_currency = item.get("currency") or currency
        flow_session = first_session_on_or_after(happened.date(), settings.market.calendar)
        flows.append({"broker": TRADING212, "reference": str(item.get("reference")),
                      "happened_at": happened.to_pydatetime(), "session": flow_session, "kind": kind,
                      "amount": amount, "currency": flow_currency,
                      "amount_usd": amount * fx_to_usd(engine, flow_currency, min(flow_session, session))})
    added = _store_flows(engine, flows)
    return (f"trading212 {session}: {len(rows)} position(s), value ${total * account_fx:,.0f} "
            f"({currency} {total:,.0f} at {account_fx:.4f})"), added


def sync_ibkr(engine: Engine, settings: Settings, client, now: datetime) -> tuple[str, int]:
    account = client.account()
    if not account.report_date:
        raise SyncError("the IBKR statement has no date")
    session = datetime.strptime(account.report_date[:8], "%Y%m%d").date()
    if session not in sessions_between(session, session, settings.market.calendar):
        raise SyncError(f"the IBKR statement date {session} is not an NYSE session")
    base = account.base_currency or "USD"
    base_fx = fx_to_usd(engine, base, session)
    pool = set(settings.market.stocks)
    rows, positions_base = [], 0.0
    for p in account.positions:
        value_base = (p.position_value or 0.0) * (p.fx_rate_to_base or 1.0)
        positions_base += value_base
        symbol = p.symbol.replace(" ", "-")
        rows.append({"session": session, "broker": IBKR, "broker_ticker": p.symbol, "symbol": symbol,
                     "name": p.description, "asset_type": p.asset_category,
                     "in_pool": int((p.asset_category in (None, "STK")) and (p.currency in (None, "USD"))
                                    and symbol in pool),
                     "quantity": p.quantity, "price": p.mark_price, "price_currency": p.currency,
                     "value_usd": value_base * base_fx})
    cash = account.ending_cash_base or 0.0
    total = positions_base + cash
    _store_snapshot(engine, {
        "session": session, "broker": IBKR, "account_id": account.account_id, "account_currency": base,
        "fx_to_usd": base_fx, "value_account_currency": total, "cash_account_currency": cash,
        "value_usd": total * base_fx, "cash_usd": cash * base_fx, "statement_date": account.report_date,
        "fetched_at": now.replace(tzinfo=None)}, rows)
    flows = []
    for flow in account.cash_flows:
        digits = "".join(ch for ch in flow.date if ch.isdigit())  # 20261015;101500 or 2026-10-15
        day = datetime.strptime(digits[:8], "%Y%m%d").date()
        flow_session = first_session_on_or_after(day, settings.market.calendar)
        kind = "DEPOSIT" if flow.amount > 0 else "WITHDRAWAL"
        amount_base = flow.amount * (flow.fx_rate_to_base or 1.0)
        flows.append({"broker": IBKR, "reference": flow.reference, "happened_at": datetime.combine(day, time()),
                      "session": flow_session, "kind": kind, "amount": flow.amount, "currency": flow.currency,
                      "amount_usd": amount_base * base_fx})
    added = _store_flows(engine, flows)
    return f"ibkr {session}: {len(rows)} position(s), value ${total * base_fx:,.0f}", added


def sync_accounts(engine: Engine, settings: Settings, only: str | None = None, now: datetime | None = None,
                  trading212_client=None, ibkr_client=None) -> SyncResult:
    """Read each broker once and store its snapshot; one broker failing does not stop the other."""
    now = now or datetime.now(timezone.utc)
    result = SyncResult()
    if only in (None, TRADING212):
        try:
            if trading212_client is None:
                from quant.broker.trading212 import Trading212Client, Trading212Credentials

                trading212_client = Trading212Client(Trading212Credentials.from_environment())
            line, added = sync_trading212(engine, settings, trading212_client, now)
            result.stored.append(line)
            result.flows_added += added
        except Exception as exc:  # noqa: BLE001  report and continue with the other broker
            result.failed[TRADING212] = f"{type(exc).__name__}: {exc}"
    if only in (None, IBKR):
        try:
            if ibkr_client is None:
                from quant.broker.ibkr_flex import IbkrFlexClient, IbkrFlexCredentials

                ibkr_client = IbkrFlexClient(IbkrFlexCredentials.from_environment())
            line, added = sync_ibkr(engine, settings, ibkr_client, now)
            result.stored.append(line)
            result.flows_added += added
        except Exception as exc:  # noqa: BLE001
            result.failed[IBKR] = f"{type(exc).__name__}: {exc}"
    return result
