"""`real` from the broker snapshots (forward-test run 2 design, proposal Phase 10).

A run started with `quant forward-test --start` (every run since run 2) has the value source
BROKER_VALUE_SOURCE. Its `real` is the owner's own portfolio, both accounts combined, in USD:

- **Value per session:** the sum of the latest snapshot of each broker on or before the session.
  A broker without a snapshot of that session carries its last one forward, and the session is
  listed as incomplete.
- **Deposits and withdrawals** never count as returns. The daily return is
  (value at T - net deposits on T) / value at T-1 - 1 (time-weighted), and `real`'s value series
  is the start value grown by those returns, so it is comparable with the paper portfolios.
- **Three parts** (descriptive): positions in the pool, positions outside it, and cash. Each part's
  contribution to a session's return is its holdings' price change since the previous session
  (quantities of T-1 times the change of their USD unit value), divided by the value at T-1; cash
  takes the rest (interest, fees, rounding). The change in EUR/USD inside those contributions is
  reported separately as the FX contribution.
- **Holdings for the signals:** the pool positions of the latest snapshots (shares summed over both
  accounts) and the cash of both, in USD.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import Engine, select

from quant.broker.tables import broker_account_snapshots, broker_cash_flows, broker_position_snapshots
from quant.config.settings import Settings
from quant.signals.positions import CASH, MANUAL, Holdings

BROKER_VALUE_SOURCE = "brokers: Trading 212 + IBKR snapshots (combined, USD)"
BROKERS = ("trading212", "ibkr")
IN_POOL, OUTSIDE_POOL, CASH_PART = "in the pool", "outside the pool", "cash"


def is_broker_run(run) -> bool:
    return run is not None and str(run["value_source"]).startswith("brokers:")


def _accounts(engine: Engine) -> pd.DataFrame:
    a = broker_account_snapshots
    with engine.connect() as conn:
        rows = conn.execute(select(a.c.session, a.c.broker, a.c.value_usd, a.c.cash_usd, a.c.fx_to_usd,
                                   a.c.cash_account_currency, a.c.account_currency)).mappings().all()
    frame = pd.DataFrame([dict(r) for r in rows], columns=["session", "broker", "value_usd", "cash_usd",
                                                           "fx_to_usd", "cash_account_currency", "account_currency"])
    frame["session"] = pd.to_datetime(frame["session"])
    return frame


def _positions(engine: Engine) -> pd.DataFrame:
    p = broker_position_snapshots
    with engine.connect() as conn:
        rows = conn.execute(select(p)).mappings().all()
    frame = pd.DataFrame([dict(r) for r in rows], columns=[c.name for c in p.columns])
    frame["session"] = pd.to_datetime(frame["session"])
    return frame


def _flows(engine: Engine) -> pd.Series:
    f = broker_cash_flows
    with engine.connect() as conn:
        rows = conn.execute(select(f.c.session, f.c.amount_usd)).all()
    frame = pd.DataFrame(rows, columns=["session", "amount_usd"])
    frame["session"] = pd.to_datetime(frame["session"])
    return frame.groupby("session")["amount_usd"].sum()


def broker_start(engine: Engine, session: pd.Timestamp) -> tuple[float, str, list[dict]]:
    """The run's start value and start rows from the snapshots of ``session`` (both brokers needed)."""
    accounts = _accounts(engine)
    today = accounts[accounts["session"] == session]
    missing = [b for b in BROKERS if b not in set(today["broker"])]
    if missing:
        raise ValueError(f"no {' and '.join(missing)} snapshot of {session.date()}: run `quant broker sync-accounts` "
                         "(IBKR's statement of a session arrives the next morning; start before that day's open)")
    positions = _positions(engine)
    held = positions[positions["session"] == session].groupby("symbol")["quantity"].sum()
    rows = [{"symbol": symbol, "quantity": float(quantity), "cost_basis": None} for symbol, quantity in held.items()]
    rows.append({"symbol": CASH, "quantity": float(today["cash_usd"].fillna(0.0).sum()), "cost_basis": None})
    return float(today["value_usd"].sum()), BROKER_VALUE_SOURCE, rows


@dataclass
class BrokerReal:
    value: pd.Series  # the start value grown by the time-weighted returns (comparable with paper portfolios)
    account_value: pd.Series  # the actual combined USD value, deposits included
    returns: pd.Series  # daily time-weighted returns
    contributions: pd.DataFrame  # session x {in the pool, outside the pool, cash, FX (inside the others)}
    weights: pd.DataFrame  # session x pool symbol: weight of the combined value (for the followed share)
    invested: pd.Series
    incomplete: list[pd.Timestamp] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _carried(frame: pd.DataFrame, sessions: pd.DatetimeIndex, broker: str) -> tuple[pd.DataFrame, list]:
    """One broker's rows of each session: its own, else its latest earlier one (carried)."""
    own = sorted(set(frame.loc[frame["broker"] == broker, "session"]))
    if not own:
        return pd.DataFrame(columns=frame.columns), list(sessions)
    pieces, carried = [], []
    for session in sessions:
        earlier = [s for s in own if s <= session]
        if not earlier:
            carried.append(session)
            continue
        source = earlier[-1]
        if source != session:
            carried.append(session)
        pieces.append(frame[(frame["broker"] == broker) & (frame["session"] == source)].assign(session=session))
    return (pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=frame.columns)), carried


def broker_real(engine: Engine, settings: Settings, sessions: pd.DatetimeIndex, start_value: float) -> BrokerReal:
    accounts, positions, flows = _accounts(engine), _positions(engine), _flows(engine)
    account_rows, position_rows, incomplete = [], [], set()
    for broker in BROKERS:
        rows, carried = _carried(accounts, sessions, broker)
        account_rows.append(rows)
        incomplete |= set(carried)
        position_rows.append(_carried(positions, sessions, broker)[0])
    account = pd.concat(account_rows, ignore_index=True)
    held = pd.concat(position_rows, ignore_index=True)
    value = account.groupby("session")["value_usd"].sum().reindex(sessions)
    net_flows = flows.reindex(sessions).fillna(0.0)
    net_flows.iloc[0] = 0.0  # flows up to the start are inside the start value
    returns = ((value - net_flows) / value.shift(1) - 1).iloc[1:]

    held = held.assign(unit_usd=held["value_usd"] / held["quantity"].where(held["quantity"] != 0))
    rows = []
    for previous, session in zip(sessions[:-1], sessions[1:]):
        before = held[held["session"] == previous].set_index(["broker", "broker_ticker"])
        after = held[held["session"] == session].set_index(["broker", "broker_ticker"])
        unit_after = after["unit_usd"].reindex(before.index).fillna(before["unit_usd"])  # sold: no change counted
        change = before["quantity"] * (unit_after - before["unit_usd"])
        base = value.loc[previous]
        in_pool = float(change[before["in_pool"] == 1].sum()) / base
        outside = float(change[before["in_pool"] != 1].sum()) / base
        fx = _fx_contribution(account, before, previous, session) / base
        rows.append({"session": session, IN_POOL: in_pool, OUTSIDE_POOL: outside,
                     CASH_PART: float(returns.loc[session]) - in_pool - outside, "FX (inside the others)": fx})
    contributions = pd.DataFrame(rows).set_index("session") if rows else pd.DataFrame()
    pool_value = held[held["in_pool"] == 1].pivot_table(index="session", columns="symbol", values="value_usd",
                                                        aggfunc="sum").reindex(sessions).fillna(0.0)
    weights = pool_value.div(value, axis=0)
    invested = held.groupby("session")["value_usd"].sum().reindex(sessions) / value
    grown = start_value * (1 + returns.fillna(0.0)).cumprod()
    real_value = pd.concat([pd.Series([start_value], index=sessions[:1]), grown])
    warnings = []
    if incomplete:
        warnings.append(f"{len(incomplete)} session(s) with a broker snapshot carried forward (no snapshot of that "
                        "session); run `quant broker sync-accounts` after each close and the next morning")
    return BrokerReal(real_value, value, returns, contributions, weights, invested, sorted(incomplete), warnings)


def _fx_contribution(account: pd.DataFrame, before: pd.DataFrame, previous, session) -> float:
    """USD change from the exchange rate alone: non-USD holdings and non-USD cash at T-1 times the
    rate change of their currency (the account currency's rate is used for the cash)."""
    rates = account.set_index(["session", "broker"])
    total = 0.0
    for broker in BROKERS:
        if (previous, broker) not in rates.index or (session, broker) not in rates.index:
            continue
        r0, r1 = rates.loc[(previous, broker), "fx_to_usd"], rates.loc[(session, broker), "fx_to_usd"]
        if not r0 or r0 == r1:
            continue
        currency = rates.loc[(previous, broker), "account_currency"]
        mine = before[before.index.get_level_values("broker") == broker]
        local = mine[mine["price_currency"] == currency]
        total += float((local["value_usd"] / r0).sum()) * (r1 - r0)
        cash = rates.loc[(previous, broker), "cash_account_currency"]
        if cash is not None and not np.isnan(cash):
            total += float(cash) * (r1 - r0)
    return total


def broker_holdings(engine: Engine, settings: Settings, session: pd.Timestamp) -> Holdings | None:
    """The owner's holdings at ``session`` as the signals need them, from the latest snapshots."""
    accounts, positions = _accounts(engine), _positions(engine)
    accounts = accounts[accounts["session"] <= session]
    if accounts.empty:
        return None
    latest = accounts.sort_values("session").groupby("broker").tail(1)
    held = pd.concat([positions[(positions["broker"] == r.broker) & (positions["session"] == r.session)]
                      for r in latest.itertuples()], ignore_index=True)
    pool = set(settings.market.stocks)
    in_pool = held[held["in_pool"] == 1].groupby("symbol")["quantity"].sum()
    outside = held[held["in_pool"] != 1].groupby("symbol")["quantity"].sum()
    warnings = [f"{s} is not in the stock pool: not managed, no signal" for s in outside.index]
    return Holdings({s: float(q) for s, q in in_pool.items() if s in pool}, {s: None for s in in_pool.index},
                    float(latest["cash_usd"].fillna(0.0).sum()), {s: float(q) for s, q in outside.items()},
                    MANUAL, Path("broker snapshots (quant broker sync-accounts)"), tuple(warnings))
