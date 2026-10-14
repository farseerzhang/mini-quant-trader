"""`quant altdata features`: the Form 4 candidate features (altdata_proposal.md 7, 8.1).

For every pool stock and NYSE session T, from filings with available_on <= T only:

- insider_net_buy_value_60d: (P value - non-10b5-1 S value) of filings listing an officer or
  director, available in the 60 sessions up to T, / market cap at T (close x shares outstanding)
- insider_buyers_60d: distinct officers / directors on filings with a P line in that window
- opportunistic_sellers_60d: distinct officers / directors with a qualifying, opportunistic sale
- insider_sell_fraction_60d: the largest (sold / (sold + held after)) over officers / directors
  with qualifying sales of direct holdings

Routine vs opportunistic (after Cohen, Malloy and Pomorski 2012): a sale in month m of year y
is classifiable if the insider had P or S lines in each of y-1, y-2, y-3 (from filings
available before it); routine if they traded in month m in each of those years.

NULL rules (8.1): before a stock's first filing, the first two are NULL; the sale features are
NULL until three years after the stock's first filing became available; the value feature is
also NULL without a market cap. The computation is pure (frames in, frame out) so it can be
tested on hand-made filings.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import Engine, delete, select

from quant.altdata import tables
from quant.altdata.form4 import Availability
from quant.altdata.shares import latest_shares, shares_frame
from quant.config.settings import Settings
from quant.persistence import queries

WINDOW = 60
FEATURES = ("insider_net_buy_value_60d", "insider_buyers_60d", "opportunistic_sellers_60d",
            "insider_sell_fraction_60d")


@dataclass
class FilingFacts:
    """What one filing contributes, computed once."""

    pos: int
    value: float = 0.0  # P minus non-10b5-1 S value, if an officer / director is on it
    buyers: set = field(default_factory=set)
    opportunistic: set = field(default_factory=set)
    direct_sales: dict = field(default_factory=dict)  # owner -> [shares sold, held after the last sale]


def _od(owner) -> bool:
    return bool(owner.is_officer) or bool(owner.is_director)


def filing_facts(filings: pd.DataFrame, owners: pd.DataFrame, lines: pd.DataFrame,
                 sessions: pd.DatetimeIndex) -> list[FilingFacts]:
    """One stock's filings (sorted by availability) turned into window contributions."""
    session_dates = np.array(sessions.date)
    owners_by = {a: list(g.itertuples(index=False)) for a, g in owners.groupby("accession_no")}
    lines_by = {a: g.sort_values("line_no") for a, g in lines.groupby("accession_no")}
    filings = filings.sort_values(["available_on", "accession_no"])
    history: dict[int, dict] = {}  # owner -> {"years": set, "months": set of (year, month)}
    # Amendment 8.1, point 3: one trade reported in several filings of the same session counts once
    # in the value feature: key (transaction date, code, shares, price) per session.
    counted: dict[int, set] = {}
    out = []
    for f in filings.itertuples(index=False):
        pos = int(np.searchsorted(session_dates, f.available_on))
        mine = lines_by.get(f.accession_no)
        listed = owners_by.get(f.accession_no, [])
        od = [o.owner_cik for o in listed if _od(o)]
        facts = FilingFacts(pos=pos)
        trades = [] if mine is None else [(l.transaction_code, l) for l in mine.itertuples(index=False)
                                          if l.transaction_code in ("P", "S")]
        planned = f.is_10b5_1 == 1
        if od and trades:
            seen = counted.setdefault(pos, set())
            value = 0.0
            for code, l in trades:
                if not (l.shares and l.price) or (code == "S" and planned):
                    continue
                key = (l.transaction_date, code, l.shares, l.price)
                if key in seen:
                    continue  # the same trade, already counted from another filing of this session
                seen.add(key)
                value += l.shares * l.price if code == "P" else -l.shares * l.price
            facts.value = value
            if any(code == "P" for code, _ in trades):
                facts.buyers = set(od)
            qualifying = [] if planned else [l for code, l in trades if code == "S"]
            for owner in od:
                past = history.get(owner, {"years": set(), "months": set()})
                for line in qualifying:
                    when = line.transaction_date or f.available_on
                    years = (when.year - 1, when.year - 2, when.year - 3)
                    if all(y in past["years"] for y in years):
                        routine = all((y, when.month) in past["months"] for y in years)
                        if not routine:
                            facts.opportunistic.add(owner)
                direct = [l for l in qualifying if l.direct_indirect == "D" and l.shares]
                if direct:
                    held = direct[-1].shares_owned_after
                    facts.direct_sales[owner] = [sum(l.shares for l in direct),
                                                 None if held is None or pd.isna(held) else float(held)]
        for owner in [o.owner_cik for o in listed]:  # every owner's trades build their history
            past = history.setdefault(owner, {"years": set(), "months": set()})
            for _code, line in trades:
                when = line.transaction_date or f.available_on
                past["years"].add(when.year)
                past["months"].add((when.year, when.month))
        out.append(facts)
    return out


def stock_features(facts: list[FilingFacts], sessions: pd.DatetimeIndex, market_cap: pd.Series,
                   first_available: date | None) -> pd.DataFrame:
    """The four features for one stock on every session (sliding 60-session window)."""
    n = len(sessions)
    result = {name: np.full(n, np.nan) for name in FEATURES}
    if first_available is None:
        return pd.DataFrame(result, index=sessions)
    sale_start = pd.Timestamp(first_available) + pd.DateOffset(years=3)
    cap = market_cap.reindex(sessions).to_numpy(dtype=float)
    window: deque = deque()
    i = 0
    first_pos = facts[0].pos if facts else n
    for t in range(n):
        while i < len(facts) and facts[i].pos <= t:
            window.append(facts[i])
            i += 1
        while window and window[0].pos <= t - WINDOW:
            window.popleft()
        if t < first_pos:
            continue
        value = sum(f.value for f in window)
        result["insider_net_buy_value_60d"][t] = value / cap[t] if cap[t] and not np.isnan(cap[t]) else np.nan
        result["insider_buyers_60d"][t] = len(set().union(*(f.buyers for f in window))) if window else 0
        if sessions[t] < sale_start:
            continue
        result["opportunistic_sellers_60d"][t] = len(set().union(*(f.opportunistic for f in window))) if window else 0
        per_owner: dict = {}
        for f in window:
            for owner, (sold, held) in f.direct_sales.items():
                total = per_owner.setdefault(owner, [0.0, None])
                total[0] += sold
                total[1] = held  # the last sale in the window sets the holding after it
        fractions = [sold / (sold + held) for sold, held in per_owner.values() if held is not None and sold + held > 0]
        result["insider_sell_fraction_60d"][t] = max(fractions) if fractions else 0.0
    return pd.DataFrame(result, index=sessions)


def load_form4(engine: Engine) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    f, o, t = tables.altdata_insider_filings, tables.altdata_insider_owners, tables.altdata_insider_transactions
    with engine.connect() as conn:
        filings = pd.DataFrame(conn.execute(select(f.c.accession_no, f.c.symbol, f.c.available_on, f.c.is_10b5_1))
                               .mappings().all(), columns=["accession_no", "symbol", "available_on", "is_10b5_1"])
        owners = pd.DataFrame(conn.execute(select(o.c.accession_no, o.c.owner_cik, o.c.is_officer, o.c.is_director))
                              .mappings().all(), columns=["accession_no", "owner_cik", "is_officer", "is_director"])
        lines = pd.DataFrame(conn.execute(select(
            t.c.accession_no, t.c.line_no, t.c.transaction_code, t.c.transaction_date, t.c.shares, t.c.price,
            t.c.direct_indirect, t.c.shares_owned_after)).mappings().all(),
            columns=["accession_no", "line_no", "transaction_code", "transaction_date", "shares", "price",
                     "direct_indirect", "shares_owned_after"])
    return filings, owners, lines


def compute_features(engine: Engine, settings: Settings) -> pd.DataFrame:
    """Long frame (symbol, date, the four features) for every pool stock and stored session."""
    stocks = list(settings.market.stocks)
    closes = queries.load_raw_closes(engine, stocks)
    sessions = closes.index
    filings, owners, lines = load_form4(engine)
    shares = shares_frame(engine)
    availability = Availability(settings.market.calendar)
    available_on = {d: availability.dataset(d) for d in shares["filed"].unique()} if len(shares) else {}
    frames = []
    for symbol in stocks:
        mine = filings[filings["symbol"] == symbol]
        accessions = set(mine["accession_no"])
        facts = filing_facts(mine, owners[owners["accession_no"].isin(accessions)],
                             lines[lines["accession_no"].isin(accessions)], sessions)
        count = latest_shares(shares, symbol, list(sessions.date), available_on) if len(shares) else \
            pd.Series(np.nan, index=sessions)
        cap = closes[symbol] * count.reindex(sessions).to_numpy() if symbol in closes else pd.Series(np.nan, sessions)
        first = mine["available_on"].min() if len(mine) else None
        frame = stock_features(facts, sessions, cap, first)
        frames.append(frame.assign(symbol=symbol).rename_axis("date").reset_index())
    return pd.concat(frames, ignore_index=True)


def store_features(engine: Engine, frame: pd.DataFrame, version: int) -> int:
    a = tables.altdata_features
    rows = frame.assign(date=pd.to_datetime(frame["date"]).dt.date, altdata_version=version)
    rows = rows.dropna(subset=list(FEATURES), how="all")
    records = [{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()}
               for r in rows[["symbol", "date", "altdata_version", *FEATURES]].to_dict("records")]
    with engine.begin() as conn:
        conn.execute(delete(a).where(a.c.altdata_version == version))
        for start in range(0, len(records), 20000):
            conn.execute(a.insert(), records[start:start + 20000])
    return len(records)


def load_features(engine: Engine, version: int) -> dict[str, pd.DataFrame]:
    """Each feature as dates x symbols."""
    a = tables.altdata_features
    with engine.connect() as conn:
        frame = pd.DataFrame(conn.execute(select(a).where(a.c.altdata_version == version)).mappings().all(),
                             columns=[c.name for c in a.columns])
    frame["date"] = pd.to_datetime(frame["date"])
    return {name: frame.pivot(index="date", columns="symbol", values=name) for name in FEATURES}
