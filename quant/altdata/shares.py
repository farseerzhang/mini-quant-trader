"""Shares outstanding from SEC XBRL company facts (altdata_proposal.md 3.4, 8.1).

`dei:EntityCommonStockSharesOutstanding` is the count on a filing's cover page. A value is
available from the session after it was filed. A filing that reports several share classes
for the same date (e.g. GOOGL A / B / C) has one value per class; they are summed when used.
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
from sqlalchemy import Engine, delete, select

from quant.altdata import tables
from quant.altdata.edgar import SecClient, SecError

FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"


def parse_shares(facts: dict) -> list[dict]:
    units = (facts.get("facts", {}).get("dei", {}).get("EntityCommonStockSharesOutstanding", {})
             .get("units", {}).get("shares", []))
    rows, seen = [], {}
    for item in units:
        try:
            as_of = datetime.strptime(item["end"], "%Y-%m-%d").date()
            filed = datetime.strptime(item["filed"], "%Y-%m-%d").date()
            shares = float(item["val"])
        except (KeyError, ValueError, TypeError):
            continue
        if shares <= 0:
            continue
        key = (item.get("accn", ""), as_of)
        seen[key] = seen.get(key, -1) + 1
        rows.append({"accession_no": item.get("accn", ""), "as_of": as_of, "filed": filed, "seq": seen[key],
                     "shares": shares, "form": item.get("form")})
    return rows


def update_shares(engine: Engine, client: SecClient, pairs: list[tuple[str, int]]) -> tuple[int, list[str]]:
    """Replace each symbol's stored values; returns (rows, errors)."""
    s = tables.altdata_shares_outstanding
    total, errors = 0, []
    by_symbol: dict[str, list[dict]] = {}
    for symbol, cik in pairs:
        try:
            rows = parse_shares(client.json(FACTS_URL.format(cik=cik)))
        except SecError as exc:
            if "HTTP 404" in str(exc):  # no XBRL facts for this CIK (e.g. a new holding company)
                continue
            errors.append(f"{symbol} shares: {exc}")
            continue
        by_symbol.setdefault(symbol, []).extend({"symbol": symbol, "cik": cik, **row} for row in rows)
    with engine.begin() as conn:
        for symbol, rows in by_symbol.items():
            conn.execute(delete(s).where(s.c.symbol == symbol))
            unique = {(r["accession_no"], r["as_of"], r["seq"]): r for r in rows}
            if unique:
                conn.execute(s.insert(), list(unique.values()))
            total += len(unique)
    return total, errors


def shares_frame(engine: Engine) -> pd.DataFrame:
    """Total shares per (symbol, filing, date): the classes of one filing and date summed."""
    s = tables.altdata_shares_outstanding
    with engine.connect() as conn:
        frame = pd.DataFrame(conn.execute(select(s.c.symbol, s.c.accession_no, s.c.as_of, s.c.filed, s.c.shares))
                             .mappings().all(), columns=["symbol", "accession_no", "as_of", "filed", "shares"])
    if frame.empty:
        return frame
    totals = frame.groupby(["symbol", "accession_no", "as_of", "filed"], as_index=False)["shares"].sum()
    return drop_wrong_counts(totals)


NEIGHBOURHOOD_DAYS = 730  # pre-registration amendment (8.1): compare with counts within two years


def drop_wrong_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop counts below 50% or above 200% of the median of the stock's counts with an as-of date
    within two years, the count itself included (so a few wrong values cannot pull the median;
    some XBRL values are off by orders of magnitude)."""
    keep = []
    for _symbol, mine in frame.groupby("symbol"):
        as_of = pd.to_datetime(mine["as_of"]).to_numpy()
        shares = mine["shares"].to_numpy(dtype=float)
        for i in range(len(mine)):
            near = abs(as_of - as_of[i]) <= pd.Timedelta(days=NEIGHBOURHOOD_DAYS)
            median = float(pd.Series(shares[near]).median())
            if 0.5 * median <= shares[i] <= 2.0 * median:
                keep.append(mine.index[i])
    return frame.loc[sorted(keep)].reset_index(drop=True)


def latest_shares(frame: pd.DataFrame, symbol: str, sessions: list[date], available_on: dict[date, date]
                  ) -> pd.Series:
    """Per session: the newest count (by its as-of date) among values available by that session."""
    mine = frame[frame["symbol"] == symbol]
    if mine.empty:
        return pd.Series(float("nan"), index=pd.DatetimeIndex(sessions))
    mine = mine.assign(available=[available_on[d] for d in mine["filed"]]).sort_values(["available", "as_of"])
    out, best, events, i = [], None, list(mine.itertuples(index=False)), 0
    for session in sessions:
        while i < len(events) and events[i].available <= session:
            if best is None or events[i].as_of >= best[0]:
                best = (events[i].as_of, events[i].shares)
            i += 1
        out.append(best[1] if best else float("nan"))
    return pd.Series(out, index=pd.DatetimeIndex(sessions))
