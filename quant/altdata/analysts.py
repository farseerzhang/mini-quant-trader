"""`quant altdata analysts`: analyst rating and price-target changes from Yahoo (altdata_proposal.md 13.3, 13.4).

Each download asks Yahoo (yfinance `upgrades_downgrades`) for every pool stock and stores only
what is new: a change (stock, timestamp, firm) is one row per version of its content.

- new change: inserted, first_seen = last_seen = the download date;
- the same change again: only its last_seen moves;
- a change whose content differs: a new row (a new version); the old one keeps its last_seen;
- a change that is no longer there: kept as it was (its last_seen stops), counted as gone.

A stock whose download failed changes nothing, so its rows are not counted as gone. The first
download stores the whole history with first_seen = that day: that history is today's view, not
point in time; from then on first_seen shows when a change was first visible here, and a change
that appears later with an old timestamp is a backfill.

Timestamps are UTC (checked in 13.3); a change counts from the first session closing after it.
Research only: not part of run-daily.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

import pandas as pd
from sqlalchemy import Engine, and_, func, select, update

from quant.altdata.tables import altdata_analyst_changes, altdata_sync
from quant.config.settings import Settings
from quant.data.calendar import sessions_closing_after

SOURCE = "analysts-yahoo"
PAUSE_SECONDS = 1.0  # between stocks (81 requests per download)
KEY = ["symbol", "grade_time", "firm"]
CONTENT = ["from_grade", "to_grade", "action", "target_action", "current_target", "prior_target"]
YAHOO = {"GradeDate": "grade_time", "Firm": "firm", "FromGrade": "from_grade", "ToGrade": "to_grade",
         "Action": "action", "priceTargetAction": "target_action", "currentPriceTarget": "current_target",
         "priorPriceTarget": "prior_target"}
Fetch = Callable[[str], pd.DataFrame]
Progress = Callable[[str], None]


def yahoo_fetch(symbol: str) -> pd.DataFrame:
    import yfinance as yf

    frame = yf.Ticker(symbol).upgrades_downgrades
    return pd.DataFrame() if frame is None else frame.reset_index()


def normalize(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Yahoo's rows as stored: naive UTC times, empty text and zero targets as NULL, one row per key."""
    columns = [*KEY, *CONTENT]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    missing = sorted(set(YAHOO) - set(frame.columns))
    if missing:
        raise ValueError(f"Yahoo's upgrades_downgrades lacks {', '.join(missing)}")
    out = frame.rename(columns=YAHOO)[list(YAHOO.values())].copy()
    times = pd.to_datetime(out["grade_time"])
    out["grade_time"] = times.dt.tz_convert("UTC").dt.tz_localize(None) if times.dt.tz is not None else times
    for column in ("firm", "from_grade", "to_grade", "action", "target_action"):
        text = out[column].astype("string").str.strip()
        out[column] = text.where(text.str.len() > 0)
    for column in ("current_target", "prior_target"):
        values = pd.to_numeric(out[column], errors="coerce")
        out[column] = values.where(values > 0)
    out["symbol"] = symbol
    out = out.dropna(subset=["grade_time", "firm"]).drop_duplicates(KEY, keep="first")
    return out[columns].sort_values(KEY).reset_index(drop=True)


def content_hash(row) -> str:
    def text(value):
        return "" if value is None or pd.isna(value) else (f"{value:.6g}" if isinstance(value, float) else str(value))

    return hashlib.sha1("|".join(text(row[c]) for c in CONTENT).encode()).hexdigest()[:16]


@dataclass
class AnalystUpdate:
    day: date
    symbols_ok: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    seen: int = 0  # changes in today's download
    new: int = 0
    changed: int = 0  # new versions of known changes
    gone: int = 0  # known changes missing from today's download

    @property
    def ok(self) -> bool:
        return not self.failed


def _stored(engine: Engine, symbol: str) -> pd.DataFrame:
    a = altdata_analyst_changes
    with engine.connect() as conn:
        rows = conn.execute(select(a.c.id, a.c.grade_time, a.c.firm, a.c.content_hash, a.c.last_seen)
                            .where(a.c.symbol == symbol)).all()
    return pd.DataFrame(rows, columns=["id", "grade_time", "firm", "content_hash", "last_seen"])


def previous_download(engine: Engine, day: date) -> date | None:
    """The date of the latest download before ``day``."""
    s = altdata_sync
    with engine.connect() as conn:
        key = conn.execute(select(func.max(s.c.key)).where(and_(s.c.source == SOURCE, s.c.key < day.isoformat()))).scalar()
    return date.fromisoformat(key) if key else None


def store_symbol(engine: Engine, settings: Settings, symbol: str, rows: pd.DataFrame, day: date,
                 previous: date | None = None) -> tuple[int, int, int]:
    """Merge one stock's download into the table; returns (new, changed, gone), where gone counts
    the changes the ``previous`` download had and this one lacks."""
    a = altdata_analyst_changes
    stored = _stored(engine, symbol)
    rows = rows.assign(content_hash=[content_hash(r) for _, r in rows.iterrows()])
    known = {(pd.Timestamp(t), f): set(g["content_hash"]) for (t, f), g in stored.groupby(["grade_time", "firm"])}
    by_version = {(pd.Timestamp(t), f, h): i for i, t, f, h in
                  stored[["id", "grade_time", "firm", "content_hash"]].itertuples(index=False)}
    seen_ids, inserts, changed = [], [], 0
    for r in rows.itertuples(index=False):
        key = (pd.Timestamp(r.grade_time), r.firm)
        if (*key, r.content_hash) in by_version:
            seen_ids.append(by_version[(*key, r.content_hash)])
            continue
        changed += key in known
        inserts.append(r)
    available = sessions_closing_after(pd.Series([r.grade_time for r in inserts], dtype="datetime64[ns]"),
                                       settings.market.calendar) if inserts else pd.Series([], dtype=object)
    with engine.begin() as conn:
        for chunk in range(0, len(seen_ids), 500):
            conn.execute(update(a).where(a.c.id.in_(seen_ids[chunk:chunk + 500])).values(last_seen=day))
        if inserts:
            conn.execute(a.insert(), [
                {"symbol": symbol, "grade_time": pd.Timestamp(r.grade_time).to_pydatetime(), "firm": r.firm,
                 **{c: (None if pd.isna(getattr(r, c)) else getattr(r, c)) for c in CONTENT},
                 "content_hash": r.content_hash, "available_on": available.iloc[i], "first_seen": day,
                 "last_seen": day}
                for i, r in enumerate(inserts)])
    gone = 0
    if previous is not None:  # changes of the previous download missing today
        today_keys = {(pd.Timestamp(r.grade_time), r.firm) for r in rows.itertuples(index=False)}
        last_round, seen = stored[stored["last_seen"] == previous], set(seen_ids)
        gone = sum(1 for i, t, f in last_round[["id", "grade_time", "firm"]].itertuples(index=False)
                   if i not in seen and (pd.Timestamp(t), f) not in today_keys)
    return len(inserts) - changed, changed, gone


def update_analysts(engine: Engine, settings: Settings, symbols: Iterable[str] | None = None,
                    fetch: Fetch = yahoo_fetch, today: date | None = None, pause: float = PAUSE_SECONDS,
                    progress: Progress | None = None) -> AnalystUpdate:
    progress = progress or (lambda line: None)
    day = today or datetime.now(timezone.utc).date()
    symbols = list(symbols or settings.market.stocks)
    result = AnalystUpdate(day)
    previous = previous_download(engine, day)
    for number, symbol in enumerate(symbols, start=1):
        try:
            rows = normalize(fetch(symbol), symbol)
            new, changed, gone = store_symbol(engine, settings, symbol, rows, day, previous)
        except Exception as exc:  # noqa: BLE001  one stock's failure must not stop the others
            result.failed[symbol] = f"{type(exc).__name__}: {exc}"[:200]
            progress(f"[{number}/{len(symbols)}] {symbol}: FAILED ({result.failed[symbol]})")
        else:
            result.symbols_ok.append(symbol)
            result.seen += len(rows)
            result.new, result.changed, result.gone = result.new + new, result.changed + changed, result.gone + gone
            if new or changed or gone:
                progress(f"[{number}/{len(symbols)}] {symbol}: {len(rows)} changes, {new} new, {changed} changed, "
                         f"{gone} gone")
        if pause and number < len(symbols):
            time.sleep(pause)
    if set(symbols) != set(settings.market.stocks):  # a partial run (--symbol) is not a download of the pool
        return result
    with engine.begin() as conn:
        s = altdata_sync
        conn.execute(s.delete().where(and_(s.c.source == SOURCE, s.c.key == day.isoformat())))
        conn.execute(s.insert().values(
            source=SOURCE, key=day.isoformat(), status="OK" if result.ok else "FAILED", filings=result.seen,
            lines=result.new + result.changed,
            message=f"{len(result.symbols_ok)} stocks; {result.new} new, {result.changed} changed, {result.gone} gone"
                    + (f"; failed: {', '.join(result.failed)}" if result.failed else ""),
            fetched_at=datetime.now(timezone.utc).replace(tzinfo=None)))
    return result


def analyst_summary(engine: Engine) -> pd.DataFrame:
    """Rows per stock: changes stored, versions, first and last timestamp, last download (read-only)."""
    a = altdata_analyst_changes
    with engine.connect() as conn:
        rows = conn.execute(select(a.c.symbol, func.count().label("rows"),
                                   func.count(func.distinct(a.c.grade_time)).label("timestamps"),
                                   func.min(a.c.grade_time).label("first"), func.max(a.c.grade_time).label("last"),
                                   func.max(a.c.last_seen).label("last_download"))
                            .group_by(a.c.symbol).order_by(a.c.symbol)).all()
    return pd.DataFrame(rows, columns=["symbol", "rows", "timestamps", "first", "last", "last_download"])
