"""`quant altdata update`: Form 4 insider transactions for the stock pool (altdata_proposal.md A1).

1. Entities: pool ticker -> CIK from company_tickers.json (all must map).
2. Backfill: every quarterly data set from altdata.sec.insider_backfill_from that is not
   loaded yet, one zip at a time (downloaded, the pool's rows kept, the zip dropped).
3. Live: once every published quarter is loaded, filings after the newest quarter's end, from
   each company's submissions list and the filing's XML. A filing whose issuer is not the pool company (e.g. Berkshire reporting
   as an owner of another company) is skipped.

Availability (section 2, decision 6): a data-set filing counts from the session after its
filing date; a live filing from the first session closing after its acceptance time.

Idempotent: a quarter is replaced as a whole when loaded again; a filing already stored from
the live feed keeps its exact acceptance time when its quarter's data set arrives later.

Quality (section 6): P / S lines with no positive price or shares, a transaction date after
the filing date, or a price more than 50% away from that day's close go to data_quarantine
and are not stored.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
from sqlalchemy import Engine, and_, case, delete, func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from quant.altdata import tables
from quant.altdata.edgar import (
    INDEX_URL,
    SUBMISSIONS_URL,
    TICKERS_URL,
    Filing,
    SecClient,
    SecError,
    dataset_links,
    document_url,
    parse_dataset,
    parse_form4_xml,
    pool_ciks,
    quarter_end,
    recent_form4,
)
from quant.config.settings import Settings
from quant.data.calendar import first_session_after, first_session_closing_after
from quant.persistence import queries

log = logging.getLogger("quant.altdata")
Progress = Callable[[str], None]
PROVIDER = "sec-form4"
PRICE_TOLERANCE = 0.5  # a P/S price more than 50% away from that day's close is quarantined


def now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


# --- entities -------------------------------------------------------------------------------

def sync_entities(engine: Engine, client: SecClient, pool: list[str]) -> dict[str, int]:
    found, missing = pool_ciks(client.json(TICKERS_URL), pool)
    if missing:
        raise SecError(f"no SEC CIK for {', '.join(missing)} in company_tickers.json")
    now = now_utc()
    with engine.begin() as conn:
        for symbol, (cik, name) in found.items():
            stmt = sqlite_insert(tables.altdata_entities).values(symbol=symbol, cik=cik, name=name, updated_at=now)
            conn.execute(stmt.on_conflict_do_update(index_elements=["symbol"],
                                                    set_={"cik": cik, "name": name, "updated_at": now}))
    return {symbol: cik for symbol, (cik, _name) in found.items()}


# --- availability and quality ---------------------------------------------------------------

class Availability:
    """Cached calendar lookups (thousands of filings share a few hundred dates)."""

    def __init__(self, calendar: str) -> None:
        self.calendar, self.after_date, self.after_moment = calendar, {}, {}

    def dataset(self, filing_date: date) -> date:
        if filing_date not in self.after_date:
            self.after_date[filing_date] = first_session_after(filing_date, self.calendar)
        return self.after_date[filing_date]

    def live(self, accepted_at: datetime) -> date:
        key = accepted_at.replace(second=0, microsecond=0)
        if key not in self.after_moment:
            self.after_moment[key] = first_session_closing_after(accepted_at.replace(tzinfo=timezone.utc),
                                                                 self.calendar)
        return self.after_moment[key]


def quality_problem(line: dict, filing_date: date, close: float | None) -> str | None:
    if line["transaction_code"] not in ("P", "S"):
        return None
    if not line["price"] or line["price"] <= 0:
        return "P/S line without a positive price"
    if not line["shares"] or line["shares"] <= 0:
        return "P/S line without positive shares"
    if line["transaction_date"] and line["transaction_date"] > filing_date:
        return "transaction date after the filing date"
    if close and abs(line["price"] / close - 1) > PRICE_TOLERANCE:
        return f"price {line['price']:g} more than 50% away from that day's close {close:g}"
    return None


def _closes(engine: Engine, symbols: list[str]) -> pd.DataFrame:
    try:
        return queries.load_raw_closes(engine, symbols)
    except (KeyError, ValueError):  # no prices stored
        return pd.DataFrame()


def _close(closes: pd.DataFrame, symbol: str, day: date | None) -> float | None:
    if day is None or closes.empty or symbol not in closes.columns:
        return None
    value = closes[symbol].get(pd.Timestamp(day))
    return None if value is None or pd.isna(value) else float(value)


# --- storing ----------------------------------------------------------------------------------

def store_filings(engine: Engine, filings: list[Filing], symbols: dict[int, str], source: str, source_key: str,
                  availability: Availability, closes: pd.DataFrame, keep_live: bool = True) -> tuple[int, int, int]:
    """Replace this source key's filings; returns (filings, lines, quarantined lines)."""
    f, o, t = tables.altdata_insider_filings, tables.altdata_insider_owners, tables.altdata_insider_transactions
    now = now_utc()
    quarantined, n_filings, n_lines = [], 0, 0
    with engine.begin() as conn:
        live = set()
        if keep_live and source == "dataset":
            accessions = [x.accession_no for x in filings]
            for start in range(0, len(accessions), 500):
                live |= {r.accession_no for r in conn.execute(select(f.c.accession_no).where(and_(
                    f.c.source == "live", f.c.accession_no.in_(accessions[start:start + 500]))))}
        old = [r.accession_no for r in conn.execute(select(f.c.accession_no).where(f.c.source_key == source_key))]
        for start in range(0, len(old), 500):
            chunk = old[start:start + 500]
            conn.execute(delete(t).where(t.c.accession_no.in_(chunk)))
            conn.execute(delete(o).where(o.c.accession_no.in_(chunk)))
            conn.execute(delete(f).where(f.c.accession_no.in_(chunk)))
        filing_rows, owner_rows, line_rows = [], [], []
        for x in filings:
            if x.accession_no in live:
                continue
            symbol = symbols[x.cik]
            available = availability.live(x.accepted_at) if x.accepted_at else availability.dataset(x.filing_date)
            filing_rows.append({"accession_no": x.accession_no, "symbol": symbol, "cik": x.cik,
                                "document_type": x.document_type, "period_of_report": x.period_of_report,
                                "date_of_original": x.date_of_original, "filing_date": x.filing_date,
                                "accepted_at": x.accepted_at, "available_on": available, "is_10b5_1": x.is_10b5_1,
                                "source": source, "source_key": source_key, "ingested_at": now})
            seen = set()
            for owner in x.owners:
                if owner["owner_cik"] not in seen:
                    seen.add(owner["owner_cik"])
                    owner_rows.append({"accession_no": x.accession_no, **owner})
            for line in x.lines:
                problem = quality_problem(line, x.filing_date, _close(closes, symbol, line["transaction_date"]))
                if problem:
                    quarantined.append({"symbol": symbol, "date": line["transaction_date"], "reason": problem,
                                        "raw": {"accession_no": x.accession_no, **line}})
                    continue
                line_rows.append({"accession_no": x.accession_no, **line})
        if filing_rows:
            conn.execute(f.insert(), filing_rows)
        if owner_rows:
            conn.execute(sqlite_insert(o).on_conflict_do_nothing(), owner_rows)
        if line_rows:
            conn.execute(t.insert(), line_rows)
        n_filings, n_lines = len(filing_rows), len(line_rows)
    if quarantined:
        queries.quarantine_rows(engine, PROVIDER, quarantined)
    return n_filings, n_lines, len(quarantined)


def record_sync(engine: Engine, source: str, key: str, status: str, filings: int, lines: int,
                message: str | None = None) -> None:
    s = tables.altdata_sync
    values = {"source": source, "key": key, "status": status, "filings": filings, "lines": lines,
              "message": message, "fetched_at": now_utc()}
    stmt = sqlite_insert(s).values(**values)
    with engine.begin() as conn:
        conn.execute(stmt.on_conflict_do_update(index_elements=["source", "key"],
                                                set_={k: stmt.excluded[k] for k in values if k not in ("source", "key")}))


def loaded_quarters(engine: Engine) -> set[str]:
    s = tables.altdata_sync
    with engine.connect() as conn:
        return {r.key for r in conn.execute(select(s.c.key).where(and_(s.c.source == "form4-dataset",
                                                                       s.c.status == "OK")))}


# --- the update -------------------------------------------------------------------------------

@dataclass
class UpdateResult:
    quarters_loaded: list[str] = field(default_factory=list)
    quarters_failed: list[tuple[str, str]] = field(default_factory=list)
    live_filings: int = 0
    live_skipped: int = 0  # not about a pool issuer
    quarantined: int = 0
    shares_rows: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.quarters_failed and not self.errors


def update_form4(engine: Engine, settings: Settings, client: SecClient, progress: Progress | None = None,
                 live: bool = True, max_quarters: int | None = None, reload: bool = False) -> UpdateResult:
    progress = progress or (lambda line: None)
    result = UpdateResult()
    pool = list(settings.market.stocks)
    ciks = sync_entities(engine, client, pool)
    # Every CIK whose filings belong to a pool stock: today's, plus earlier ones from config.
    pairs = list(ciks.items()) + [(symbol, cik) for symbol, extra in settings.altdata.sec.extra_ciks.items()
                                  if symbol in ciks for cik in extra if cik != ciks[symbol]]  # pool stocks only
    by_cik = {cik: symbol for symbol, cik in pairs}
    progress(f"Entities: {len(ciks)} pool stocks mapped to SEC CIKs"
             + (f" (+{len(pairs) - len(ciks)} earlier CIK(s) from altdata.sec.extra_ciks)" if len(pairs) > len(ciks)
                else ""))
    availability = Availability(settings.market.calendar)
    closes = _closes(engine, pool)

    links = dataset_links(client.get(INDEX_URL).decode("utf-8", "replace"))
    start = settings.altdata.sec.insider_backfill_from
    wanted = [key for key in links if quarter_end(key) >= start]
    done = set() if reload else loaded_quarters(engine)
    missing = [key for key in wanted if key not in done]
    todo = missing[:max_quarters] if max_quarters is not None else missing
    progress(f"Quarterly data sets: {len(wanted)} from {start}, {len(wanted) - len(missing)} loaded, "
             f"{len(todo)} to load this run" + (f" ({len(missing) - len(todo)} left for later runs)"
                                                 if len(todo) < len(missing) else ""))
    for number, key in enumerate(todo, start=1):
        try:
            filings = parse_dataset(client.get(links[key], timeout=300), set(by_cik))
            n_filings, n_lines, bad = store_filings(engine, filings, by_cik, "dataset", key, availability, closes)
        except (SecError, KeyError, ValueError, OSError) as exc:  # one bad quarter must not stop the others
            result.quarters_failed.append((key, str(exc)))
            record_sync(engine, "form4-dataset", key, "FAILED", 0, 0, str(exc)[:500])
            progress(f"  [{number}/{len(todo)}] {key}: FAILED ({exc})")
            continue
        record_sync(engine, "form4-dataset", key, "OK", n_filings, n_lines,
                    f"{bad} line(s) quarantined" if bad else None)
        result.quarters_loaded.append(key)
        result.quarantined += bad
        progress(f"  [{number}/{len(todo)}] {key}: {n_filings} filings, {n_lines} transaction lines"
                 + (f", {bad} quarantined" if bad else ""))

    # Live filings only fill the time after the newest published quarter, and only once every
    # quarter is loaded: otherwise "after the newest loaded quarter" could reach back years and
    # mean tens of thousands of single-filing requests.
    still_missing = [key for key in wanted if key not in loaded_quarters(engine)]
    if live and still_missing:
        progress(f"Live: skipped until the backfill is complete ({len(still_missing)} quarter(s) left)")
    elif live and wanted:
        after = quarter_end(wanted[-1])
        progress(f"Live: Form 4 filings after {after} from {len(pairs)} submissions lists")
        known = _known_accessions(engine)
        filings = []
        for symbol, cik in pairs:
            try:
                entries = recent_form4(client.json(SUBMISSIONS_URL.format(cik=cik)), after)
            except SecError as exc:
                result.errors.append(f"{symbol}: {exc}")
                continue
            for entry in entries:
                if entry["accession_no"] in known:
                    continue
                try:
                    filing = parse_form4_xml(client.get(document_url(cik, entry["accession_no"], entry["document"])),
                                             entry)
                except SecError as exc:
                    result.errors.append(f"{symbol} {entry['accession_no']}: {exc}")
                    continue
                except ET.ParseError as exc:
                    result.errors.append(f"{symbol} {entry['accession_no']}: bad XML ({exc})")
                    continue
                if filing is None or filing.cik != cik:
                    result.live_skipped += 1
                    continue
                filings.append(filing)
                known.add(entry["accession_no"])
        merged = _merge_live(engine, filings)
        n_filings, n_lines, bad = store_filings(engine, merged, by_cik, "live", "live", availability, closes)
        record_sync(engine, "form4-live", "live", "OK" if not result.errors else "FAILED", n_filings, n_lines,
                    "; ".join(result.errors)[:500] or None)
        result.live_filings = len(filings)
        result.quarantined += bad
        progress(f"  live: {len(filings)} new filings ({result.live_skipped} skipped: issuer not the pool company)"
                 + (f", {bad} lines quarantined" if bad else "") + (f", {len(result.errors)} error(s)"
                                                                     if result.errors else ""))
    if live:
        from quant.altdata.shares import update_shares

        result.shares_rows, errors = update_shares(engine, client, pairs)
        result.errors.extend(errors)
        progress(f"Shares outstanding (XBRL company facts): {result.shares_rows} values for {len(ciks)} stocks"
                 + (f", {len(errors)} error(s)" if errors else ""))
    return result


def _known_accessions(engine: Engine) -> set[str]:
    f = tables.altdata_insider_filings
    with engine.connect() as conn:
        return {r.accession_no for r in conn.execute(select(f.c.accession_no))}


def _merge_live(engine: Engine, new: list[Filing]) -> list[Filing]:
    """The live source key is replaced as a whole, so earlier live filings are reloaded with the new ones."""
    f, o, t = tables.altdata_insider_filings, tables.altdata_insider_owners, tables.altdata_insider_transactions
    with engine.connect() as conn:
        rows = conn.execute(select(f).where(f.c.source == "live")).mappings().all()
        owners = conn.execute(select(o).join(f, f.c.accession_no == o.c.accession_no)
                              .where(f.c.source == "live")).mappings().all()
        lines = conn.execute(select(t).join(f, f.c.accession_no == t.c.accession_no)
                             .where(f.c.source == "live")).mappings().all()
    old = {}
    for r in rows:
        old[r["accession_no"]] = Filing(accession_no=r["accession_no"], cik=r["cik"], document_type=r["document_type"],
                                        filing_date=r["filing_date"], period_of_report=r["period_of_report"],
                                        date_of_original=r["date_of_original"], accepted_at=r["accepted_at"],
                                        is_10b5_1=r["is_10b5_1"])
    for r in owners:
        old[r["accession_no"]].owners.append({k: r[k] for k in r.keys() if k != "accession_no"})
    for r in lines:
        old[r["accession_no"]].lines.append({k: r[k] for k in r.keys() if k != "accession_no"})
    for filing in new:
        old[filing.accession_no] = filing
    return list(old.values())


# --- coverage ---------------------------------------------------------------------------------

def coverage(engine: Engine) -> pd.DataFrame:
    """Per symbol and year: filings, P lines and S lines."""
    f, t = tables.altdata_insider_filings, tables.altdata_insider_transactions
    stmt = (select(f.c.symbol, func.strftime("%Y", f.c.filing_date).label("year"),
                   func.count(func.distinct(f.c.accession_no)).label("filings"),
                   func.sum(case((t.c.transaction_code == "P", 1), else_=0)).label("P"),
                   func.sum(case((t.c.transaction_code == "S", 1), else_=0)).label("S"))
            .select_from(f.outerjoin(t, t.c.accession_no == f.c.accession_no))
            .group_by(f.c.symbol, "year"))
    with engine.connect() as conn:
        return pd.DataFrame(conn.execute(stmt).mappings().all(), columns=["symbol", "year", "filings", "P", "S"])


def write_coverage(frame: pd.DataFrame, reports_dir: str | Path) -> Path:
    path = Path(reports_dir) / "altdata" / "coverage-form4.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Form 4 coverage, {now_utc():%Y-%m-%d %H:%M} UTC", ""]
    if frame.empty:
        lines.append("No filings stored yet.")
    else:
        years = sorted(frame["year"].unique())
        per = frame.pivot_table(index="symbol", columns="year", values="filings", aggfunc="sum", fill_value=0)
        buys = frame.groupby("year")["P"].sum()
        sells = frame.groupby("year")["S"].sum()
        lines += [f"{frame['symbol'].nunique()} stocks, {int(frame['filings'].sum()):,} filings, "
                  f"{int(frame['P'].sum()):,} open-market purchase lines (P), {int(frame['S'].sum()):,} sale lines (S).",
                  "", "## Per year (all stocks)", "", "| Year | Filings | P lines | S lines |", "|---|---|---|---|"]
        filings = frame.groupby("year")["filings"].sum()
        for year in years:
            lines.append(f"| {year} | {int(filings[year]):,} | {int(buys[year]):,} | {int(sells[year]):,} |")
        lines += ["", "## Filings per stock and year", "",
                  "| Stock | " + " | ".join(years) + " |", "|---|" + "---|" * len(years)]
        for symbol, row in per.iterrows():
            lines.append(f"| {symbol} | " + " | ".join(str(int(row.get(y, 0))) for y in years) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
