"""`quant altdata options`: one read of the pool's option chains per session (altdata_proposal.md 13.5, 13.7).

Yahoo (yfinance `option_chain`) gives, per contract, that session's volume, the open interest,
bid / ask and its own IV; no trades and no side. Per stock and session this stores a summary row
and at most 24 contracts:

- the 30-day grid: on the expiry nearest 30 calendar days, the strike nearest to 85 .. 115% of
  spot for calls and puts (14 contracts), so our own IVs can be computed later;
- the unusual: the 5 largest volumes, and the 5 largest volume / open interest among contracts
  with a volume of at least 100 (any expiry within 60 days).

A read counts for the session whose close it follows, only until the next open; it is stored only
within altdata.options_max_hours_after_close of that close (13.7: overnight, Yahoo's open interest
and volume are reset or half-loaded), and a missed window is never filled in later. The first stored
read of a session is kept: a later read changes nothing (a stock missing from it is filled in), so
the stored values are what was known that evening. `compare_options` reads again without storing
and shows what moved, which tells when Yahoo's volume and open interest are final.

Research only: run-daily stores it as its last stage; nothing live reads these tables.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sqlalchemy import Engine, and_, select

from quant.altdata.tables import altdata_option_contracts, altdata_option_summaries, altdata_sync
from quant.config.settings import Settings
from quant.data.calendar import last_completed_session, next_open_after, session_close
from quant.persistence import queries

SOURCE = "options-yahoo"
NEAR_DAYS = 60  # expiries read, in calendar days after the session
IV_TARGET_DAYS = 30
GRID = (0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15)  # strike / spot
TOP = 5
MIN_UNUSUAL_VOLUME = 100
THREADS = 4
VALID_IV = (0.01, 5.0)
CONTRACT_COLUMNS = ["contract_symbol", "expiry", "side", "strike", "bid", "ask", "last_price", "last_trade_at",
                    "volume", "open_interest", "yahoo_iv"]
YAHOO = {"contractSymbol": "contract_symbol", "strike": "strike", "bid": "bid", "ask": "ask",
         "lastPrice": "last_price", "lastTradeDate": "last_trade_at", "volume": "volume",
         "openInterest": "open_interest", "impliedVolatility": "yahoo_iv"}


@dataclass(frozen=True)
class Chains:
    spot: float | None  # Yahoo's last price; the stored close of the session is used when there is one
    contracts: pd.DataFrame  # Yahoo's columns plus `expiry` and `side`


Fetch = Callable[[str, date], Chains]
Progress = Callable[[str], None]


class OutsideWindowError(ValueError):
    """The market is open (a read now would have partial volumes), or, for storing, the session's
    read window after its close has passed."""


def options_session(now: datetime, calendar_name: str, max_hours_after_close: float | None = None) -> date:
    """The session a read at ``now`` belongs to: the last one closed, until the next open. With
    ``max_hours_after_close`` (storing reads), only until that long after its close (13.7)."""
    session = last_completed_session(now, calendar_name)
    opens = next_open_after(session, calendar_name)
    if now >= opens:
        raise OutsideWindowError(f"the market is open ({opens:%Y-%m-%d %H:%M} UTC .. the close): option volumes "
                                 "are partial; read after the close")
    if max_hours_after_close is not None:
        window_end = session_close(session, calendar_name) + timedelta(hours=max_hours_after_close)
        if now > window_end:
            raise OutsideWindowError(f"too late for the session of {session} (its read window closed at "
                                     f"{window_end:%Y-%m-%d %H:%M} UTC); no option data is stored for it")
    return session


def yahoo_fetch(symbol: str, session: date) -> Chains:
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    frames = []
    for expiry in ticker.options:
        if (date.fromisoformat(expiry) - session).days > NEAR_DAYS:
            continue
        chain = ticker.option_chain(expiry)
        frames += [chain.calls.assign(expiry=expiry, side="call"), chain.puts.assign(expiry=expiry, side="put")]
    try:
        spot = float(ticker.fast_info["last_price"])
    except Exception:  # noqa: BLE001  the stored close is used instead
        spot = None
    return Chains(spot, pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Yahoo's contracts as stored: dates, naive UTC trade times, no volume as NaN (no trade)."""
    if frame.empty:
        return pd.DataFrame(columns=CONTRACT_COLUMNS)
    missing = sorted(set(YAHOO) - set(frame.columns))
    if missing:
        raise ValueError(f"Yahoo's option chain lacks {', '.join(missing)}")
    out = frame.rename(columns=YAHOO)[[*YAHOO.values(), "expiry", "side"]].copy()
    out["expiry"] = pd.to_datetime(out["expiry"]).dt.date
    times = pd.to_datetime(out["last_trade_at"], utc=True)
    out["last_trade_at"] = times.dt.tz_localize(None)
    for column in ("strike", "bid", "ask", "last_price", "volume", "open_interest", "yahoo_iv"):
        out[column] = pd.to_numeric(out[column], errors="coerce").astype(float)
    out.loc[out["volume"] <= 0, "volume"] = np.nan
    return out.dropna(subset=["contract_symbol", "strike"]).drop_duplicates("contract_symbol")[CONTRACT_COLUMNS]


def iv_expiry(contracts: pd.DataFrame, session: date) -> date | None:
    if contracts.empty:
        return None
    expiries = sorted(set(contracts["expiry"]))
    return min(expiries, key=lambda e: (abs((e - session).days - IV_TARGET_DAYS), e))


def _mean_iv(rows: pd.DataFrame) -> float | None:
    valid = rows.loc[rows["volume"].notna() & rows["yahoo_iv"].between(*VALID_IV), "yahoo_iv"]
    return float(valid.mean()) if len(valid) else None


def summarize(contracts: pd.DataFrame, spot: float, session: date) -> dict:
    expiry = iv_expiry(contracts, session)
    near = contracts[contracts["expiry"] == expiry].assign(moneyness=lambda f: f["strike"] / spot)
    calls, puts = near[near["side"] == "call"], near[near["side"] == "put"]
    side = contracts["side"]
    return {
        "spot": spot, "expiries": int(contracts["expiry"].nunique()), "contracts": len(contracts),
        "traded_contracts": int(contracts["volume"].notna().sum()),
        "call_volume": float(contracts.loc[side == "call", "volume"].sum()),
        "put_volume": float(contracts.loc[side == "put", "volume"].sum()),
        "call_open_interest": float(contracts.loc[side == "call", "open_interest"].sum()),
        "put_open_interest": float(contracts.loc[side == "put", "open_interest"].sum()),
        "iv_expiry": expiry,
        "atm_call_iv": _mean_iv(calls[calls["moneyness"].between(0.95, 1.05)]),
        "atm_put_iv": _mean_iv(puts[puts["moneyness"].between(0.95, 1.05)]),
        "otm_put_iv": _mean_iv(puts[puts["moneyness"].between(0.80, 0.95, inclusive="left")]),
        "otm_call_iv": _mean_iv(calls[calls["moneyness"].between(1.05, 1.20, inclusive="right")]),
    }


def select_contracts(contracts: pd.DataFrame, spot: float, session: date) -> pd.DataFrame:
    """The grid and the unusual contracts, each once, with why it was kept."""
    if contracts.empty:
        return contracts.assign(in_grid=[], top_volume=[], top_volume_over_oi=[])
    near = contracts[contracts["expiry"] == iv_expiry(contracts, session)]
    grid = set()
    for side in ("call", "put"):
        strikes = near[near["side"] == side]
        if strikes.empty:
            continue
        distance = (strikes["strike"] / spot).to_numpy()
        grid |= {strikes.index[int(np.argmin(np.abs(distance - target)))] for target in GRID}
    traded = contracts[contracts["volume"].notna()]
    top_volume = set(traded.nlargest(TOP, "volume").index)
    large = traded[traded["volume"] >= MIN_UNUSUAL_VOLUME]
    ratio = large["volume"] / large["open_interest"].fillna(0).clip(lower=1)
    top_ratio = set(ratio.nlargest(TOP).index)
    kept = contracts.loc[sorted(grid | top_volume | top_ratio)].copy()
    kept["in_grid"] = kept.index.isin(grid).astype(int)
    kept["top_volume"] = kept.index.isin(top_volume).astype(int)
    kept["top_volume_over_oi"] = kept.index.isin(top_ratio).astype(int)
    return kept.reset_index(drop=True)


def _none(value):
    return None if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NaT else value


def stored_symbols(engine: Engine, session: date) -> set[str]:
    s = altdata_option_summaries
    with engine.connect() as conn:
        return set(conn.execute(select(s.c.symbol).where(s.c.session == session)).scalars())


def store_symbol(engine: Engine, symbol: str, session: date, read_at: datetime, spot: float,
                 contracts: pd.DataFrame) -> int:
    """One stock's summary and kept contracts; returns the number of contracts kept."""
    summary = summarize(contracts, spot, session)
    kept = select_contracts(contracts, spot, session)
    with engine.begin() as conn:
        conn.execute(altdata_option_summaries.insert().values(
            session=session, symbol=symbol, read_at=read_at, **{k: _none(v) for k, v in summary.items()}))
        if len(kept):
            conn.execute(altdata_option_contracts.insert(), [
                {"session": session, "symbol": symbol,
                 **{c: _none(r[c].to_pydatetime() if c == "last_trade_at" and pd.notna(r[c]) else r[c])
                    for c in [*CONTRACT_COLUMNS, "in_grid", "top_volume", "top_volume_over_oi"]}}
                for _, r in kept.iterrows()])
    return len(kept)


def _spots(engine: Engine, symbols: list[str], session: date) -> dict[str, float]:
    closes = queries.load_raw_closes(engine, symbols, since=session)
    if closes.empty or pd.Timestamp(session) not in closes.index:
        return {}
    row = closes.loc[pd.Timestamp(session)]
    return {s: float(v) for s, v in row.items() if pd.notna(v)}


@dataclass
class OptionsUpdate:
    session: date
    stored: list[str] = field(default_factory=list)  # stocks stored by this read
    already: int = 0  # stocks the session already had (kept as they were)
    contracts_kept: int = 0
    failed: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failed


def _read(symbols: list[str], session: date, fetch: Fetch) -> dict[str, Chains | Exception]:
    def one(symbol):
        try:
            return symbol, fetch(symbol, session)
        except Exception as exc:  # noqa: BLE001  one stock's failure must not stop the others
            return symbol, exc

    with ThreadPoolExecutor(THREADS) as pool:
        return dict(pool.map(one, symbols))


def update_options(engine: Engine, settings: Settings, now: datetime | None = None, fetch: Fetch = yahoo_fetch,
                   symbols: Iterable[str] | None = None, progress: Progress | None = None) -> OptionsUpdate:
    """Store the session's first read for every stock that has none yet (raises OutsideWindowError
    while the market is open or after the session's read window, before anything is read)."""
    progress = progress or (lambda line: None)
    now = now or datetime.now(timezone.utc)
    session = options_session(now, settings.market.calendar, settings.altdata.options_max_hours_after_close)
    pool = list(symbols or settings.market.stocks)
    already = stored_symbols(engine, session)
    result = OptionsUpdate(session, already=len(already & set(pool)))
    todo = [s for s in pool if s not in already]
    spots = _spots(engine, todo, session) if todo else {}
    read_at = now.astimezone(timezone.utc).replace(tzinfo=None)
    for symbol, chains in _read(todo, session, fetch).items():
        try:
            if isinstance(chains, Exception):
                raise chains
            spot = spots.get(symbol) or chains.spot
            if not spot:
                raise ValueError("no close of the session stored and no price from Yahoo")
            result.contracts_kept += store_symbol(engine, symbol, session, read_at, spot, normalize(chains.contracts))
            result.stored.append(symbol)
        except Exception as exc:  # noqa: BLE001
            result.failed[symbol] = f"{type(exc).__name__}: {exc}"[:200]
            progress(f"{symbol}: FAILED ({result.failed[symbol]})")
    if symbols is None:  # a read of the whole pool
        _log(engine, result, len(pool))
    return result


def _log(engine: Engine, result: OptionsUpdate, pool_size: int) -> None:
    s = altdata_sync
    have = result.already + len(result.stored)
    with engine.begin() as conn:
        previous = conn.execute(select(s.c.lines).where(and_(s.c.source == SOURCE, s.c.key == result.session.isoformat()))
                                ).scalar() or 0
        conn.execute(s.delete().where(and_(s.c.source == SOURCE, s.c.key == result.session.isoformat())))
        conn.execute(s.insert().values(
            source=SOURCE, key=result.session.isoformat(), status="OK" if have == pool_size else "FAILED",
            filings=have, lines=previous + result.contracts_kept,
            message=f"{have}/{pool_size} stocks stored for the session; this read: {len(result.stored)} new"
                    + (f"; failed: {', '.join(result.failed)}" if result.failed else ""),
            fetched_at=datetime.now(timezone.utc).replace(tzinfo=None)))


def compare_options(engine: Engine, settings: Settings, now: datetime | None = None, fetch: Fetch = yahoo_fetch,
                    symbols: Iterable[str] | None = None) -> tuple[date, pd.DataFrame]:
    """Read the chains again (nothing stored) and show, per stock, what moved since the stored read
    of the same session: total volume and open interest, and how many kept contracts changed."""
    now = now or datetime.now(timezone.utc)
    session = options_session(now, settings.market.calendar)
    s, c = altdata_option_summaries, altdata_option_contracts
    with engine.connect() as conn:
        summaries = pd.DataFrame(conn.execute(select(s).where(s.c.session == session)).mappings().all())
        kept = pd.DataFrame(conn.execute(select(c.c.symbol, c.c.contract_symbol, c.c.volume, c.c.open_interest)
                                         .where(c.c.session == session)).mappings().all())
    if summaries.empty:
        raise ValueError(f"no option chains stored for {session}; run `quant altdata options` first")
    wanted = [x for x in (symbols or summaries["symbol"]) if x in set(summaries["symbol"])]
    rows = []
    for symbol, chains in _read(wanted, session, fetch).items():
        before = summaries.set_index("symbol").loc[symbol]
        if isinstance(chains, Exception):
            rows.append({"symbol": symbol, "error": str(chains)[:80]})
            continue
        now_contracts = normalize(chains.contracts).set_index("contract_symbol")
        old = kept[kept["symbol"] == symbol].set_index("contract_symbol")
        both = old.join(now_contracts[["volume", "open_interest"]], rsuffix="_now", how="inner")
        rows.append({
            "symbol": symbol, "stored_read_at": before["read_at"],
            "volume_stored": before["call_volume"] + before["put_volume"],
            "volume_now": float(now_contracts["volume"].sum()),
            "oi_stored": before["call_open_interest"] + before["put_open_interest"],
            "oi_now": float(now_contracts["open_interest"].sum()),
            "kept_contracts": len(old), "found_again": len(both),
            "volume_changed": int((both["volume"].fillna(0) != both["volume_now"].fillna(0)).sum()),
            "oi_changed": int((both["open_interest"].fillna(0) != both["open_interest_now"].fillna(0)).sum()),
        })
    return session, pd.DataFrame(rows)
