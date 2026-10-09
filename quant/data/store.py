"""Store historical market data locally (proposal sections 6, 9, 10)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

from sqlalchemy import Engine

from quant.data.calendar import last_completed_session, next_session_after
from quant.data.data_quality import validate_prices
from quant.data.provider import MarketDataProvider
from quant.data.yfinance_provider import MarketDataError
from quant.persistence import queries


@dataclass
class IngestSummary:
    symbol: str
    fetched_prices: int
    stored_prices: int
    quarantined: int
    fetched_actions: int
    stored_actions: int


@dataclass
class SyncFailure:
    symbol: str
    error: str


@dataclass
class SyncReport:
    as_of: date
    summaries: list[IngestSummary]
    skipped: list[str]
    failures: list[SyncFailure]


def _actions_to_rows(symbol: str, provider_name: str, actions) -> list[dict]:
    rows: list[dict] = []
    for _, row in actions.iterrows():
        rows.append(
            {
                "symbol": symbol,
                "ex_date": row["date"],
                "action_type": row["action_type"],
                "value": float(row["value"]),
                "provider": provider_name,
            }
        )
    return rows


def store_market_data(
    engine: Engine,
    provider: MarketDataProvider,
    symbol: str,
    start: date,
    end: date,
    provider_name: str = "yfinance",
) -> IngestSummary:
    """Fetch, validate, and persist daily prices and corporate actions."""
    symbol = symbol.strip().upper()
    if start >= end:
        raise ValueError("start must be earlier than end")

    queries.upsert_symbol(engine, symbol)

    # Fetch both datasets before writing, so an API failure is safe to retry.
    prices = provider.get_daily_data(symbol, start, end)
    actions = provider.get_corporate_actions(symbol, start, end)
    good, bad = validate_prices(symbol, prices)
    for row in good:
        row["provider"] = provider_name

    stored_prices = queries.upsert_daily_prices(engine, good)
    quarantined = queries.quarantine_rows(engine, provider_name, bad)
    action_rows = _actions_to_rows(symbol, provider_name, actions)
    stored_actions = queries.upsert_corporate_actions(engine, action_rows)

    last_date = good[-1]["date"] if good else None
    queries.record_pipeline_run(
        engine,
        command="store-market-data",
        stage="ingest",
        status="OK",
        data_as_of_date=last_date,
        message=(
            f"{symbol}: stored {stored_prices} prices, "
            f"{quarantined} quarantined, {stored_actions} actions"
        ),
    )

    return IngestSummary(
        symbol=symbol,
        fetched_prices=len(prices),
        stored_prices=stored_prices,
        quarantined=quarantined,
        fetched_actions=len(action_rows),
        stored_actions=stored_actions,
    )


def sync_market_data(
    engine: Engine,
    provider: MarketDataProvider,
    symbols: list[str],
    initial_date: date,
    now: datetime | None = None,
    provider_name: str = "yfinance",
    calendar_name: str = "XNYS",
) -> SyncReport:
    """Incrementally fetch all configured symbols through the latest completed NYSE session.

    Unseen symbols start at initial_date. Existing symbols start at their next
    NYSE session after the latest stored bar. The provider end is exclusive.
    API failures are recorded per symbol and do not abort the remaining pool.
    """
    as_of = last_completed_session(now, calendar_name)
    if initial_date > as_of:
        raise ValueError("initial_date must not be after the last completed session")
    end_exclusive = as_of + timedelta(days=1)
    summaries: list[IngestSummary] = []
    skipped: list[str] = []
    failures: list[SyncFailure] = []

    normalized_symbols = list(dict.fromkeys(s.strip().upper() for s in symbols if s.strip()))
    if not normalized_symbols:
        raise ValueError("the configured symbol pool is empty")

    for symbol in normalized_symbols:
        latest = queries.latest_stored_date(engine, symbol)
        if latest is None:
            start = initial_date
        else:
            start = next_session_after(latest, as_of, calendar_name)
            if start is None:
                skipped.append(symbol)
                continue

        try:
            summary = store_market_data(
                engine, provider, symbol, start, end_exclusive, provider_name
            )
            summaries.append(summary)
        except MarketDataError as exc:
            failures.append(SyncFailure(symbol=symbol, error=str(exc)))

    return SyncReport(as_of=as_of, summaries=summaries, skipped=skipped, failures=failures)
