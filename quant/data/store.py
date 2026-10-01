"""Store historical market data locally (proposal sections 6, 9, 10).

Ties the market-data provider to the local SQLite store:

    provider -> validate -> upsert(good) / quarantine(bad) -> upsert(actions)

Prices are stored exactly as traded (unadjusted). The flow is idempotent: a
second run for the same range inserts no duplicate rows and never overwrites
existing history.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from sqlalchemy import Engine

from quant.data.data_quality import validate_prices
from quant.data.provider import MarketDataProvider
from quant.persistence import queries


@dataclass
class IngestSummary:
    symbol: str
    fetched_prices: int
    stored_prices: int
    quarantined: int
    fetched_actions: int
    stored_actions: int


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

    queries.upsert_symbol(engine, symbol)

    prices = provider.get_daily_data(symbol, start, end)
    good, bad = validate_prices(symbol, prices)
    for row in good:
        row["provider"] = provider_name

    stored_prices = queries.upsert_daily_prices(engine, good)
    quarantined = queries.quarantine_rows(engine, provider_name, bad)

    actions = provider.get_corporate_actions(symbol, start, end)
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
