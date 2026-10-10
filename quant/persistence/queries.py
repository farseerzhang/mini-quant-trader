"""Thin data-access layer: plain functions over SQLAlchemy Core."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import date, datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
from sqlalchemy import Engine, func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from quant.persistence import tables


def _now() -> datetime:
    return datetime.now(timezone.utc)


def upsert_symbol(engine: Engine, symbol: str, **attrs: Any) -> None:
    now = _now()
    values = {"symbol": symbol, "active": attrs.pop("active", 1), "created_at": now, "updated_at": now, **attrs}
    stmt = sqlite_insert(tables.symbols).values(**values)
    update_cols = {
        column.name: stmt.excluded[column.name]
        for column in tables.symbols.columns
        if column.name not in ("symbol", "created_at")
    }
    with engine.begin() as conn:
        conn.execute(stmt.on_conflict_do_update(index_elements=["symbol"], set_=update_cols))


def upsert_daily_prices(engine: Engine, rows: Sequence[Mapping[str, Any]]) -> int:
    rows = [dict(row) for row in rows]
    if not rows:
        return 0
    now = _now()
    for row in rows:
        row.setdefault("ingested_at", now)
    stmt = sqlite_insert(tables.daily_prices).on_conflict_do_nothing(index_elements=["symbol", "date"])
    with engine.begin() as conn:
        result = conn.execute(stmt, rows)
        return result.rowcount if result.rowcount is not None else 0


def upsert_corporate_actions(engine: Engine, rows: Sequence[Mapping[str, Any]]) -> int:
    rows = [dict(row) for row in rows]
    if not rows:
        return 0
    now = _now()
    for row in rows:
        row.setdefault("ingested_at", now)
    stmt = sqlite_insert(tables.corporate_actions).on_conflict_do_nothing(
        index_elements=["symbol", "ex_date", "action_type"]
    )
    with engine.begin() as conn:
        result = conn.execute(stmt, rows)
        return result.rowcount if result.rowcount is not None else 0


def quarantine_rows(engine: Engine, provider: str, bad_rows: Iterable[Mapping[str, Any]]) -> int:
    now = _now()
    payload = [
        {
            "symbol": item["symbol"],
            "date": item.get("date"),
            "provider": provider,
            "reason": item["reason"],
            "raw": json.dumps(item.get("raw"), default=str),
            "quarantined_at": now,
        }
        for item in bad_rows
    ]
    if not payload:
        return 0
    with engine.begin() as conn:
        conn.execute(tables.data_quarantine.insert(), payload)
    return len(payload)


def latest_stored_date(engine: Engine, symbol: str) -> date | None:
    stmt = select(func.max(tables.daily_prices.c.date)).where(tables.daily_prices.c.symbol == symbol)
    with engine.connect() as conn:
        return conn.execute(stmt).scalar_one_or_none()


def count_daily_prices(engine: Engine, symbol: str | None = None) -> int:
    stmt = select(func.count()).select_from(tables.daily_prices)
    if symbol is not None:
        stmt = stmt.where(tables.daily_prices.c.symbol == symbol)
    with engine.connect() as conn:
        return int(conn.execute(stmt).scalar_one())


def record_pipeline_run(
    engine: Engine,
    command: str,
    stage: str,
    status: str,
    data_as_of_date: date | None = None,
    message: str | None = None,
) -> None:
    now = _now()
    with engine.begin() as conn:
        conn.execute(
            tables.pipeline_runs.insert().values(
                command=command, stage=stage, status=status,
                data_as_of_date=data_as_of_date, started_at=now, finished_at=now, message=message,
            )
        )


def load_market_data(engine: Engine, symbols: Sequence[str] | None = None):
    """Load ordered price and corporate-action frames for feature calculation."""
    price_stmt = select(tables.daily_prices).order_by(tables.daily_prices.c.symbol, tables.daily_prices.c.date)
    action_stmt = select(tables.corporate_actions).order_by(
        tables.corporate_actions.c.symbol, tables.corporate_actions.c.ex_date
    )
    if symbols is not None:
        normalized = [symbol.strip().upper() for symbol in symbols]
        price_stmt = price_stmt.where(tables.daily_prices.c.symbol.in_(normalized))
        action_stmt = action_stmt.where(tables.corporate_actions.c.symbol.in_(normalized))
    with engine.connect() as conn:
        price_rows = conn.execute(price_stmt).mappings().all()
        action_rows = conn.execute(action_stmt).mappings().all()

    grouped_prices: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in price_rows:
        grouped_prices[row["symbol"]].append(dict(row))
    prices_by_symbol: dict[str, pd.DataFrame] = {}
    for symbol, rows in grouped_prices.items():
        frame = pd.DataFrame(rows).set_index("date")
        frame.index = pd.DatetimeIndex(frame.index)
        prices_by_symbol[symbol] = frame[["open", "high", "low", "close", "volume"]]

    grouped_actions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in action_rows:
        grouped_actions[row["symbol"]].append(
            {key: row[key] for key in ("ex_date", "action_type", "value")}
        )
    actions_by_symbol = {
        symbol: pd.DataFrame(rows) for symbol, rows in grouped_actions.items()
    }
    return prices_by_symbol, actions_by_symbol


def list_price_symbols(engine: Engine) -> list[str]:
    stmt = select(tables.daily_prices.c.symbol).distinct().order_by(tables.daily_prices.c.symbol)
    with engine.connect() as conn:
        return list(conn.execute(stmt).scalars())


def save_features(
    engine: Engine,
    symbol_rows: Sequence[Mapping[str, Any]],
    market_rows: Sequence[Mapping[str, Any]],
) -> tuple[int, int]:
    """Insert or refresh deterministic feature rows for the same version/date."""
    symbol_rows = [dict(row) for row in symbol_rows]
    market_rows = [dict(row) for row in market_rows]
    with engine.begin() as conn:
        symbol_count = 0
        if symbol_rows:
            stmt = sqlite_insert(tables.features)
            stmt = stmt.on_conflict_do_update(
                index_elements=["symbol", "date", "feature_version"],
                set_={column.name: stmt.excluded[column.name]
                      for column in tables.features.columns
                      if column.name not in ("symbol", "date", "feature_version")},
            )
            result = conn.execute(stmt, symbol_rows)
            symbol_count = result.rowcount or 0
        market_count = 0
        if market_rows:
            stmt = sqlite_insert(tables.market_features)
            stmt = stmt.on_conflict_do_update(
                index_elements=["date", "feature_version"],
                set_={column.name: stmt.excluded[column.name]
                      for column in tables.market_features.columns
                      if column.name not in ("date", "feature_version")},
            )
            result = conn.execute(stmt, market_rows)
            market_count = result.rowcount or 0
    return symbol_count, market_count

def load_regime_inputs(engine: Engine, feature_version: str, market_feature_version: str) -> pd.DataFrame:
    """Market features joined with SPY's per-symbol trend and momentum, indexed by date."""
    market = tables.market_features
    spy = tables.features
    stmt = (
        select(
            market.c.date,
            market.c.spy_trend,
            market.c.qqq_trend,
            market.c.vix,
            spy.c.sma50_vs_sma200.label("spy_sma50_vs_sma200"),
            spy.c.return_3m.label("spy_return_3m"),
        )
        .select_from(
            market.outerjoin(
                spy,
                (spy.c.date == market.c.date)
                & (spy.c.symbol == "SPY")
                & (spy.c.feature_version == feature_version),
            )
        )
        .where(market.c.feature_version == market_feature_version)
        .order_by(market.c.date)
    )
    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
    columns = ["spy_trend", "qqq_trend", "spy_sma50_vs_sma200", "spy_return_3m", "vix"]
    frame = pd.DataFrame([dict(row) for row in rows], columns=["date", *columns])
    frame = frame.set_index("date")
    frame.index = pd.DatetimeIndex(frame.index)
    return frame.astype(float)


def replace_regimes(
    engine: Engine, regime_method: str, model_version: str, rows: Sequence[Mapping[str, Any]]
) -> int:
    """Replace the stored history of one method and version with freshly computed rows.

    The regime is path-dependent, so a recomputed history replaces the old one
    as a whole (in one transaction) rather than being merged row by row.
    """
    now = _now()
    payload = [
        {**row, "regime_method": regime_method, "model_version": model_version, "created_at": now}
        for row in rows
    ]
    regimes = tables.market_regimes
    with engine.begin() as conn:
        conn.execute(
            regimes.delete().where(
                regimes.c.regime_method == regime_method, regimes.c.model_version == model_version
            )
        )
        if payload:
            conn.execute(regimes.insert(), payload)
    return len(payload)
