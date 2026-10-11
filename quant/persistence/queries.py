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
    # Only the columns given here are updated, so a bare upsert during ingestion
    # never clears the sector or asset type synchronized from config.
    update_cols = {name: stmt.excluded[name] for name in values if name not in ("symbol", "created_at")}
    with engine.begin() as conn:
        conn.execute(stmt.on_conflict_do_update(index_elements=["symbol"], set_=update_cols))


def sync_symbols(engine: Engine, rows: Sequence[Mapping[str, Any]]) -> None:
    """Make the symbols table match config (section 6.1): upsert these rows as active,
    and mark every other stored symbol inactive. History is kept either way."""
    now = _now()
    payload = [
        {"active": 1, "created_at": now, "updated_at": now, **dict(row)}
        for row in rows
    ]
    with engine.begin() as conn:
        for values in payload:
            stmt = sqlite_insert(tables.symbols).values(**values)
            update_cols = {name: stmt.excluded[name] for name in values if name not in ("symbol", "created_at")}
            conn.execute(stmt.on_conflict_do_update(index_elements=["symbol"], set_=update_cols))
        conn.execute(
            tables.symbols.update()
            .where(tables.symbols.c.symbol.not_in([values["symbol"] for values in payload]))
            .values(active=0, updated_at=now)
        )


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


def load_feature_panel(
    engine: Engine, feature_version: str, symbols: Sequence[str], columns: Sequence[str]
) -> dict[str, pd.DataFrame]:
    """Per feature column, a dates x symbols frame (symbols in the given order)."""
    feature_columns = [tables.features.c[name] for name in columns]
    stmt = (
        select(tables.features.c.symbol, tables.features.c.date, *feature_columns)
        .where(
            tables.features.c.feature_version == feature_version,
            tables.features.c.symbol.in_(list(symbols)),
        )
        .order_by(tables.features.c.date)
    )
    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
    frame = pd.DataFrame([dict(row) for row in rows], columns=["symbol", "date", *columns])
    frame["date"] = pd.DatetimeIndex(frame["date"])
    panel = {}
    for name in columns:
        wide = frame.pivot(index="date", columns="symbol", values=name)
        panel[name] = wide.reindex(columns=list(symbols)).astype(float)
    return panel


def load_regimes(engine: Engine, regime_method: str, model_version: str) -> pd.Series:
    """Stored regime per date for one method and version."""
    regimes = tables.market_regimes
    stmt = (
        select(regimes.c.date, regimes.c.regime)
        .where(regimes.c.regime_method == regime_method, regimes.c.model_version == model_version)
        .order_by(regimes.c.date)
    )
    with engine.connect() as conn:
        rows = conn.execute(stmt).all()
    return pd.Series(
        [row.regime for row in rows],
        index=pd.DatetimeIndex([row.date for row in rows]),
        name="regime",
        dtype=object,
    )


def count_pipeline_runs(engine: Engine, command: str, stage: str) -> int:
    """How many pipeline_runs rows a command/stage has, e.g. holdout evaluations."""
    stmt = select(func.count()).select_from(tables.pipeline_runs).where(
        tables.pipeline_runs.c.command == command, tables.pipeline_runs.c.stage == stage
    )
    with engine.connect() as conn:
        return int(conn.execute(stmt).scalar_one())


def save_backtest_runs(
    engine: Engine, runs: Sequence[tuple[Mapping[str, Any], Sequence[Mapping[str, Any]]]]
) -> list[int]:
    """Insert backtest runs with their equity curves in one transaction; returns run ids."""
    run_ids = []
    with engine.begin() as conn:
        for run, equity in runs:
            run_id = conn.execute(tables.backtest_runs.insert().values(**run)).inserted_primary_key[0]
            if equity:
                conn.execute(tables.backtest_equity.insert(), [{**row, "run_id": run_id} for row in equity])
            run_ids.append(run_id)
    return run_ids


def load_features_long(
    engine: Engine, feature_version: str, symbols: Sequence[str], columns: Sequence[str]
) -> pd.DataFrame:
    """Per-symbol features as rows indexed by (date, symbol)."""
    stmt = (
        select(tables.features.c.date, tables.features.c.symbol, *[tables.features.c[name] for name in columns])
        .where(tables.features.c.feature_version == feature_version, tables.features.c.symbol.in_(list(symbols)))
        .order_by(tables.features.c.date, tables.features.c.symbol)
    )
    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
    frame = pd.DataFrame([dict(row) for row in rows], columns=["date", "symbol", *columns])
    frame["date"] = pd.DatetimeIndex(frame["date"])
    return frame.set_index(["date", "symbol"]).astype(float)


def load_market_features(engine: Engine, market_feature_version: str, columns: Sequence[str]) -> pd.DataFrame:
    """Market features indexed by date."""
    stmt = (
        select(tables.market_features.c.date, *[tables.market_features.c[name] for name in columns])
        .where(tables.market_features.c.feature_version == market_feature_version)
        .order_by(tables.market_features.c.date)
    )
    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
    frame = pd.DataFrame([dict(row) for row in rows], columns=["date", *columns])
    frame["date"] = pd.DatetimeIndex(frame["date"])
    return frame.set_index("date").astype(float)


def count_holdout_evaluations(engine: Engine) -> int:
    """How many times the holdout has been evaluated, by any command (it is one shared resource)."""
    stmt = select(func.count()).select_from(tables.pipeline_runs).where(tables.pipeline_runs.c.stage == "holdout")
    with engine.connect() as conn:
        return int(conn.execute(stmt).scalar_one())


def active_model(engine: Engine, role: str):
    """The ACTIVE model_versions row of a role, or None."""
    stmt = select(tables.model_versions).where(
        tables.model_versions.c.role == role, tables.model_versions.c.status == "ACTIVE"
    )
    with engine.connect() as conn:
        return conn.execute(stmt).mappings().first()


def replace_positions(engine: Engine, as_of: date, source: str, rows: Sequence[Mapping[str, Any]]) -> int:
    """Store the holdings snapshot for one session and source, replacing any earlier one
    (a holding removed from the file must not survive a re-run)."""
    now = _now()
    payload = [{**row, "as_of_date": as_of, "source": source, "created_at": now} for row in rows]
    positions = tables.positions
    with engine.begin() as conn:
        conn.execute(positions.delete().where(positions.c.as_of_date == as_of, positions.c.source == source))
        if payload:
            conn.execute(positions.insert(), payload)
    return len(payload)


def load_positions(engine: Engine, as_of: date, source: str) -> pd.DataFrame:
    stmt = (
        select(tables.positions)
        .where(tables.positions.c.as_of_date == as_of, tables.positions.c.source == source)
        .order_by(tables.positions.c.symbol)
    )
    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
    return pd.DataFrame([dict(row) for row in rows], columns=[c.name for c in tables.positions.columns])


def load_raw_closes(engine: Engine, symbols: Sequence[str], since: date | None = None) -> pd.DataFrame:
    """Stored (as-traded, unadjusted) closes: dates x symbols."""
    prices = tables.daily_prices
    stmt = select(prices.c.date, prices.c.symbol, prices.c.close).where(prices.c.symbol.in_(list(symbols)))
    if since is not None:
        stmt = stmt.where(prices.c.date >= since)
    with engine.connect() as conn:
        rows = conn.execute(stmt).all()
    frame = pd.DataFrame(rows, columns=["date", "symbol", "close"])
    wide = frame.pivot(index="date", columns="symbol", values="close")
    wide.index = pd.DatetimeIndex(wide.index)
    return wide.sort_index().astype(float)


def replace_signals(
    engine: Engine, session: date, strategy: str, model_version: str, rows: Sequence[Mapping[str, Any]]
) -> int:
    """Store one session's signals for a strategy and version, replacing any earlier run of it
    (a stock that no longer has a signal must not keep yesterday's row)."""
    now = _now()
    payload = [
        {**row, "date": session, "strategy": strategy, "model_version": model_version, "created_at": now}
        for row in rows
    ]
    signals = tables.signals
    with engine.begin() as conn:
        conn.execute(signals.delete().where(
            signals.c.date == session, signals.c.strategy == strategy, signals.c.model_version == model_version
        ))
        if payload:
            conn.execute(signals.insert(), payload)
    return len(payload)


def load_signals(engine: Engine, session: date, strategy: str | None = None) -> pd.DataFrame:
    stmt = select(tables.signals).where(tables.signals.c.date == session).order_by(tables.signals.c.symbol)
    if strategy is not None:
        stmt = stmt.where(tables.signals.c.strategy == strategy)
    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
    return pd.DataFrame([dict(row) for row in rows], columns=[c.name for c in tables.signals.columns])
