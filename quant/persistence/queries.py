"""Thin data-access layer: plain functions over SQLAlchemy Core."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import date, datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
from sqlalchemy import Engine, func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from quant.persistence import tables


log = logging.getLogger("quant.pipeline")
LEVELS = {"OK": logging.INFO, "SKIPPED": logging.INFO, "STARTED": logging.INFO, "FAILED": logging.ERROR}


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


def _quarantine_date(value: Any) -> date | None:
    """The date as the Date column stores it, so a new row compares equal to a stored one."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def quarantine_rows(engine: Engine, provider: str, bad_rows: Iterable[Mapping[str, Any]]) -> int:
    """Store the bad rows; returns how many were added. A row whose (provider, symbol, date, reason,
    raw) is already stored, or repeated within this call, is skipped, so re-loading the same data
    does not quarantine it again (altdata section 13.8)."""
    quarantine = tables.data_quarantine
    now = _now()
    payload = [
        {
            "symbol": item["symbol"],
            "date": _quarantine_date(item.get("date")),
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
        symbols = sorted({row["symbol"] for row in payload})
        seen: set[tuple] = set()
        for start in range(0, len(symbols), 500):
            stored = select(quarantine.c.symbol, quarantine.c.date, quarantine.c.reason, quarantine.c.raw).where(
                quarantine.c.provider == provider, quarantine.c.symbol.in_(symbols[start:start + 500]))
            seen |= {tuple(row) for row in conn.execute(stored)}
        new_rows = []
        for row in payload:
            key = (row["symbol"], row["date"], row["reason"], row["raw"])
            if key not in seen:
                seen.add(key)
                new_rows.append(row)
        if new_rows:
            conn.execute(quarantine.insert(), new_rows)
    return len(new_rows)


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
    log.log(LEVELS.get(status, logging.INFO), "%s/%s %s%s: %s", command, stage, status,
            f" (data to {data_as_of_date})" if data_as_of_date else "", message or "-")
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


def latest_price_dates(engine: Engine) -> dict[str, date]:
    """Last stored session per symbol."""
    prices = tables.daily_prices
    stmt = select(prices.c.symbol, func.max(prices.c.date)).group_by(prices.c.symbol)
    with engine.connect() as conn:
        return {symbol: last for symbol, last in conn.execute(stmt).all()}


def latest_pipeline_runs(engine: Engine) -> pd.DataFrame:
    """The most recent pipeline_runs row of every (command, stage, status)."""
    runs = tables.pipeline_runs
    latest = (
        select(func.max(runs.c.run_id).label("run_id"))
        .group_by(runs.c.command, runs.c.stage, runs.c.status)
        .subquery()
    )
    stmt = select(runs).where(runs.c.run_id.in_(select(latest.c.run_id))).order_by(runs.c.run_id)
    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
    return pd.DataFrame([dict(row) for row in rows], columns=[c.name for c in runs.columns])


def count_quarantined(engine: Engine, since: datetime | None = None, provider: str | None = None) -> int:
    quarantine = tables.data_quarantine
    stmt = select(func.count()).select_from(quarantine)
    if since is not None:
        stmt = stmt.where(quarantine.c.quarantined_at >= since)
    if provider is not None:
        stmt = stmt.where(quarantine.c.provider == provider)
    with engine.connect() as conn:
        return int(conn.execute(stmt).scalar_one())


def latest_date(engine: Engine, column, *conditions) -> date | None:
    """max(column) over the rows matching ``conditions``."""
    stmt = select(func.max(column))
    for condition in conditions:
        stmt = stmt.where(condition)
    with engine.connect() as conn:
        return conn.execute(stmt).scalar_one()


def model_counts(engine: Engine) -> pd.DataFrame:
    """Registered models per role and status, with how many passed their gates."""
    models = tables.model_versions
    stmt = select(
        models.c.role, models.c.status, func.count().label("models"),
        func.sum(models.c.gates_passed).label("gates_passed"),
    ).group_by(models.c.role, models.c.status)
    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
    return pd.DataFrame([dict(row) for row in rows], columns=["role", "status", "models", "gates_passed"])


def replace_predictions(engine: Engine, model_version: str, since: date, rows: Sequence[Mapping[str, Any]]) -> int:
    """Store a model's predictions from ``since`` on, replacing what was stored for those sessions."""
    now = _now()
    payload = [{**row, "model_version": model_version, "created_at": now} for row in rows]
    predictions = tables.predictions
    with engine.begin() as conn:
        conn.execute(predictions.delete().where(
            predictions.c.model_version == model_version, predictions.c.date >= since))
        if payload:
            conn.execute(predictions.insert(), payload)
    return len(payload)


def load_predictions(engine: Engine, model_version: str) -> pd.DataFrame:
    """A model's stored predictions, indexed by (date, symbol)."""
    p = tables.predictions
    stmt = select(p.c.date, p.c.symbol, p.c.probability, p.c.input_hash).where(
        p.c.model_version == model_version).order_by(p.c.date, p.c.symbol)
    with engine.connect() as conn:
        rows = conn.execute(stmt).all()
    frame = pd.DataFrame(rows, columns=["date", "symbol", "probability", "input_hash"])
    frame["date"] = pd.DatetimeIndex(frame["date"])
    return frame.set_index(["date", "symbol"])


def forward_run(engine: Engine, run_id: int | None = None):
    """A forward-test run (default: the open one), or None."""
    runs = tables.forward_runs
    stmt = select(runs).where(runs.c.run_id == run_id) if run_id is not None else \
        select(runs).where(runs.c.is_open == 1)
    with engine.connect() as conn:
        return conn.execute(stmt).mappings().first()


forward_start = forward_run  # the open run is the forward test in progress


def list_forward_runs(engine: Engine) -> pd.DataFrame:
    runs = tables.forward_runs
    with engine.connect() as conn:
        rows = conn.execute(select(runs).order_by(runs.c.run_id)).mappings().all()
    return pd.DataFrame([dict(r) for r in rows], columns=[c.name for c in runs.columns])


def _run_id(engine: Engine, run_id: int | None) -> int | None:
    if run_id is not None:
        return run_id
    run = forward_run(engine)
    return None if run is None else run["run_id"]


def record_forward_start(engine: Engine, started_on: date, start_value: float, value_source: str,
                         positions: Sequence[Mapping[str, Any]] = ()) -> bool:
    """Open a new run with its start and starting holdings; False if a run is open already."""
    runs = tables.forward_runs
    with engine.begin() as conn:
        if conn.execute(select(runs.c.run_id).where(runs.c.is_open == 1)).first() is not None:
            return False
        run_id = conn.execute(runs.insert().values(
            started_on=started_on, start_value=start_value, value_source=value_source, created_at=_now(),
            is_open=1)).inserted_primary_key[0]
        if positions:
            conn.execute(tables.forward_start_positions.insert(), [{**dict(r), "run_id": run_id} for r in positions])
    return True


def end_forward_run(engine: Engine, reason: str):
    """End the open run (kept, never deleted); its derived equity is cleared. Returns the run, or None."""
    runs = tables.forward_runs
    with engine.begin() as conn:
        run = conn.execute(select(runs).where(runs.c.is_open == 1)).mappings().first()
        if run is None:
            return None
        conn.execute(runs.update().where(runs.c.run_id == run["run_id"])
                     .values(ended_at=_now(), end_reason=reason, is_open=None))
        conn.execute(tables.forward_equity.delete())
    return run


def forward_start_positions(engine: Engine, run_id: int | None = None) -> pd.DataFrame:
    run_id = _run_id(engine, run_id)
    fsp = tables.forward_start_positions
    with engine.connect() as conn:
        rows = conn.execute(select(fsp.c.symbol, fsp.c.quantity, fsp.c.cost_basis)
                            .where(fsp.c.run_id == run_id)).mappings().all()
    return pd.DataFrame([dict(r) for r in rows], columns=["symbol", "quantity", "cost_basis"])


def frozen_sessions(engine: Engine, portfolio: str | None = None, run_id: int | None = None) -> pd.DataFrame:
    """forward_sessions rows of a run (default: the open one; none without one), ordered by session."""
    run_id = _run_id(engine, run_id)
    fs = tables.forward_sessions
    stmt = select(fs).where(fs.c.run_id == run_id).order_by(fs.c.session, fs.c.portfolio)
    if portfolio is not None:
        stmt = stmt.where(fs.c.portfolio == portfolio)
    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
    return pd.DataFrame([dict(r) for r in rows], columns=[c.name for c in fs.columns])


def freeze_forward(engine: Engine, header: Mapping[str, Any], targets: Sequence[Mapping[str, Any]]) -> bool:
    """Write one portfolio's session once, in the open run. Returns False (and writes nothing) if
    it is already frozen: a frozen session is never updated, whatever later data says."""
    run_id = _run_id(engine, header.get("run_id"))
    if run_id is None:
        raise ValueError("no open forward-test run: `quant forward-test --start` first")
    fs, ft = tables.forward_sessions, tables.forward_targets
    header = {**header, "run_id": run_id}
    with engine.begin() as conn:
        exists = conn.execute(select(fs.c.session).where(
            fs.c.run_id == run_id, fs.c.session == header["session"], fs.c.portfolio == header["portfolio"])).first()
        if exists is not None:
            return False
        conn.execute(fs.insert().values(**header, frozen_at=_now()))
        rows = [{**row, "run_id": run_id, "session": header["session"], "portfolio": header["portfolio"]}
                for row in targets]
        if rows:
            conn.execute(ft.insert(), rows)
    return True


def load_forward_targets(engine: Engine, portfolio: str, run_id: int | None = None) -> pd.DataFrame:
    """Frozen target weights of one portfolio in a run, sessions x symbols (0 where not held)."""
    run_id = _run_id(engine, run_id)
    ft = tables.forward_targets
    with engine.connect() as conn:
        rows = conn.execute(select(ft.c.session, ft.c.symbol, ft.c.target_weight)
                            .where(ft.c.run_id == run_id, ft.c.portfolio == portfolio)).all()
    frame = pd.DataFrame(rows, columns=["session", "symbol", "target_weight"])
    wide = frame.pivot(index="session", columns="symbol", values="target_weight")
    wide.index = pd.DatetimeIndex(wide.index)
    sessions = frozen_sessions(engine, portfolio, run_id)["session"]
    return wide.reindex(pd.DatetimeIndex(sessions)).fillna(0.0).sort_index()


def load_raw_opens(engine: Engine, symbols: Sequence[str], since: date | None = None) -> pd.DataFrame:
    """Stored (as-traded) opens: dates x symbols."""
    prices = tables.daily_prices
    stmt = select(prices.c.date, prices.c.symbol, prices.c.open).where(prices.c.symbol.in_(list(symbols)))
    if since is not None:
        stmt = stmt.where(prices.c.date >= since)
    with engine.connect() as conn:
        rows = conn.execute(stmt).all()
    frame = pd.DataFrame(rows, columns=["date", "symbol", "open"])
    wide = frame.pivot(index="date", columns="symbol", values="open")
    wide.index = pd.DatetimeIndex(wide.index)
    return wide.sort_index().astype(float)


def replace_forward_equity(engine: Engine, rows: Sequence[Mapping[str, Any]]) -> int:
    with engine.begin() as conn:
        conn.execute(tables.forward_equity.delete())
        if rows:
            conn.execute(tables.forward_equity.insert(), list(rows))
    return len(rows)


def load_forward_equity(engine: Engine) -> pd.DataFrame:
    fe = tables.forward_equity
    with engine.connect() as conn:
        rows = conn.execute(select(fe).order_by(fe.c.portfolio, fe.c.date)).mappings().all()
    frame = pd.DataFrame([dict(r) for r in rows], columns=[c.name for c in fe.columns])
    frame["date"] = pd.DatetimeIndex(frame["date"])
    return frame
