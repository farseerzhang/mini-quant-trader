"""Thin data-access layer: plain functions over SQLAlchemy Core.

Idempotency and the "never silently overwrite" rule (sections 4.4, 6.2) are
enforced here: writes use ``INSERT ... ON CONFLICT DO NOTHING``, so re-running
ingestion never creates duplicates and never rewrites an existing historical
row. Corrections must go through an explicit, logged path instead.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from sqlalchemy import Engine, func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from quant.persistence import tables


def _now() -> datetime:
    return datetime.now(timezone.utc)


def upsert_symbol(engine: Engine, symbol: str, **attrs: Any) -> None:
    """Insert a symbol or update its descriptive attributes and ``updated_at``."""
    now = _now()
    values = {
        "symbol": symbol,
        "active": attrs.pop("active", 1),
        "created_at": now,
        "updated_at": now,
        **attrs,
    }
    stmt = sqlite_insert(tables.symbols).values(**values)
    update_cols = {
        c.name: stmt.excluded[c.name]
        for c in tables.symbols.columns
        if c.name not in ("symbol", "created_at")
    }
    stmt = stmt.on_conflict_do_update(index_elements=["symbol"], set_=update_cols)
    with engine.begin() as conn:
        conn.execute(stmt)


def upsert_daily_prices(
    engine: Engine, rows: Sequence[Mapping[str, Any]]
) -> int:
    """Insert daily price rows idempotently.

    Existing ``(symbol, date)`` rows are left untouched (no silent overwrite).
    Returns the number of newly inserted rows.
    """
    rows = [dict(r) for r in rows]
    if not rows:
        return 0
    now = _now()
    for r in rows:
        r.setdefault("ingested_at", now)
    stmt = sqlite_insert(tables.daily_prices).on_conflict_do_nothing(
        index_elements=["symbol", "date"]
    )
    with engine.begin() as conn:
        result = conn.execute(stmt, rows)
        return result.rowcount if result.rowcount is not None else 0


def upsert_corporate_actions(
    engine: Engine, rows: Sequence[Mapping[str, Any]]
) -> int:
    """Insert corporate-action rows idempotently. Returns rows inserted."""
    rows = [dict(r) for r in rows]
    if not rows:
        return 0
    now = _now()
    for r in rows:
        r.setdefault("ingested_at", now)
    stmt = sqlite_insert(tables.corporate_actions).on_conflict_do_nothing(
        index_elements=["symbol", "ex_date", "action_type"]
    )
    with engine.begin() as conn:
        result = conn.execute(stmt, rows)
        return result.rowcount if result.rowcount is not None else 0


def quarantine_rows(
    engine: Engine, provider: str, bad_rows: Iterable[Mapping[str, Any]]
) -> int:
    """Persist rows that failed validation. Returns rows quarantined."""
    now = _now()
    payload = []
    for item in bad_rows:
        payload.append(
            {
                "symbol": item["symbol"],
                "date": item.get("date"),
                "provider": provider,
                "reason": item["reason"],
                "raw": json.dumps(item.get("raw"), default=str),
                "quarantined_at": now,
            }
        )
    if not payload:
        return 0
    with engine.begin() as conn:
        conn.execute(tables.data_quarantine.insert(), payload)
    return len(payload)


def latest_stored_date(engine: Engine, symbol: str) -> date | None:
    """Return the most recent stored bar date for a symbol, or None."""
    stmt = select(func.max(tables.daily_prices.c.date)).where(
        tables.daily_prices.c.symbol == symbol
    )
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
                command=command,
                stage=stage,
                status=status,
                data_as_of_date=data_as_of_date,
                started_at=now,
                finished_at=now,
                message=message,
            )
        )
