"""Engine creation and connection configuration for the local SQLite store.

Per section 4.2 of the proposal:
- The database is a single local file, set by ``database.url`` in config/settings.yaml.
- WAL mode is enabled so the writing CLI and the reading Streamlit app can
  coexist.
- A busy timeout is set so a brief lock does not immediately raise.
"""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import Engine, create_engine, event

from quant.persistence.tables import metadata


def _configure_connection(dbapi_connection, busy_timeout_ms: int) -> None:
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA synchronous=NORMAL")
    finally:
        cursor.close()


def create_db_engine(db_path: str | os.PathLike[str], busy_timeout_ms: int) -> Engine:
    """Create an engine for the local SQLite database.

    Pass ``":memory:"`` for an in-memory database (used by tests).
    """
    if str(db_path) == ":memory:":
        url = "sqlite:///:memory:"
    else:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{path}"

    engine = create_engine(url, future=True)
    event.listen(
        engine,
        "connect",
        lambda dbapi_connection, _record: _configure_connection(dbapi_connection, busy_timeout_ms),
    )
    return engine


def init_db(engine: Engine) -> None:
    """Create all tables and indexes that do not exist yet.

    create_all creates an index only together with its table, so an index added to an
    existing table (e.g. the one-ACTIVE-per-role index) is created here explicitly.
    """
    _upgrade_forward_tables(engine)
    metadata.create_all(engine)
    for table in metadata.tables.values():
        for index in table.indexes:
            index.create(engine, checkfirst=True)


FORWARD_TABLES = ("forward_sessions", "forward_targets", "forward_start_positions")


def _upgrade_forward_tables(engine: Engine) -> None:
    """The first forward-test layout had no runs (no ``run_id``; a ``forward_start`` table). Its
    tables are rebuilt in the run layout while they are still empty; with data in them, stop."""
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    names = set(inspector.get_table_names())
    old = [n for n in FORWARD_TABLES if n in names and "run_id" not in {c["name"] for c in inspector.get_columns(n)}]
    if "forward_start" in names:
        old.append("forward_start")
    if not old:
        return
    with engine.begin() as conn:
        full = [n for n in old if conn.execute(text(f'SELECT COUNT(*) FROM "{n}"')).scalar_one()]
        if full:
            raise RuntimeError(f"forward-test tables {', '.join(full)} hold data in the old layout (no runs); "
                               "move the database aside or migrate them by hand")
        for name in old:
            conn.execute(text(f'DROP TABLE "{name}"'))
