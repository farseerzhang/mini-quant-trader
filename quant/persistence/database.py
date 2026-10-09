"""Engine creation and connection configuration for the local SQLite store.

Per section 4.2 of the proposal:
- The database is a single local file (``data/quant.db`` by default).
- WAL mode is enabled so the writing CLI and the reading Streamlit app can
  coexist.
- A busy timeout is set so a brief lock does not immediately raise.
"""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import Engine, create_engine, event

from quant.persistence.tables import metadata

DEFAULT_BUSY_TIMEOUT_MS = 5000


def _configure_connection(dbapi_connection, busy_timeout_ms: int) -> None:
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA synchronous=NORMAL")
    finally:
        cursor.close()


def create_db_engine(
    db_path: str | os.PathLike[str], busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS
) -> Engine:
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
    """Create all tables if they do not already exist."""
    metadata.create_all(engine)
