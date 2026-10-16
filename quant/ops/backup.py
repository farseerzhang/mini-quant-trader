"""`quant backup-database`: a consistent copy of the SQLite database, run by hand.

The copy is made with SQLite's own online backup, so it is consistent even while the dashboard
or another command has the database open. Each copy is checked (`PRAGMA integrity_check`) and its
forward-test rows are counted against the original before older copies are deleted. Only the
newest `database.backups_to_keep` copies are kept. The forward-test frozen log cannot be rebuilt
(no backfill), which is the main reason for keeping copies.

Restore: close the dashboard and stop any running command, then copy a backup over
`data/quant.db` (keep the current file aside first).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PREFIX, SUFFIX = "quant-", ".db"
COUNTED_TABLES = ("forward_sessions", "forward_targets", "forward_runs", "daily_prices")


class BackupError(RuntimeError):
    pass


@dataclass(frozen=True)
class BackupResult:
    path: Path
    size_bytes: int
    row_counts: dict[str, int]
    deleted: list[Path]
    kept: int


def _row_counts(connection: sqlite3.Connection) -> dict[str, int]:
    present = {name for (name,) in connection.execute("select name from sqlite_master where type = 'table'")}
    return {table: connection.execute(f"select count(*) from {table}").fetchone()[0]
            for table in COUNTED_TABLES if table in present}


def existing_backups(directory: Path) -> list[Path]:
    """The backups in ``directory``, oldest first (the timestamp in the name sorts them)."""
    return sorted(directory.glob(f"{PREFIX}*{SUFFIX}"))


def backup_database(database_path: str | Path, directory: str | Path, keep: int,
                    now: datetime | None = None) -> BackupResult:
    source_path = Path(database_path)
    if not source_path.exists():
        raise BackupError(f"no database at {source_path}")
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    target = target_dir / f"{PREFIX}{stamp}{SUFFIX}"
    partial = target.with_suffix(".partial")
    source = sqlite3.connect(f"file:{source_path.as_posix()}?mode=ro", uri=True)
    try:
        copy = sqlite3.connect(partial)
        try:
            source.backup(copy)
            check = copy.execute("pragma integrity_check").fetchone()[0]
            if check != "ok":
                raise BackupError(f"the copy failed its integrity check: {check}")
            copied, original = _row_counts(copy), _row_counts(source)
            if copied != original:
                raise BackupError(f"row counts differ between the copy and the database: {copied} vs {original}")
        finally:
            copy.close()
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    finally:
        source.close()
    partial.replace(target)  # only a checked copy gets the backup name
    backups = existing_backups(target_dir)
    deleted = backups[:-keep] if len(backups) > keep else []
    for old in deleted:
        old.unlink()
    return BackupResult(target, target.stat().st_size, copied, deleted, len(backups) - len(deleted))
