"""`quant backup-database`: consistent, checked, dated copies; only the newest are kept."""

import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant.ops.backup import BackupError, backup_database, existing_backups

START = datetime(2026, 10, 16, 21, 0, tzinfo=timezone.utc)


class BackupTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.database = self.root / "quant.db"
        with closing(sqlite3.connect(self.database)) as conn, conn:
            conn.execute("create table forward_sessions (session text, portfolio text)")
            conn.executemany("insert into forward_sessions values (?, ?)",
                             [("2026-10-15", "live"), ("2026-10-15", "shadow_hmm")])
        self.backups = self.root / "backups"

    def tearDown(self):
        self.temp.cleanup()

    def test_a_checked_copy_with_the_same_rows(self):
        result = backup_database(self.database, self.backups, keep=3, now=START)
        self.assertEqual(result.path.name, "quant-20261016T210000Z.db")
        self.assertEqual(result.row_counts, {"forward_sessions": 2})
        with closing(sqlite3.connect(result.path)) as conn:
            self.assertEqual(conn.execute("select count(*) from forward_sessions").fetchone()[0], 2)
        self.assertEqual(list(self.backups.glob("*.partial")), [])

    def test_only_the_newest_are_kept(self):
        for day in range(5):
            result = backup_database(self.database, self.backups, keep=3, now=START + timedelta(days=day))
        names = [p.name for p in existing_backups(self.backups)]
        self.assertEqual(names, ["quant-20261018T210000Z.db", "quant-20261019T210000Z.db", "quant-20261020T210000Z.db"])
        self.assertEqual((len(result.deleted), result.kept), (1, 3))

    def test_the_database_stays_usable_while_open(self):
        open_connection = sqlite3.connect(self.database)  # e.g. the dashboard
        try:
            result = backup_database(self.database, self.backups, keep=3, now=START)
        finally:
            open_connection.close()
        self.assertTrue(result.path.exists())

    def test_a_missing_database_is_an_error(self):
        with self.assertRaises(BackupError):
            backup_database(self.root / "missing.db", self.backups, keep=3)


if __name__ == "__main__":
    unittest.main()
