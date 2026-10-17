import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from quant import cli
from quant.forward.freeze import freeze_session
from quant.forward.report import build_report, format_report
from quant.persistence import queries, tables
from quant.persistence.database import create_db_engine, init_db
from tests.backtest_fixture import DATES
from tests.test_forward_paper import WINDOW, PaperFixture
from tests.test_forward_freeze import EVENING
from tests.test_settings import PROJECT_SETTINGS


class RunTests(PaperFixture):
    def test_ending_keeps_the_run_and_a_new_one_starts_clean(self):
        self.freeze(WINDOW[:10])  # run 1
        ended = queries.end_forward_run(self.engine, "wrong positions")
        self.assertEqual(ended["run_id"], 1)
        self.assertIsNone(queries.forward_run(self.engine))
        self.assertTrue(queries.frozen_sessions(self.engine).empty)  # nothing open
        self.assertEqual(len(queries.frozen_sessions(self.engine, "live", run_id=1)), 10)  # kept
        self.assertTrue(queries.load_forward_equity(self.engine).empty)

        self.freeze(WINDOW[10:])  # run 2, on later sessions
        run = queries.forward_run(self.engine)
        self.assertEqual((run["run_id"], run["started_on"]), (2, WINDOW[10].date()))
        self.assertEqual(len(queries.frozen_sessions(self.engine, "live")), 20)
        runs = queries.list_forward_runs(self.engine)
        self.assertEqual(list(runs["end_reason"].fillna("")), ["wrong positions", ""])

    def test_reports_of_the_running_and_an_ended_run(self):
        self.freeze(WINDOW[:10])
        queries.end_forward_run(self.engine, "wrong positions")
        self.freeze(WINDOW[10:])
        current = build_report(self.engine, self.settings)
        self.assertEqual((current.run_id, current.started_on, current.ended), (2, WINDOW[10], None))
        text = format_report(current)
        self.assertIn("Forward test run 2 since", text)
        self.assertIn(f"Other run 1: started {WINDOW[0].date()}, ended", text)
        self.assertIn("wrong positions", text)
        old = build_report(self.engine, self.settings, run_id=1)
        self.assertEqual((old.run_id, old.last_session), (1, WINDOW[9]))  # stops at its last frozen session
        self.assertIn("(ENDED)", format_report(old))
        with self.assertRaisesRegex(ValueError, "no forward-test run 7"):
            build_report(self.engine, self.settings, run_id=7)

    def test_only_one_run_is_open(self):
        self.assertTrue(queries.record_forward_start(self.engine, WINDOW[0].date(), 1.0, "a"))
        self.assertFalse(queries.record_forward_start(self.engine, WINDOW[1].date(), 1.0, "b"))
        with self.assertRaises(IntegrityError), self.engine.begin() as conn:
            conn.execute(tables.forward_runs.insert().values(
                started_on=WINDOW[2].date(), start_value=1.0, value_source="c", created_at=WINDOW[2], is_open=1))

    def test_start_freezes_into_the_new_run(self):
        Path(self.settings.positions.path).write_text("symbol,quantity\nCASH,1000\n", encoding="utf-8")
        queries.record_forward_start(self.engine, DATES[-5].date(), 1.0, "old")
        queries.end_forward_run(self.engine, "restart")
        result = freeze_session(self.engine, self.settings, now=EVENING, start=True)
        self.assertEqual(result.frozen, ["live"])
        self.assertEqual(queries.forward_run(self.engine)["run_id"], 2)
        self.assertEqual(queries.forward_start_positions(self.engine).to_dict("records"),
                         [{"symbol": "CASH", "quantity": 1000.0, "cost_basis": None}])


    def test_a_broker_start_needs_both_snapshots(self):
        """`quant forward-test --start` always starts from the brokers (since 2026-10-17); without
        snapshots of the session it is refused, whatever the positions file holds."""
        import quant.broker.tables  # noqa: F401  the broker tables exist, empty

        init_db(self.engine)
        Path(self.settings.positions.path).write_text("symbol,quantity\nCASH,1000\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "no trading212 and ibkr snapshot.*sync-accounts"):
            freeze_session(self.engine, self.settings, now=EVENING, start=True, real_from_brokers=True)
        self.assertIsNone(queries.forward_run(self.engine))


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "cli.db"
        engine = create_db_engine(self.db, 5000)
        init_db(engine)
        queries.record_forward_start(engine, DATES[-1].date(), 1000.0, "test")
        engine.dispose()

    def tearDown(self):
        self.temp.cleanup()

    def run_cli(self, *args):
        output, errors = StringIO(), StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            code = cli.main(["--config", str(PROJECT_SETTINGS), "forward-test", *args, "--db", str(self.db)])
        return code, output.getvalue(), errors.getvalue()

    def test_end_needs_a_reason_and_is_logged(self):
        code, _, errors = self.run_cli("--end")
        self.assertEqual(code, 1)
        self.assertIn("--end needs --reason", errors)
        code, output, _ = self.run_cli("--end", "--reason", "wrong positions")
        self.assertEqual(code, 0)
        self.assertIn("Forward test run 1", output)
        self.assertIn("ENDED: wrong positions", output)
        code, _, errors = self.run_cli("--end", "--reason", "again")
        self.assertEqual(code, 1)
        self.assertIn("No forward test is running", errors)
        engine = create_db_engine(self.db, 5000)
        with engine.connect() as conn:
            stage = conn.execute(tables.pipeline_runs.select().where(tables.pipeline_runs.c.stage == "end")).first()
        engine.dispose()
        self.assertIsNotNone(stage)

    def test_end_and_start_together_is_refused(self):
        code, _, errors = self.run_cli("--end", "--start", "--reason", "x")
        self.assertEqual(code, 1)
        self.assertIn("separate steps", errors)


class UpgradeTests(unittest.TestCase):
    OLD = ("CREATE TABLE forward_sessions (session DATE, portfolio VARCHAR)",
           "CREATE TABLE forward_start (started_on DATE PRIMARY KEY, start_value FLOAT)")

    def test_empty_tables_of_the_old_layout_are_rebuilt(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = create_db_engine(Path(directory) / "old.db", 5000)
            with engine.begin() as conn:
                for statement in self.OLD:
                    conn.execute(text(statement))
            init_db(engine)
            names = set(inspect(engine).get_table_names())
            columns = {c["name"] for c in inspect(engine).get_columns("forward_sessions")}
            engine.dispose()
        self.assertNotIn("forward_start", names)
        self.assertIn("forward_runs", names)
        self.assertIn("run_id", columns)

    def test_old_tables_with_data_are_never_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = create_db_engine(Path(directory) / "old.db", 5000)
            with engine.begin() as conn:
                for statement in self.OLD:
                    conn.execute(text(statement))
                conn.execute(text("INSERT INTO forward_start VALUES ('2026-10-12', 1.0)"))
            with self.assertRaisesRegex(RuntimeError, "hold data in the old layout"):
                init_db(engine)
            engine.dispose()


if __name__ == "__main__":
    unittest.main()
