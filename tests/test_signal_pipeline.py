import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import date, datetime
from io import StringIO
from pathlib import Path

import dataclasses
import yaml
from sqlalchemy import select

from quant import cli
from quant.config.settings import DEFAULT_CONFIG_PATH
from quant.persistence import queries, tables
from quant.persistence.database import create_db_engine
from quant.signals.pipeline import generate_and_store_signals
from quant.signals.positions import MANUAL
from tests.backtest_fixture import POOL, build_market_db, small_settings

PROJECT_SETTINGS = Path(__file__).resolve().parents[1] / DEFAULT_CONFIG_PATH


class SignalPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        base = small_settings()
        self.positions = self.root / "positions.csv"
        self.settings = dataclasses.replace(base, positions=dataclasses.replace(base.positions, path=str(self.positions)))
        build_market_db(self.root / "pipeline.db", self.settings)
        self.engine = create_db_engine(self.root / "pipeline.db", 5000)

    def tearDown(self):
        self.engine.dispose()
        self.temp.cleanup()

    def stages(self):
        with self.engine.connect() as conn:
            rows = conn.execute(select(tables.pipeline_runs).where(tables.pipeline_runs.c.command == "generate-signals")
                                .order_by(tables.pipeline_runs.c.run_id)).mappings().all()
        return [(row["stage"], row["status"]) for row in rows]

    def test_signals_and_positions_are_stored_with_their_stages(self):
        self.positions.write_text("symbol,quantity\nAAPL,100\nXOM,50\nCASH,5000\n", encoding="utf-8")
        run = generate_and_store_signals(self.engine, self.settings)
        session = run.signals.session.date()
        stored = queries.load_signals(self.engine, session).set_index("symbol")
        self.assertEqual(sorted(stored.index), sorted(run.signals.rows.index))
        self.assertEqual(run.signals_stored, len(stored))
        self.assertEqual(set(stored["model_version"]), {"rule_based-v1"})
        self.assertEqual(set(stored["strategy"]), {"regime_switch"})
        self.assertTrue(stored["reason"].str.contains("->").all())
        self.assertTrue(stored["score"].isna().all())  # no ACTIVE predictor
        for symbol, row in run.signals.rows.iterrows():
            self.assertEqual(stored.loc[symbol, "signal"], row["signal"])
            self.assertAlmostEqual(stored.loc[symbol, "suggested_shares"], row["suggested_shares"])
        positions = queries.load_positions(self.engine, session, MANUAL)
        self.assertEqual(sorted(positions["symbol"]), ["AAPL", "CASH", "XOM"])
        self.assertEqual(self.stages(), [("predictor", "SKIPPED"), ("positions", "OK"), ("signals", "OK")])

    def test_a_rerun_of_the_same_session_replaces_signals_and_positions(self):
        self.positions.write_text("symbol,quantity\nAAPL,100\nXOM,50\nCASH,5000\n", encoding="utf-8")
        first = generate_and_store_signals(self.engine, self.settings)
        self.positions.write_text("symbol,quantity\nCASH,1000\n", encoding="utf-8")  # sold everything
        second = generate_and_store_signals(self.engine, self.settings)
        session = second.signals.session.date()
        stored = queries.load_signals(self.engine, session)
        self.assertEqual(sorted(stored["symbol"]), sorted(second.signals.rows.index))
        self.assertTrue((stored["signal"] == "BUY").all())  # nothing held any more
        self.assertNotEqual(len(first.signals.rows), 0)
        self.assertEqual(sorted(queries.load_positions(self.engine, session, MANUAL)["symbol"]), ["CASH"])

    def test_without_a_positions_file_nothing_is_stored_as_positions(self):
        run = generate_and_store_signals(self.engine, self.settings)
        self.assertEqual(run.positions_stored, 0)
        self.assertTrue(queries.load_positions(self.engine, run.signals.session.date(), MANUAL).empty)
        self.assertGreater(run.signals_stored, 0)


class GenerateSignalsCommandTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        settings = small_settings()
        build_market_db(root / "cli.db", settings)
        self.positions = root / "positions.csv"
        document = yaml.safe_load(PROJECT_SETTINGS.read_text(encoding="utf-8"))
        document["database"]["url"] = f"sqlite:///{(root / 'cli.db').as_posix()}"
        document["market"]["symbols"] = {sector: list(stocks) for sector, stocks in POOL.items()}
        document["risk"].update(max_position_size=settings.risk.max_position_size,
                                max_sector_weight=settings.risk.max_sector_weight)
        document["positions"]["path"] = self.positions.as_posix()
        self.config = root / "settings.yaml"
        self.config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def run_cli(self, *args):
        output, errors = StringIO(), StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            code = cli.main(["--config", str(self.config), "generate-signals", *args])
        return code, output.getvalue(), errors.getvalue()

    def test_prints_the_table_warnings_and_optionally_the_reasons(self):
        self.positions.write_text("symbol,quantity\nAAPL,100\nTSLA,1\nCASH,5000\n", encoding="utf-8")
        code, output, _ = self.run_cli()
        self.assertEqual(code, 0, output)
        self.assertIn("(fill at the next open): regime_switch (rule_based-v1)", output)
        self.assertIn("warning: TSLA is not in the stock pool", output)
        self.assertIn("Saved ", output)
        self.assertNotIn("AAPL: ", output)
        code, output, _ = self.run_cli("--reasons")
        self.assertIn("AAPL: ", output)
        self.assertIn("regime (rule_based-v1)", output)

    def test_an_invalid_positions_file_fails_with_its_line(self):
        self.positions.write_text("symbol,quantity\nAAPL,-1\nCASH,0\n", encoding="utf-8")
        code, _, errors = self.run_cli()
        self.assertEqual(code, 1)
        self.assertIn("Signal generation failed", errors)
        self.assertIn("line 2: AAPL quantity -1 is negative", errors)


if __name__ == "__main__":
    unittest.main()
