import dataclasses
import os
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from datetime import date, datetime, timezone
from io import StringIO
from pathlib import Path

import yaml

from quant import cli
from quant.config.settings import DEFAULT_CONFIG_PATH
from quant.ops.status import collect_status, format_status
from quant.persistence import queries, tables
from quant.persistence.database import create_db_engine
from quant.signals.pipeline import generate_and_store_signals
from tests.backtest_fixture import DATES, POOL, build_market_db, small_settings

PROJECT_SETTINGS = Path(__file__).resolve().parents[1] / DEFAULT_CONFIG_PATH
LAST = DATES[-1].date()
# The evening after the fixture's last session (a Friday close has passed by then).
EVENING = datetime(LAST.year, LAST.month, LAST.day, 23, 0, tzinfo=timezone.utc)


class StatusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        base = small_settings()
        self.positions = self.root / "positions.csv"
        self.settings = dataclasses.replace(base, positions=dataclasses.replace(base.positions, path=str(self.positions)))
        build_market_db(self.root / "status.db", self.settings)
        self.engine = create_db_engine(self.root / "status.db", 5000)

    def tearDown(self):
        self.engine.dispose()
        self.temp.cleanup()

    def status(self, now=EVENING, settings=None):
        return collect_status(self.engine, settings or self.settings, now=now)

    def test_a_fresh_market_without_signals_or_positions(self):
        report = self.status()
        self.assertEqual((report.prices_as_of, report.features_as_of, report.regime_as_of), (LAST, LAST, LAST))
        self.assertIsNone(report.signals_as_of)
        self.assertIn("signals end nowhere, behind the prices", " | ".join(report.warnings))
        self.assertIn("no positions file", " | ".join(report.warnings))
        self.assertIsNone(report.active["REGIME"])
        self.assertIn("none ACTIVE: rule-based regime", format_status(report))

    def test_after_signals_with_a_positions_file_there_is_nothing_to_warn_about(self):
        self.positions.write_text("symbol,quantity\nAAPL,10\nCASH,1000\n", encoding="utf-8")
        os.utime(self.positions, (time.time() - 60, time.time() - 60))
        generate_and_store_signals(self.engine, self.settings)
        report = self.status()
        self.assertEqual(report.warnings, [], report.warnings)
        self.assertEqual((report.signals_as_of, report.positions_as_of), (LAST, LAST))
        self.assertEqual(sum(report.signal_counts.values()), len(queries.load_signals(self.engine, LAST)))
        self.assertIn("No warnings.", format_status(report))

    def test_an_edited_positions_file_asks_for_new_signals(self):
        self.positions.write_text("symbol,quantity\nCASH,1000\n", encoding="utf-8")
        generate_and_store_signals(self.engine, self.settings)
        later = time.time() + 3600
        os.utime(self.positions, (later, later))
        self.assertIn("changed after the last signals", " | ".join(self.status().warnings))

    def test_stale_prices_and_a_symbol_behind_the_others(self):
        with self.engine.begin() as conn:
            conn.execute(tables.daily_prices.delete().where(
                tables.daily_prices.c.symbol == "XOM", tables.daily_prices.c.date > DATES[-3].date()))
        report = self.status(now=datetime(2026, 10, 9, 23, 0, tzinfo=timezone.utc))
        text = " | ".join(report.warnings)
        self.assertIn(f"prices end {LAST}, the last completed session is 2026-10-09", text)
        self.assertIn(f"behind the others: XOM ({DATES[-3].date()})", text)

    def test_a_symbol_without_prices(self):
        pool = {**{k: tuple(v) for k, v in POOL.items()}, "utilities": ("NEE",)}
        settings = dataclasses.replace(self.settings, market=dataclasses.replace(self.settings.market, symbols=pool))
        self.assertIn("no prices stored for: NEE", " | ".join(self.status(settings=settings).warnings))

    def test_a_failed_run_after_the_last_success(self):
        queries.record_pipeline_run(self.engine, command="calculate-features", stage="features", status="FAILED")
        self.assertIn("features: the last run FAILED", " | ".join(self.status().warnings))
        queries.record_pipeline_run(self.engine, command="calculate-features", stage="features", status="OK")
        self.assertNotIn("features: the last run FAILED", " | ".join(self.status().warnings))

    def test_an_active_predictor_trained_on_another_regime_is_stale(self):
        with self.engine.begin() as conn:
            conn.execute(tables.model_versions.insert().values(
                model_id="lr-x", role="PREDICTOR", model_type="logistic_regression", version=1, status="ACTIVE",
                depends_on="hmm-old", training_start=date(2010, 1, 1), training_end=date(2020, 1, 1),
                feature_version="technical-v1", target_definition="{}", hyperparameters="{}", metrics="{}",
                gates_passed=1, gate_results="{}", trial_count=1, holdout_evaluated=0, model_path="x",
                library_versions="{}", config_hash="h", created_at=datetime.now(timezone.utc),
            ))
        report = self.status()
        self.assertEqual(report.active["PREDICTOR"]["model_id"], "lr-x")
        self.assertIn("trained on regime hmm-old, but the regime in use is rule_based-v1",
                      " | ".join(report.warnings))


class ForwardStatusTests(StatusTests):
    """Reuses the status fixture; the inherited status tests run once, in StatusTests."""

    locals().update({name: None for name in dir(StatusTests) if name.startswith("test_")})

    def start(self):
        queries.record_forward_start(self.engine, DATES[-3].date(), 1000.0, "test",
                                     [{"symbol": "CASH", "quantity": 1000.0, "cost_basis": None}])
        queries.freeze_forward(self.engine, {
            "session": DATES[-3].date(), "portfolio": "live", "strategy": "regime_switch", "regime": "BULL",
            "regime_source": "rule_based-v1", "model_version": "rule_based-v1", "invested": 0.0, "positions": 0}, [])

    def test_missed_sessions_and_tonight(self):
        self.start()
        report = self.status()  # the evening of the last session: not frozen yet
        text = " | ".join(report.warnings)
        self.assertIn(f"forward test: {LAST} is not frozen yet; run `quant run-daily` before", text)
        self.assertIn(f"forward test: 1 missed session(s) since {DATES[-3].date()} ({DATES[-2].date()})", text)
        self.assertEqual((report.forward.frozen, report.forward.missed), (1, [DATES[-2].date()]))
        self.assertIn("forward test            since", format_status(report))

    def test_a_broken_trade_log_is_a_warning(self):
        self.start()
        trades = self.root / "trades.csv"
        trades.write_text("date,symbol,side,quantity,price\n2020-01-02,AAPL,BUY,1,1\n", encoding="utf-8")
        settings = dataclasses.replace(self.settings, positions=dataclasses.replace(
            self.settings.positions, trades_path=str(trades)))
        self.assertIn("trade log:", " | ".join(collect_status(self.engine, settings, now=EVENING).warnings))

    def test_before_the_start(self):
        self.assertIn("forward test            not started", format_status(self.status()))


class StatusCommandTests(unittest.TestCase):
    def test_status_command_prints_every_section(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_market_db(root / "cli.db", small_settings())
            document = yaml.safe_load(PROJECT_SETTINGS.read_text(encoding="utf-8"))
            document["database"]["url"] = f"sqlite:///{(root / 'cli.db').as_posix()}"
            document["market"]["symbols"] = {sector: list(stocks) for sector, stocks in POOL.items()}
            document["positions"]["path"] = (root / "positions.csv").as_posix()
            config = root / "settings.yaml"
            config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
            output = StringIO()
            with redirect_stdout(output):
                code = cli.main(["--config", str(config), "status"])
        self.assertEqual(code, 0)
        for heading in ("Data", "Last successful runs", "Models", "Warnings"):
            self.assertIn(heading, output.getvalue())


if __name__ == "__main__":
    unittest.main()
