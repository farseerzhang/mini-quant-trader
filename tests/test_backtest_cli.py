import dataclasses
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import yaml
from sqlalchemy import func, select

from quant.backtest.persist import config_hash, git_commit
from quant.cli import main
from quant.config.settings import DEFAULT_CONFIG_PATH
from quant.persistence import queries, tables
from quant.persistence.database import create_db_engine
from tests.backtest_fixture import POOL, build_market_db, small_settings

PROJECT_SETTINGS = Path(__file__).resolve().parents[1] / DEFAULT_CONFIG_PATH


class BacktestCommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_dir = tempfile.TemporaryDirectory()
        cls.fixture_db = Path(cls.fixture_dir.name) / "market.db"
        build_market_db(cls.fixture_db, small_settings())

    @classmethod
    def tearDownClass(cls):
        cls.fixture_dir.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "bt.db"
        shutil.copy(self.fixture_db, self.db)
        document = yaml.safe_load(PROJECT_SETTINGS.read_text(encoding="utf-8"))
        document["database"]["url"] = f"sqlite:///{self.db.as_posix()}"
        document["market"]["symbols"] = POOL
        document["backtest"]["random_baseline_runs"] = 3
        self.config = Path(self.temp.name) / "settings.yaml"
        self.config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def run_backtest(self, *extra):
        output, errors = StringIO(), StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            code = main(["--config", str(self.config), "backtest", *extra])
        return code, output.getvalue(), errors.getvalue()

    def rows(self):
        engine = create_db_engine(self.db, 5000)
        with engine.connect() as conn:
            runs = conn.execute(select(tables.backtest_runs)).mappings().all()
            last_equity = conn.execute(select(func.max(tables.backtest_equity.c.date))).scalar_one()
            holdout_views = queries.count_pipeline_runs(engine, "backtest", "holdout")
        engine.dispose()
        return runs, last_equity, holdout_views

    def test_default_run_shows_and_saves_development_only(self):
        code, output, _ = self.run_backtest()
        self.assertEqual(code, 0)
        self.assertIn("DEVELOPMENT", output)
        self.assertIn("Random baseline", output)
        self.assertIn("not shown (evaluated 0 time(s) so far", output)
        self.assertNotIn("evaluation #", output)

        runs, last_equity, holdout_views = self.rows()
        self.assertEqual({run["period"] for run in runs}, {"DEVELOPMENT"})
        self.assertEqual(
            [run["strategy"] for run in runs],
            ["trend", "mean_reversion", "regime_switch", "regime_switch_bear50", "SPY", "equal_weight"],
        )
        development_end = runs[0]["end_date"]
        self.assertLessEqual(last_equity, development_end)  # no holdout values stored
        self.assertEqual(holdout_views, 0)

        trend = runs[0]
        self.assertEqual(trend["kind"], "STRATEGY")
        self.assertEqual(trend["strategy_version"], "strategies-v1")
        self.assertEqual(trend["model_version"], "rule_based-v1")
        self.assertEqual(trend["random_runs"], 3)
        self.assertEqual(json.loads(trend["universe"]), ["AAPL", "MSFT", "XOM"])
        self.assertIn("mean_reversion", json.loads(trend["parameters"])["strategies"])
        self.assertEqual(len(trend["config_hash"]), 64)
        self.assertEqual((trend["fee_bps"], trend["slippage_bps"]), (1.0, 5.0))
        spy = runs[4]
        self.assertEqual((spy["kind"], spy["strategy_version"], spy["random_runs"]), ("BENCHMARK", None, None))
        self.assertAlmostEqual(trend["benchmark_total_return"], spy["total_return"])

    def test_evaluate_holdout_shows_saves_and_logs_every_evaluation(self):
        code, output, _ = self.run_backtest("--evaluate-holdout")
        self.assertEqual(code, 0)
        self.assertIn("(evaluation #1; logged in pipeline_runs)", output)
        runs, last_equity, holdout_views = self.rows()
        holdout = [run for run in runs if run["period"] == "HOLDOUT"]
        self.assertEqual(len(holdout), 6)
        self.assertEqual(last_equity, holdout[0]["end_date"])
        self.assertIsNone(holdout[0]["random_runs"])  # the random baseline is development-only
        self.assertEqual(holdout_views, 1)

        _, output, _ = self.run_backtest("--evaluate-holdout")
        self.assertIn("(evaluation #2;", output)
        _, output, _ = self.run_backtest()
        self.assertIn("not shown (evaluated 2 time(s) so far", output)

    def test_strategy_filter_keeps_the_benchmarks(self):
        code, _, _ = self.run_backtest("--strategy", "trend")
        self.assertEqual(code, 0)
        runs, _, _ = self.rows()
        self.assertEqual([run["strategy"] for run in runs], ["trend", "SPY", "equal_weight"])

    def test_unknown_strategy_is_rejected_before_running(self):
        code, _, errors = self.run_backtest("--strategy", "momentum")
        self.assertEqual(code, 1)
        self.assertIn("Unknown strategy: momentum", errors)
        self.assertEqual(self.rows()[0], [])

    def test_missing_regimes_are_a_clear_error(self):
        engine = create_db_engine(self.db, 5000)
        with engine.begin() as conn:
            conn.execute(tables.market_regimes.delete())
        engine.dispose()
        code, _, errors = self.run_backtest()
        self.assertEqual(code, 1)
        self.assertIn("calculate-regime", errors)


class ReproducibilityTests(unittest.TestCase):
    def test_config_hash_is_stable_and_changes_with_any_setting(self):
        settings = small_settings()
        self.assertEqual(config_hash(settings), config_hash(small_settings()))
        changed = dataclasses.replace(
            settings, backtest=dataclasses.replace(settings.backtest, slippage_bps=6)
        )
        self.assertNotEqual(config_hash(settings), config_hash(changed))

    def test_git_commit_is_recorded(self):
        commit = git_commit()
        self.assertIsNotNone(commit)
        self.assertRegex(commit, r"^[0-9a-f]{40}(\+dirty)?$")


if __name__ == "__main__":
    unittest.main()
