"""Section 17.1.5 trial C: volatility targeting on regime_switch and its one-time trial; offline, hand-made data."""

import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import yaml
from sqlalchemy import select

from quant.cli import main
from quant.config.settings import DEFAULT_CONFIG_PATH
from quant.persistence import tables
from quant.persistence.database import create_db_engine
from quant.strategies.volatility_target import (
    realized_volatility,
    volatility_multiplier,
    volatility_target_weights,
)
from quant.strategies.volatility_target_trial import multiplier_table, point_in_time_checks, trial_weights
from settings_helper import SETTINGS
from tests.backtest_fixture import POOL, build_market_db, small_settings

PROJECT_SETTINGS = Path(__file__).resolve().parents[1] / DEFAULT_CONFIG_PATH
SESSIONS = pd.bdate_range("2010-01-04", periods=120)


class VolatilityTest(unittest.TestCase):
    def test_annualized_sample_volatility_of_the_last_20_log_returns(self):
        returns = np.tile([0.01, -0.01], 60)[:len(SESSIONS)]
        closes = pd.Series(100 * np.exp(np.cumsum(returns)), index=SESSIONS)
        volatility = realized_volatility(closes)
        self.assertTrue(volatility.iloc[:20].isna().all())  # 20 returns need 21 closes
        expected = np.std(np.diff(np.log(closes.to_numpy()))[-20:], ddof=1) * np.sqrt(252)
        self.assertAlmostEqual(volatility.iloc[-1], expected)
        self.assertAlmostEqual(volatility.iloc[-1], 0.01 * np.sqrt(20 / 19) * np.sqrt(252))

    def test_the_multiplier_only_reduces(self):
        sigma = pd.Series([0.30, 0.15, 0.10, 0.0, np.nan], index=SESSIONS[:5])
        m = volatility_multiplier(sigma)
        self.assertEqual(list(m.iloc[:4]), [0.5, 1.0, 1.0, 1.0])
        self.assertTrue(np.isnan(m.iloc[4]))

    def test_weights_are_scaled_per_session(self):
        weights = pd.DataFrame({"A": [0.1, 0.1], "B": [0.2, 0.0]}, index=SESSIONS[:2])
        scaled = volatility_target_weights(weights, pd.Series([0.5, 1.0], index=SESSIONS[:2]))
        self.assertEqual(scaled.to_numpy().tolist(), [[0.05, 0.1], [0.1, 0.0]])
        with self.assertRaisesRegex(ValueError, "no SPY volatility"):
            volatility_target_weights(weights, pd.Series([0.5], index=SESSIONS[:1]))

    def test_multiplier_table(self):
        m = pd.Series([1.0, 0.5, 0.8, 1.0], index=SESSIONS[:4])
        regime = pd.Series(["BULL", "BULL", "SIDEWAYS", "BEAR"], index=SESSIONS[:4])
        table = multiplier_table(m, regime)
        self.assertEqual(table.loc["all sessions", "binds (m < 1)"], "50.0%")
        self.assertEqual(table.loc["regime BULL", "lowest m"], "0.50")


def stored_frame(values, index):
    values = np.asarray(values, dtype=float)
    return pd.DataFrame({"open": values, "high": values * 1.01, "low": values * 0.99, "close": values,
                         "volume": 1000.0}, index=index)


class PointInTimeTest(unittest.TestCase):
    """The regime_switch part is replaced by fixed weights; the overlay's own inputs are rebuilt at each cut."""

    def setUp(self):
        rng = np.random.default_rng(5)
        spy = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, len(SESSIONS))))
        spy[100:] = spy[100:] / 2  # a split after the first cut
        self.stored = {"SPY": stored_frame(spy, SESSIONS)}
        self.actions = {"SPY": pd.DataFrame([{"ex_date": SESSIONS[100].date(), "action_type": "SPLIT", "value": 2.0},
                                             {"ex_date": SESSIONS[70].date(), "action_type": "DIVIDEND",
                                              "value": 0.4}])}
        self.base = pd.DataFrame({"A": 0.5, "B": 0.3}, index=SESSIONS)
        self.index = SESSIONS[40:]

    def run_checks(self, weights, cuts):
        fake = lambda panel, regime, index, settings: self.base.reindex(index)  # noqa: E731
        with mock.patch("quant.strategies.volatility_target_trial.regime_switch_weights_over", side_effect=fake):
            return point_in_time_checks({}, pd.Series(dtype=object), self.stored, self.actions, weights, SETTINGS,
                                        cuts)

    def full_weights(self):
        from quant.strategies.strategy_trial import total_return_closes

        spy = total_return_closes(self.stored, self.actions, ["SPY"])["SPY"]
        fake = lambda panel, regime, index, settings: self.base.reindex(index)  # noqa: E731
        with mock.patch("quant.strategies.volatility_target_trial.regime_switch_weights_over", side_effect=fake):
            return trial_weights({}, pd.Series(dtype=object), spy, self.index, SETTINGS)[0]

    def test_later_splits_and_dividends_change_nothing_before_the_cut(self):
        weights = self.full_weights()
        self.assertTrue((weights["A"] <= 0.5 + 1e-12).all())
        checks = self.run_checks(weights, [SESSIONS[80], SESSIONS[110]])
        self.assertTrue(all(c.identical for c in checks), checks)

    def test_a_changed_past_target_is_caught(self):
        weights = self.full_weights()
        weights.iloc[5, 0] += 0.01
        self.assertFalse(self.run_checks(weights, [SESSIONS[80]])[0].identical)


class TrialCommandTest(unittest.TestCase):
    """The whole trial on the small stored market, with short folds."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        cls.db = root / "trial.db"
        build_market_db(cls.db, small_settings())
        document = yaml.safe_load(PROJECT_SETTINGS.read_text(encoding="utf-8"))
        document["market"]["symbols"] = POOL
        document["market"]["market_symbols"] = ["SPY", "QQQ", "^VIX"]
        document["retrain"]["validation"].update(min_train_years=1, fold_length_months=1, min_folds=2,
                                                 holdout_months=2)
        document["retrain"]["reports_dir"] = (root / "reports").as_posix()
        cls.reports = root / "reports"
        cls.config = root / "settings.yaml"
        cls.config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_the_trial_runs_and_writes_its_report(self):
        output, errors = StringIO(), StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            code = main(["--config", str(self.config), "strategy-trial", "volatility-target", "--db", str(self.db)])
        self.assertEqual(code, 0, errors.getvalue())
        text = output.getvalue()
        self.assertIn("trial C: regime_switch_voltarget", text)
        self.assertIn("point_in_time                PASS", text)
        reports = list((self.reports / "strategies").glob("volatility-target-trial-*.md"))
        self.assertEqual(len(reports), 1)
        report = reports[0].read_text(encoding="utf-8")
        for part in ("## Gates:", "| regime_switch_voltarget |", "| regime_switch [rule_based-v1] |",
                     "## The multiplier m over the window", "| all sessions |"):
            self.assertIn(part, report)
        engine = create_db_engine(self.db, 5000)
        try:
            with engine.connect() as conn:
                runs = conn.execute(select(tables.pipeline_runs).where(
                    tables.pipeline_runs.c.command == "strategy-trial")).mappings().all()
        finally:
            engine.dispose()
        self.assertEqual([(r["stage"], r["status"]) for r in runs], [("volatility-target", "OK")])


if __name__ == "__main__":
    unittest.main()
