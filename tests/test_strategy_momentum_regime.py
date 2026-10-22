"""Section 17.1.5 trial B: 12-1 momentum inside the regime timing and its one-time trial; offline, hand-made data."""

import dataclasses
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sqlalchemy import select

from quant.cli import main
from quant.config.settings import DEFAULT_CONFIG_PATH
from quant.persistence import tables
from quant.persistence.database import create_db_engine
from quant.regime.rule_based import BEAR, BULL, SIDEWAYS
from quant.strategies.momentum_regime import (
    momentum_holdings,
    momentum_regime_weights,
    momentum_scores,
)
from quant.strategies.momentum_regime_trial import point_in_time_checks, trial_weights
from quant.strategies.strategy_trial import total_return_closes
from settings_helper import SETTINGS
from tests.backtest_fixture import POOL, build_market_db, small_settings

PROJECT_SETTINGS = Path(__file__).resolve().parents[1] / DEFAULT_CONFIG_PATH
SESSIONS = pd.bdate_range("2010-01-04", periods=330)
STOCKS = [f"S{i:02d}" for i in range(40)]
# Eight sectors of five stocks, so the 30% sector cap never binds at 5% per stock.
SECTORS = {f"sector_{k}": tuple(STOCKS[5 * k:5 * k + 5]) for k in range(8)}
WIDE = dataclasses.replace(SETTINGS, market=dataclasses.replace(SETTINGS.market, symbols=SECTORS))


def growing_closes(rates, index=SESSIONS):
    """Each stock grows at its own steady daily rate, so its momentum rank is fixed."""
    t = np.arange(len(index))[:, None]
    return pd.DataFrame(100 * np.exp(np.asarray(rates)[None, :] * t), index=index, columns=STOCKS[:len(rates)])


class ScoreTest(unittest.TestCase):
    def test_return_from_252_to_21_sessions_back(self):
        closes = growing_closes(np.linspace(0.0, 0.002, 40))
        scores = momentum_scores(closes)
        t = 300
        expected = closes.iloc[t - 21] / closes.iloc[t - 252] - 1
        pd.testing.assert_series_equal(scores.iloc[t], expected, check_names=False)
        self.assertTrue(scores.iloc[251].isna().all())  # no close 252 sessions back yet
        self.assertFalse(scores.iloc[252].isna().any())

    def test_a_stock_without_a_close_today_or_back_then_is_not_ranked(self):
        closes = growing_closes(np.linspace(0.0, 0.002, 40))
        closes.iloc[300, 3] = np.nan  # not trading today
        closes.iloc[:100, 5] = np.nan  # listed later
        scores = momentum_scores(closes)
        self.assertTrue(np.isnan(scores.iloc[300, 3]))
        self.assertTrue(np.isnan(scores.iloc[300, 5]))  # 252 sessions back falls before its listing
        self.assertFalse(np.isnan(scores.iloc[300, 4]))


class HoldingsTest(unittest.TestCase):
    def test_top_20_on_rebalances_held_in_between(self):
        closes = growing_closes(np.linspace(0.0, 0.002, 40))  # S39 strongest
        rebalances = SESSIONS[[260, 280]]
        held = momentum_holdings(momentum_scores(closes), rebalances)
        self.assertFalse(held.iloc[:260].any().any())  # nothing before the first rebalance
        self.assertEqual(set(held.columns[held.iloc[260]]), set(STOCKS[20:]))
        self.assertTrue((held.iloc[260:280] == held.iloc[260]).all().all())

    def test_the_keep_band_keeps_a_held_stock_ranked_21_to_30(self):
        scores = pd.DataFrame(np.nan, index=SESSIONS[:3], columns=STOCKS)
        first = np.arange(40, 0, -1, dtype=float)  # S00 ranks 1st ... S39 ranks 40th
        scores.iloc[0] = first
        second = first.copy()
        second[[0, 1]] = [15.5, 5.5]  # S00 drops to rank 25 (kept), S01 to rank 35 (sold)
        scores.iloc[1] = second
        held = momentum_holdings(scores, SESSIONS[:2])
        day2 = set(held.columns[held.iloc[1]])
        self.assertEqual(len(day2), 20)
        self.assertIn("S00", day2)
        self.assertNotIn("S01", day2)
        self.assertIn("S20", day2)  # S01's place goes to the best stock not held


class WeightsTest(unittest.TestCase):
    def test_equal_weight_times_the_regime_exposure(self):
        closes = growing_closes(np.linspace(0.0, 0.002, 40))
        held = momentum_holdings(momentum_scores(closes), SESSIONS[[260]])
        regimes = pd.Series(BULL, index=SESSIONS)
        regimes.iloc[270:280] = SIDEWAYS
        regimes.iloc[280:290] = BEAR
        regimes.iloc[290:] = np.nan
        weights = momentum_regime_weights(held, regimes, WIDE)
        self.assertAlmostEqual(weights.iloc[265].sum(), 1.0)
        self.assertAlmostEqual(weights.iloc[265].max(), 0.05)
        self.assertAlmostEqual(weights.iloc[275].sum(), 1.0)  # SIDEWAYS: invested
        self.assertEqual(weights.iloc[285].sum(), 0.0)  # BEAR: cash
        self.assertEqual(weights.iloc[295].sum(), 0.0)  # no regime: nothing
        always = momentum_regime_weights(held, None, WIDE)
        self.assertAlmostEqual(always.iloc[285].sum(), 1.0)  # the descriptive always-invested variant


def stored_frame(values, index=SESSIONS):
    values = np.asarray(values, dtype=float)
    return pd.DataFrame({"open": values, "high": values * 1.01, "low": values * 0.99, "close": values,
                         "volume": 1000.0}, index=index)


class PointInTimeTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.stocks = STOCKS
        self.stored, self.actions = {}, {}
        for i, stock in enumerate(self.stocks):
            values = 100 * np.exp(np.cumsum(rng.normal(0.0002 * (i % 7), 0.01, len(SESSIONS))))
            self.stored[stock] = stored_frame(values)
        values = self.stored["S07"]["close"].to_numpy().copy()
        values[310:] = values[310:] / 4  # a 4:1 split after the first cut
        self.stored["S07"] = stored_frame(values)
        self.actions = {"S07": pd.DataFrame([{"ex_date": SESSIONS[310].date(), "action_type": "SPLIT", "value": 4.0}]),
                        "S11": pd.DataFrame([{"ex_date": SESSIONS[d].date(), "action_type": "DIVIDEND", "value": 0.5}
                                             for d in (262, 295, 320)])}
        self.regime = pd.Series(BULL, index=SESSIONS)
        self.regime.iloc[290:300] = BEAR
        self.first = SESSIONS[255]

    def weights(self):
        closes = total_return_closes(self.stored, self.actions, self.stocks)
        return trial_weights(closes, self.regime, self.first, SESSIONS[-1], WIDE)[1]

    def test_later_splits_and_dividends_change_nothing_before_the_cut(self):
        checks = point_in_time_checks(self.stored, self.actions, self.stocks, self.regime, self.weights(),
                                      self.first, WIDE, [SESSIONS[300], SESSIONS[325]])
        self.assertTrue(all(c.identical for c in checks), checks)

    def test_a_stock_listed_after_the_cut(self):
        listed = SESSIONS[305]
        self.stored["S03"] = self.stored["S03"].loc[listed:]
        self.actions["S03"] = pd.DataFrame([{"ex_date": SESSIONS[320].date(), "action_type": "DIVIDEND",
                                             "value": 0.2}])
        weights = self.weights()
        self.assertEqual(weights["S03"].abs().sum(), 0.0)  # never ranked: no close 252 sessions back
        checks = point_in_time_checks(self.stored, self.actions, self.stocks, self.regime, weights, self.first,
                                      WIDE, [SESSIONS[300], SESSIONS[325]])
        self.assertTrue(all(c.identical for c in checks), checks)

    def test_a_changed_past_target_is_caught(self):
        tampered = self.weights()
        tampered.iloc[270, 0] += 0.01
        checks = point_in_time_checks(self.stored, self.actions, self.stocks, self.regime, tampered, self.first,
                                      WIDE, [SESSIONS[300]])
        self.assertFalse(checks[0].identical)


class TrialCommandTest(unittest.TestCase):
    """The whole trial on the small stored market (3 stocks: all held), with short folds."""

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
            code = main(["--config", str(self.config), "strategy-trial", "momentum-regime", "--db", str(self.db)])
        self.assertEqual(code, 0, errors.getvalue())
        text = output.getvalue()
        self.assertIn("trial B: momentum_regime", text)
        self.assertIn("point_in_time                PASS", text)
        reports = list((self.reports / "strategies").glob("momentum-regime-trial-*.md"))
        self.assertEqual(len(reports), 1)
        report = reports[0].read_text(encoding="utf-8")
        for part in ("## Gates:", "| momentum_regime |", "| momentum_12_1 (always invested) |",
                     "| regime_switch [rule_based-v1] |", "## Stocks held most often", "| stocks held (average) |"):
            self.assertIn(part, report)
        engine = create_db_engine(self.db, 5000)
        try:
            with engine.connect() as conn:
                runs = conn.execute(select(tables.pipeline_runs).where(
                    tables.pipeline_runs.c.command == "strategy-trial")).mappings().all()
        finally:
            engine.dispose()
        self.assertEqual([(r["stage"], r["status"]) for r in runs], [("momentum-regime", "OK")])


if __name__ == "__main__":
    unittest.main()
