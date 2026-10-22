"""Section 17.1.6 trial E: the three strongest stocks of the last 12 months, quarterly; offline, hand-made data."""

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
from quant.strategies.strategy_trial import total_return_closes
from quant.strategies.top3_momentum import (
    quarter_rebalance_sessions,
    top3_holdings,
    top3_order_targets,
    twelve_month_scores,
)
from quant.strategies.top3_momentum_trial import point_in_time_checks, trial_holdings
from tests.backtest_fixture import POOL, build_market_db, small_settings

PROJECT_SETTINGS = Path(__file__).resolve().parents[1] / DEFAULT_CONFIG_PATH
SESSIONS = pd.bdate_range("2010-01-04", periods=400)
STOCKS = [f"S{i:02d}" for i in range(10)]


def growing(rates, index=SESSIONS):
    t = np.arange(len(index))[:, None]
    return pd.DataFrame(100 * np.exp(np.asarray(rates)[None, :] * t), index=index, columns=STOCKS[:len(rates)])


class RebalanceAndScoreTest(unittest.TestCase):
    def test_first_session_then_each_quarter_start(self):
        dates = quarter_rebalance_sessions(SESSIONS, pd.Timestamp("2010-02-17"))
        self.assertEqual(list(dates[:4]), [pd.Timestamp("2010-02-17"), pd.Timestamp("2010-04-01"),
                                           pd.Timestamp("2010-07-01"), pd.Timestamp("2010-10-01")])
        self.assertTrue(all(d.month in (1, 4, 7, 10) for d in dates[1:]))
        on_start = quarter_rebalance_sessions(SESSIONS, pd.Timestamp("2010-04-01"))
        self.assertEqual(list(on_start[:2]), [pd.Timestamp("2010-04-01"), pd.Timestamp("2010-07-01")])

    def test_twelve_month_return_without_a_skipped_month(self):
        closes = growing(np.linspace(0.0, 0.002, 10))
        closes.iloc[300, 2] = np.nan
        scores = twelve_month_scores(closes)
        self.assertAlmostEqual(scores.iloc[300, 4], closes.iloc[300, 4] / closes.iloc[48, 4] - 1)
        self.assertTrue(scores.iloc[251].isna().all())
        self.assertTrue(np.isnan(scores.iloc[300, 2]))  # no close today: not ranked


class HoldingsAndOrdersTest(unittest.TestCase):
    def test_top_3_held_until_the_next_quarter(self):
        closes = growing(np.linspace(0.0, 0.002, 10))  # S09 strongest
        rebalances = SESSIONS[[260, 325]]
        held = top3_holdings(closes, rebalances)
        self.assertFalse(held.iloc[:260].any().any())
        self.assertEqual(set(held.columns[held.iloc[260]]), {"S07", "S08", "S09"})
        self.assertTrue((held.iloc[260:325] == held.iloc[260]).all().all())

    def test_a_new_leader_replaces_the_weakest_and_targets_are_reset(self):
        closes = growing(np.linspace(0.0, 0.002, 10))
        closes.loc[SESSIONS[270]:, "S00"] *= np.exp(0.03 * np.arange(len(closes.loc[SESSIONS[270]:])))
        rebalances = SESSIONS[[260, 325]]
        held = top3_holdings(closes, rebalances)
        self.assertEqual(set(held.columns[held.iloc[325]]), {"S00", "S08", "S09"})
        orders = top3_order_targets(held, rebalances)
        self.assertTrue(orders.drop(index=rebalances).isna().all().all())  # no orders between rebalances
        row = orders.loc[SESSIONS[325]]
        self.assertAlmostEqual(row["S09"], 1 / 3)  # kept: set back to one third
        self.assertAlmostEqual(row["S00"], 1 / 3)  # bought
        self.assertEqual(row["S07"], 0.0)  # dropped: sold
        self.assertAlmostEqual(row.sum(), 1.0)  # no single-stock or sector cap


def stored_frame(values, index=SESSIONS):
    values = np.asarray(values, dtype=float)
    return pd.DataFrame({"open": values, "high": values * 1.01, "low": values * 0.99, "close": values,
                         "volume": 1000.0}, index=index)


class PointInTimeTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        self.stored = {s: stored_frame(100 * np.exp(np.cumsum(rng.normal(0.0003 * i, 0.012, len(SESSIONS)))))
                       for i, s in enumerate(STOCKS)}
        values = self.stored["S08"]["close"].to_numpy().copy()
        values[360:] = values[360:] / 3
        self.stored["S08"] = stored_frame(values)
        self.stored["S03"] = self.stored["S03"].loc[SESSIONS[355]:]  # listed after the first cut
        self.actions = {"S08": pd.DataFrame([{"ex_date": SESSIONS[360].date(), "action_type": "SPLIT", "value": 3.0}]),
                        "S05": pd.DataFrame([{"ex_date": SESSIONS[d].date(), "action_type": "DIVIDEND", "value": 0.5}
                                             for d in (270, 330, 380)])}
        self.first = SESSIONS[255]

    def holdings(self):
        closes = total_return_closes(self.stored, self.actions, STOCKS)
        return trial_holdings(closes, self.first, SESSIONS[-1])[0]

    def test_later_listings_splits_and_dividends_change_nothing_before_the_cut(self):
        checks = point_in_time_checks(self.stored, self.actions, STOCKS, self.holdings(), self.first,
                                      [SESSIONS[340], SESSIONS[395]])
        self.assertTrue(all(c.identical for c in checks), checks)

    def test_a_changed_past_pick_is_caught(self):
        tampered = self.holdings()
        tampered.iloc[270] = ~tampered.iloc[270]
        checks = point_in_time_checks(self.stored, self.actions, STOCKS, tampered, self.first, [SESSIONS[340]])
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
            code = main(["--config", str(self.config), "strategy-trial", "top3-momentum", "--db", str(self.db)])
        self.assertEqual(code, 0, errors.getvalue())
        text = output.getvalue()
        self.assertIn("Section 17.1.6 trial E: top3_momentum_12m_quarterly", text)
        self.assertIn("point_in_time                PASS", text)
        reports = list((self.reports / "strategies").glob("top3-momentum-trial-*.md"))
        self.assertEqual(len(reports), 1)
        report = reports[0].read_text(encoding="utf-8")
        for part in ("## Gates:", "| top3_momentum_12m_quarterly |", "| regime_switch [rule_based-v1] |",
                     "## Picks per quarter", "## Stocks held most often"):
            self.assertIn(part, report)
        engine = create_db_engine(self.db, 5000)
        try:
            with engine.connect() as conn:
                runs = conn.execute(select(tables.pipeline_runs).where(
                    tables.pipeline_runs.c.command == "strategy-trial")).mappings().all()
        finally:
            engine.dispose()
        self.assertEqual([(r["stage"], r["status"], r["message"]) for r in runs][0][:2], ("top3-momentum", "OK"))
        self.assertIn("section 17.1.6 trial E", runs[0]["message"])


if __name__ == "__main__":
    unittest.main()
