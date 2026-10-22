"""Section 17.1.5 trial A: the dual momentum rule and its one-time trial; offline, hand-made data."""

import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sqlalchemy import select

from quant.cli import main
from quant.config.settings import DEFAULT_CONFIG_PATH
from quant.persistence import queries, tables
from quant.persistence.database import create_db_engine
from quant.strategies.dual_momentum import (
    BONDS,
    CASH,
    STOCKS,
    dual_momentum_choices,
    dual_momentum_weights,
    rebalance_sessions,
)
from quant.strategies.dual_momentum_trial import fund_closes, point_in_time_checks, trial_choices
from tests.backtest_fixture import DATES, POOL, build_market_db, small_settings

PROJECT_SETTINGS = Path(__file__).resolve().parents[1] / DEFAULT_CONFIG_PATH
SESSIONS = pd.bdate_range("2010-01-04", periods=400)


def closes(spy_growth, ief_growth, index=SESSIONS):
    t = np.arange(len(index))
    return pd.DataFrame({STOCKS: 100 * (1 + spy_growth) ** (t / 252), BONDS: 100 * (1 + ief_growth) ** (t / 252)},
                        index=index)


def flat_yield(percent, index=SESSIONS):
    return pd.Series(float(percent), index=index)


class RebalanceTest(unittest.TestCase):
    def test_first_session_then_first_of_each_month(self):
        first = pd.Timestamp("2010-03-17")
        dates = rebalance_sessions(SESSIONS, first)
        self.assertEqual(list(dates[:3]), [first, pd.Timestamp("2010-04-01"), pd.Timestamp("2010-05-03")])
        self.assertTrue(all(d >= first for d in dates))
        months = dates[1:].to_period("M")
        self.assertEqual(len(months), len(set(months)))  # one per month

    def test_first_session_on_a_month_start_is_not_doubled(self):
        dates = rebalance_sessions(SESSIONS, pd.Timestamp("2010-03-01"))
        self.assertEqual(list(dates[:2]), [pd.Timestamp("2010-03-01"), pd.Timestamp("2010-04-01")])


class ChoiceTest(unittest.TestCase):
    def choice(self, spy, ief, cash_percent):
        session = SESSIONS[[300]]
        return dual_momentum_choices(closes(spy, ief), flat_yield(cash_percent), session).iloc[0]

    def test_stocks_bonds_or_cash(self):
        self.assertEqual(self.choice(0.10, 0.05, 2.0)["choice"], STOCKS)
        self.assertEqual(self.choice(-0.10, 0.05, 2.0)["choice"], BONDS)
        self.assertEqual(self.choice(-0.10, -0.02, 2.0)["choice"], CASH)
        # Stocks below cash but bonds above it: bonds, even when stocks beat bonds.
        self.assertEqual(self.choice(0.03, 0.04, 3.5)["choice"], BONDS)
        self.assertEqual(self.choice(0.03, 0.02, 3.5)["choice"], CASH)

    def test_returns_over_252_sessions_and_the_average_yield(self):
        index = SESSIONS
        prices = closes(0.10, 0.05)
        yields = pd.Series(np.linspace(1.0, 5.0, len(index)), index=index)
        yields.iloc[290] = np.nan  # a missing session is left out of the average
        row = dual_momentum_choices(prices, yields, index[[300]]).iloc[0]
        self.assertAlmostEqual(row["r_stocks"], prices[STOCKS].iloc[300] / prices[STOCKS].iloc[48] - 1)
        self.assertAlmostEqual(row["r_bonds"], prices[BONDS].iloc[300] / prices[BONDS].iloc[48] - 1)
        self.assertAlmostEqual(row["r_cash"], yields.iloc[49:301].mean() / 100)

    def test_a_tie_with_cash_is_not_a_win(self):
        prices = closes(0.0, 0.0)
        self.assertEqual(dual_momentum_choices(prices, flat_yield(0.0), SESSIONS[[300]]).iloc[0]["choice"], CASH)

    def test_too_little_history_raises(self):
        with self.assertRaisesRegex(ValueError, "fewer than 252"):
            dual_momentum_choices(closes(0.1, 0.0), flat_yield(1.0), SESSIONS[[251]])
        with self.assertRaisesRegex(ValueError, "no \\^IRX"):
            dual_momentum_choices(closes(0.1, 0.0), pd.Series(np.nan, index=SESSIONS), SESSIONS[[300]])

    def test_weights_hold_until_the_next_rebalance(self):
        index = SESSIONS[300:340]
        choices = pd.Series([STOCKS, CASH, BONDS], index=index[[2, 10, 25]])
        weights = dual_momentum_weights(choices, index)
        self.assertEqual(weights.iloc[:2].to_numpy().sum(), 0.0)  # before the first decision
        self.assertTrue((weights.iloc[2:10][STOCKS] == 1.0).all())
        self.assertEqual(weights.iloc[10:25].to_numpy().sum(), 0.0)  # cash
        self.assertTrue((weights.iloc[25:][BONDS] == 1.0).all())
        self.assertTrue((weights.sum(axis=1) <= 1.0).all())


def stored_frame(values, index=SESSIONS):
    values = np.asarray(values, dtype=float)
    return pd.DataFrame({"open": values, "high": values * 1.01, "low": values * 0.99, "close": values,
                         "volume": 1000.0}, index=index)


class PointInTimeTest(unittest.TestCase):
    def test_later_splits_and_dividends_change_nothing_before_the_cut(self):
        rng = np.random.default_rng(7)
        spy = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(SESSIONS))))
        ief = 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.004, len(SESSIONS))))
        spy[350:] = spy[350:] / 2  # a 2:1 split after the first cut
        stored = {STOCKS: stored_frame(spy), BONDS: stored_frame(ief)}
        actions = {
            STOCKS: pd.DataFrame([{"ex_date": SESSIONS[350].date(), "action_type": "SPLIT", "value": 2.0},
                                  {"ex_date": SESSIONS[280].date(), "action_type": "DIVIDEND", "value": 0.4}]),
            BONDS: pd.DataFrame([{"ex_date": SESSIONS[d].date(), "action_type": "DIVIDEND", "value": 0.25}
                                 for d in (260, 290, 320, 360, 390)]),
        }
        cash = pd.Series(rng.uniform(0.5, 4.0, len(SESSIONS)), index=SESSIONS)
        first, last = SESSIONS[270], SESSIONS[-1]
        choices = trial_choices(fund_closes(stored, actions), cash, first, last)
        checks = point_in_time_checks(stored, actions, cash, choices, first, [SESSIONS[330], SESSIONS[385]])
        self.assertTrue(all(c.identical for c in checks), checks)
        self.assertGreater(checks[0].decisions, 1)

    def test_a_changed_past_decision_is_caught(self):
        stored = {STOCKS: stored_frame(np.linspace(100, 130, len(SESSIONS))),
                  BONDS: stored_frame(np.linspace(100, 104, len(SESSIONS)))}
        cash = flat_yield(1.0)
        first = SESSIONS[270]
        choices = trial_choices(fund_closes(stored, {}), cash, first, SESSIONS[-1])
        tampered = choices.copy()
        tampered.iloc[0, tampered.columns.get_loc("choice")] = CASH
        checks = point_in_time_checks(stored, {}, cash, tampered, first, [SESSIONS[330]])
        self.assertFalse(checks[0].identical)


class TrialCommandTest(unittest.TestCase):
    """The whole trial on the small stored market, with short folds; nothing is stored but the run log."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        cls.db = root / "trial.db"
        build_market_db(cls.db, small_settings())
        t = np.arange(len(DATES))
        ief = 100 * np.exp(0.0001 * t + 0.01 * np.sin(t / 60))
        rows = [{"symbol": BONDS, "date": d.date(), "open": float(c) * 0.999, "high": float(c) * 1.005,
                 "low": float(c) * 0.995, "close": float(c), "volume": 1000, "provider": "test"}
                for d, c in zip(DATES, ief)]
        rows += [{"symbol": "^IRX", "date": d.date(), "open": 2.0, "high": 2.0, "low": 2.0, "close": 2.0,
                  "volume": 0, "provider": "test"} for d in DATES]
        engine = create_db_engine(cls.db, 5000)
        queries.upsert_daily_prices(engine, rows)
        engine.dispose()
        document = yaml.safe_load(PROJECT_SETTINGS.read_text(encoding="utf-8"))
        document["market"]["symbols"] = POOL
        document["market"]["market_symbols"] = ["SPY", "QQQ", "^VIX", BONDS, "^IRX"]
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
            code = main(["--config", str(self.config), "strategy-trial", "dual-momentum", "--db", str(self.db)])
        self.assertEqual(code, 0, errors.getvalue())
        text = output.getvalue()
        for gate in ("sharpe_higher", "max_drawdown_no_deeper", "years_higher", "point_in_time"):
            self.assertIn(gate, text)
        self.assertIn("point_in_time                PASS", text)
        reports = list((self.reports / "strategies").glob("dual-momentum-trial-*.md"))
        self.assertEqual(len(reports), 1)
        report = reports[0].read_text(encoding="utf-8")
        self.assertIn("## Gates:", report)
        self.assertIn("| dual_momentum |", report)
        self.assertIn("| regime_switch [rule_based-v1] |", report)
        self.assertIn("## Monthly decisions", report)
        engine = create_db_engine(self.db, 5000)
        try:
            with engine.connect() as conn:
                run = conn.execute(select(tables.pipeline_runs).where(
                    tables.pipeline_runs.c.command == "strategy-trial")).mappings().all()
        finally:
            engine.dispose()
        self.assertEqual([(r["stage"], r["status"]) for r in run], [("dual-momentum", "OK")])


if __name__ == "__main__":
    unittest.main()
