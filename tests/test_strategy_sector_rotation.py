"""Section 17.1.5 trial D: sector rotation with sector ETFs and its one-time trial; offline, hand-made data."""

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
from quant.persistence import queries, tables
from quant.persistence.database import create_db_engine
from quant.regime.rule_based import BEAR, BULL, SIDEWAYS
from quant.strategies.sector_rotation import SECTOR_FUNDS, sector_holdings, sector_rotation_weights
from quant.strategies.sector_rotation_trial import point_in_time_checks, run_sector_rotation_trial, trial_weights
from quant.strategies.strategy_trial import total_return_closes
from settings_helper import SETTINGS
from tests.backtest_fixture import DATES, POOL, build_market_db, small_settings

PROJECT_SETTINGS = Path(__file__).resolve().parents[1] / DEFAULT_CONFIG_PATH
SESSIONS = pd.bdate_range("2010-01-04", periods=330)


def growing(rates, index=SESSIONS):
    t = np.arange(len(index))[:, None]
    return pd.DataFrame(100 * np.exp(np.asarray(rates)[None, :] * t), index=index, columns=list(SECTOR_FUNDS))


class HoldingsAndWeightsTest(unittest.TestCase):
    def test_top_3_on_rebalances_one_third_each(self):
        closes = growing(np.linspace(0.0, 0.0008, 9))  # XLU strongest, then XLB, XLI
        held = sector_holdings(closes, SESSIONS[[260, 290]])
        self.assertFalse(held.iloc[:260].any().any())
        self.assertEqual(set(held.columns[held.iloc[260]]), {"XLU", "XLB", "XLI"})
        weights = sector_rotation_weights(held, None, SETTINGS)
        self.assertAlmostEqual(weights.iloc[270].sum(), 1.0)
        self.assertAlmostEqual(weights.iloc[270]["XLU"], 1 / 3)  # no single-stock cap on funds

    def test_no_keep_band_the_new_top_3_is_held(self):
        closes = growing(np.linspace(0.0, 0.0008, 9))
        closes.loc[SESSIONS[250]:, "XLK"] *= np.exp(0.02 * np.arange(len(closes.loc[SESSIONS[250]:])))  # late surge
        held = sector_holdings(closes, SESSIONS[[260, 320]])
        self.assertNotIn("XLK", set(held.columns[held.iloc[260]]))  # its surge is still inside the skipped month
        self.assertIn("XLK", set(held.columns[held.iloc[320]]))
        self.assertEqual(int(held.iloc[320].sum()), 3)

    def test_the_regime_exposure(self):
        held = sector_holdings(growing(np.linspace(0.0, 0.0008, 9)), SESSIONS[[260]])
        regimes = pd.Series(BULL, index=SESSIONS)
        regimes.iloc[270:280] = SIDEWAYS
        regimes.iloc[280:290] = BEAR
        weights = sector_rotation_weights(held, regimes, SETTINGS)
        self.assertAlmostEqual(weights.iloc[275].sum(), 1.0)
        self.assertEqual(weights.iloc[285].sum(), 0.0)


def stored_frame(values, index=SESSIONS):
    values = np.asarray(values, dtype=float)
    return pd.DataFrame({"open": values, "high": values * 1.01, "low": values * 0.99, "close": values,
                         "volume": 1000.0}, index=index)


class PointInTimeTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(9)
        self.stored = {fund: stored_frame(100 * np.exp(np.cumsum(rng.normal(0.0002 * i, 0.012, len(SESSIONS)))))
                       for i, fund in enumerate(SECTOR_FUNDS)}
        values = self.stored["XLE"]["close"].to_numpy().copy()
        values[310:] = values[310:] / 2
        self.stored["XLE"] = stored_frame(values)
        self.actions = {"XLE": pd.DataFrame([{"ex_date": SESSIONS[310].date(), "action_type": "SPLIT", "value": 2.0}]),
                        "XLU": pd.DataFrame([{"ex_date": SESSIONS[d].date(), "action_type": "DIVIDEND", "value": 0.6}
                                             for d in (265, 290, 315)])}
        self.regime = pd.Series(BULL, index=SESSIONS)
        self.regime.iloc[285:295] = BEAR
        self.first = SESSIONS[255]

    def weights(self):
        closes = total_return_closes(self.stored, self.actions, list(SECTOR_FUNDS))
        return trial_weights(closes, self.regime, self.first, SESSIONS[-1], SETTINGS)[1]

    def test_later_splits_and_dividends_change_nothing_before_the_cut(self):
        checks = point_in_time_checks(self.stored, self.actions, self.regime, self.weights(), self.first, SETTINGS,
                                      [SESSIONS[300], SESSIONS[325]])
        self.assertTrue(all(c.identical for c in checks), checks)

    def test_a_changed_past_target_is_caught(self):
        tampered = self.weights()
        tampered.iloc[270, 0] += 0.01
        checks = point_in_time_checks(self.stored, self.actions, self.regime, tampered, self.first, SETTINGS,
                                      [SESSIONS[300]])
        self.assertFalse(checks[0].identical)


class TrialCommandTest(unittest.TestCase):
    """The whole trial on the small stored market plus nine fake sector funds, with short folds."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        cls.db = root / "trial.db"
        build_market_db(cls.db, small_settings())
        t = np.arange(len(DATES))
        rows = []
        for i, fund in enumerate(SECTOR_FUNDS):
            closes = 100 * np.exp(0.0001 * i * t + 0.03 * np.sin(t / (20 + 5 * i)))
            rows += [{"symbol": fund, "date": d.date(), "open": float(c) * 0.999, "high": float(c) * 1.01,
                      "low": float(c) * 0.99, "close": float(c), "volume": 1000, "provider": "test"}
                     for d, c in zip(DATES, closes)]
        engine = create_db_engine(cls.db, 5000)
        queries.upsert_daily_prices(engine, rows)
        engine.dispose()
        document = yaml.safe_load(PROJECT_SETTINGS.read_text(encoding="utf-8"))
        document["market"]["symbols"] = POOL
        document["market"]["market_symbols"] = ["SPY", "QQQ", "^VIX", *SECTOR_FUNDS]
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
            code = main(["--config", str(self.config), "strategy-trial", "sector-rotation", "--db", str(self.db)])
        self.assertEqual(code, 0, errors.getvalue())
        text = output.getvalue()
        self.assertIn("trial D: sector_rotation", text)
        self.assertIn("point_in_time                PASS", text)
        reports = list((self.reports / "strategies").glob("sector-rotation-trial-*.md"))
        self.assertEqual(len(reports), 1)
        report = reports[0].read_text(encoding="utf-8")
        for part in ("## Gates:", "| sector_rotation |", "| sector_rotation (always invested) |",
                     "| regime_switch [rule_based-v1] |", "## Funds held", "| funds bought |"):
            self.assertIn(part, report)
        engine = create_db_engine(self.db, 5000)
        try:
            with engine.connect() as conn:
                runs = conn.execute(select(tables.pipeline_runs).where(
                    tables.pipeline_runs.c.command == "strategy-trial")).mappings().all()
        finally:
            engine.dispose()
        self.assertEqual([(r["stage"], r["status"]) for r in runs], [("sector-rotation", "OK")])

    def test_the_funds_must_be_configured(self):
        settings = dataclasses.replace(SETTINGS, market=dataclasses.replace(
            SETTINGS.market, market_symbols=("SPY", "QQQ", "^VIX")))
        engine = create_db_engine(self.db, 5000)
        try:
            with self.assertRaisesRegex(ValueError, "add XLK, XLF"):
                run_sector_rotation_trial(engine, settings)
        finally:
            engine.dispose()


if __name__ == "__main__":
    unittest.main()
