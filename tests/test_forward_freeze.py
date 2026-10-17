import dataclasses
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import joblib
import yaml

from quant.config.settings import DEFAULT_CONFIG_PATH, ConfigError, ForwardTestSettings, ShadowSettings, load_settings
from quant.forward.freeze import NotStartedError, freeze_session
from quant.persistence import queries, tables
from quant.persistence.database import create_db_engine
from quant.regime.hmm import fit_hmm, load_observations
from tests.backtest_fixture import DATES, build_market_db
from tests.test_active_models import TRAIN_END, _register, _settings, _train_predictor

LAST = DATES[-1]  # a Friday
EVENING = datetime(LAST.year, LAST.month, LAST.day, 22, 0, tzinfo=timezone.utc)
MONDAY_OPEN = datetime(2025, 6, 30, 13, 30, tzinfo=timezone.utc)
PROJECT_SETTINGS = Path(__file__).resolve().parents[1] / DEFAULT_CONFIG_PATH


class FreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        base = _settings(cls.root)
        build_market_db(cls.root / "base.db", base)
        engine = create_db_engine(cls.root / "base.db", 5000)
        cls.model = _train_predictor(engine, base)
        cls.hmm = fit_hmm(load_observations(engine, base).loc[:TRAIN_END], base.regime.hmm, 42)
        engine.dispose()
        cls.model_path, cls.hmm_path = cls.root / "lr.joblib", cls.root / "hmm.joblib"
        cls.model.save(cls.model_path)
        joblib.dump(cls.hmm, cls.hmm_path)
        shadows = {"shadow_hmm": ShadowSettings("regime_switch", regime_model="hmm-1"),
                   "shadow_lr": ShadowSettings("ml_select", predictor="lr-1")}
        cls.settings = dataclasses.replace(base, forward_test=ForwardTestSettings(shadows))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.db = self.root / f"{self._testMethodName}.db"
        build_market_db(self.db, self.settings)
        self.engine = create_db_engine(self.db, 5000)
        _register(self.engine, "hmm-1", "REGIME", "hmm", self.hmm_path, TRAIN_END, status="RETIRED")
        _register(self.engine, "lr-1", "PREDICTOR", "logistic_regression", self.model_path,
                  self.model.training_end, depends_on="rule_based-v1", status="CANDIDATE")

    def tearDown(self):
        self.engine.dispose()

    def test_every_portfolio_is_frozen_once_on_the_evening(self):
        result = freeze_session(self.engine, self.settings, now=EVENING, start=True)
        self.assertEqual((result.session, result.missed), (LAST, False))
        self.assertEqual(sorted(result.frozen), ["live", "shadow_hmm", "shadow_lr"])
        self.assertEqual(result.deadline, MONDAY_OPEN)
        sessions = queries.frozen_sessions(self.engine).set_index("portfolio")
        self.assertEqual(sessions.loc["shadow_hmm", "regime_source"], "hmm-1")
        self.assertEqual(sessions.loc["shadow_lr", "model_version"], "lr-1")
        self.assertEqual(sessions.loc["live", "strategy"], "regime_switch")
        for name in ("live", "shadow_hmm", "shadow_lr"):
            targets = queries.load_forward_targets(self.engine, name)
            self.assertEqual(int((targets.iloc[0] > 0).sum()), sessions.loc[name, "positions"])
            self.assertAlmostEqual(targets.iloc[0].sum(), sessions.loc[name, "invested"])
        with self.engine.connect() as conn:
            scores = conn.execute(tables.forward_targets.select().where(
                tables.forward_targets.c.portfolio == "shadow_lr")).mappings().all()
        self.assertTrue(all(row["score"] is not None for row in scores))

    def test_a_rerun_and_later_data_never_change_a_frozen_session(self):
        freeze_session(self.engine, self.settings, now=EVENING, start=True)
        before = queries.load_forward_targets(self.engine, "live")
        with self.engine.begin() as conn:  # a data revision after the freeze
            conn.execute(tables.features.update().where(tables.features.c.date == LAST.date())
                         .values(price_vs_sma200=-1.0))
        again = freeze_session(self.engine, self.settings, now=EVENING, start=True)
        self.assertEqual((again.frozen, sorted(again.already)), ([], ["live", "shadow_hmm", "shadow_lr"]))
        self.assertTrue(queries.load_forward_targets(self.engine, "live").equals(before))

    def test_after_the_next_open_the_session_is_missed_for_good(self):
        queries.record_forward_start(self.engine, DATES[-2].date(), 1000.0, "test")  # started the day before
        result = freeze_session(self.engine, self.settings, now=MONDAY_OPEN)
        self.assertTrue(result.missed)
        self.assertEqual(result.frozen, [])
        self.assertTrue(queries.frozen_sessions(self.engine).empty)
        with self.engine.connect() as conn:
            message = conn.execute(tables.pipeline_runs.select().where(
                tables.pipeline_runs.c.command == "forward-test")).mappings().first()["message"]
        self.assertIn("no backfill", message)

    def test_nothing_is_written_before_the_start(self):
        result = freeze_session(self.engine, self.settings, now=EVENING)
        self.assertFalse(result.started)
        self.assertTrue(queries.frozen_sessions(self.engine).empty)
        self.assertIsNone(queries.forward_start(self.engine))

    def test_a_start_after_the_deadline_is_refused(self):
        with self.assertRaisesRegex(NotStartedError, "cannot start on .*next open .* has passed"):
            freeze_session(self.engine, self.settings, now=MONDAY_OPEN, start=True)
        self.assertIsNone(queries.forward_start(self.engine))

    def test_a_portfolio_that_targets_nothing_still_gets_its_session_row(self):
        with self.engine.begin() as conn:
            conn.execute(tables.market_regimes.update().where(tables.market_regimes.c.date == LAST.date())
                         .values(regime="BEAR"))
        freeze_session(self.engine, self.settings, now=EVENING, start=True)
        live = queries.frozen_sessions(self.engine, "live").iloc[0]
        self.assertEqual((live["regime"], live["positions"], live["invested"]), ("BEAR", 0, 0.0))


    def _with_rule_bonds(self, drop_last_hyg=False):
        """shadow_rule_bonds (section 12.5.3) with TLT / IEF / HYG closes added to the fixture."""
        import numpy as np

        from tests.backtest_fixture import _rows

        t = np.arange(len(DATES))
        bonds = {"TLT": 100 * np.exp(0.0001 * t + 0.05 * np.sin(t / 30)), "IEF": 100 * np.exp(0.0001 * t),
                 "HYG": 100 * np.exp(0.0002 * t + 0.02 * np.sin(t / 20))}
        rows = [row for symbol, closes in bonds.items() for row in _rows(symbol, closes)]
        if drop_last_hyg:
            rows = [r for r in rows if not (r["symbol"] == "HYG" and r["date"] == LAST.date())]
        queries.upsert_daily_prices(self.engine, rows)
        market = dataclasses.replace(self.settings.market,
                                     market_symbols=(*self.settings.market.market_symbols, "TLT", "IEF", "HYG"))
        shadows = {"shadow_rule_bonds": ShadowSettings("regime_switch", regime_rule="rule_bonds-v1")}
        return dataclasses.replace(self.settings, market=market, forward_test=ForwardTestSettings(shadows))

    def test_rule_bonds_shadow_is_computed_each_evening(self):
        from quant.regime.rule_bonds import rule_bonds_regime

        settings = self._with_rule_bonds()
        result = freeze_session(self.engine, settings, now=EVENING, start=True)
        self.assertEqual(sorted(result.frozen), ["live", "shadow_rule_bonds"])
        row = queries.frozen_sessions(self.engine, "shadow_rule_bonds").iloc[0]
        self.assertEqual((row["strategy"], row["regime_source"]), ("regime_switch", "rule_bonds-v1"))
        self.assertEqual(row["regime"], rule_bonds_regime(self.engine, settings).iloc[-1])
        self.assertEqual(rule_bonds_regime(self.engine, settings).index[-1], LAST)

    def test_rule_bonds_shadow_without_the_latest_bond_close_is_not_frozen(self):
        from quant.regime.active import ActiveModelError

        settings = self._with_rule_bonds(drop_last_hyg=True)
        with self.assertRaisesRegex(ActiveModelError, "shadow_rule_bonds: its data ends"):
            freeze_session(self.engine, settings, now=EVENING, start=True)
        self.assertEqual(list(queries.frozen_sessions(self.engine)["portfolio"]), ["live"])


    def test_a_run_started_from_the_broker_snapshots(self):
        import pandas as pd

        from quant.broker.tables import broker_account_snapshots, broker_position_snapshots
        from quant.forward.broker_real import BROKER_VALUE_SOURCE
        from quant.forward.paper import update_paper_equity
        from quant.forward.report import build_report, format_report
        from quant.persistence.database import init_db

        init_db(self.engine)  # the broker tables (new tables only)
        session = LAST.date()
        snapshot = {"session": session, "account_id": "x", "fx_to_usd": 1.0, "statement_date": str(session),
                    "fetched_at": pd.Timestamp(EVENING).tz_localize(None)}
        with self.engine.begin() as conn:
            conn.execute(broker_account_snapshots.insert(), [
                {**snapshot, "broker": "ibkr", "account_currency": "USD", "value_account_currency": 6000.0,
                 "cash_account_currency": 1000.0, "value_usd": 6000.0, "cash_usd": 1000.0},
                {**snapshot, "broker": "trading212", "account_currency": "USD", "value_account_currency": 4000.0,
                 "cash_account_currency": 500.0, "value_usd": 4000.0, "cash_usd": 500.0}])
            conn.execute(broker_position_snapshots.insert(), [
                {"session": session, "broker": "ibkr", "broker_ticker": "AAPL", "symbol": "AAPL", "name": "Apple",
                 "asset_type": "STK", "in_pool": 1, "quantity": 10.0, "price": 500.0, "price_currency": "USD",
                 "value_usd": 5000.0}])
        settings = self.settings
        freeze_session(self.engine, settings, now=EVENING, start=True, real_from_brokers=True)
        run = queries.forward_run(self.engine)
        self.assertEqual((run["start_value"], run["value_source"]), (10000.0, BROKER_VALUE_SOURCE))
        update_paper_equity(self.engine, settings)
        report = build_report(self.engine, settings)
        self.assertTrue(report.real_from_brokers)
        self.assertEqual(report.values["real"].iloc[0], 10000.0)
        self.assertIn("real = your own portfolio, both broker accounts", format_report(report))


class ShadowSettingsTests(unittest.TestCase):
    def test_project_shadows(self):
        shadows = load_settings(PROJECT_SETTINGS).forward_test.shadows
        self.assertEqual(sorted(shadows), ["shadow_hmm", "shadow_lr", "shadow_rule_bonds"])
        self.assertEqual(shadows["shadow_lr"].strategy, "ml_select")
        self.assertEqual(shadows["shadow_rule_bonds"].regime_rule, "rule_bonds-v1")

    def test_invalid_shadows_are_rejected(self):
        bad = {
            "reserved name": {"live": {"strategy": "regime_switch"}},
            "ml_select without predictor": {"x": {"strategy": "ml_select"}},
            "predictor on a rule strategy": {"x": {"strategy": "trend", "predictor": "lr"}},
            "unknown strategy": {"x": {"strategy": "momentum"}},
            "unknown key": {"x": {"strategy": "trend", "model": "lr"}},
            "unknown regime rule": {"x": {"strategy": "regime_switch", "regime_rule": "rule_based-v9"}},
            "regime model and rule": {"x": {"strategy": "regime_switch", "regime_model": "hmm-1",
                                            "regime_rule": "rule_bonds-v1"}},
        }
        for name, shadows in bad.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as directory:
                document = yaml.safe_load(PROJECT_SETTINGS.read_text(encoding="utf-8"))
                document["forward_test"]["shadows"] = shadows
                path = Path(directory) / "settings.yaml"
                path.write_text(yaml.safe_dump(document), encoding="utf-8")
                with self.assertRaises(ConfigError):
                    load_settings(path)


if __name__ == "__main__":
    unittest.main()
