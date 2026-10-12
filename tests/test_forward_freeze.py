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


class ShadowSettingsTests(unittest.TestCase):
    def test_project_shadows(self):
        shadows = load_settings(PROJECT_SETTINGS).forward_test.shadows
        self.assertEqual(sorted(shadows), ["shadow_hmm", "shadow_lr"])
        self.assertEqual(shadows["shadow_lr"].strategy, "ml_select")

    def test_invalid_shadows_are_rejected(self):
        bad = {
            "reserved name": {"live": {"strategy": "regime_switch"}},
            "ml_select without predictor": {"x": {"strategy": "ml_select"}},
            "predictor on a rule strategy": {"x": {"strategy": "trend", "predictor": "lr"}},
            "unknown strategy": {"x": {"strategy": "momentum"}},
            "unknown key": {"x": {"strategy": "trend", "model": "lr"}},
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
