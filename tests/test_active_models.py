"""ACTIVE models in daily use (Phase 9 step 3): an HMM regime and a predictor driving ml_select."""

import dataclasses
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sqlalchemy import update

from quant import cli
from quant.config.settings import DEFAULT_CONFIG_PATH
from quant.features.technical import FEATURE_COLUMNS, MARKET_FEATURE_COLUMNS
from quant.models.calibration import Calibrator
from quant.models.dataset import INPUT_COLUMNS, load_dataset
from quant.models.final import FinalModel
from quant.models.live import load_probabilities, predict_sessions, predictor_regimes
from quant.models.predictors import build_model
from quant.persistence import queries, tables
from quant.persistence.database import create_db_engine
from quant.regime.active import (
    ActiveModelError, active_hmm_regime, calculate_and_store_active_regime, regime_in_use,
)
from quant.regime.hmm import fit_hmm, load_observations
from quant.regime.pipeline import RULE_METHOD
from quant.signals.pipeline import generate_and_store_signals
from quant.signals.targets import load_targets
from quant.strategies.ml import ml_select_weights
from tests.backtest_fixture import DATES, POOL, build_market_db, small_settings

PROJECT_SETTINGS = Path(__file__).resolve().parents[1] / DEFAULT_CONFIG_PATH
TRAIN_END = DATES[400]


def _settings(root: Path):
    base = small_settings()
    return dataclasses.replace(
        base,
        positions=dataclasses.replace(base.positions, path=str(root / "positions.csv")),
        strategies=dataclasses.replace(
            base.strategies, ml_select=dataclasses.replace(base.strategies.ml_select, top_n=2, keep_rank=2)),
    )


def _register(engine, model_id, role, model_type, path, training_end, depends_on=None, feature_version=None,
              status="ACTIVE"):
    with engine.begin() as conn:
        version = len(conn.execute(tables.model_versions.select()).all()) + 1
        conn.execute(tables.model_versions.insert().values(
            model_id=model_id, role=role, model_type=model_type, version=version, status=status,
            depends_on=depends_on, training_start=DATES[0].date(), training_end=training_end.date(),
            feature_version=feature_version or "technical-v1", target_definition=json.dumps({"horizon_days": 5}),
            hyperparameters="{}", metrics="{}", gates_passed=0, gate_results="{}", trial_count=1,
            holdout_evaluated=0, model_path=str(path), library_versions="{}", config_hash="h",
            created_at=datetime.now(timezone.utc),
        ))


def _train_predictor(engine, settings) -> FinalModel:
    labelled = load_dataset(engine, settings).labelled()
    rows = labelled.frame[labelled.frame.index.get_level_values("date") <= TRAIN_END]
    x, y = rows.loc[:, list(INPUT_COLUMNS)].to_numpy(dtype=float), rows["label"].to_numpy(dtype=float)
    model = build_model("logistic_regression", {"C": 0.1}, 42).fit(x, y)
    calibrator = Calibrator("sigmoid").fit(model.predict_proba(x)[:, 1], y)
    dates = rows.index.get_level_values("date")
    return FinalModel("logistic_regression", {"C": 0.1}, tuple(INPUT_COLUMNS), model, calibrator,
                      dates.min(), dates.max(), len(y))


class ActiveRegimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.settings = _settings(self.root)
        build_market_db(self.root / "hmm.db", self.settings)
        self.engine = create_db_engine(self.root / "hmm.db", 5000)
        observations = load_observations(self.engine, self.settings)
        self.observations = observations
        self.fit = fit_hmm(observations.loc[:TRAIN_END], self.settings.regime.hmm, 42)
        self.path = self.root / "hmm-1.joblib"
        joblib.dump(self.fit, self.path)

    def tearDown(self):
        self.engine.dispose()
        self.temp.cleanup()

    def test_without_an_active_model_the_rule_based_regime_is_used(self):
        self.assertIsNone(calculate_and_store_active_regime(self.engine, self.settings))
        in_use = regime_in_use(self.engine, self.settings)
        self.assertEqual((in_use.method, in_use.model_version), (RULE_METHOD, "rule_based-v1"))

    def test_an_active_hmm_is_stored_and_drives_the_signals(self):
        _register(self.engine, "hmm-1", "REGIME", "hmm", self.path, TRAIN_END)
        summary = calculate_and_store_active_regime(self.engine, self.settings)
        stored = queries.load_regimes(self.engine, "HMM", "hmm-1")
        self.assertEqual(summary.rows, len(stored))
        self.assertEqual(stored.index[-1], DATES[-1])
        expected, _ = active_hmm_regime(self.fit, self.observations, self.settings.regime.min_dwell_days)
        pd.testing.assert_series_equal(stored, expected, check_names=False)
        targets = load_targets(self.engine, self.settings)
        self.assertEqual(targets.regime_source, "hmm-1")
        self.assertEqual(targets.regime, stored.iloc[-1])
        run = generate_and_store_signals(self.engine, self.settings)
        self.assertTrue(all("regime (hmm-1)" in r for r in run.signals.rows["reason"]))
        from quant.ops.status import collect_status

        report = collect_status(self.engine, self.settings)
        self.assertEqual(report.regime_in_use, "hmm-1")
        self.assertFalse(any("ACTIVE regime" in w for w in report.warnings), report.warnings)

    def test_the_hmm_regime_is_point_in_time(self):
        full, _ = active_hmm_regime(self.fit, self.observations, self.settings.regime.min_dwell_days)
        for cut in (DATES[300], DATES[500]):
            part, _ = active_hmm_regime(self.fit, self.observations.loc[:cut], self.settings.regime.min_dwell_days)
            pd.testing.assert_series_equal(part, full.loc[:cut])

    def test_refusals(self):
        _register(self.engine, "hmm-1", "REGIME", "hmm", self.path, TRAIN_END)
        changed = dataclasses.replace(self.settings, regime=dataclasses.replace(
            self.settings.regime, hmm=dataclasses.replace(self.settings.regime.hmm, observations=("log_vix",))))
        with self.assertRaisesRegex(ActiveModelError, "but regime.hmm.observations is log_vix"):
            calculate_and_store_active_regime(self.engine, changed)
        with self.assertRaisesRegex(ActiveModelError, "no stored regime for the ACTIVE model hmm-1"):
            regime_in_use(self.engine, self.settings)
        self.path.unlink()
        with self.assertRaisesRegex(ActiveModelError, "file .* is missing"):
            calculate_and_store_active_regime(self.engine, self.settings)


class ActivePredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.settings = _settings(cls.root)
        build_market_db(cls.root / "base.db", cls.settings)
        engine = create_db_engine(cls.root / "base.db", 5000)
        cls.model = _train_predictor(engine, cls.settings)
        engine.dispose()
        cls.path = cls.root / "lr-1.joblib"
        cls.model.save(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.db = self.root / f"{self._testMethodName}.db"
        build_market_db(self.db, self.settings)
        self.engine = create_db_engine(self.db, 5000)
        _register(self.engine, "lr-1", "PREDICTOR", "logistic_regression", self.path,
                  self.model.training_end, depends_on="rule_based-v1")

    def tearDown(self):
        self.engine.dispose()

    def test_predictions_are_stored_and_ml_select_drives_the_signals(self):
        run = generate_and_store_signals(self.engine, self.settings)
        stored = queries.load_predictions(self.engine, "lr-1")
        dates = stored.index.get_level_values("date")
        self.assertGreater(dates.min(), self.model.training_end)
        self.assertEqual(dates.max(), DATES[-1])
        self.assertEqual(run.predictions.rows, len(stored))
        targets = run.signals.targets
        self.assertEqual((targets.strategy, targets.model_version), ("ml_select", "lr-1"))
        # live = backtest: ml_select over the stored predictions, at T
        regimes = queries.load_regimes(self.engine, RULE_METHOD, "rule_based-v1")
        weights = ml_select_weights(load_probabilities(self.engine, "lr-1"), regimes, self.settings)
        pd.testing.assert_series_equal(targets.weights, weights.loc[targets.session].reindex(targets.weights.index),
                                       check_names=False)
        signals = queries.load_signals(self.engine, targets.session.date()).set_index("symbol")
        self.assertEqual(set(signals["model_version"]), {"lr-1"})
        self.assertTrue(signals["score"].notna().all())
        self.assertTrue(((signals["confidence"] > 0) & (signals["confidence"] <= 1)).all())
        self.assertTrue(all("ML selection by lr-1" in r for r in signals["reason"]))
        held = targets.stocks[targets.stocks["rule"] == "ml_select"]
        self.assertLessEqual(len(held), 2)

    def test_rerunning_reproduces_the_same_predictions_and_hashes(self):
        generate_and_store_signals(self.engine, self.settings)
        first = queries.load_predictions(self.engine, "lr-1")
        generate_and_store_signals(self.engine, self.settings)
        second = queries.load_predictions(self.engine, "lr-1")
        pd.testing.assert_frame_equal(first, second)  # no duplicates, identical values and hashes

    def test_predictions_are_point_in_time(self):
        stocks = list(self.settings.market.stocks)
        features = queries.load_features_long(self.engine, "technical-v1", stocks, FEATURE_COLUMNS)
        market = queries.load_market_features(self.engine, "market-v1", MARKET_FEATURE_COLUMNS)
        regimes = predictor_regimes(self.engine, self.settings, {"depends_on": "rule_based-v1", "model_id": "lr-1"})
        start = self.model.training_end + pd.Timedelta(days=1)
        full = predict_sessions(self.model, features, market, regimes, start)
        cut = DATES[560]
        dates = features.index.get_level_values("date")
        part = predict_sessions(self.model, features[dates <= cut], market.loc[:cut], regimes.loc[:cut], start)
        pd.testing.assert_frame_equal(part, full[full.index.get_level_values("date") <= cut])

    def test_a_feature_version_mismatch_is_refused(self):
        with self.engine.begin() as conn:
            conn.execute(update(tables.model_versions).values(feature_version="technical-v9"))
        with self.assertRaisesRegex(ActiveModelError, "trained on technical-v9"):
            generate_and_store_signals(self.engine, self.settings)

    def test_a_predictor_on_an_unstored_regime_is_refused(self):
        with self.engine.begin() as conn:
            conn.execute(update(tables.model_versions).values(depends_on="hmm-gone"))
        with self.assertRaisesRegex(ActiveModelError, "needs the regime hmm-gone"):
            generate_and_store_signals(self.engine, self.settings)

    def test_status_names_the_predictor_in_use_and_stale_predictions(self):
        from quant.ops.status import collect_status, format_status

        generate_and_store_signals(self.engine, self.settings)
        report = collect_status(self.engine, self.settings)
        self.assertEqual((report.strategy_in_use, report.signal_strategy), ("ml_select", "ml_select"))
        self.assertIn("signals use             ml_select with the rule_based-v1 regime", format_status(report))
        self.assertFalse(any("predictions of" in w for w in report.warnings))
        with self.engine.begin() as conn:
            conn.execute(tables.predictions.delete().where(tables.predictions.c.date == DATES[-1].date()))
        warnings = " | ".join(collect_status(self.engine, self.settings).warnings)
        self.assertIn(f"predictions of lr-1 end {DATES[-2].date()}, before {DATES[-1].date()}", warnings)

    def test_retiring_it_falls_back_to_the_rule_based_strategy(self):
        with self.engine.begin() as conn:
            conn.execute(update(tables.model_versions).values(status="RETIRED"))
        run = generate_and_store_signals(self.engine, self.settings)
        self.assertEqual((run.signals.targets.strategy, run.model_version), ("regime_switch", "rule_based-v1"))
        self.assertIsNone(run.predictions)

    def test_command_line_prints_the_active_predictor(self):
        document = yaml.safe_load(PROJECT_SETTINGS.read_text(encoding="utf-8"))
        document["database"]["url"] = f"sqlite:///{self.db.as_posix()}"
        document["market"]["symbols"] = {sector: list(stocks) for sector, stocks in POOL.items()}
        document["risk"].update(max_position_size=0.10, max_sector_weight=0.30)
        document["strategies"]["ml_select"].update(top_n=2, keep_rank=2)
        document["positions"]["path"] = (self.root / "none.csv").as_posix()
        config = self.root / f"{self._testMethodName}.yaml"
        config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        output = StringIO()
        with redirect_stdout(output), redirect_stderr(StringIO()):
            code = cli.main(["--config", str(config), "generate-signals", "--reasons"])
        self.assertEqual(code, 0, output.getvalue())
        text = output.getvalue()
        self.assertIn("ACTIVE predictor lr-1 (logistic_regression):", text)
        self.assertIn("ml_select drives the signals", text)
        self.assertIn("ML pick: probability", text)


if __name__ == "__main__":
    unittest.main()
