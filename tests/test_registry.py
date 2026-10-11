import dataclasses
import json
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import select

from quant.backtest.data import load_prices
from quant.backtest.engine import split_periods
from quant.models.dataset import INPUT_COLUMNS, load_dataset
from quant.models.final import explain, load_final_model, refit_final
from quant.models.folds import development_sessions, walk_forward_folds
from quant.models.gates import collect_evidence, evaluate_predictor_gates
from quant.models.ml_backtest import run_ml_backtest
from quant.models.registry import ACTIVE, CANDIDATE, register_candidate
from quant.models.search import search
from quant.persistence import queries, tables
from quant.persistence.database import create_db_engine, init_db
from quant.strategies import PANEL_COLUMNS
from tests.backtest_fixture import build_market_db, small_settings


def _tiny_settings():
    base = small_settings(random_runs=2)
    return dataclasses.replace(
        base,
        model=dataclasses.replace(base.model, calibration_months=2),
        retrain=dataclasses.replace(
            base.retrain,
            validation=dataclasses.replace(
                base.retrain.validation, min_train_years=1, fold_length_months=2, min_folds=1, holdout_months=2
            ),
            search=dataclasses.replace(base.retrain.search, space={"logistic_regression": {"C": (0.01, 0.1)}}),
        ),
        strategies=dataclasses.replace(
            base.strategies, ml_select=dataclasses.replace(base.strategies.ml_select, top_n=2, keep_rank=2)
        ),
    )


class RegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.settings = settings = _tiny_settings()
        build_market_db(cls.root / "registry.db", settings)
        engine = create_db_engine(cls.root / "registry.db", settings.database.busy_timeout_ms)
        try:
            stocks = list(settings.market.stocks)
            cls.dataset = load_dataset(engine, settings)
            cls.holdout_start = split_periods(cls.dataset.sessions, settings.holdout_months).holdout_start
            cls.folds = walk_forward_folds(
                cls.dataset.labelled().sessions, cls.holdout_start, settings.retrain.validation, settings.model.embargo_days
            )
            cls.search = search(cls.dataset, cls.folds, "logistic_regression", settings)
            prices = load_prices(engine, [*stocks, "SPY"])
            panel = queries.load_feature_panel(engine, settings.features.feature_version, stocks, PANEL_COLUMNS)
            regimes = queries.load_regimes(engine, "RULE", settings.regime.rule_version)
            cls.backtest = run_ml_backtest(
                {"ml_logistic_regression": cls.search.best}, cls.folds, prices, panel, regimes, settings
            )
            cls.evidence = collect_evidence(
                "ml_logistic_regression", cls.search.best, cls.backtest, cls.folds, prices.close.index,
                cls.holdout_start, settings.model.embargo_days,
            )
            cls.gates = evaluate_predictor_gates(settings.retrain.gates["predictor"], cls.evidence)
            cls.final = refit_final(cls.dataset, cls.holdout_start, cls.search.best, settings)
        finally:
            engine.dispose()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.db = self.root / f"{self._testMethodName}.db"
        self.engine = create_db_engine(self.db, 5000)
        init_db(self.engine)

    def tearDown(self):
        self.engine.dispose()

    def register(self):
        return register_candidate(
            self.engine, self.settings, self.final, self.search, self.folds, self.backtest,
            "ml_logistic_regression", self.evidence, self.gates,
            models_dir=self.root / "models", reports_dir=self.root / "reports",
        )

    def rows(self):
        with self.engine.connect() as conn:
            return conn.execute(select(tables.model_versions).order_by(tables.model_versions.c.version)).mappings().all()

    def test_final_refit_uses_only_purged_development_data(self):
        usable = development_sessions(self.dataset.labelled().sessions, self.holdout_start, self.settings.model.embargo_days)
        self.assertLessEqual(self.final.training_end, usable[-1])
        self.assertLess(self.final.training_end, self.holdout_start)
        self.assertGreater(self.final.training_rows, 0)
        self.assertEqual(self.final.parameters, self.search.best_parameters)
        self.assertEqual(set(explain(self.final).index), set(INPUT_COLUMNS))

    def test_saved_model_reloads_with_identical_predictions(self):
        registration = self.register()
        reloaded = load_final_model(registration.model_path)
        inputs = self.dataset.inputs.iloc[-50:]
        pd.testing.assert_series_equal(self.final.predict(inputs), reloaded.predict(inputs))
        self.assertTrue(((reloaded.predict(inputs) > 0) & (reloaded.predict(inputs) < 1)).all())

    def test_registration_records_a_candidate_and_never_touches_other_models(self):
        now = datetime.now(timezone.utc)
        active = {
            "model_id": "existing-active", "role": "PREDICTOR", "model_type": "logistic_regression", "version": 1,
            "status": ACTIVE, "training_start": date(2010, 1, 1), "training_end": date(2020, 1, 1),
            "feature_version": "technical-v1", "target_definition": "{}", "hyperparameters": "{}", "metrics": "{}",
            "gates_passed": 1, "gate_results": "{}", "trial_count": 1, "holdout_evaluated": 0,
            "model_path": "x", "library_versions": "{}", "config_hash": "h", "created_at": now,
        }
        with self.engine.begin() as conn:
            conn.execute(tables.model_versions.insert().values(**active))

        first, second = self.register(), self.register()
        rows = {row["model_id"]: row for row in self.rows()}
        self.assertEqual(rows["existing-active"]["status"], ACTIVE)  # unchanged
        self.assertEqual((first.version, second.version), (2, 3))
        self.assertNotEqual(first.model_id, second.model_id)

        row = rows[first.model_id]
        self.assertEqual((row["status"], row["role"]), (CANDIDATE, "PREDICTOR"))
        self.assertEqual(row["gates_passed"], int(self.gates.passed))
        self.assertEqual(row["trial_count"], 2)
        self.assertEqual(row["holdout_evaluated"], 0)
        self.assertEqual(row["depends_on"], "rule_based-v1")
        self.assertEqual(json.loads(row["hyperparameters"]), self.search.best_parameters)
        self.assertEqual(json.loads(row["target_definition"])["horizon_days"], 5)
        self.assertEqual(set(json.loads(row["gate_results"])), set(self.settings.retrain.gates["predictor"]))
        self.assertIn("scikit-learn", json.loads(row["library_versions"]))
        self.assertEqual(len(row["config_hash"]), 64)

    def test_report_is_written_with_every_section(self):
        registration = self.register()
        text = registration.report_path.read_text(encoding="utf-8")
        for heading in ("# Validation report: " + registration.model_id, "## Gates", "## Walk-forward ML backtest",
                        "## Probability quality per fold", "## Hyperparameter search", "## What the model leans on",
                        "## Leak checks", "## Caveats"):
            self.assertIn(heading, text)
        self.assertIn("The holdout was not evaluated.", text)


if __name__ == "__main__":
    unittest.main()
