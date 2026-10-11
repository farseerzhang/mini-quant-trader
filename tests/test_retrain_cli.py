import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import date, datetime, timezone
from io import StringIO
from pathlib import Path

import yaml
from sqlalchemy import select

from quant.cli import main
from quant.config.settings import DEFAULT_CONFIG_PATH
from quant.persistence import queries, tables
from quant.persistence.database import create_db_engine
from tests.backtest_fixture import POOL, build_market_db
from tests.test_registry import _tiny_settings

PROJECT_SETTINGS = Path(__file__).resolve().parents[1] / DEFAULT_CONFIG_PATH


class RetrainCommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_dir = tempfile.TemporaryDirectory()
        cls.fixture_db = Path(cls.fixture_dir.name) / "market.db"
        build_market_db(cls.fixture_db, _tiny_settings())

    @classmethod
    def tearDownClass(cls):
        cls.fixture_dir.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.db = root / "retrain.db"
        shutil.copy(self.fixture_db, self.db)
        document = yaml.safe_load(PROJECT_SETTINGS.read_text(encoding="utf-8"))
        document["database"]["url"] = f"sqlite:///{self.db.as_posix()}"
        document["market"]["symbols"] = POOL
        document["model"]["calibration_months"] = 2
        document["retrain"]["validation"].update(min_train_years=1, fold_length_months=2, min_folds=1, holdout_months=2)
        document["retrain"]["search"]["space"] = {"logistic_regression": {"C": [0.01, 0.1]}}
        document["retrain"]["models_dir"] = (root / "models").as_posix()
        document["retrain"]["reports_dir"] = (root / "reports").as_posix()
        document["strategies"]["ml_select"].update(top_n=2, keep_rank=2)
        document["backtest"]["random_baseline_runs"] = 2
        self.config = root / "settings.yaml"
        self.config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def run_cli(self, *args):
        output, errors = StringIO(), StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            code = main(["--config", str(self.config), *args])
        return code, output.getvalue(), errors.getvalue()

    def models(self):
        engine = create_db_engine(self.db, 5000)
        with engine.connect() as conn:
            rows = conn.execute(select(tables.model_versions).order_by(tables.model_versions.c.created_at)).mappings().all()
        views = queries.count_holdout_evaluations(engine)
        engine.dispose()
        return rows, views

    def test_retrain_registers_a_candidate_and_reports_without_touching_the_holdout(self):
        code, output, _ = self.run_cli("retrain", "--role", "predictor", "--model", "logistic_regression")
        self.assertEqual(code, 0)
        for text in ("Search (2 trials", "Walk-forward backtest", "Gates:", "as CANDIDATE", "No ACTIVE predictor"):
            self.assertIn(text, output)
        self.assertNotIn("HOLDOUT", output)
        rows, views = self.models()
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["status"], rows[0]["holdout_evaluated"], rows[0]["trial_count"]), ("CANDIDATE", 0, 2))
        self.assertTrue(Path(rows[0]["model_path"]).exists())
        self.assertTrue(Path(rows[0]["validation_report_path"]).exists())
        self.assertEqual(views, 0)

    def test_evaluate_holdout_checks_its_gate_and_shares_the_count_with_backtest(self):
        code, output, _ = self.run_cli("retrain", "--role", "predictor", "--model", "logistic_regression", "--evaluate-holdout")
        self.assertEqual(code, 0)
        self.assertIn("(evaluation #1; logged in pipeline_runs)", output)
        rows, views = self.models()
        self.assertEqual((rows[0]["holdout_evaluated"], views), (1, 1))
        self.assertNotIn('"holdout_not_worse_than_baseline": {"status": "NOT EVALUATED"', rows[0]["gate_results"])
        _, output, _ = self.run_cli("backtest", "--evaluate-holdout")
        self.assertIn("(evaluation #2;", output)  # one shared holdout, one shared count

    def test_an_active_model_is_reported_and_left_unchanged(self):
        engine = create_db_engine(self.db, 5000)
        with engine.begin() as conn:
            conn.execute(tables.model_versions.insert().values(
                model_id="logistic_regression-old", role="PREDICTOR", model_type="logistic_regression", version=1,
                status="ACTIVE", training_start=date(2020, 1, 1), training_end=date(2024, 1, 1),
                feature_version="technical-v1", target_definition="{}", hyperparameters="{}", metrics="{}",
                gates_passed=1, gate_results="{}", trial_count=1, holdout_evaluated=0, model_path="x",
                library_versions="{}", config_hash="h", created_at=datetime.now(timezone.utc),
            ))
        engine.dispose()
        code, output, _ = self.run_cli("retrain", "--role", "predictor", "--model", "logistic_regression")
        self.assertEqual(code, 0)
        self.assertIn("ACTIVE predictor stays logistic_regression-old", output)
        rows, _ = self.models()
        self.assertEqual({row["model_id"]: row["status"] for row in rows}["logistic_regression-old"], "ACTIVE")
        self.assertEqual([row["version"] for row in rows], [1, 2])

    def test_unavailable_roles_and_models_are_clear_errors(self):
        code, _, errors = self.run_cli("retrain", "--role", "regime")
        self.assertEqual(code, 1)
        self.assertIn("Phase 7", errors)
        code, _, errors = self.run_cli("retrain", "--role", "predictor", "--model", "xgboost")
        self.assertEqual(code, 1)
        self.assertIn("Phase 6", errors)
        self.assertEqual(self.models()[0], [])


if __name__ == "__main__":
    unittest.main()
