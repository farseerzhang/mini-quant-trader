import dataclasses
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import date, datetime, timezone
from io import StringIO
from pathlib import Path

import yaml
from sqlalchemy import inspect, select, text
from sqlalchemy.exc import IntegrityError

from quant import cli
from quant.config.settings import DEFAULT_CONFIG_PATH
from quant.ops.promote import PromotionError, promote_model, retire_model
from quant.ops.status import collect_status
from quant.persistence import tables
from quant.persistence.database import create_db_engine, init_db
from tests.settings_helper import SETTINGS

PROJECT_SETTINGS = Path(__file__).resolve().parents[1] / DEFAULT_CONFIG_PATH
FAILED_GATES = {"beats_baseline_after_costs": {"status": "FAIL", "observed": "Sharpe 0.72 vs 0.86", "required": "-"},
                "leak_tests_pass": {"status": "PASS", "observed": "4/4", "required": "-"}}


class PromoteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.engine = create_db_engine(self.root / "promote.db", 5000)
        init_db(self.engine)
        self.settings = dataclasses.replace(SETTINGS, positions=dataclasses.replace(
            SETTINGS.positions, path=str(self.root / "positions.csv")))

    def tearDown(self):
        self.engine.dispose()
        self.temp.cleanup()

    def add(self, model_id, role="PREDICTOR", status="CANDIDATE", passed=True, depends_on="rule_based-v1",
            version=None, file=True):
        path = self.root / f"{model_id}.joblib"
        if file:
            path.write_bytes(b"model")
        model_type = "hmm" if role == "REGIME" else "logistic_regression"
        with self.engine.begin() as conn:
            count = conn.execute(select(tables.model_versions.c.model_id)).all()
            conn.execute(tables.model_versions.insert().values(
                model_id=model_id, role=role, model_type=model_type, version=version or len(count) + 1,
                status=status, depends_on=depends_on if role == "PREDICTOR" else None,
                training_start=date(2010, 1, 1), training_end=date(2020, 1, 1), feature_version="technical-v1",
                target_definition="{}", hyperparameters="{}", metrics="{}", gates_passed=int(passed),
                gate_results=json.dumps({} if passed else FAILED_GATES), trial_count=1, holdout_evaluated=0,
                model_path=str(path), library_versions="{}", config_hash="h", created_at=datetime.now(timezone.utc),
            ))

    def status_of(self):
        with self.engine.connect() as conn:
            rows = conn.execute(select(tables.model_versions.c.model_id, tables.model_versions.c.status)).all()
        return dict(rows)

    def promotions_logged(self):
        with self.engine.connect() as conn:
            return [r.message for r in conn.execute(
                select(tables.pipeline_runs).where(tables.pipeline_runs.c.command == "promote")).all()]

    def test_promotes_a_passing_candidate_and_retires_the_previous_one_of_its_role_only(self):
        self.add("lr-old", status="ACTIVE")
        self.add("hmm-1", role="REGIME", status="ACTIVE", passed=True)
        self.add("lr-new")
        promotion = promote_model(self.engine, self.settings, "lr-new")
        self.assertEqual((promotion.role, promotion.retired, promotion.override_reason), ("PREDICTOR", "lr-old", None))
        self.assertEqual(self.status_of(), {"lr-old": "RETIRED", "hmm-1": "ACTIVE", "lr-new": "ACTIVE"})
        self.assertIn("lr-new ACTIVE for PREDICTOR; lr-old RETIRED", self.promotions_logged()[-1])

    def test_failed_gates_are_refused_without_an_override(self):
        self.add("lr-bad", passed=False)
        with self.assertRaisesRegex(PromotionError, r"failed its gates: beats_baseline_after_costs \(Sharpe 0.72"):
            promote_model(self.engine, self.settings, "lr-bad")
        self.assertEqual(self.status_of(), {"lr-bad": "CANDIDATE"})
        self.assertEqual(self.promotions_logged(), [])

    def test_an_override_promotes_and_logs_the_reason(self):
        self.add("lr-bad", passed=False)
        promotion = promote_model(self.engine, self.settings, "lr-bad", "forward test only")
        self.assertEqual(promotion.override_reason, "forward test only")
        self.assertEqual(promotion.failed_gates, ("beats_baseline_after_costs (Sharpe 0.72 vs 0.86)",))
        self.assertIn("OVERRIDE of failed gates (beats_baseline_after_costs (Sharpe 0.72 vs 0.86)): forward test only",
                      self.promotions_logged()[-1])

    def test_an_override_on_a_passing_candidate_is_not_logged_as_an_override(self):
        self.add("lr-good")
        self.assertIsNone(promote_model(self.engine, self.settings, "lr-good", "just because").override_reason)
        self.assertNotIn("OVERRIDE", self.promotions_logged()[-1])

    def test_refusals(self):
        self.add("lr-active", status="ACTIVE")
        self.add("lr-bad", passed=False)
        self.add("lr-nofile", file=False)
        cases = {
            "unknown": ("nope", None, "no model 'nope'"),
            "already active": ("lr-active", None, "already the ACTIVE PREDICTOR model"),
            "empty override": ("lr-bad", "  ", "needs a reason"),
            "missing file": ("lr-nofile", None, "model file .* is missing"),
        }
        for name, (model_id, override, message) in cases.items():
            with self.subTest(name), self.assertRaisesRegex(PromotionError, message):
                promote_model(self.engine, self.settings, model_id, override)
        self.assertEqual(self.status_of()["lr-active"], "ACTIVE")

    def test_a_retired_model_can_be_promoted_again(self):
        self.add("lr-a", status="ACTIVE")
        self.add("lr-b")
        promote_model(self.engine, self.settings, "lr-b")
        promote_model(self.engine, self.settings, "lr-a")
        self.assertEqual(self.status_of(), {"lr-a": "ACTIVE", "lr-b": "RETIRED"})

    def test_promoting_a_regime_makes_the_active_predictor_stale(self):
        self.add("lr-1", status="ACTIVE", depends_on="rule_based-v1")
        self.add("hmm-1", role="REGIME")
        promotion = promote_model(self.engine, self.settings, "hmm-1")
        self.assertEqual(promotion.stale_predictor, "lr-1")
        self.assertEqual(self.status_of()["lr-1"], "ACTIVE")  # it keeps running meanwhile (section 16.1)
        warnings = " | ".join(collect_status(self.engine, self.settings).warnings)
        self.assertIn("lr-1 was trained on regime rule_based-v1, but the regime in use is hmm-1", warnings)

    def test_a_predictor_trained_on_another_regime_is_flagged_when_promoted(self):
        self.add("lr-hmm", depends_on="hmm-1")
        self.assertEqual(promote_model(self.engine, self.settings, "lr-hmm").stale_predictor, "lr-hmm")

    def test_retire_falls_back_to_rule_based_and_promote_undoes_it(self):
        self.add("hmm-1", role="REGIME")
        self.add("lr-1", status="ACTIVE")
        promote_model(self.engine, self.settings, "hmm-1")
        self.assertEqual(retire_model(self.engine, self.settings, "hmm-1", "too much turnover"), "REGIME")
        self.assertEqual(self.status_of(), {"hmm-1": "RETIRED", "lr-1": "ACTIVE"})  # the other role untouched
        with self.engine.connect() as conn:
            message = conn.execute(select(tables.pipeline_runs.c.message)
                                   .where(tables.pipeline_runs.c.command == "retire")).scalar_one()
        self.assertEqual(message, "hmm-1 RETIRED; REGIME falls back to the rule-based regime; reason: too much turnover")
        promote_model(self.engine, self.settings, "hmm-1")
        self.assertEqual(self.status_of()["hmm-1"], "ACTIVE")

    def test_retire_refuses_a_model_that_is_not_active(self):
        self.add("lr-1")
        for model_id, message in (("lr-1", "is CANDIDATE, not ACTIVE"), ("nope", "no model 'nope'")):
            with self.subTest(model_id), self.assertRaisesRegex(PromotionError, message):
                retire_model(self.engine, self.settings, model_id)

    def test_the_database_allows_one_active_model_per_role(self):
        self.add("lr-1", status="ACTIVE")
        with self.assertRaises(IntegrityError):
            self.add("lr-2", status="ACTIVE")
        self.add("hmm-1", role="REGIME", status="ACTIVE")  # another role is fine

    def test_init_db_adds_the_index_to_an_existing_database(self):
        with self.engine.begin() as conn:
            conn.execute(text("DROP INDEX uq_model_versions_one_active_per_role"))
        init_db(self.engine)
        names = {index["name"] for index in inspect(self.engine).get_indexes("model_versions")}
        self.assertIn("uq_model_versions_one_active_per_role", names)


class PromoteCommandTests(unittest.TestCase):
    def test_refusal_and_override_from_the_command_line(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = yaml.safe_load(PROJECT_SETTINGS.read_text(encoding="utf-8"))
            document["database"]["url"] = f"sqlite:///{(root / 'cli.db').as_posix()}"
            config = root / "settings.yaml"
            config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
            engine = create_db_engine(root / "cli.db", 5000)
            init_db(engine)
            model = root / "hmm-1.joblib"
            model.write_bytes(b"model")
            with engine.begin() as conn:
                conn.execute(tables.model_versions.insert().values(
                    model_id="hmm-1", role="REGIME", model_type="hmm", version=1, status="CANDIDATE",
                    training_start=date(2010, 1, 1), training_end=date(2020, 1, 1), feature_version="market-v1",
                    target_definition="{}", hyperparameters="{}", metrics="{}", gates_passed=0,
                    gate_results=json.dumps(FAILED_GATES), trial_count=1, holdout_evaluated=0,
                    model_path=str(model), library_versions="{}", config_hash="h",
                    created_at=datetime.now(timezone.utc),
                ))
            engine.dispose()

            def run_command(*args):
                output, errors = StringIO(), StringIO()
                with redirect_stdout(output), redirect_stderr(errors):
                    code = cli.main(["--config", str(config), *args])
                return code, output.getvalue(), errors.getvalue()

            def run(*args):
                return run_command("promote", *args)

            code, _, errors = run("hmm-1")
            self.assertEqual(code, 1)
            self.assertIn("Promotion refused: hmm-1 failed its gates", errors)
            code, output, _ = run("hmm-1", "--override", "forward test")
            self.assertEqual(code, 0)
            self.assertIn("hmm-1 is now the ACTIVE REGIME model.", output)
            self.assertIn("OVERRIDE logged", output)
            self.assertIn("Undo with `quant retire hmm-1` (back to the rule-based regime)", output)
            code, output, _ = run_command("retire", "hmm-1", "--reason", "test")
            self.assertEqual(code, 0)
            self.assertIn("hmm-1 is RETIRED. REGIME now uses the rule-based regime", output)
            code, _, errors = run_command("retire", "hmm-1")
            self.assertEqual(code, 1)
            self.assertIn("Retire refused: hmm-1 is RETIRED, not ACTIVE", errors)


if __name__ == "__main__":
    unittest.main()
