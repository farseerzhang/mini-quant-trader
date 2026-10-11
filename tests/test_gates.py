import dataclasses
import unittest

import pandas as pd

from quant.models.folds import Fold, walk_forward_folds
from quant.models.gates import (
    DISABLED,
    FAIL,
    NOT_EVALUATED,
    PASS,
    LeakCheck,
    PredictorEvidence,
    evaluate_predictor_gates,
    leak_checks,
)
from quant.models.walk_forward import run_walk_forward
from tests.settings_helper import SETTINGS
from tests.test_models import SHORT, _synthetic

GATES = dict(SETTINGS.retrain.gates["predictor"])  # the configured gates


def _evidence(**changes) -> PredictorEvidence:
    """Evidence that passes every configured gate; override fields to probe one gate."""
    base = PredictorEvidence(
        leak_checks=[LeakCheck("a", True, ""), LeakCheck("b", True, "")],
        calibration_error=0.02,
        sharpe=1.0,
        baseline_sharpe=0.9,
        max_drawdown=-0.20,
        baseline_max_drawdown=-0.21,
        folds=10,
        folds_higher_sharpe=6,
    )
    return dataclasses.replace(base, **changes)


def _status(report, name):
    return report.table().loc[name, "status"]


class GateEvaluationTests(unittest.TestCase):
    def test_passing_evidence_passes_and_the_holdout_is_not_evaluated(self):
        report = evaluate_predictor_gates(GATES, _evidence())
        self.assertTrue(report.passed)
        self.assertEqual(list(report.table().index), list(GATES))  # config order
        self.assertEqual(_status(report, "holdout_not_worse_than_baseline"), NOT_EVALUATED)

    def test_each_threshold_at_its_boundary(self):
        cases = {
            "min_fraction_of_folds_beating_baseline": (dict(folds_higher_sharpe=6), dict(folds_higher_sharpe=5)),
            "max_calibration_error": (dict(calibration_error=0.05), dict(calibration_error=0.0501)),
            "max_drawdown_not_worse_than_baseline": (dict(max_drawdown=-0.21), dict(max_drawdown=-0.2101)),
            "beats_baseline_after_costs": (dict(sharpe=0.9001), dict(sharpe=0.9)),
        }
        for gate, (passing, failing) in cases.items():
            with self.subTest(gate):
                self.assertEqual(_status(evaluate_predictor_gates(GATES, _evidence(**passing)), gate), PASS)
                failed = evaluate_predictor_gates(GATES, _evidence(**failing))
                self.assertEqual(_status(failed, gate), FAIL)
                self.assertFalse(failed.passed)

    def test_one_failed_leak_check_fails_the_leak_gate(self):
        report = evaluate_predictor_gates(GATES, _evidence(leak_checks=[LeakCheck("a", True, ""), LeakCheck("b", False, "")]))
        self.assertEqual(_status(report, "leak_tests_pass"), FAIL)
        self.assertIn("failed: b", report.table().loc["leak_tests_pass", "observed"])

    def test_holdout_gate_when_the_holdout_was_evaluated(self):
        worse = evaluate_predictor_gates(GATES, _evidence(holdout_sharpe=0.5, holdout_baseline_sharpe=0.6))
        self.assertEqual(_status(worse, "holdout_not_worse_than_baseline"), FAIL)
        equal = evaluate_predictor_gates(GATES, _evidence(holdout_sharpe=0.6, holdout_baseline_sharpe=0.6))
        self.assertEqual(_status(equal, "holdout_not_worse_than_baseline"), PASS)

    def test_disabled_gates_do_not_block_and_unknown_gates_are_errors(self):
        disabled = {**GATES, "max_drawdown_not_worse_than_baseline": False}
        report = evaluate_predictor_gates(disabled, _evidence(max_drawdown=-0.9))
        self.assertEqual(_status(report, "max_drawdown_not_worse_than_baseline"), DISABLED)
        self.assertTrue(report.passed)
        with self.assertRaisesRegex(ValueError, "no evaluator"):
            evaluate_predictor_gates({**GATES, "beats_spy": True}, _evidence())

    def test_same_evidence_same_report(self):
        evidence = _evidence(folds_higher_sharpe=5)
        pd.testing.assert_frame_equal(
            evaluate_predictor_gates(GATES, evidence).table(), evaluate_predictor_gates(GATES, evidence).table()
        )


class LeakCheckTests(unittest.TestCase):
    def setUp(self):
        self.dataset = _synthetic(signal=0.5)
        self.sessions = self.dataset.sessions
        self.holdout_start = self.sessions[-120]
        self.folds = walk_forward_folds(
            self.sessions, self.holdout_start, SHORT.retrain.validation, SHORT.model.embargo_days
        )
        self.result = run_walk_forward(
            self.dataset, self.folds, "logistic_regression", {"C": 0.1}, SHORT
        )

    def test_a_correct_walk_forward_passes_every_check(self):
        checks = leak_checks(self.folds, self.result, self.sessions, self.holdout_start, SHORT.model.embargo_days)
        self.assertEqual(len(checks), 4)
        self.assertTrue(all(check.passed for check in checks), [c for c in checks if not c.passed])

    def test_an_unpurged_fold_is_caught(self):
        broken = list(self.folds)
        first = broken[0]
        position = self.sessions.get_loc(first.validation_start)
        broken[0] = Fold(first.number, self.sessions[: position - 2], first.validation)  # only 2 sessions purged
        checks = {c.name: c for c in leak_checks(broken, self.result, self.sessions, self.holdout_start, 6)}
        self.assertFalse(checks["training purged before validation"].passed)
        self.assertFalse(checks["predictions out of sample"].passed)


if __name__ == "__main__":
    unittest.main()
