import dataclasses
import unittest

import pandas as pd

from quant.models.gates import DISABLED, FAIL, PASS, LeakCheck
from quant.regime.gates import RegimeEvidence, default_cuts, evaluate_regime_gates, point_in_time_checks
from quant.regime.hmm_walk_forward import run_hmm_walk_forward
from tests.settings_helper import SETTINGS
from tests.test_hmm_walk_forward import SHORT, _setup

GATES = dict(SETTINGS.retrain.gates["regime"])  # the configured gates


def _evidence(**changes) -> RegimeEvidence:
    base = RegimeEvidence(
        point_in_time_checks=[LeakCheck("a", True, ""), LeakCheck("b", True, "")],
        agreement=pd.Series([0.95, 0.99, 0.97], index=[2, 3, 4]),
        average_dwell=32.0,
        sharpe=0.90,
        baseline_sharpe=0.86,
    )
    return dataclasses.replace(base, **changes)


def _status(report, name):
    return report.table().loc[name, "status"]


class RegimeGateTests(unittest.TestCase):
    def test_passing_evidence_passes_in_config_order(self):
        report = evaluate_regime_gates(GATES, _evidence())
        self.assertTrue(report.passed)
        self.assertEqual(list(report.table().index), list(GATES))

    def test_each_threshold_at_its_boundary(self):
        cases = {
            "min_avg_regime_dwell_days": (dict(average_dwell=10.0), dict(average_dwell=9.99)),
            "beats_rule_based_regime_after_costs": (dict(sharpe=0.8601), dict(sharpe=0.86)),
            "state_mapping_stable_across_refits": (
                dict(agreement=pd.Series([0.80, 0.99], index=[2, 3])),
                dict(agreement=pd.Series([0.7999, 0.99], index=[2, 3])),  # one pair below is enough to fail
            ),
        }
        for gate, (passing, failing) in cases.items():
            with self.subTest(gate):
                self.assertEqual(_status(evaluate_regime_gates(GATES, _evidence(**passing)), gate), PASS)
                failed = evaluate_regime_gates(GATES, _evidence(**failing))
                self.assertEqual(_status(failed, gate), FAIL)
                self.assertFalse(failed.passed)

    def test_point_in_time_gate_needs_every_cut_identical(self):
        report = evaluate_regime_gates(GATES, _evidence(point_in_time_checks=[LeakCheck("a", True, ""), LeakCheck("b", False, "")]))
        self.assertEqual(_status(report, "filtered_only_test_pass"), FAIL)
        none = evaluate_regime_gates(GATES, _evidence(point_in_time_checks=[]))
        self.assertEqual(_status(none, "filtered_only_test_pass"), FAIL)  # no evidence is not a pass

    def test_disabled_gates_do_not_block_and_unknown_gates_are_errors(self):
        report = evaluate_regime_gates({**GATES, "beats_rule_based_regime_after_costs": False}, _evidence(sharpe=0.1))
        self.assertEqual(_status(report, "beats_rule_based_regime_after_costs"), DISABLED)
        self.assertTrue(report.passed)
        with self.assertRaisesRegex(ValueError, "no evaluator"):
            evaluate_regime_gates({**GATES, "beats_spy": True}, _evidence())

    def test_same_evidence_same_report(self):
        evidence = _evidence(sharpe=0.5)
        pd.testing.assert_frame_equal(
            evaluate_regime_gates(GATES, evidence).table(), evaluate_regime_gates(GATES, evidence).table()
        )


class PointInTimeCheckTests(unittest.TestCase):
    def test_a_correct_walk_forward_passes_and_a_tampered_one_fails(self):
        observations, _, folds, _ = _setup()
        result = run_hmm_walk_forward(observations, folds, SHORT)
        cuts = default_cuts(folds)
        self.assertTrue(all(c.passed for c in point_in_time_checks(observations, folds, SHORT, result, cuts)))

        # Tamper: pretend a regime early in the window had been revised after the fact.
        changed = result.raw_regime.copy()
        first = changed.index[5]
        changed.loc[first] = "BEAR" if changed.loc[first] != "BEAR" else "BULL"
        tampered = dataclasses.replace(result, raw_regime=changed)
        self.assertFalse(any(c.passed for c in point_in_time_checks(observations, folds, SHORT, tampered, cuts)))


if __name__ == "__main__":
    unittest.main()
