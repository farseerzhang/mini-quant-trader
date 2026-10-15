"""Section 13.3 trial: the rule-based regime with an HMM BEAR trigger; offline, hand-made data."""

import tempfile
import unittest
from datetime import datetime, timezone

import pandas as pd

from quant.models.folds import Fold
from quant.regime.combo_trial import (
    BOTH,
    COMBO,
    HMM,
    V1,
    ComboTrialOutcome,
    PointInTime,
    both_bear,
    combine,
    evaluate,
    rule_point_in_time,
    sell_off_spells,
    write_combo_report,
)
from quant.regime.comparison import RegimeComparison, portfolio_name
from quant.regime.gates import LeakCheck
from quant.regime.rule_based import BEAR, BULL, SIDEWAYS, calculate_rule_regimes
from settings_helper import SETTINGS

DAYS = pd.bdate_range("2018-09-03", periods=8)


def series(values, index=DAYS):
    return pd.Series(values, index=index, dtype=object)


class CombineTest(unittest.TestCase):
    def test_either_bear_else_the_rule(self):
        rule = series([BULL, BULL, SIDEWAYS, BEAR, BEAR, SIDEWAYS, BULL, BULL])
        hmm = series([SIDEWAYS, BEAR, BEAR, BULL, BEAR, SIDEWAYS, SIDEWAYS, BULL])
        self.assertEqual(list(combine(rule, hmm)), [BULL, BEAR, BEAR, BEAR, BEAR, SIDEWAYS, BULL, BULL])
        # the HMM's SIDEWAYS is ignored (first and last two sessions)
        self.assertEqual(list(both_bear(rule, hmm)),
                         [BULL, BULL, SIDEWAYS, SIDEWAYS, BEAR, SIDEWAYS, BULL, BULL])

    def test_only_on_the_hmm_sessions_and_the_rule_must_cover_them(self):
        rule = series([BULL] * 8)
        hmm = series([BEAR] * 3, DAYS[5:])
        self.assertEqual(list(combine(rule, hmm).index), list(DAYS[5:]))
        with self.assertRaises(ValueError):
            combine(rule.iloc[:6], hmm)

    def test_rule_point_in_time(self):
        index = pd.bdate_range("2010-01-04", periods=300)
        wave = pd.Series(range(300), index=index).map(lambda i: 1.0 if (i // 40) % 2 == 0 else -1.0)
        inputs = pd.DataFrame({"spy_trend": wave, "qqq_trend": wave, "spy_sma50_vs_sma200": wave,
                               "spy_return_3m": wave, "vix": 25.0}, index=index)
        rule = calculate_rule_regimes(inputs, SETTINGS.regime)["regime"]
        hmm = series([SIDEWAYS] * 200, index[100:])
        combined = combine(rule, hmm)
        checks = rule_point_in_time(inputs, SETTINGS, rule, hmm, combined, [index[150], index[250]])
        self.assertTrue(all(c.identical for c in checks))
        self.assertEqual(checks[0].sessions, 51)


class SellOffTest(unittest.TestCase):
    def test_entry_and_exit(self):
        days = pd.bdate_range("2018-10-01", periods=60)
        spy = pd.Series([100.0] * 10 + [90.0] * 10 + [80.0] * 10 + [88.0] * 30, index=days)
        regime = series([BULL] * 15 + [BEAR] * 20 + [SIDEWAYS] * 25, days)
        cell = sell_off_spells({"x": regime}, spy).loc["2018 Q4", "x"]
        self.assertEqual(cell, f"in {days[15].date()} (-10%), out {days[35].date()} (+10% off the low)")


class GatesAndReportTest(unittest.TestCase):
    def test_gates_and_report(self):
        sessions = pd.bdate_range("2014-01-01", periods=400)
        index = sessions[100:]
        folds = [Fold(1, sessions[:100], index[:150]), Fold(2, sessions[:250], index[150:])]
        rule = series([BULL] * 400, sessions)
        hmm = series([SIDEWAYS] * 300, index)
        hmm.iloc[120:160] = BEAR
        combined = combine(rule, hmm)
        regimes = {V1: rule, COMBO: combined, BOTH: both_bear(rule, hmm)}
        names = [portfolio_name(s, r) for s in ("regime_switch", "regime_switch_bear50") for r in regimes]
        sharpe = {n: 0.9 for n in names} | {portfolio_name("regime_switch", COMBO): 0.95}
        drawdown = {n: -0.20 for n in names} | {portfolio_name("regime_switch", COMBO): -0.22}
        overall = pd.DataFrame({"sharpe": sharpe, "max_drawdown": drawdown, "cagr": 0.1})
        by_fold = pd.DataFrame([{"fold": f, "portfolio": n, "sharpe": sharpe[n] + f / 10, "total_return": 0.1,
                                 "max_drawdown": -0.1} for f in (1, 2) for n in names]).set_index(["fold", "portfolio"])
        comparison = RegimeComparison(index[0], index[-1], {}, overall, by_fold)
        cut = PointInTime(index[100], 101, True)
        gates = evaluate(comparison, combined, [LeakCheck("x", True, "")], [cut])
        self.assertEqual([r.status for r in gates.results], ["PASS", "FAIL", "FAIL", "PASS", "PASS"])
        # deeper drawdown fails; 2 of 2 years is below the 6 required
        outcome = ComboTrialOutcome(folds, None, {**regimes, HMM: hmm}, comparison, gates,
                                    pd.Series(100.0, index=sessions))
        with tempfile.TemporaryDirectory() as directory:
            path = write_combo_report(outcome, directory, datetime(2026, 10, 15, tzinfo=timezone.utc))
            text = path.read_text(encoding="utf-8")
        self.assertEqual(path.name, "combo-trial-20261015T000000Z.md")
        self.assertIn("## Gates: FAILED", text)
        self.assertIn("regime_switch Sharpe 0.9500 vs 0.9000 with rule_based-v1", text)


if __name__ == "__main__":
    unittest.main()
