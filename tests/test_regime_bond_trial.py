"""Section 12.5 trial: bond-market fear inputs in the rule score; offline, hand-made data."""

import tempfile
import unittest
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from quant.models.folds import Fold
from quant.regime.bond_trial import (
    BOND_INPUTS,
    CREDIT,
    SAFE_HAVEN,
    SINGLES,
    V1,
    V2,
    BondTrialOutcome,
    bond_inputs,
    candidate_scores,
    evaluate,
    point_in_time,
    trial_regimes,
    write_trial_report,
)
from quant.regime.comparison import RegimeComparison, portfolio_name
from quant.regime.rule_based import BEAR, BULL, SIDEWAYS
from settings_helper import SETTINGS

SESSIONS = pd.bdate_range("2010-01-04", periods=400)
REGIME = SETTINGS.regime  # E = 3, X = 1, VIX 20 / 30, dwell 5


def v1_inputs(score_sign: float, vix: float = 25.0, index=SESSIONS) -> pd.DataFrame:
    """The five v1 inputs: four signs all `score_sign`, VIX between the thresholds (scores 0)."""
    return pd.DataFrame({"spy_trend": score_sign, "qqq_trend": score_sign, "spy_sma50_vs_sma200": score_sign,
                         "spy_return_3m": score_sign, "vix": vix}, index=index)


def closes(seed=0, index=SESSIONS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0003, 0.01, size=(len(index), 4))
    return pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)), index=index, columns=["SPY", "TLT", "IEF", "HYG"])


class InputsTest(unittest.TestCase):
    def test_twenty_session_differences_up_to_t(self):
        close = closes()
        inputs = bond_inputs(close)
        t = 30
        hyg, ief = close["HYG"], close["IEF"]
        expected = (hyg.iloc[t] / hyg.iloc[t - 20] - 1) - (ief.iloc[t] / ief.iloc[t - 20] - 1)
        self.assertAlmostEqual(inputs[CREDIT].iloc[t], expected)
        self.assertTrue(inputs.iloc[:20].isna().all().all())  # warm-up: NULL, not 0
        later = close.copy()
        later.iloc[t + 1:] *= 3  # what happens after T does not change T
        self.assertAlmostEqual(bond_inputs(later)[SAFE_HAVEN].iloc[t], inputs[SAFE_HAVEN].iloc[t])

    def test_scores_add_the_signs_and_need_every_input(self):
        inputs = v1_inputs(1.0).iloc[:3].assign(**{CREDIT: [0.01, -0.02, np.nan], SAFE_HAVEN: [0.03, -0.01, 0.0]})
        scores = candidate_scores(inputs, REGIME, BOND_INPUTS)
        self.assertEqual(scores.iloc[0], 6)  # 4 + 0 (VIX) + 1 + 1
        self.assertEqual(scores.iloc[1], 2)  # 4 - 1 - 1
        self.assertTrue(np.isnan(scores.iloc[2]))  # a missing input is never 0
        self.assertEqual(candidate_scores(inputs, REGIME, (SAFE_HAVEN,)).iloc[2], 4)


class RegimesTest(unittest.TestCase):
    def test_bond_fear_keeps_the_candidate_out_of_bull(self):
        # v1 scores +4 (>= E = 3: BULL); the bonds say risk-off, so v2 scores +2 (< E = 4).
        inputs = v1_inputs(1.0).assign(**{CREDIT: -0.01, SAFE_HAVEN: -0.01})
        inputs.iloc[:5, inputs.columns.get_loc(CREDIT)] = np.nan  # the first sessions lack HYG
        regimes = trial_regimes(inputs, REGIME)
        self.assertEqual(set(regimes), {V1, V2, *SINGLES})
        starts = {regime.index[0] for regime in regimes.values()}
        self.assertEqual(starts, {SESSIONS[5]})  # every recursion starts on the first complete session
        self.assertEqual(regimes[V1].iloc[-1], BULL)
        self.assertEqual(regimes[V2].iloc[-1], SIDEWAYS)
        self.assertEqual(regimes[f"v1 + {SAFE_HAVEN}"].iloc[-1], SIDEWAYS)  # six inputs: 4 - 1 = 3 < 4

    def test_bear_needs_minus_four(self):
        inputs = v1_inputs(-1.0).assign(**{CREDIT: 0.01, SAFE_HAVEN: -0.01})  # -4 + 1 - 1 = -4
        regimes = trial_regimes(inputs, REGIME)
        self.assertEqual(regimes[V2].iloc[-1], BEAR)
        flipped = v1_inputs(-1.0).assign(**{CREDIT: 0.01, SAFE_HAVEN: 0.01})  # -4 + 2 = -2
        self.assertEqual(trial_regimes(flipped, REGIME)[V2].iloc[-1], SIDEWAYS)

    def test_point_in_time(self):
        close = closes(1)
        rng = np.random.default_rng(2)
        inputs = v1_inputs(1.0).assign(spy_trend=rng.normal(size=len(SESSIONS)),
                                       vix=rng.uniform(12, 40, len(SESSIONS))).join(bond_inputs(close))
        regimes = trial_regimes(inputs, REGIME)
        checks = point_in_time(inputs, close, REGIME, regimes, [SESSIONS[150], SESSIONS[300]])
        self.assertTrue(all(c.identical for c in checks))
        self.assertEqual(checks[0].sessions, 150 - 20 + 1)


class GatesAndReportTest(unittest.TestCase):
    def test_gates_and_report(self):
        index = SESSIONS[100:]
        regimes = {name: pd.Series(BULL, index=SESSIONS) for name in (V1, V2, *SINGLES)}
        regimes[V2].iloc[200:260] = SIDEWAYS
        folds = [Fold(1, SESSIONS[:100], index[:150]), Fold(2, SESSIONS[:250], index[150:])]
        names = [portfolio_name(s, r) for s in ("regime_switch", "regime_switch_bear50") for r in regimes]
        sharpe = {name: 0.9 for name in names} | {portfolio_name("regime_switch", V2): 0.95}
        overall = pd.DataFrame({"sharpe": sharpe, "cagr": 0.1, "max_drawdown": -0.2})
        by_fold = pd.DataFrame([{"fold": f, "portfolio": n, "sharpe": sharpe[n] + f / 10, "total_return": 0.1,
                                 "max_drawdown": -0.1} for f in (1, 2) for n in names]).set_index(["fold", "portfolio"])
        comparison = RegimeComparison(index[0], index[-1], {}, overall, by_fold)
        gates = evaluate(comparison, regimes, [])
        self.assertEqual([r.status for r in gates.results], ["PASS", "PASS", "FAIL"])  # no cut checked: FAIL
        close = closes(3)
        inputs = v1_inputs(1.0).join(bond_inputs(close))
        outcome = BondTrialOutcome(folds, SESSIONS[0], regimes, comparison, [], gates, close["SPY"], inputs)
        with tempfile.TemporaryDirectory() as directory:
            path = write_trial_report(outcome, directory, datetime(2026, 10, 15, tzinfo=timezone.utc))
            text = path.read_text(encoding="utf-8")
        self.assertEqual(path.name, "bond-trial-20261015T000000Z.md")
        self.assertIn("## Gates: FAILED", text)
        self.assertIn("regime_switch Sharpe 0.9500 vs 0.9000", text)
        self.assertIn("higher in 2 of 2 validation years", text)


if __name__ == "__main__":
    unittest.main()
