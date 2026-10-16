"""Section 12.7: the interest-rate shock inputs and their trial; offline, hand-made data."""

import tempfile
import unittest
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from quant.data.data_quality import validate_prices
from quant.models.folds import Fold
from quant.regime.comparison import RegimeComparison, portfolio_name
from quant.regime.rate_shocks_trial import (
    BASELINE,
    SINGLE_INPUT_VARIANTS,
    RateShocksTrialOutcome,
    evaluate,
    point_in_time_checks,
    trial_regimes,
    write_rate_shocks_report,
)
from quant.regime.rule_based import BEAR, BULL, SIDEWAYS, score_inputs
from quant.regime.rule_rate_shocks import (
    LONG_RATE_JUMP,
    NAME,
    SHORT_RATE_DROP,
    rate_shock_inputs,
    rate_shock_scores,
)
from settings_helper import SETTINGS

SESSIONS = pd.bdate_range("2010-01-04", periods=300)
REGIME = SETTINGS.regime  # E = 3, X = 1, VIX 20 / 30, dwell 5


def yields(short, long, index=SESSIONS):
    return pd.DataFrame({"^IRX": short, "^TNX": long}, index=index, dtype=float)


def v1_inputs(sign: float, index=SESSIONS) -> pd.DataFrame:
    return pd.DataFrame({"spy_trend": sign, "qqq_trend": sign, "spy_sma50_vs_sma200": sign, "spy_return_3m": sign,
                         "vix": 25.0}, index=index)


class InputsTest(unittest.TestCase):
    def test_thresholds_warm_up_and_never_plus_one(self):
        short = np.full(len(SESSIONS), 2.0)
        short[100:] = 1.4  # a 0.60-point drop
        short[200:] = 1.0  # a further 0.40 drop: not enough
        long = np.full(len(SESSIONS), 3.0)
        long[150:] = 3.8  # a 0.80-point jump
        inputs = rate_shock_inputs(yields(short, long))
        self.assertTrue(inputs[SHORT_RATE_DROP].iloc[:20].isna().all())  # warm-up: NULL, not 0
        self.assertTrue(inputs[LONG_RATE_JUMP].iloc[:60].isna().all())
        self.assertEqual(inputs[SHORT_RATE_DROP].iloc[100], -1)
        self.assertEqual(inputs[SHORT_RATE_DROP].iloc[119], -1)  # still within 20 sessions of the drop
        self.assertEqual(inputs[SHORT_RATE_DROP].iloc[120], 0)
        self.assertEqual(inputs[SHORT_RATE_DROP].iloc[200], 0)  # 0.40 points: below the threshold
        self.assertEqual(inputs[LONG_RATE_JUMP].iloc[150], -1)
        self.assertEqual(inputs[LONG_RATE_JUMP].iloc[210], 0)
        values = inputs.to_numpy().ravel()
        self.assertTrue(set(values[~np.isnan(values)]) <= {0.0, -1.0})

    def test_one_missing_session_is_filled_two_are_not(self):
        short = pd.Series(2.0, index=SESSIONS)
        short.iloc[50] = np.nan
        short.iloc[80:82] = np.nan
        inputs = rate_shock_inputs(yields(short, 3.0))
        self.assertEqual(inputs[SHORT_RATE_DROP].iloc[50], 0)  # the previous session's value
        self.assertEqual(inputs[SHORT_RATE_DROP].iloc[80], 0)
        self.assertTrue(np.isnan(inputs[SHORT_RATE_DROP].iloc[81]))  # a second missing session: NULL

    def test_scores_only_subtract(self):
        inputs = v1_inputs(1.0).join(pd.DataFrame({SHORT_RATE_DROP: -1.0, LONG_RATE_JUMP: 0.0}, index=SESSIONS))
        self.assertTrue((rate_shock_scores(inputs, REGIME) == score_inputs(inputs, REGIME) - 1).all())


class RegimesTest(unittest.TestCase):
    def test_rate_shocks_push_the_candidate_out_of_bull(self):
        # Both start in BULL (+4). From session 100 VIX is above 30 and both rate inputs fire:
        # v1 scores +3, the candidate +1, which keeps BULL (>= X = 1, hysteresis).
        inputs = v1_inputs(1.0).join(pd.DataFrame({SHORT_RATE_DROP: 0.0, LONG_RATE_JUMP: 0.0}, index=SESSIONS))
        inputs.iloc[:30, inputs.columns.get_loc(SHORT_RATE_DROP)] = np.nan  # warm-up
        inputs.iloc[100:, inputs.columns.get_loc(SHORT_RATE_DROP)] = -1.0
        inputs.iloc[100:, inputs.columns.get_loc(LONG_RATE_JUMP)] = -1.0
        inputs.iloc[100:, inputs.columns.get_loc("vix")] = 35.0
        regimes = trial_regimes(inputs, REGIME)
        self.assertEqual(set(regimes), {BASELINE, NAME, *SINGLE_INPUT_VARIANTS})
        self.assertEqual({r.index[0] for r in regimes.values()}, {SESSIONS[30]})  # one common first session
        self.assertEqual(regimes[BASELINE].iloc[-1], BULL)
        self.assertEqual(regimes[NAME].iloc[-1], BULL)
        inputs.iloc[100:, inputs.columns.get_loc("spy_trend")] = -1.0  # now v1 scores +1 (BULL), the candidate -1
        regimes = trial_regimes(inputs, REGIME)
        self.assertEqual(regimes[BASELINE].iloc[-1], BULL)
        self.assertEqual(regimes[NAME].iloc[-1], SIDEWAYS)

    def test_bear_is_reached_with_the_rate_inputs(self):
        inputs = v1_inputs(-1.0).join(pd.DataFrame({SHORT_RATE_DROP: 0.0, LONG_RATE_JUMP: 0.0}, index=SESSIONS))
        inputs["spy_trend"] = 1.0  # v1: -3 + 1 = -2, not BEAR
        self.assertNotEqual(trial_regimes(inputs, REGIME)[BASELINE].iloc[-1], BEAR)
        inputs[SHORT_RATE_DROP] = -1.0  # candidate: -3
        self.assertEqual(trial_regimes(inputs, REGIME)[NAME].iloc[-1], BEAR)

    def test_point_in_time(self):
        rng = np.random.default_rng(4)
        series = yields(2 + np.cumsum(rng.normal(0, 0.05, len(SESSIONS))), 3 + np.cumsum(rng.normal(0, 0.05, len(SESSIONS))))
        series.iloc[120, 0] = np.nan
        inputs = v1_inputs(1.0).assign(spy_trend=rng.normal(size=len(SESSIONS)),
                                       vix=rng.uniform(12, 40, len(SESSIONS))).join(rate_shock_inputs(series))
        regimes = trial_regimes(inputs, REGIME)
        checks = point_in_time_checks(inputs, series, REGIME, regimes, [SESSIONS[150], SESSIONS[250]])
        self.assertTrue(all(c.identical for c in checks))


class GatesAndReportTest(unittest.TestCase):
    def test_gates_and_report(self):
        index = SESSIONS[100:]
        names = [BASELINE, NAME, *SINGLE_INPUT_VARIANTS]
        regimes = {name: pd.Series(BULL, index=SESSIONS) for name in names}
        regimes[NAME].iloc[150:200] = SIDEWAYS
        folds = [Fold(1, SESSIONS[:100], index[:100]), Fold(2, SESSIONS[:200], index[100:])]
        portfolios = [portfolio_name(s, r) for s in ("regime_switch", "regime_switch_bear50") for r in names]
        sharpe = {p: 0.9 for p in portfolios} | {portfolio_name("regime_switch", NAME): 0.95}
        overall = pd.DataFrame({"sharpe": sharpe, "max_drawdown": -0.2, "cagr": 0.1})
        by_fold = pd.DataFrame([{"fold": f, "portfolio": p, "sharpe": sharpe[p] + f / 10, "total_return": 0.1,
                                 "max_drawdown": -0.1} for f in (1, 2) for p in portfolios]
                               ).set_index(["fold", "portfolio"])
        comparison = RegimeComparison(index[0], index[-1], {}, overall, by_fold)
        gates = evaluate(comparison, regimes[NAME], [])
        self.assertEqual([r.status for r in gates.results], ["PASS", "PASS", "FAIL", "PASS", "FAIL"])
        inputs = pd.DataFrame({SHORT_RATE_DROP: 0.0, LONG_RATE_JUMP: -1.0}, index=SESSIONS)
        outcome = RateShocksTrialOutcome(folds, SESSIONS[0], regimes, comparison, gates, inputs,
                                         pd.Series(100.0, index=SESSIONS))
        with tempfile.TemporaryDirectory() as directory:
            path = write_rate_shocks_report(outcome, directory, datetime(2026, 10, 16, tzinfo=timezone.utc))
            text = path.read_text(encoding="utf-8")
        self.assertEqual(path.name, "rate-shocks-trial-20261016T000000Z.md")
        self.assertIn("## Gates: FAILED", text)
        self.assertIn("| 2010 | 0% | 100% |", text)


class YieldQualityTest(unittest.TestCase):
    def test_negative_yields_are_kept_negative_prices_are_not(self):
        bars = pd.DataFrame({"Open": [-0.03], "High": [-0.02], "Low": [-0.04], "Close": [-0.03], "Volume": [0.0]},
                            index=pd.to_datetime(["2020-03-19"]))
        good, bad = validate_prices("^IRX", bars)
        self.assertEqual((len(good), len(bad)), (1, 0))
        good, bad = validate_prices("AAPL", bars)
        self.assertEqual((len(good), bad[0]["reason"]), (0, "non-positive price"))


if __name__ == "__main__":
    unittest.main()
