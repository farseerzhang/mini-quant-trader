import dataclasses
import unittest

import numpy as np
import pandas as pd

from quant.backtest.data import Prices
from quant.models.folds import Fold
from quant.models.ml_backtest import BASELINE, check_out_of_sample, probability_panel, run_ml_backtest
from quant.models.walk_forward import WalkForwardResult
from quant.regime.rule_based import BEAR, BULL, SIDEWAYS
from quant.strategies import PANEL_COLUMNS
from quant.strategies.ml import ml_regime_weights, top_n_holdings
from tests.settings_helper import SETTINGS

POOL = {"information_technology": ("AAA", "BBB"), "energy": ("CCC",), "utilities": ("DDD",)}
SETTINGS_4 = dataclasses.replace(
    SETTINGS,
    market=dataclasses.replace(SETTINGS.market, symbols=POOL),
    strategies=dataclasses.replace(
        SETTINGS.strategies, ml_regime=dataclasses.replace(SETTINGS.strategies.ml_regime, top_n=2)
    ),
    backtest=dataclasses.replace(SETTINGS.backtest, random_baseline_runs=2),
)
STOCKS = ["AAA", "BBB", "CCC", "DDD"]


def _dates(n, start="2020-01-01"):
    return pd.bdate_range(start, periods=n)


class TopNTests(unittest.TestCase):
    def test_selection_refreshes_only_every_n_sessions(self):
        probabilities = pd.DataFrame(
            [[0.5, 0.4, 0.3, 0.2], [0.1, 0.2, 0.9, 0.8], [0.1, 0.2, 0.9, 0.8], [0.1, 0.2, 0.9, 0.8]],
            index=_dates(4), columns=STOCKS,
        )
        held = top_n_holdings(probabilities, top_n=2, every=3)
        self.assertEqual(list(held.iloc[0]), [True, True, False, False])
        self.assertEqual(list(held.iloc[2]), [True, True, False, False])  # still the session-0 choice
        self.assertEqual(list(held.iloc[3]), [False, False, True, True])  # re-chosen on session 3

    def test_missing_probabilities_are_never_selected(self):
        probabilities = pd.DataFrame([[np.nan, 0.4, np.nan, 0.2]], index=_dates(1), columns=STOCKS)
        self.assertEqual(list(top_n_holdings(probabilities, 3, 5).iloc[0]), [False, True, False, True])


class MLRegimeWeightTests(unittest.TestCase):
    def test_regime_sets_exposure_every_session_and_caps_apply(self):
        probabilities = pd.DataFrame([[0.6, 0.5, 0.4, 0.3]] * 4, index=_dates(4), columns=STOCKS)
        regimes = pd.Series([BULL, SIDEWAYS, BEAR, np.nan], index=_dates(4), dtype=object)
        weights = ml_regime_weights(probabilities, regimes, SETTINGS_4)
        self.assertEqual(list(weights.iloc[0]), [0.10, 0.10, 0.0, 0.0])  # top 2, 10% cap each
        self.assertEqual(list(weights.iloc[1]), [0.10, 0.10, 0.0, 0.0])
        self.assertEqual(weights.iloc[2].sum(), 0.0)  # BEAR between rebalances: cash at once
        self.assertEqual(weights.iloc[3].sum(), 0.0)  # no regime: nothing

    def test_truncation_does_not_change_weights_up_to_t(self):
        rng = np.random.default_rng(4)
        probabilities = pd.DataFrame(rng.random((40, 4)), index=_dates(40), columns=STOCKS)
        regimes = pd.Series(np.array([BULL, SIDEWAYS, BEAR])[(np.arange(40) // 7) % 3], index=_dates(40))
        full = ml_regime_weights(probabilities, regimes, SETTINGS_4)
        cut = _dates(40)[23]
        pd.testing.assert_frame_equal(full.loc[:cut], ml_regime_weights(probabilities.loc[:cut], regimes, SETTINGS_4))


def _fold(number, train, validation):
    return Fold(number=number, train=pd.DatetimeIndex(train), validation=pd.DatetimeIndex(validation))


class OutOfSampleGuardTests(unittest.TestCase):
    def setUp(self):
        self.sessions = _dates(60)
        self.fold = _fold(1, self.sessions[:30], self.sessions[36:60])
        index = pd.MultiIndex.from_product([self.sessions[36:60], STOCKS], names=["date", "symbol"])
        self.predictions = pd.DataFrame({"raw": 0.4, "probability": 0.4, "label": 0.0, "fold": 1}, index=index)

    def test_accepts_purged_out_of_sample_predictions(self):
        check_out_of_sample(self.predictions, [self.fold], self.sessions, embargo=6)
        self.assertEqual(probability_panel(self.predictions).shape, (24, 4))

    def test_rejects_a_fold_trained_too_close_to_its_window(self):
        too_close = _fold(1, self.sessions[:33], self.sessions[36:60])
        with self.assertRaisesRegex(ValueError, "trained 3 session"):
            check_out_of_sample(self.predictions, [too_close], self.sessions, embargo=6)

    def test_rejects_predictions_outside_the_fold_window(self):
        late = self.predictions.copy()
        late.index = pd.MultiIndex.from_product([self.sessions[30:54], STOCKS], names=["date", "symbol"])
        with self.assertRaisesRegex(ValueError, "outside its validation window"):
            check_out_of_sample(late, [self.fold], self.sessions, embargo=6)


class RunMLBacktestTests(unittest.TestCase):
    def test_every_portfolio_on_the_same_window_with_per_fold_comparison(self):
        n = 400
        sessions = _dates(n, "2019-01-01")
        rng = np.random.default_rng(9)
        drift = {"AAA": 0.0008, "BBB": 0.0002, "CCC": -0.0002, "DDD": 0.0001, "SPY": 0.0003}
        close = pd.DataFrame(
            {s: 100 * np.cumprod(1 + d + rng.normal(0, 0.01, n)) for s, d in drift.items()}, index=sessions
        )
        prices = Prices(open=close.shift(1).fillna(close.iloc[0]), close=close)
        folds = [_fold(1, sessions[:200], sessions[206:300]), _fold(2, sessions[:294], sessions[300:390])]
        rows = []
        for fold in folds:
            for date in fold.validation:
                for symbol, p in zip(STOCKS, (0.45, 0.40, 0.30, 0.35)):  # always prefers AAA and BBB
                    rows.append({"date": date, "symbol": symbol, "raw": p, "probability": p, "label": 0.0, "fold": fold.number})
        predictions = pd.DataFrame(rows).set_index(["date", "symbol"])
        result = WalkForwardResult("logistic_regression", {"C": 0.01}, pd.DataFrame(), predictions)
        panel = {name: pd.DataFrame(0.05, index=sessions, columns=STOCKS) for name in PANEL_COLUMNS}
        regimes = pd.Series(BULL, index=sessions, dtype=object)

        report = run_ml_backtest({"ml_logistic_regression": result}, folds, prices, panel, regimes, SETTINGS_4)

        self.assertEqual((report.start, report.end), (sessions[206], sessions[389]))
        for name, simulation in report.simulations.items():
            self.assertEqual((simulation.value.index[0], simulation.value.index[-1]), (report.start, report.end), name)
        self.assertEqual(set(report.overall.index), {"ml_logistic_regression", BASELINE, "SPY", "equal_weight"})
        self.assertEqual(len(report.by_fold), 2 * 4)
        held = report.simulations["ml_logistic_regression"].orders["Column"].unique()
        self.assertEqual(sorted(held), ["AAA", "BBB"])
        comparison = report.versus_baseline("ml_logistic_regression")
        self.assertEqual(comparison["folds"], 2)
        self.assertIn(comparison["fraction_higher_sharpe"], (0.0, 0.5, 1.0))
        self.assertEqual(int(report.random.loc["ml_logistic_regression", "runs"]), 2)


if __name__ == "__main__":
    unittest.main()
