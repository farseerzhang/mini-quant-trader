import unittest

import numpy as np
import pandas as pd

from quant.backtest.data import Prices
from quant.models.folds import Fold
from quant.regime.comparison import compare_regimes, portfolio_name
from quant.regime.rule_based import BEAR, BULL
from quant.strategies import PANEL_COLUMNS
from tests.test_ml_backtest import SETTINGS_4, STOCKS


def _market(n=300, seed=4):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2019-01-01", periods=n)
    close = pd.DataFrame(100 * np.cumprod(1 + 0.0004 + rng.normal(0, 0.01, (n, 5)), axis=0),
                         index=index, columns=[*STOCKS, "SPY"])
    prices = Prices(open=close.shift(1).fillna(close.iloc[0]), close=close)
    # Every stock in a clean uptrend, so `trend` (used in BULL and BEAR) holds them all.
    panel = {name: pd.DataFrame(0.05, index=index, columns=STOCKS) for name in PANEL_COLUMNS}
    panel["rsi14"] = pd.DataFrame(60.0, index=index, columns=STOCKS)
    folds = [Fold(1, index[:100], index[110:200]), Fold(2, index[:194], index[200:290])]
    return prices, panel, folds, index


class CompareRegimesTests(unittest.TestCase):
    def test_only_the_regime_differs_and_everything_shares_one_window(self):
        prices, panel, folds, index = _market()
        always_bull = pd.Series(BULL, index=index, dtype=object)
        always_bear = pd.Series(BEAR, index=index, dtype=object)
        result = compare_regimes({"bull": always_bull, "bear": always_bear}, folds, prices, panel, SETTINGS_4)

        self.assertEqual((result.start, result.end), (index[110], index[289]))
        for name, sim in result.simulations.items():
            self.assertEqual((sim.value.index[0], sim.value.index[-1]), (result.start, result.end), name)
        bull, bear = portfolio_name("regime_switch", "bull"), portfolio_name("regime_switch", "bear")
        # BULL: trend holdings at full size; BEAR with the baseline exposure 0: all cash.
        self.assertGreater(result.overall.loc[bull, "avg_invested"], 0.3)
        self.assertEqual(result.overall.loc[bear, "avg_invested"], 0.0)
        # The bear50 variant keeps half the trend positions in BEAR.
        half = portfolio_name("regime_switch_bear50", "bear")
        self.assertAlmostEqual(result.overall.loc[half, "avg_invested"],
                               result.overall.loc[portfolio_name("regime_switch_bear50", "bull"), "avg_invested"] / 2, places=2)
        comparison = result.versus(bull, bear)
        self.assertEqual(comparison["folds"], 2)
        self.assertEqual(len(result.by_fold), 2 * len(result.simulations))

    def test_a_regime_with_gaps_in_the_window_is_rejected(self):
        prices, panel, folds, index = _market()
        gappy = pd.Series(BULL, index=index[:150], dtype=object)
        with self.assertRaisesRegex(ValueError, "no value"):
            compare_regimes({"gappy": gappy}, folds, prices, panel, SETTINGS_4)


if __name__ == "__main__":
    unittest.main()
