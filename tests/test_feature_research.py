import unittest

import numpy as np
import pandas as pd

from quant.backtest.data import Prices
from quant.models.feature_research import candidate_features, ic_summary, rank_ic

STOCKS = [f"S{i}" for i in range(12)]
SECTORS = {stock: ("energy" if i % 2 else "utilities") for i, stock in enumerate(STOCKS)}


def _market(n=320, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", periods=n)
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (n, 13)), axis=0), index=index, columns=[*STOCKS, "SPY"])
    open_ = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.002, (n, 13)))
    volume = pd.DataFrame(rng.integers(1_000, 10_000, (n, 12)).astype(float), index=index, columns=STOCKS)
    return Prices(open=open_, close=close), volume


class RankICTests(unittest.TestCase):
    def test_perfect_and_inverse_rankings(self):
        index = pd.bdate_range("2020-01-01", periods=5)
        target = pd.DataFrame(np.tile(np.arange(12.0), (5, 1)), index=index, columns=STOCKS)
        self.assertTrue(np.allclose(rank_ic(target * 2 + 1, target), 1.0))
        self.assertTrue(np.allclose(rank_ic(-target, target), -1.0))

    def test_a_feature_equal_for_every_stock_has_no_ic(self):
        index = pd.bdate_range("2020-01-01", periods=5)
        target = pd.DataFrame(np.random.default_rng(1).random((5, 12)), index=index, columns=STOCKS)
        same = pd.DataFrame(3.0, index=index, columns=STOCKS)  # like every market feature
        self.assertTrue(rank_ic(same, target).isna().all())

    def test_summary_counts_years_with_the_same_sign(self):
        ic = pd.Series([0.02] * 300 + [-0.01] * 200, index=pd.bdate_range("2020-01-01", periods=500))
        summary = ic_summary(ic, horizon=5)
        self.assertGreater(summary["mean_ic"], 0)
        self.assertEqual(summary["years"], 2)


class CandidateFeatureTests(unittest.TestCase):
    def test_every_candidate_at_t_uses_data_up_to_t_only(self):
        prices, volume = _market()
        full = candidate_features(prices, volume, STOCKS, SECTORS)
        cut = prices.close.index[290]
        truncated = candidate_features(
            Prices(open=prices.open.loc[:cut], close=prices.close.loc[:cut]), volume.loc[:cut], STOCKS, SECTORS
        )
        for name, frame in full.items():
            with self.subTest(name):
                pd.testing.assert_series_equal(frame.loc[cut], truncated[name].loc[cut], check_names=False)
                self.assertTrue(frame.loc[cut].notna().all())

    def test_sector_relative_returns_sum_to_zero_within_a_sector(self):
        prices, volume = _market()
        relative = candidate_features(prices, volume, STOCKS, SECTORS)["sector_relative_return_5d"].iloc[-1]
        totals = relative.groupby(pd.Series(SECTORS)).sum()
        self.assertTrue(np.allclose(totals, 0.0))



class InFoldSelectionTests(unittest.TestCase):
    def test_selection_sees_only_the_fold_fit_rows(self):
        from tests.test_models import SHORT, _folds, _synthetic
        from quant.models.dataset import Dataset
        from quant.models.feature_research import ic_selector, with_candidates
        from quant.models.walk_forward import run_walk_forward

        dataset = _synthetic(signal=0.0, seed=8)
        folds = _folds(dataset)
        dates = dataset.frame.index.get_level_values("date")
        excess = pd.Series(np.random.default_rng(2).normal(size=len(dataset.frame)), index=dataset.frame.index)
        frame = dataset.frame.copy()
        frame["excess"] = excess
        frame["label"] = (excess > 0.35).astype(float)
        noise = np.random.default_rng(3).normal(size=len(frame))
        # "late" predicts the target only after the first fold's training window ends.
        late = np.where(dates > folds[0].train_end, excess.to_numpy(), noise)
        # "early" predicts it everywhere.
        candidates = {
            "late": pd.Series(late, index=frame.index).unstack("symbol"),
            "early": (excess + 0.5 * pd.Series(noise, index=frame.index)).unstack("symbol"),
        }
        extended = with_candidates(Dataset(frame), candidates)
        result = run_walk_forward(
            extended, folds[:1], "logistic_regression", {"C": 0.1}, SHORT,
            select_inputs=ic_selector(["late", "early"], horizon=5),
        )
        self.assertEqual(result.folds.iloc[0]["inputs"], "early")  # "late" would need the future

    def test_at_least_one_input_is_kept(self):
        from quant.models.dataset import Dataset
        from quant.models.feature_research import ic_selector

        index = pd.MultiIndex.from_product([pd.bdate_range("2020-01-01", periods=60), STOCKS], names=["date", "symbol"])
        rng = np.random.default_rng(5)
        frame = pd.DataFrame({"a": rng.normal(size=len(index)), "b": rng.normal(size=len(index)),
                              "excess": rng.normal(size=len(index))}, index=index)
        self.assertEqual(len(ic_selector(["a", "b"], horizon=5, min_abs_t=99)(Dataset(frame))), 1)


if __name__ == "__main__":
    unittest.main()
