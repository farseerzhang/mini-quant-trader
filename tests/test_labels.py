import dataclasses
import unittest

import numpy as np
import pandas as pd

from quant.models.labels import future_window_return, make_labels, volatility_threshold
from tests.settings_helper import SETTINGS

TARGET = SETTINGS.model.target  # h = 5, k = 0.25, 20-session volatility, 0.5% floor


def _dates(n):
    return pd.bdate_range("2026-01-05", periods=n)


class FutureReturnTests(unittest.TestCase):
    def test_window_runs_from_the_open_after_t_to_h_sessions_later(self):
        opens = pd.Series([100.0, 101, 102, 103, 104, 105, 106, 107, 108], index=_dates(9))
        returns = future_window_return(opens, horizon=5)
        # T = day 0: entry at the open of day 1 (101), exit at the open of day 1 + 5 = day 6 (106).
        self.assertAlmostEqual(returns.iloc[0], 106 / 101 - 1)
        self.assertTrue(returns.iloc[-6:].isna().all())  # the last h + 1 sessions have no exit yet
        self.assertFalse(np.isnan(returns.iloc[-7]))


class LabelTests(unittest.TestCase):
    def setUp(self):
        n = 40
        self.index = _dates(n)
        rng = np.random.default_rng(3)
        self.close = pd.DataFrame(
            {"A": 100 * np.cumprod(1 + rng.normal(0, 0.01, n)), "B": 50 * np.cumprod(1 + rng.normal(0, 0.03, n))},
            index=self.index,
        )
        self.open = self.close.shift(1).fillna(self.close.iloc[0]) * 1.001
        self.spy = pd.Series(200 * np.cumprod(1 + rng.normal(0, 0.008, n)), index=self.index)

    def test_hand_calculation(self):
        labels = make_labels(self.open, self.close, self.spy, TARGET)
        t = 25
        stock = self.open["B"].iloc[t + 6] / self.open["B"].iloc[t + 1] - 1
        bench = self.spy.iloc[t + 6] / self.spy.iloc[t + 1] - 1
        sigma = self.close["B"].pct_change().iloc[t - 19 : t + 1].std()
        threshold = max(TARGET.k * sigma * np.sqrt(5), TARGET.min_threshold)
        self.assertAlmostEqual(labels.excess["B"].iloc[t], stock - bench)
        self.assertAlmostEqual(labels.threshold["B"].iloc[t], threshold)
        self.assertEqual(labels.label["B"].iloc[t], float(stock - bench > threshold))

    def test_floor_applies_to_low_volatility_stocks(self):
        flat = pd.DataFrame({"A": 100 + 0.001 * np.arange(40)}, index=self.index)
        threshold = volatility_threshold(flat, TARGET)
        self.assertTrue((threshold.dropna() == TARGET.min_threshold).all().all())

    def test_unknown_labels_stay_missing(self):
        labels = make_labels(self.open, self.close, self.spy, TARGET)
        self.assertTrue(labels.label.iloc[:19].isna().all().all())  # volatility warm-up (20 returns need 21 closes)
        self.assertTrue(labels.label.iloc[-6:].isna().all().all())  # no exit open yet
        self.assertTrue(labels.label.iloc[20:-6].notna().all().all())
        self.assertTrue(set(np.unique(labels.label.dropna(how="all").stack())) <= {0.0, 1.0})

    def test_label_at_t_uses_only_opens_after_t_and_threshold_only_data_up_to_t(self):
        t = 25
        base = make_labels(self.open, self.close, self.spy, TARGET)

        # Changing the open AT T (or earlier) does not change the window return at T.
        open_at_t = self.open.copy()
        open_at_t.iloc[: t + 1] *= 1.5
        self.assertAlmostEqual(
            make_labels(open_at_t, self.close, self.spy, TARGET).excess["A"].iloc[t], base.excess["A"].iloc[t]
        )
        # Changing prices AFTER T does not change the threshold at T.
        close_after = self.close.copy()
        close_after.iloc[t + 1 :] *= 3.0
        self.assertAlmostEqual(
            make_labels(self.open, close_after, self.spy, TARGET).threshold["A"].iloc[t], base.threshold["A"].iloc[t]
        )
        # The exit open (T+1+h) does change it: the label really looks h sessions ahead.
        open_exit = self.open.copy()
        open_exit.iloc[t + 6] *= 1.2
        self.assertNotAlmostEqual(
            make_labels(open_exit, self.close, self.spy, TARGET).excess["A"].iloc[t], base.excess["A"].iloc[t]
        )
        # ... and nothing after the exit does.
        open_late = self.open.copy()
        open_late.iloc[t + 7 :] *= 1.2
        self.assertAlmostEqual(
            make_labels(open_late, self.close, self.spy, TARGET).excess["A"].iloc[t], base.excess["A"].iloc[t]
        )

    def test_other_target_kinds_are_not_silently_used(self):
        fixed = dataclasses.replace(TARGET, kind="fixed_return")
        with self.assertRaises(NotImplementedError):
            make_labels(self.open, self.close, self.spy, fixed)


if __name__ == "__main__":
    unittest.main()
