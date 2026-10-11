import dataclasses
import unittest

import numpy as np
import pandas as pd

from quant.regime.hmm import LABELS_BY_MEAN_RETURN, fit_hmm, load_observations
from tests.settings_helper import SETTINGS
from tests.test_hmm_filter import _regime_data

HMM = SETTINGS.regime.hmm  # spy_return_1d, log_market_volatility, log_vix; 3 states; 5 starts
COLUMNS = list(HMM.observations)


def _observations(sessions=1500, seed=1):
    """Synthetic sessions in roughly real units; true state 0 = calm-up, 2 = crash."""
    x, states = _regime_data(sessions=sessions, seed=seed, separation=1.0)
    frame = pd.DataFrame(
        {
            "spy_return_1d": 0.0005 + 0.004 * x[:, 0],
            "log_market_volatility": -1.9 + 0.3 * x[:, 1],
            "log_vix": 2.9 + 0.2 * x[:, 2],
        },
        index=pd.bdate_range("2010-01-01", periods=sessions),
    )
    return frame, states


class FitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.observations, cls.states = _observations()
        cls.train = cls.observations.iloc[:1000]
        cls.fit = fit_hmm(cls.train, HMM, seed=42)

    def test_standardization_comes_from_the_training_window_only(self):
        np.testing.assert_allclose(self.fit.mean, self.train[COLUMNS].mean().to_numpy())
        np.testing.assert_allclose(self.fit.std, self.train[COLUMNS].std(ddof=0).to_numpy())
        self.assertEqual((self.fit.train_start, self.fit.train_end), (self.train.index[0], self.train.index[-1]))

    def test_states_are_labelled_by_mean_return(self):
        described = self.fit.describe()
        self.assertEqual(list(described.index), list(LABELS_BY_MEAN_RETURN))
        returns = described["spy_return_1d mean"]
        self.assertGreater(returns["BULL"], returns["SIDEWAYS"])
        self.assertGreater(returns["SIDEWAYS"], returns["BEAR"])
        self.assertEqual(sorted(self.fit.labels), sorted(LABELS_BY_MEAN_RETURN))
        # The crash state is also the volatile one, as in the synthetic data.
        self.assertEqual(described["log_market_volatility mean"].idxmax(), "BEAR")

    def test_the_best_start_is_kept_and_the_fit_is_reproducible(self):
        self.assertEqual(len(self.fit.start_log_likelihoods), HMM.starts)
        self.assertAlmostEqual(self.fit.log_likelihood, max(self.fit.start_log_likelihoods))
        again = fit_hmm(self.train, HMM, seed=42)
        # Equal up to floating-point rounding (multi-threaded maths differs around 1e-14),
        # and the regimes it gives are identical.
        np.testing.assert_allclose(again.parameters.means, self.fit.parameters.means, rtol=0, atol=1e-10)
        self.assertEqual(again.labels, self.fit.labels)
        pd.testing.assert_series_equal(
            again.filtered(self.observations).idxmax(axis=1), self.fit.filtered(self.observations).idxmax(axis=1)
        )

    def test_filtered_regimes_on_later_sessions_follow_the_true_states(self):
        later = self.observations.iloc[1000:]
        probabilities = self.fit.filtered(later)
        self.assertEqual(list(probabilities.columns), list(LABELS_BY_MEAN_RETURN))
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0))
        regime = probabilities.idxmax(axis=1).to_numpy()
        truth = np.array(["BULL", "SIDEWAYS", "BEAR"])[self.states[1000:]]
        self.assertGreater((regime == truth).mean(), 0.85)  # out of sample, filtered only

    def test_too_little_data_and_unknown_observations_are_errors(self):
        with self.assertRaisesRegex(ValueError, "too few"):
            fit_hmm(self.train.iloc[:100], HMM, seed=42)
        bad = dataclasses.replace(
            SETTINGS, regime=dataclasses.replace(SETTINGS.regime, hmm=dataclasses.replace(HMM, observations=("rsi14",)))
        )
        with self.assertRaisesRegex(ValueError, "not market features"):
            load_observations(None, bad)


if __name__ == "__main__":
    unittest.main()
