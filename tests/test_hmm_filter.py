import unittest
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM

from quant.regime.hmm import HMMParameters, filtered_probabilities


def _regime_data(sessions=1500, seed=0, separation=1.0):
    """Synthetic market-like observations from three known hidden states.

    ``separation`` < 1 makes the states overlap, so the filter is often unsure."""
    rng = np.random.default_rng(seed)
    means = np.array([[0.5, -0.6, -0.6], [0.0, 0.0, 0.0], [-1.0, 1.4, 1.4]]) * separation
    transition = np.array([[0.97, 0.025, 0.005], [0.03, 0.95, 0.02], [0.02, 0.05, 0.93]])
    state, states, x = 0, [], []
    for _ in range(sessions):
        state = rng.choice(3, p=transition[state])
        states.append(state)
        x.append(rng.normal(means[state], 0.6))
    return np.array(x), np.array(states)


def _fit(x, seed=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # hmmlearn on numpy 2.5 (section 13.1)
        return GaussianHMM(n_components=3, covariance_type="diag", n_iter=200, random_state=seed).fit(x)


class ForwardFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.states = _regime_data(separation=0.6)  # overlapping states: a hard case
        cls.model = _fit(cls.x)
        cls.parameters = HMMParameters.from_hmmlearn(cls.model)
        cls.filtered = filtered_probabilities(cls.parameters, cls.x)

    def test_probabilities_are_valid(self):
        self.assertEqual(self.filtered.shape, (len(self.x), 3))
        self.assertTrue(np.allclose(self.filtered.sum(axis=1), 1.0))
        self.assertTrue(((self.filtered >= 0) & (self.filtered <= 1)).all())

    def test_appending_later_sessions_never_changes_the_state_at_t(self):
        """The core point-in-time property (section 13, rule 1)."""
        for t in (0, 1, 250, 777, len(self.x) - 2):
            truncated = filtered_probabilities(self.parameters, self.x[: t + 1])
            np.testing.assert_array_equal(truncated[t], self.filtered[t])

    def test_matches_hmmlearn_where_filtering_and_smoothing_coincide(self):
        """At the last session of a sequence the smoothed posterior equals the filtered one,
        which checks this implementation against hmmlearn's."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            for t in (0, 10, 400, 1200, len(self.x) - 1):
                smoothed_last = self.model.predict_proba(self.x[: t + 1])[-1]
                np.testing.assert_allclose(self.filtered[t], smoothed_last, atol=1e-8)

    def test_smoothed_probabilities_look_ahead_and_filtered_do_not(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            smoothed = self.model.predict_proba(self.x)
        # With overlapping states, knowing the future often changes the estimate a lot.
        self.assertGreater(np.abs(smoothed - self.filtered).max(), 0.3)
        self.assertGreater((np.abs(smoothed - self.filtered).max(axis=1) > 0.1).mean(), 0.05)

    def test_long_histories_and_extreme_observations_stay_finite(self):
        x, _ = _regime_data(sessions=20_000, seed=3)
        x[5_000] = [40.0, -40.0, 40.0]  # a wildly unusual session
        probabilities = filtered_probabilities(self.parameters, x)
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0))

    def test_recovers_clearly_separated_states_with_the_best_of_several_starts(self):
        # One start can stop in a poor local optimum (here seed 0: 59% recovery); as
        # pre-registered (section 13.1), the best of 5 starts by log-likelihood is used.
        x, states = _regime_data(seed=1, separation=1.0)
        model = max((_fit(x, seed) for seed in range(5)), key=lambda m: m.score(x))
        guessed = filtered_probabilities(HMMParameters.from_hmmlearn(model), x).argmax(axis=1)
        # States are numbered arbitrarily by the fit: map each to its most common true state.
        mapping = {g: np.bincount(states[guessed == g], minlength=3).argmax() for g in np.unique(guessed)}
        accuracy = (np.vectorize(mapping.get)(guessed) == states).mean()
        self.assertGreater(accuracy, 0.90)  # 93% filtered; smoothing (with look-ahead) reaches 98%


if __name__ == "__main__":
    unittest.main()
