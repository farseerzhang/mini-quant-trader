import dataclasses
import unittest

import numpy as np
import pandas as pd

from quant.models.folds import walk_forward_folds
from quant.regime.hmm_walk_forward import apply_min_dwell, mapping_agreement, run_hmm_walk_forward
from tests.settings_helper import SETTINGS
from tests.test_hmm_fit import _observations

SHORT = dataclasses.replace(
    SETTINGS,
    retrain=dataclasses.replace(
        SETTINGS.retrain,
        validation=dataclasses.replace(SETTINGS.retrain.validation, min_train_years=2, min_folds=2),
    ),
)


def _setup(sessions=1500):
    observations, states = _observations(sessions=sessions, seed=1)
    holdout_start = observations.index[-120]
    folds = walk_forward_folds(observations.index, holdout_start, SHORT.retrain.validation, SHORT.model.embargo_days)
    return observations, states, folds, holdout_start


class DwellTests(unittest.TestCase):
    def test_same_rule_as_the_rule_based_regime(self):
        index = pd.bdate_range("2020-01-01", periods=9)
        proposed = pd.Series(["BULL", "BEAR", "BEAR", "SIDEWAYS", "SIDEWAYS", "SIDEWAYS", "BEAR", "BEAR", "BEAR"], index=index)
        # min 3: BEAR on days 1-2 is blocked (BULL held 1, then 2 sessions); on day 3 BULL has been
        # held 3 sessions, so the proposal there (SIDEWAYS) is taken - the earlier BEAR was not queued;
        # SIDEWAYS is held 3 sessions (days 3-5), so BEAR on day 6 is taken.
        regime = apply_min_dwell(proposed, 3)
        self.assertEqual(list(regime), ["BULL", "BULL", "BULL", "SIDEWAYS", "SIDEWAYS", "SIDEWAYS", "BEAR", "BEAR", "BEAR"])
        self.assertEqual(list(apply_min_dwell(proposed, 1)), list(proposed))  # min 1 = no smoothing


class WalkForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.observations, cls.states, cls.folds, cls.holdout_start = _setup()
        cls.result = run_hmm_walk_forward(cls.observations, cls.folds, SHORT)

    def test_each_fit_uses_only_its_training_sessions_and_never_the_holdout(self):
        for fit, fold in zip(self.result.fits, self.folds):
            self.assertLessEqual(fit.train_end, fold.train_end)
            self.assertLess(fit.train_end, fold.validation_start)
            self.assertLess(fit.train_end, self.holdout_start)
        self.assertLess(self.result.regime.index.max(), self.holdout_start)

    def test_every_validation_session_has_one_regime_from_its_own_fold(self):
        expected = pd.DatetimeIndex(np.concatenate([fold.validation for fold in self.folds]))
        self.assertTrue(self.result.regime.index.equals(expected))
        for fold in self.folds:
            owners = self.result.fold_of_session.loc[fold.validation]
            self.assertTrue((owners == fold.number).all())

    def test_regime_up_to_t_is_unchanged_when_later_observations_are_removed(self):
        """Point in time across refits: the whole walk-forward on data truncated at T."""
        cut = self.folds[-1].validation[60]
        truncated = run_hmm_walk_forward(self.observations.loc[:cut], self.folds, SHORT)
        common = truncated.raw_regime.index
        pd.testing.assert_series_equal(truncated.raw_regime, self.result.raw_regime.loc[common])
        pd.testing.assert_series_equal(truncated.regime, self.result.regime.loc[common])

    def test_out_of_sample_regimes_follow_the_true_states(self):
        truth = pd.Series(np.array(["BULL", "SIDEWAYS", "BEAR"])[self.states], index=self.observations.index)
        accuracy = (self.result.raw_regime == truth.reindex(self.result.raw_regime.index)).mean()
        self.assertGreater(accuracy, 0.85)

    def test_consecutive_fits_agree_and_a_fit_agrees_fully_with_itself(self):
        self.assertEqual(list(self.result.agreement.index), [fold.number for fold in self.folds[1:]])
        self.assertTrue((self.result.agreement > 0.8).all())
        fit = self.result.fits[0]
        self.assertEqual(mapping_agreement(fit, fit, self.observations), 1.0)


if __name__ == "__main__":
    unittest.main()
