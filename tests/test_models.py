import dataclasses
import unittest

import numpy as np
import pandas as pd

from quant.models.calibration import Calibrator
from quant.models.dataset import INPUT_COLUMNS, Dataset
from quant.models.evaluation import calibration_error, probability_metrics
from quant.models.folds import walk_forward_folds
from quant.models.walk_forward import run_walk_forward, split_calibration
from tests.settings_helper import SETTINGS

# Short history for tests: 2 training years, yearly folds, at least 3 folds.
SHORT = dataclasses.replace(
    SETTINGS,
    retrain=dataclasses.replace(
        SETTINGS.retrain,
        validation=dataclasses.replace(SETTINGS.retrain.validation, min_train_years=2, min_folds=3),
    ),
)
PARAMETERS = {"logistic_regression": {"C": 0.1}, "random_forest": {"max_depth": 4, "min_samples_leaf": 20}}


def _synthetic(signal: float, sessions=1500, symbols=12, seed=0) -> Dataset:
    """Random inputs; the label depends on input 0 with strength ``signal`` (0 = pure noise)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2014-01-01", periods=sessions)
    index = pd.MultiIndex.from_product([dates, [f"S{i}" for i in range(symbols)]], names=["date", "symbol"])
    inputs = rng.normal(size=(len(index), len(INPUT_COLUMNS)))
    score = signal * inputs[:, 0] + rng.normal(size=len(index))
    label = (score > np.quantile(score, 0.63)).astype(float)  # ~37% positive, like the real label
    frame = pd.DataFrame(inputs, index=index, columns=list(INPUT_COLUMNS))
    frame["label"], frame["excess"], frame["threshold"] = label, 0.0, 0.005
    return Dataset(frame)


def _folds(dataset: Dataset):
    sessions = dataset.sessions
    holdout_start = sessions[-120]
    return walk_forward_folds(sessions, holdout_start, SHORT.retrain.validation, SHORT.model.embargo_days)


class MetricTests(unittest.TestCase):
    def test_calibration_error_on_known_cases(self):
        labels = np.array([0, 1] * 50, dtype=float)
        self.assertAlmostEqual(calibration_error(labels, np.full(100, 0.5)), 0.0)
        self.assertAlmostEqual(calibration_error(np.ones(100), np.full(100, 0.5)), 0.5)
        # Two groups, each predicted at its true rate: perfectly calibrated.
        probabilities = np.r_[np.full(50, 0.2), np.full(50, 0.8)]
        labels = np.r_[np.repeat([1.0, 0.0], [10, 40]), np.repeat([1.0, 0.0], [40, 10])]
        self.assertAlmostEqual(calibration_error(labels, probabilities, bins=2), 0.0)

    def test_the_base_rate_reference_has_zero_skill(self):
        labels = np.random.default_rng(1).random(1000) < 0.37
        metrics = probability_metrics(labels, np.full(1000, 0.37), base_rate=0.37)
        self.assertAlmostEqual(metrics["log_loss_skill"], 0.0)
        self.assertAlmostEqual(metrics["brier_skill"], 0.0)
        self.assertAlmostEqual(metrics["auc"], 0.5)


class CalibratorTests(unittest.TestCase):
    def test_isotonic_and_sigmoid_map_scores_to_observed_rates(self):
        rng = np.random.default_rng(2)
        raw = rng.random(5000)
        labels = (rng.random(5000) < raw**2).astype(float)  # raw scores overstate the probability
        for method in ("isotonic", "sigmoid"):
            with self.subTest(method):
                calibrated = Calibrator(method).fit(raw, labels).transform(raw)
                self.assertTrue(((calibrated >= 0) & (calibrated <= 1)).all())
                self.assertTrue((np.diff(calibrated[np.argsort(raw)]) >= -1e-12).all())  # monotone
                self.assertLess(calibration_error(labels, calibrated), calibration_error(labels, raw))


class CalibrationSplitTests(unittest.TestCase):
    def test_last_months_calibrate_and_the_fit_part_is_purged_before_them(self):
        train = pd.bdate_range("2015-01-01", "2020-12-31")
        split = split_calibration(train, months=12, embargo=6)
        self.assertGreater(split.calibration[0], train[-1] - pd.DateOffset(months=12))
        gap = train.get_loc(split.calibration[0]) - train.get_loc(split.fit[-1]) - 1
        self.assertEqual(gap, 6)
        self.assertEqual(split.fit[0], train[0])


class WalkForwardTests(unittest.TestCase):
    def test_both_models_learn_a_real_signal_and_are_calibrated(self):
        dataset = _synthetic(signal=1.0)
        folds = _folds(dataset)
        for model_type, parameters in PARAMETERS.items():
            with self.subTest(model_type):
                result = run_walk_forward(dataset, folds, model_type, parameters, SHORT)
                summary = result.summary()
                self.assertGreater(summary["auc_mean"], 0.7)
                self.assertGreater(summary["log_loss_skill_mean"], 0.05)
                self.assertGreater(summary["brier_skill_mean"], 0.05)
                self.assertLess(summary["calibration_error_mean"], 0.05)
                self.assertEqual(summary["folds_beating_reference"], len(folds))

    def test_pure_noise_shows_no_edge(self):
        """A leak (any future or validation information in training) would show up as skill here."""
        dataset = _synthetic(signal=0.0, seed=5)
        folds = _folds(dataset)
        for model_type, parameters in PARAMETERS.items():
            with self.subTest(model_type):
                summary = run_walk_forward(dataset, folds, model_type, parameters, SHORT).summary()
                self.assertLess(abs(summary["auc_mean"] - 0.5), 0.03)
                self.assertLess(summary["log_loss_skill_mean"], 0.01)

    def test_predictions_are_out_of_sample_and_reproducible(self):
        dataset = _synthetic(signal=0.5)
        folds = _folds(dataset)
        first = run_walk_forward(dataset, folds, "random_forest", PARAMETERS["random_forest"], SHORT)
        second = run_walk_forward(dataset, folds, "random_forest", PARAMETERS["random_forest"], SHORT)
        pd.testing.assert_frame_equal(first.predictions, second.predictions)

        for fold in folds:
            in_fold = first.predictions[first.predictions["fold"] == fold.number]
            fold_dates = in_fold.index.get_level_values("date")
            self.assertTrue(fold_dates.isin(fold.validation).all())
            self.assertFalse(fold_dates.isin(fold.train).any())
        self.assertFalse(first.predictions.index.duplicated().any())  # each stock-day predicted once
        row = first.folds.iloc[0]
        self.assertLess(row["fit_end"], row["calibration_start"])
        self.assertLess(row["calibration_start"], row["validation_start"])


if __name__ == "__main__":
    unittest.main()
