import dataclasses
import unittest

import pandas as pd

from quant.models.search import candidates, search, selection_score
from tests.test_models import SHORT, _folds, _synthetic


class CandidateTests(unittest.TestCase):
    def test_small_spaces_try_every_combination(self):
        space = {"max_depth": (4, 6, 8), "min_samples_leaf": (20, 50, 100)}
        grid = candidates(space, max_trials=30, seed=42)
        self.assertEqual(len(grid), 9)
        self.assertEqual(grid[0], {"max_depth": 4, "min_samples_leaf": 20})
        self.assertEqual(len({tuple(sorted(c.items())) for c in grid}), 9)

    def test_large_spaces_draw_a_reproducible_random_subset(self):
        space = {"a": tuple(range(10)), "b": tuple(range(10))}
        first = candidates(space, max_trials=7, seed=42)
        self.assertEqual(len(first), 7)
        self.assertEqual(first, candidates(space, max_trials=7, seed=42))
        self.assertNotEqual(first, candidates(space, max_trials=7, seed=1))
        self.assertEqual(len({tuple(sorted(c.items())) for c in first}), 7)


class SelectionTests(unittest.TestCase):
    def test_consistency_beats_a_higher_but_erratic_mean(self):
        steady = pd.Series([0.004, 0.005, 0.004, 0.005])  # mean 0.45%, small spread
        erratic = pd.Series([0.030, -0.010, 0.020, -0.015])  # mean 0.625%, large spread
        self.assertGreater(erratic.mean(), steady.mean())
        self.assertGreater(selection_score(steady), selection_score(erratic))


class SearchTests(unittest.TestCase):
    def test_search_runs_every_candidate_and_returns_the_best(self):
        dataset = _synthetic(signal=1.0)
        folds = _folds(dataset)
        settings = dataclasses.replace(
            SHORT,
            retrain=dataclasses.replace(
                SHORT.retrain,
                search=dataclasses.replace(SHORT.retrain.search, space={"logistic_regression": {"C": (0.001, 1.0)}}),
            ),
        )
        result = search(dataset, folds, "logistic_regression", settings)
        self.assertEqual(result.trial_count, 2)
        self.assertEqual(list(result.trials["param_C"]), [0.001, 1.0])
        best_trial = result.trials["score"].idxmax()
        self.assertEqual(result.best_parameters, {"C": result.trials.loc[best_trial, "param_C"]})
        self.assertAlmostEqual(result.trials.loc[best_trial, "score"], selection_score(result.best.folds["log_loss_skill"]))

    def test_a_model_without_a_search_space_is_an_error(self):
        settings = dataclasses.replace(
            SHORT, retrain=dataclasses.replace(SHORT.retrain, search=dataclasses.replace(SHORT.retrain.search, space={}))
        )
        with self.assertRaisesRegex(ValueError, "no entry"):
            search(_synthetic(signal=1.0), [], "random_forest", settings)


if __name__ == "__main__":
    unittest.main()
