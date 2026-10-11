import dataclasses
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from quant.backtest.data import load_prices
from quant.config.settings import ConfigError
from quant.models.experiments import TrialSpec, run_trial, trial_settings
from quant.persistence import queries
from quant.persistence.database import create_db_engine
from quant.strategies import PANEL_COLUMNS
from quant.strategies.ml import top_n_holdings
from tests.backtest_fixture import build_market_db, small_settings
from tests.settings_helper import SETTINGS

STOCKS = ["AAA", "BBB", "CCC", "DDD", "EEE"]


class KeepBandTests(unittest.TestCase):
    def test_a_held_stock_stays_while_it_ranks_within_keep_rank(self):
        index = pd.bdate_range("2026-01-05", periods=3)
        probabilities = pd.DataFrame(
            [
                [0.9, 0.8, 0.1, 0.2, 0.3],  # buy AAA, BBB
                [0.1, 0.8, 0.9, 0.7, 0.2],  # AAA now 5th: dropped; BBB 2nd: kept; CCC fills
                [0.1, 0.5, 0.9, 0.8, 0.7],  # BBB now 4th: kept with keep_rank 4
            ],
            index=index, columns=STOCKS,
        )
        held = top_n_holdings(probabilities, top_n=2, every=1, keep_rank=4)
        self.assertEqual(list(held.iloc[0]), [True, True, False, False, False])
        self.assertEqual(list(held.iloc[1]), [False, True, True, False, False])
        self.assertEqual(list(held.iloc[2]), [False, True, True, False, False])
        # Without a band the same rows simply follow the top 2.
        plain = top_n_holdings(probabilities, top_n=2, every=1)
        self.assertEqual(list(plain.iloc[2]), [False, False, True, True, False])

    def test_the_band_lowers_turnover_on_noisy_rankings(self):
        rng = np.random.default_rng(3)
        base = np.linspace(0.45, 0.30, 40)
        probabilities = pd.DataFrame(base + rng.normal(0, 0.02, (200, 40)), index=pd.bdate_range("2020-01-01", periods=200))
        changes = lambda held: int((held != held.shift()).iloc[1:].to_numpy().sum())  # noqa: E731
        self.assertLess(
            changes(top_n_holdings(probabilities, 10, 5, keep_rank=20)), changes(top_n_holdings(probabilities, 10, 5))
        )

    def test_keep_rank_below_top_n_is_rejected(self):
        with self.assertRaises(ConfigError):
            dataclasses.replace(SETTINGS.strategies.ml_select, keep_rank=SETTINGS.strategies.ml_select.top_n - 1)


class TrialSettingsTests(unittest.TestCase):
    def test_horizon_sets_the_embargo_and_rebalance_and_keeps_everything_else(self):
        spec = TrialSpec("logistic_regression", {"C": 0.01}, horizon_days=20, keep_rank=20)
        trial = trial_settings(SETTINGS, spec)
        self.assertEqual(trial.model.target.horizon_days, 20)
        self.assertEqual(trial.model.embargo_days, 21)
        self.assertEqual(trial.strategies.ml_select.rebalance_every_sessions, 20)
        self.assertEqual(trial.strategies.ml_select.keep_rank, 20)
        self.assertEqual(trial.retrain.gates, SETTINGS.retrain.gates)  # gates never change
        self.assertEqual(trial.model.target.k, SETTINGS.model.target.k)
        self.assertEqual(spec.name, "logistic_regression h20 keep20")


class RunTrialTests(unittest.TestCase):
    def test_a_trial_runs_the_whole_chain_on_a_small_market(self):
        base = small_settings(random_runs=2)
        settings = dataclasses.replace(
            base,
            model=dataclasses.replace(base.model, calibration_months=2),
            retrain=dataclasses.replace(
                base.retrain,
                validation=dataclasses.replace(
                    base.retrain.validation, min_train_years=1, fold_length_months=2, min_folds=1, holdout_months=2
                ),
            ),
            strategies=dataclasses.replace(
                base.strategies, ml_select=dataclasses.replace(base.strategies.ml_select, top_n=2, keep_rank=2)
            ),
        )
        spec = TrialSpec("logistic_regression", {"C": 0.01}, horizon_days=10, keep_rank=3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trial.db"
            build_market_db(path, settings)
            engine = create_db_engine(path, settings.database.busy_timeout_ms)
            try:
                stocks = list(settings.market.stocks)
                prices = load_prices(engine, [*stocks, "SPY"])
                panel = queries.load_feature_panel(engine, settings.features.feature_version, stocks, PANEL_COLUMNS)
                regimes = queries.load_regimes(engine, "RULE", settings.regime.rule_version)
                result = run_trial(engine, settings, spec, prices, panel, regimes)
            finally:
                engine.dispose()

        self.assertEqual(result.settings.model.embargo_days, 11)
        self.assertTrue(all(check.passed for check in result.evidence.leak_checks))
        self.assertEqual(list(result.gates.table().index), list(settings.retrain.gates["predictor"]))
        self.assertIn("ml_logistic_regression", result.backtest.overall.index)


if __name__ == "__main__":
    unittest.main()
