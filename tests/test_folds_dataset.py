import dataclasses
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from quant.features.technical import FEATURE_COLUMNS, MARKET_FEATURE_COLUMNS
from quant.models.dataset import INPUT_COLUMNS, REGIME_COLUMNS, TARGET_COLUMNS, build_dataset, load_dataset
from quant.models.folds import holdout_fold, sample_sessions, walk_forward_folds
from quant.models.labels import Labels
from quant.persistence import queries
from quant.persistence.database import create_db_engine
from tests.backtest_fixture import build_market_db, small_settings
from tests.settings_helper import SETTINGS

VALIDATION = SETTINGS.retrain.validation  # expanding, 8 training years, 12-month folds, >= 5 folds
EMBARGO = SETTINGS.model.embargo_days  # 6 = h + 1
H = SETTINGS.model.target.horizon_days


class FoldTests(unittest.TestCase):
    def setUp(self):
        self.sessions = pd.bdate_range("2006-01-02", "2026-10-09")
        self.holdout_start = pd.Timestamp("2025-10-10")
        self.folds = walk_forward_folds(self.sessions, self.holdout_start, VALIDATION, EMBARGO)
        self.position = {stamp: i for i, stamp in enumerate(self.sessions)}

    def test_training_always_precedes_validation_and_folds_do_not_overlap(self):
        self.assertGreaterEqual(len(self.folds), VALIDATION.min_folds)
        for fold in self.folds:
            self.assertLess(fold.train_end, fold.validation_start)
        for earlier, later in zip(self.folds, self.folds[1:]):
            self.assertLess(earlier.validation_end, later.validation_start)
            # Adjacent: the next window starts right after the previous one ends.
            self.assertEqual(self.position[later.validation_start], self.position[earlier.validation_end] + 1)
            # Expanding: each fold trains on more history, from the same first session.
            self.assertEqual(earlier.train[0], later.train[0])
            self.assertGreater(len(later.train), len(earlier.train))

    def test_no_training_label_window_reaches_its_validation_period(self):
        for fold in self.folds:
            last = self.position[fold.train_end]
            exit_session = self.sessions[last + 1 + H]  # the label of the last training row ends here
            self.assertLess(exit_session, fold.validation_start)
            # Exactly `embargo` sessions are dropped between training and validation.
            self.assertEqual(self.position[fold.validation_start] - last - 1, EMBARGO)

    def test_no_label_window_crosses_into_the_holdout(self):
        last = self.folds[-1].validation_end
        self.assertLess(self.sessions[self.position[last] + 1 + H], self.holdout_start)
        for fold in self.folds:
            self.assertFalse((fold.train >= self.holdout_start).any())
            self.assertFalse((fold.validation >= self.holdout_start).any())

    def test_windows_are_anchored_to_the_development_end_and_need_min_train_years(self):
        self.assertEqual(self.folds[-1].validation_end, self.sessions[self.position[self.holdout_start] - EMBARGO - 1])
        first = self.folds[0]
        self.assertGreaterEqual(first.train_end, self.sessions[0] + pd.DateOffset(years=VALIDATION.min_train_years))
        for fold in self.folds:
            self.assertLessEqual(fold.validation_end - fold.validation_start, pd.Timedelta(days=366))

    def test_too_little_history_is_a_clear_error(self):
        with self.assertRaisesRegex(ValueError, "min_folds"):
            walk_forward_folds(pd.bdate_range("2016-01-04", "2026-10-09"), self.holdout_start, VALIDATION, EMBARGO)

    def test_rolling_scheme_limits_the_training_window(self):
        rolling = dataclasses.replace(VALIDATION, scheme="rolling_walk_forward", rolling_years=5)
        for fold in walk_forward_folds(self.sessions, self.holdout_start, rolling, EMBARGO):
            self.assertGreater(fold.train[0], fold.validation_start - pd.DateOffset(years=5))

    def test_holdout_fold_trains_on_all_purged_development_sessions(self):
        fold = holdout_fold(self.sessions, self.holdout_start, EMBARGO)
        self.assertEqual(fold.validation_start, self.holdout_start)
        self.assertEqual(fold.train_end, self.folds[-1].validation_end)

    def test_sampling_picks_the_same_calendar_sessions_in_every_fold(self):
        sampled = [set(sample_sessions(f.train, 5, self.sessions)) for f in self.folds]
        for earlier, later in zip(sampled, sampled[1:]):
            self.assertTrue(earlier <= later)
        self.assertEqual(len(sample_sessions(self.sessions[:100], 5, self.sessions)), 20)


class BuildDatasetTests(unittest.TestCase):
    def test_rows_join_features_market_regime_and_labels(self):
        dates = pd.bdate_range("2026-01-05", periods=4)
        index = pd.MultiIndex.from_product([dates, ["AAA", "BBB"]], names=["date", "symbol"])
        features = pd.DataFrame(1.0, index=index, columns=list(FEATURE_COLUMNS))
        features.loc[(dates[0], "AAA"), "rsi14"] = np.nan  # warm-up: dropped
        market = pd.DataFrame(2.0, index=dates, columns=list(MARKET_FEATURE_COLUMNS))
        regimes = pd.Series([np.nan, "BULL", "SIDEWAYS", "BEAR"], index=dates, dtype=object)
        wide = pd.DataFrame({"AAA": [1.0, 0.0, 1.0, np.nan], "BBB": [0.0, 1.0, 0.0, np.nan]}, index=dates)
        labels = Labels(label=wide, excess=wide * 0.01, threshold=wide * 0 + 0.005)

        frame = build_dataset(features, market, regimes, labels).frame
        self.assertEqual(list(frame.columns), [*INPUT_COLUMNS, *TARGET_COLUMNS])
        # Day 0 has no regime (and AAA a missing feature): dropped. The last day keeps rows without labels.
        self.assertEqual(len(frame), 6)
        self.assertEqual(frame.loc[(dates[1], "BBB"), "label"], 1.0)
        self.assertTrue(frame.loc[dates[3], "label"].isna().all())
        self.assertEqual(list(frame.loc[(dates[2], "AAA"), list(REGIME_COLUMNS)]), [0.0, 1.0, 0.0])
        self.assertEqual(frame.loc[(dates[1], "AAA"), "vix"], 2.0)


class LoadDatasetTests(unittest.TestCase):
    def test_dataset_from_a_stored_market(self):
        settings = small_settings()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ds.db"
            build_market_db(path, settings)
            engine = create_db_engine(path, settings.database.busy_timeout_ms)
            dataset = load_dataset(engine, settings)
            stored = queries.load_features_long(engine, settings.features.feature_version, ["XOM"], FEATURE_COLUMNS)
            engine.dispose()

        frame = dataset.frame
        self.assertEqual(set(frame.index.get_level_values("symbol")), {"AAPL", "MSFT", "XOM"})  # no SPY/QQQ/^VIX
        self.assertFalse(frame.loc[:, list(INPUT_COLUMNS)].isna().any().any())
        some_row = frame.xs("XOM", level="symbol").index[50]
        pd.testing.assert_series_equal(
            frame.loc[(some_row, "XOM"), list(FEATURE_COLUMNS)],
            stored.loc[(some_row, "XOM"), list(FEATURE_COLUMNS)],
            check_names=False,
        )
        self.assertTrue(frame["label"].iloc[-3:].isna().all())  # the newest rows await their labels
        self.assertEqual(len(dataset.labelled().frame), int(frame["label"].notna().sum()))


if __name__ == "__main__":
    unittest.main()
