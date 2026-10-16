"""A4 round 2: the analyst features (altdata_proposal.md section 8.2); offline, hand-made data."""

import unittest
from datetime import date, datetime

import numpy as np
import pandas as pd

import quant.altdata.tables  # noqa: F401  registers the altdata tables
from quant.altdata.analyst_features import load_changes, percentile_ranks, stock_features
from quant.altdata.tables import altdata_analyst_changes
from quant.persistence.database import create_db_engine, init_db

SESSIONS = pd.bdate_range("2015-01-05", periods=400)
NO_SPLITS = pd.DataFrame(columns=["symbol", "ex_date", "ratio"])


def day(i):
    return SESSIONS[i]


def changes(*rows):
    """(session index, firm, action, current target, prior target, minute)."""
    return pd.DataFrame([{"symbol": "X", "firm": f, "action": a, "current_target": c, "prior_target": p,
                          "available_on": day(i), "grade_time": day(i) + pd.Timedelta(minutes=m)}
                         for i, f, a, c, p, m in rows])


def features(rows, splits=NO_SPLITS, close=100.0):
    return stock_features(rows, splits, pd.Series(close, index=SESSIONS), SESSIONS)


class FeatureTest(unittest.TestCase):
    def test_revision_window_null_before_and_clip(self):
        out = features(changes((10, "JPM", "main", 360.0, 340.0, 0), (12, "MS", "main", 1000.0, 100.0, 0)))
        revision = out["target_revision_20d"]
        self.assertTrue(np.isnan(revision.iloc[9]))  # before the stock's first change
        self.assertAlmostEqual(revision.iloc[10], np.log(360 / 340))
        self.assertAlmostEqual(revision.iloc[12], (np.log(360 / 340) + np.log(2)) / 2)  # 10x clipped to ln 2
        self.assertAlmostEqual(revision.iloc[29], (np.log(360 / 340) + np.log(2)) / 2)  # session 10 still inside
        self.assertAlmostEqual(revision.iloc[30], np.log(2))  # 20 sessions later, JPM's change has left
        self.assertEqual(revision.iloc[40], 0.0)  # no change in the window: 0

    def test_net_rating_changes(self):
        out = features(changes((5, "A", "up", None, None, 0), (6, "B", "down", None, None, 0),
                               (7, "C", "up", 50.0, None, 0)))
        self.assertEqual(list(out["net_rating_changes_20d"].iloc[5:8]), [1.0, 0.0, 1.0])
        self.assertEqual(out["net_rating_changes_20d"].iloc[30], 0.0)
        self.assertEqual(out["target_revision_20d"].iloc[7], 0.0)  # no prior target: not a revision

    def test_upside_median_of_latest_targets_and_expiry(self):
        out = features(changes((10, "A", "main", 120.0, 110.0, 0), (20, "B", "main", 140.0, 130.0, 0),
                               (30, "A", "main", 160.0, 120.0, 0), (31, "C", "main", 100.0, 90.0, 0)))
        upside = out["target_upside"]
        self.assertAlmostEqual(upside.iloc[25], 1.30 - 1)  # median(120, 140) / 100
        self.assertAlmostEqual(upside.iloc[31], 1.40 - 1)  # A now 160: median(160, 140, 100)
        self.assertAlmostEqual(upside.iloc[20 + 249], 1.40 - 1)  # B's 250th session: median(160, 140, 100)
        self.assertAlmostEqual(upside.iloc[20 + 250], 1.30 - 1)  # B expired: median(160, 100)
        self.assertTrue(np.isnan(features(changes((0, "A", "main", 120.0, 110.0, 0)))["target_upside"].iloc[250]))

    def test_split_adjustment_and_exclusion(self):
        splits = pd.DataFrame({"symbol": ["X"], "ex_date": [day(30)], "ratio": [10.0]})
        rows = changes((15, "A", "main", 1000.0, 900.0, 0), (31, "B", "main", 110.0, 1000.0, 0),
                       (40, "C", "main", 120.0, 100.0, 0))
        out = stock_features(rows, splits, pd.Series(np.where(np.arange(400) < 30, 500.0, 50.0), index=SESSIONS),
                             SESSIONS)
        self.assertAlmostEqual(out["target_upside"].iloc[20], 1000 / 500 - 1)  # before the split
        self.assertAlmostEqual(out["target_upside"].iloc[35], np.median([100.0, 110.0]) / 50 - 1)  # A's 1000 -> 100
        revisions = out["target_revision_20d"]
        self.assertAlmostEqual(revisions.iloc[31], np.log(1000 / 900))  # B (1 session after the split) left out
        self.assertAlmostEqual(revisions.iloc[40], np.log(120 / 100))  # C counts (A has left the window)

    def test_percentile_ranks(self):
        frame = pd.DataFrame({"A": [1.0, np.nan], "B": [3.0, 2.0], "C": [3.0, 1.0]})
        ranks = percentile_ranks(frame)
        self.assertEqual(list(ranks.iloc[0]), [1 / 3, 2.5 / 3, 2.5 / 3])
        self.assertTrue(np.isnan(ranks.iloc[1, 0]))


class LoadTest(unittest.TestCase):
    def test_as_of_view_and_one_change_per_firm_and_session(self):
        engine = create_db_engine(":memory:", 1000)
        init_db(engine)
        base = {"symbol": "X", "firm": "JPM", "action": "main", "current_target": 1.0, "prior_target": 1.0,
                "available_on": date(2026, 10, 12), "content_hash": "h", "first_seen": date(2026, 10, 16),
                "last_seen": date(2026, 10, 16)}
        with engine.begin() as conn:
            conn.execute(altdata_analyst_changes.insert(), [
                {**base, "grade_time": datetime(2026, 10, 12, 12), "current_target": 10.0},
                {**base, "grade_time": datetime(2026, 10, 12, 15), "current_target": 20.0},  # same session: latest
                {**base, "grade_time": datetime(2026, 10, 9, 12), "available_on": date(2026, 10, 9),
                 "first_seen": date(2026, 10, 20), "last_seen": date(2026, 10, 20)},  # after the as-of download
            ])
        frame = load_changes(engine, date(2026, 10, 16))
        engine.dispose()
        self.assertEqual(list(frame["current_target"]), [20.0])


if __name__ == "__main__":
    unittest.main()
