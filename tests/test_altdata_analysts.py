"""Analyst rating and price-target changes (altdata_proposal.md 13.4): offline, a fake Yahoo."""

import dataclasses
import unittest
from datetime import date

import pandas as pd
from sqlalchemy import select

import quant.altdata.tables  # noqa: F401  registers the altdata tables
from quant.altdata.analysts import normalize, update_analysts
from quant.altdata.tables import altdata_analyst_changes, altdata_sync
from quant.persistence.database import create_db_engine, init_db
from settings_helper import SETTINGS

POOL = {"information_technology": ("AAPL", "MSFT")}
SETTINGS_2 = dataclasses.replace(SETTINGS, market=dataclasses.replace(SETTINGS.market, symbols=POOL))


def yahoo(*rows):
    """Rows as yfinance gives them: (time, firm, from, to, action, target action, current, prior)."""
    columns = ["GradeDate", "Firm", "ToGrade", "FromGrade", "Action", "priceTargetAction", "currentPriceTarget",
               "priorPriceTarget"]
    frame = pd.DataFrame([(pd.Timestamp(t), f, to, fr, a, ta, c, p) for t, f, fr, to, a, ta, c, p in rows],
                         columns=columns)
    return frame.set_index("GradeDate").reset_index()


JPM = ("2026-10-12 18:47:46", "JP Morgan", "Overweight", "Overweight", "main", "Raises", 360.0, 340.0)
NEEDHAM = ("2026-10-01 11:00:30", "Needham", "Hold", "Hold", "reit", "", 0.0, 0.0)
LATE = ("2026-10-13 20:30:00", "Citi", "Buy", "Buy", "main", "Raises", 400.0, 380.0)  # after the 20:00 UTC close


class AnalystTest(unittest.TestCase):
    def setUp(self):
        self.engine = create_db_engine(":memory:", 1000)
        init_db(self.engine)
        self.answers = {}

    def tearDown(self):
        self.engine.dispose()

    def fetch(self, symbol):
        answer = self.answers[symbol]
        if isinstance(answer, Exception):
            raise answer
        return answer

    def run_day(self, day):
        return update_analysts(self.engine, SETTINGS_2, fetch=self.fetch, today=day, pause=0)

    def rows(self):
        a = altdata_analyst_changes
        with self.engine.connect() as conn:
            return pd.DataFrame(conn.execute(select(a).order_by(a.c.id)).mappings().all())

    def test_normalize(self):
        rows = normalize(yahoo(JPM, NEEDHAM, JPM), "AAPL")
        self.assertEqual(len(rows), 2)  # the repeated row counts once
        needham = rows.set_index("firm").loc["Needham"]
        self.assertTrue(pd.isna(needham["current_target"]) and pd.isna(needham["target_action"]))  # 0 and "" -> NULL
        aware = yahoo(JPM)
        aware["GradeDate"] = aware["GradeDate"].dt.tz_localize("America/New_York")
        self.assertEqual(normalize(aware, "AAPL")["grade_time"].iloc[0], pd.Timestamp("2026-10-12 22:47:46"))

    def test_first_download_then_only_changes(self):
        self.answers = {"AAPL": yahoo(JPM, NEEDHAM), "MSFT": yahoo(LATE)}
        first = self.run_day(date(2026, 10, 14))
        self.assertEqual((first.seen, first.new, first.changed, first.gone), (3, 3, 0, 0))
        stored = self.rows().set_index("firm")
        self.assertEqual(stored.loc["JP Morgan", "available_on"], date(2026, 10, 12))  # before that day's close
        self.assertEqual(stored.loc["Citi", "available_on"], date(2026, 10, 14))  # after the close: next session

        # Next download: JPM unchanged, Needham's row now has a target (a new version), Citi gone, one new.
        revised = NEEDHAM[:5] + ("Announces", 250.0, 0.0)
        new = ("2026-10-15 12:00:00", "UBS", "Neutral", "Buy", "up", "Raises", 380.0, 330.0)
        self.answers = {"AAPL": yahoo(JPM, revised, new), "MSFT": yahoo()}
        second = self.run_day(date(2026, 10, 16))
        self.assertEqual((second.seen, second.new, second.changed, second.gone), (3, 1, 1, 1))
        rows = self.rows()
        self.assertEqual(len(rows), 5)  # nothing deleted or overwritten
        jpm = rows[rows["firm"] == "JP Morgan"].iloc[0]
        self.assertEqual((jpm["first_seen"], jpm["last_seen"]), (date(2026, 10, 14), date(2026, 10, 16)))
        needham = rows[rows["firm"] == "Needham"]
        self.assertEqual(list(needham["last_seen"]), [date(2026, 10, 14), date(2026, 10, 16)])  # two versions
        self.assertEqual(needham.iloc[1]["current_target"], 250.0)
        citi = rows[rows["firm"] == "Citi"].iloc[0]
        self.assertEqual(citi["last_seen"], date(2026, 10, 14))  # gone, kept

        again = self.run_day(date(2026, 10, 16))  # a rerun the same day stores nothing new
        self.assertEqual((again.new, again.changed, again.gone), (0, 0, 1))  # Citi: still gone since 10-14
        self.assertEqual(len(self.rows()), 5)
        later = self.run_day(date(2026, 10, 19))  # gone is counted against the previous download only
        self.assertEqual((later.new, later.changed, later.gone), (0, 0, 0))

    def test_a_failed_stock_changes_nothing(self):
        self.answers = {"AAPL": yahoo(JPM), "MSFT": yahoo(LATE)}
        self.run_day(date(2026, 10, 14))
        self.answers = {"AAPL": yahoo(JPM), "MSFT": ConnectionError("rate limited")}
        result = self.run_day(date(2026, 10, 16))
        self.assertEqual((result.symbols_ok, list(result.failed), result.gone), (["AAPL"], ["MSFT"], 0))
        citi = self.rows().set_index("firm").loc["Citi"]
        self.assertEqual(citi["last_seen"], date(2026, 10, 14))
        with self.engine.connect() as conn:
            sync = conn.execute(select(altdata_sync).where(altdata_sync.c.key == "2026-10-16")).mappings().one()
        self.assertEqual(sync["status"], "FAILED")
        self.assertIn("failed: MSFT", sync["message"])

    def test_a_partial_run_is_not_a_download(self):
        self.answers = {"AAPL": yahoo(JPM), "MSFT": yahoo(LATE)}
        self.run_day(date(2026, 10, 14))
        update_analysts(self.engine, SETTINGS_2, ["AAPL"], fetch=self.fetch, today=date(2026, 10, 15), pause=0)
        with self.engine.connect() as conn:
            keys = [r.key for r in conn.execute(select(altdata_sync.c.key))]
        self.assertEqual(keys, ["2026-10-14"])
        self.answers = {"AAPL": yahoo(JPM), "MSFT": yahoo()}
        self.assertEqual(self.run_day(date(2026, 10, 16)).gone, 1)  # against 10-14, not the partial 10-15


if __name__ == "__main__":
    unittest.main()
