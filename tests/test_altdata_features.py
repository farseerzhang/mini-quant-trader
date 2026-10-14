"""A4 round 1: the Form 4 features (altdata_proposal.md 7, 8.1) and stage 1; offline, hand-made data."""

import unittest
from datetime import date

import numpy as np
import pandas as pd

from quant.altdata.features import filing_facts, stock_features
from quant.altdata.research import FeatureResult, stage1
from quant.altdata.shares import latest_shares, parse_shares
from quant.models.folds import Fold

SESSIONS = pd.bdate_range("2010-01-04", "2014-12-31")
CEO, CFO, FUND = 1, 2, 3


def _filings(rows):
    return pd.DataFrame(rows, columns=["accession_no", "symbol", "available_on", "is_10b5_1"])


def _owners(rows):
    return pd.DataFrame(rows, columns=["accession_no", "owner_cik", "is_officer", "is_director"])


def _lines(rows):
    return pd.DataFrame(rows, columns=["accession_no", "line_no", "transaction_code", "transaction_date", "shares",
                                       "price", "direct_indirect", "shares_owned_after"])


def day(i):
    return SESSIONS[i].date()


def features(filings, owners, lines, cap=1e6):
    facts = filing_facts(_filings(filings), _owners(owners), _lines(lines), SESSIONS)
    first = min(f[2] for f in filings) if filings else None
    return stock_features(facts, SESSIONS, pd.Series(cap, index=SESSIONS), first)


class WindowTest(unittest.TestCase):
    def setUp(self):
        self.result = features(
            [("a1", "X", day(10), 0)],
            [("a1", CEO, 1, 0), ("a1", FUND, 0, 0)],
            [("a1", 1, "P", day(8), 100.0, 50.0, "D", 1100.0)])

    def test_visible_only_from_its_session_and_for_60_sessions(self):
        buyers = self.result["insider_buyers_60d"]
        self.assertTrue(np.isnan(buyers.iloc[9]))  # before the stock's first filing: NULL
        self.assertEqual(buyers.iloc[10], 1)  # the 10% owner alone does not count
        self.assertEqual(buyers.iloc[69], 1)  # still inside the 60-session window
        self.assertEqual(buyers.iloc[70], 0)  # gone after 60 sessions
        self.assertAlmostEqual(self.result["insider_net_buy_value_60d"].iloc[10], 100 * 50 / 1e6)

    def test_sale_features_wait_three_years(self):
        sellers = self.result["opportunistic_sellers_60d"]
        three_years = pd.Timestamp(day(10)) + pd.DateOffset(years=3)
        self.assertTrue(sellers[SESSIONS < three_years].isna().all())
        self.assertEqual(sellers[SESSIONS >= three_years].iloc[0], 0)

    def test_value_needs_a_market_cap(self):
        result = features([("a1", "X", day(10), 0)], [("a1", CEO, 1, 0)],
                          [("a1", 1, "P", day(8), 100.0, 50.0, "D", 1100.0)], cap=np.nan)
        self.assertTrue(np.isnan(result["insider_net_buy_value_60d"].iloc[10]))
        self.assertEqual(result["insider_buyers_60d"].iloc[10], 1)


class SaleTest(unittest.TestCase):
    def history(self, owner, months):
        """Earlier sales of `owner`, one filing per (year, month)."""
        filings, owners, lines = [], [], []
        for k, (year, month) in enumerate(months):
            when = date(year, month, 15)
            pos = SESSIONS.searchsorted(pd.Timestamp(when))
            accession = f"h{owner}-{k}"
            filings.append((accession, "X", SESSIONS[pos].date(), 0))
            owners.append((accession, owner, 1, 0))
            lines.append((accession, 1, "S", when, 10.0, 20.0, "D", 5000.0))
        return filings, owners, lines

    def test_routine_and_opportunistic_and_fraction(self):
        f1, o1, l1 = self.history(CEO, [(2011, 3), (2012, 3), (2013, 3)])  # sells every March
        f2, o2, l2 = self.history(CFO, [(2011, 1), (2012, 5), (2013, 9)])  # trades every year, other months
        sale_day = SESSIONS[SESSIONS.searchsorted(pd.Timestamp("2014-03-20"))].date()
        filings = f1 + f2 + [("s1", "X", sale_day, 0), ("s2", "X", sale_day, 0), ("s3", "X", sale_day, 1)]
        owners = o1 + o2 + [("s1", CEO, 1, 0), ("s2", CFO, 1, 0), ("s3", CFO, 1, 0)]
        lines = l1 + l2 + [("s1", 1, "S", date(2014, 3, 18), 100.0, 20.0, "D", 900.0),   # CEO: routine, 10%
                           ("s2", 1, "S", date(2014, 3, 18), 400.0, 20.0, "D", 600.0),   # CFO: opportunistic, 40%
                           ("s3", 1, "S", date(2014, 3, 18), 999.0, 20.0, "D", 1.0)]     # 10b5-1: ignored
        result = features(filings, owners, lines)
        t = SESSIONS.get_loc(pd.Timestamp(sale_day))
        self.assertEqual(result["opportunistic_sellers_60d"].iloc[t], 1)  # only the CFO
        self.assertAlmostEqual(result["insider_sell_fraction_60d"].iloc[t], 0.4)
        self.assertEqual(result["opportunistic_sellers_60d"].iloc[t - 1], 0)  # not visible before

    def test_not_classifiable_without_three_years(self):
        sale_day = SESSIONS[SESSIONS.searchsorted(pd.Timestamp("2014-06-10"))].date()
        f0, o0, l0 = self.history(FUND, [(2010, 2)])  # gives the stock its first filing early
        result = features(f0 + [("s1", "X", sale_day, 0)], o0 + [("s1", CEO, 1, 0)],
                          l0 + [("s1", 1, "S", date(2014, 6, 9), 100.0, 20.0, "D", 100.0)])
        t = SESSIONS.get_loc(pd.Timestamp(sale_day))
        self.assertEqual(result["opportunistic_sellers_60d"].iloc[t], 0)  # no history: neither
        self.assertAlmostEqual(result["insider_sell_fraction_60d"].iloc[t], 0.5)  # still counts for the fraction

    def test_one_trade_in_several_filings_counts_once(self):
        trade = (1, "S", day(9), 100.0, 50.0, "D", 0.0)
        result = features([("a1", "X", day(10), 0), ("a2", "X", day(10), 0)],
                          [("a1", CEO, 1, 0), ("a2", CFO, 0, 1)],
                          [("a1", *trade), ("a2", *trade)])
        self.assertAlmostEqual(result["insider_net_buy_value_60d"].iloc[10], -100 * 50 / 1e6)

    def test_planned_sales_do_not_reduce_the_value(self):
        result = features([("a1", "X", day(10), 1)], [("a1", CEO, 1, 0)],
                          [("a1", 1, "S", day(9), 100.0, 50.0, "D", 0.0)])
        self.assertEqual(result["insider_net_buy_value_60d"].iloc[10], 0.0)


class SharesTest(unittest.TestCase):
    def test_classes_summed_and_available_after_filing(self):
        facts = {"facts": {"dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
            {"end": "2012-01-31", "val": 100, "accn": "x1", "filed": "2012-02-10", "form": "10-K"},
            {"end": "2012-01-31", "val": 50, "accn": "x1", "filed": "2012-02-10", "form": "10-K"},
            {"end": "2012-04-30", "val": 160, "accn": "x2", "filed": "2012-05-08", "form": "10-Q"},
        ]}}}}}
        rows = parse_shares(facts)
        self.assertEqual([r["seq"] for r in rows], [0, 1, 0])
        frame = pd.DataFrame([{"symbol": "X", **r} for r in rows])
        frame = frame.groupby(["symbol", "accession_no", "as_of", "filed"], as_index=False)["shares"].sum()
        sessions = [date(2012, 2, 10), date(2012, 2, 13), date(2012, 5, 9)]
        available = {date(2012, 2, 10): date(2012, 2, 13), date(2012, 5, 8): date(2012, 5, 9)}
        series = latest_shares(frame, "X", sessions, available)
        self.assertTrue(np.isnan(series.iloc[0]))  # filed that day: not yet
        self.assertEqual(series.iloc[1], 150)  # two classes summed
        self.assertEqual(series.iloc[2], 160)


class WrongCountsTest(unittest.TestCase):
    def test_orders_of_magnitude_off_are_dropped(self):
        from quant.altdata.shares import drop_wrong_counts

        frame = pd.DataFrame({
            "symbol": ["PSA"] * 5 + ["ONE"], "accession_no": list("abcdef"),
            "as_of": [date(2015, 2, 1), date(2015, 5, 1), date(2015, 8, 1), date(2015, 11, 1), date(2016, 2, 1),
                      date(2015, 1, 1)],
            "filed": [date(2015, 2, 5)] * 6,
            "shares": [175e6, 176e6, 173.0, 177e6, 171858.0, 5e6]})
        kept = drop_wrong_counts(frame)
        self.assertEqual(sorted(kept["accession_no"]), ["a", "b", "d", "f"])  # a lone value has nothing to compare


class Stage1Test(unittest.TestCase):
    def test_pass_and_fail(self):
        sessions = pd.bdate_range("2014-01-01", periods=400)
        rng = np.random.default_rng(0)
        target = pd.DataFrame(rng.normal(size=(400, 30)), index=sessions)
        good = target + rng.normal(scale=3.0, size=(400, 30))  # weakly informative everywhere
        noise = pd.DataFrame(rng.normal(size=(400, 30)), index=sessions)
        folds = [Fold(i + 1, sessions[:1], sessions[i * 40:(i + 1) * 40]) for i in range(10)]
        results = {r.name: r for r in stage1({"good": good, "noise": noise}, target, folds, horizon=5)}
        self.assertTrue(results["good"].passed)
        self.assertGreaterEqual(results["good"].folds_same_sign, 7)
        self.assertFalse(results["noise"].passed)
        self.assertEqual(len(results["good"].coverage), 10)

    def test_pass_rule(self):
        self.assertFalse(FeatureResult("x", 0.01, 2.5, 6, 11, 100, [], []).passed)  # too few folds
        self.assertFalse(FeatureResult("x", 0.01, 1.9, 9, 11, 100, [], []).passed)  # t too small
        self.assertTrue(FeatureResult("x", -0.01, -2.1, 7, 11, 100, [], []).passed)  # negative also counts


if __name__ == "__main__":
    unittest.main()
