"""Option chains loader (altdata_proposal.md 13.7): the window, the first read kept, the grid and the unusual
contracts, the comparison; offline with a fake Yahoo, in memory."""

import argparse
import dataclasses
import unittest
from contextlib import redirect_stdout
from datetime import date, datetime, timedelta, timezone
from io import StringIO
from unittest import mock

import pandas as pd
from sqlalchemy import select

import quant.altdata.tables  # noqa: F401  registers the altdata tables
from quant.altdata.options import (
    Chains,
    OutsideWindowError,
    compare_options,
    normalize,
    options_session,
    update_options,
)
from quant.altdata.tables import altdata_option_contracts, altdata_option_summaries, altdata_sync
from quant.persistence import queries
from quant.persistence.database import create_db_engine, init_db
from settings_helper import SETTINGS

FRIDAY = date(2026, 10, 16)
FRIDAY_EVENING = datetime(2026, 10, 16, 21, 30, tzinfo=timezone.utc)
POOL = {"information_technology": ("MSFT", "NVDA")}
SMALL = dataclasses.replace(SETTINGS, market=dataclasses.replace(SETTINGS.market, symbols=POOL))


def chain(symbol, session=FRIDAY, volume_scale=1.0, open_interest=500.0):
    """Three expiries (3, 31 and 45 days), strikes 80 .. 120 around a spot of 100, calls and puts;
    the 31-day 110 call trades 5,000 against an open interest of 50 (the unusual one)."""
    rows = []
    for days in (3, 31, 45):
        expiry = (session + timedelta(days=days)).isoformat()
        tag = (session + timedelta(days=days)).strftime("%y%m%d")
        for side, letter in (("call", "C"), ("put", "P")):
            for strike in range(80, 121, 5):
                unusual = days == 31 and side == "call" and strike == 110
                rows.append({
                    "contractSymbol": f"{symbol}{tag}{letter}{strike * 1000:08d}", "strike": float(strike),
                    "lastPrice": 1.0, "bid": 0.9, "ask": 1.1, "lastTradeDate": "2026-10-16 19:59:00+00:00",
                    "volume": (5000.0 if unusual else 0.0 if strike == 80 else 10.0 * strike / 100) * volume_scale,
                    "openInterest": 50.0 if unusual else open_interest,
                    "impliedVolatility": 0.30 + (0.05 if side == "put" and strike < 95 else 0.0),
                    "expiry": expiry, "side": side})
    return pd.DataFrame(rows)


class FakeYahoo:
    def __init__(self, failing=(), **options):
        self.failing, self.options, self.calls = set(failing), options, []

    def __call__(self, symbol, session):
        self.calls.append(symbol)
        if symbol in self.failing:
            raise ConnectionError("Yahoo did not answer")
        return Chains(spot=999.0, contracts=chain(symbol, session, **self.options))  # 999: not the stored close


class OptionsTest(unittest.TestCase):
    def setUp(self):
        self.engine = create_db_engine(":memory:", 1000)
        init_db(self.engine)
        queries.upsert_daily_prices(self.engine, [
            {"symbol": s, "date": FRIDAY, "open": 100, "high": 101, "low": 99, "close": 100.0, "volume": 1,
             "provider": "test"} for s in ("MSFT", "NVDA")])

    def tearDown(self):
        self.engine.dispose()

    def rows(self, table):
        with self.engine.connect() as conn:
            return pd.DataFrame(conn.execute(select(table)).mappings().all())

    def test_the_window(self):
        self.assertEqual(options_session(FRIDAY_EVENING, "XNYS"), FRIDAY)
        self.assertEqual(options_session(datetime(2026, 10, 18, 14, 0, tzinfo=timezone.utc), "XNYS"), FRIDAY)
        tuesday_morning = datetime(2026, 10, 20, 7, 0, tzinfo=timezone.utc)
        self.assertEqual(options_session(tuesday_morning, "XNYS"), date(2026, 10, 19))
        with self.assertRaises(OutsideWindowError):
            options_session(datetime(2026, 10, 19, 15, 0, tzinfo=timezone.utc), "XNYS")  # Monday, market open

    def test_the_read_window_after_the_close(self):
        friday_close = datetime(2026, 10, 16, 20, 0, tzinfo=timezone.utc)
        self.assertEqual(options_session(friday_close + timedelta(hours=4), "XNYS", 4), FRIDAY)
        for late in (friday_close + timedelta(hours=4, minutes=1),
                     datetime(2026, 10, 18, 14, 0, tzinfo=timezone.utc)):  # Sunday: Friday's window is over
            with self.subTest(late=late), self.assertRaisesRegex(OutsideWindowError, "too late for the session of 2026-10-16"):
                options_session(late, "XNYS", 4)
        self.assertEqual(options_session(datetime(2026, 10, 18, 14, 0, tzinfo=timezone.utc), "XNYS"), FRIDAY)  # no limit
        # 2026-11-27, the day after Thanksgiving, closes early at 13:00 New York (18:00 UTC).
        early = date(2026, 11, 27)
        self.assertEqual(options_session(datetime(2026, 11, 27, 21, 30, tzinfo=timezone.utc), "XNYS", 4), early)
        with self.assertRaises(OutsideWindowError):
            options_session(datetime(2026, 11, 27, 22, 30, tzinfo=timezone.utc), "XNYS", 4)

    def test_a_late_read_is_refused_before_any_request(self):
        fake = FakeYahoo()
        with self.assertRaisesRegex(OutsideWindowError, "too late"):
            update_options(self.engine, SMALL, now=datetime(2026, 10, 17, 9, 0, tzinfo=timezone.utc), fetch=fake)
        self.assertEqual(fake.calls, [])
        self.assertTrue(self.rows(altdata_option_summaries).empty)
        self.assertTrue(self.rows(altdata_sync).empty)

    def test_a_late_read_is_a_note_not_a_failure(self):
        # As in run-daily's stage 7: the command prints why and returns 0.
        from quant.altdata import cli as altdata_cli

        late = OutsideWindowError("too late for the session of 2026-10-16 (its read window closed at "
                                  "2026-10-17 00:00 UTC); no option data is stored for it")
        args = argparse.Namespace(altdata_command="options", compare=False, symbol=None, db=":memory:")
        output = StringIO()
        with mock.patch("quant.altdata.options.update_options", side_effect=late), redirect_stdout(output):
            self.assertEqual(altdata_cli.run(args, SMALL), 0)
        self.assertIn("Option chains not read: too late for the session of 2026-10-16", output.getvalue())

    def test_normalize(self):
        contracts = normalize(chain("MSFT"))
        self.assertTrue(contracts.loc[contracts["strike"] == 80, "volume"].isna().all())  # 0 = no trade
        self.assertEqual(contracts["expiry"].iloc[0], date(2026, 10, 19))
        self.assertIsNone(contracts["last_trade_at"].dt.tz)  # naive UTC

    def test_first_read_stored_with_grid_and_unusual(self):
        result = update_options(self.engine, SMALL, now=FRIDAY_EVENING, fetch=FakeYahoo())
        self.assertEqual((result.session, sorted(result.stored), result.failed), (FRIDAY, ["MSFT", "NVDA"], {}))
        summary = self.rows(altdata_option_summaries).set_index("symbol").loc["MSFT"]
        self.assertEqual(summary["spot"], 100.0)  # the stored close, not Yahoo's 999
        self.assertEqual((summary["expiries"], summary["contracts"]), (3, 54))
        self.assertEqual(summary["iv_expiry"], date(2026, 11, 16))  # 31 days: nearest 30
        self.assertAlmostEqual(summary["atm_call_iv"], 0.30)
        self.assertAlmostEqual(summary["otm_put_iv"], 0.35)  # strikes 85 and 90 (80 did not trade)
        kept = self.rows(altdata_option_contracts)
        msft = kept[kept["symbol"] == "MSFT"]
        self.assertEqual(int(msft["in_grid"].sum()), 14)  # 85 .. 115, calls and puts
        unusual = msft[msft["top_volume_over_oi"] == 1]
        self.assertIn("MSFT261116C00110000", set(unusual["contract_symbol"]))
        self.assertTrue((msft.loc[msft["top_volume_over_oi"] == 1, "volume"] >= 100).all())
        self.assertEqual(int(msft["top_volume"].sum()), 5)
        self.assertLessEqual(len(msft), 24)
        sync = self.rows(altdata_sync).iloc[0]
        self.assertEqual((sync["source"], sync["key"], sync["status"], sync["filings"]),
                         ("options-yahoo", "2026-10-16", "OK", 2))

    def test_a_later_read_keeps_the_first(self):
        update_options(self.engine, SMALL, now=FRIDAY_EVENING, fetch=FakeYahoo(failing={"NVDA"}))
        self.assertEqual(self.rows(altdata_sync).iloc[0]["status"], "FAILED")
        later = datetime(2026, 10, 16, 23, 30, tzinfo=timezone.utc)  # 3.5 h after the close: still in the window
        fake = FakeYahoo(volume_scale=3.0, open_interest=900.0)
        result = update_options(self.engine, SMALL, now=later, fetch=fake)
        self.assertEqual((fake.calls, result.stored, result.already), (["NVDA"], ["NVDA"], 1))  # only the gap
        summaries = self.rows(altdata_option_summaries).set_index("symbol")
        self.assertEqual(summaries.loc["MSFT", "read_at"], FRIDAY_EVENING.replace(tzinfo=None))  # kept
        self.assertEqual(summaries.loc["MSFT", "call_open_interest"], 500.0 * 26 + 50)
        self.assertEqual(self.rows(altdata_sync).iloc[0]["status"], "OK")

    def test_compare_stores_nothing(self):
        update_options(self.engine, SMALL, now=FRIDAY_EVENING, fetch=FakeYahoo())
        before = len(self.rows(altdata_option_contracts))
        later = datetime(2026, 10, 17, 9, 0, tzinfo=timezone.utc)
        session, frame = compare_options(self.engine, SMALL, now=later, fetch=FakeYahoo(open_interest=600.0))
        row = frame.set_index("symbol").loc["MSFT"]
        self.assertEqual(session, FRIDAY)
        self.assertEqual(row["volume_now"], row["volume_stored"])
        self.assertGreater(row["oi_now"], row["oi_stored"])
        self.assertEqual(row["found_again"], row["kept_contracts"])
        self.assertGreater(row["oi_changed"], 0)
        self.assertEqual(row["volume_changed"], 0)
        self.assertEqual(len(self.rows(altdata_option_contracts)), before)

    def test_compare_needs_a_stored_read(self):
        with self.assertRaisesRegex(ValueError, "no option chains stored"):
            compare_options(self.engine, SMALL, now=FRIDAY_EVENING, fetch=FakeYahoo())


if __name__ == "__main__":
    unittest.main()
