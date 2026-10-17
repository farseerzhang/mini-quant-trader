"""Broker snapshots (forward-test run 2 design): sessions, USD values, cash flows; fake brokers, offline."""

import dataclasses
import unittest
from datetime import date, datetime, timezone

from sqlalchemy import select

import quant.broker.tables  # noqa: F401  registers the broker tables
from quant.broker.ibkr_flex import IbkrAccount, IbkrCashFlow, IbkrPosition
from quant.broker.sync import (
    SyncError,
    sync_accounts,
    trading212_session,
    project_symbol_trading212,
)
from quant.broker.tables import broker_account_snapshots, broker_cash_flows, broker_position_snapshots
from quant.broker.trading212 import Trading212Account, Trading212Position
from quant.persistence import queries
from quant.persistence.database import create_db_engine, init_db
from settings_helper import SETTINGS

FRIDAY_EVENING = datetime(2026, 10, 16, 21, 30, tzinfo=timezone.utc)  # after the US close
POOL = {"information_technology": ("MSFT", "META"), "financials": ("BRK-B",)}
SETTINGS_SMALL = dataclasses.replace(SETTINGS, market=dataclasses.replace(SETTINGS.market, symbols=POOL))
INSTRUMENTS = {
    "MSFT_US_EQ": {"shortName": "MSFT", "name": "Microsoft", "currencyCode": "USD", "type": "STOCK"},
    "FB_US_EQ": {"shortName": "META", "name": "Meta Platforms", "currencyCode": "USD", "type": "STOCK"},
    "BMWd_EQ": {"shortName": "BMW", "name": "BMW", "currencyCode": "EUR", "type": "STOCK"},
}


class FakeTrading212:
    def __init__(self, transactions=()):
        self._transactions = list(transactions)

    def account(self):
        return Trading212Account("123", "EUR", cash_free=1000.0, cash_total=11000.0, positions=[
            Trading212Position("MSFT_US_EQ", 3.0, 400.0, 500.0, None, None),
            Trading212Position("FB_US_EQ", 0.5, 600.0, 700.0, None, None),
            Trading212Position("BMWd_EQ", 10.0, 50.0, 60.0, None, None)])

    def instruments(self):
        return INSTRUMENTS

    def transactions(self, known_references=None):
        return [t for t in self._transactions if t["reference"] not in (known_references or set())]


class FakeIbkr:
    def account(self):
        return IbkrAccount("U1", "USD", "20261016", 2000.0, positions=[
            IbkrPosition("BRK B", "STK", 2.0, 480.0, 960.0, None, "USD", 1.0, "Berkshire"),
            IbkrPosition("SPY", "STK", 1.0, 700.0, 700.0, None, "USD", 1.0, "SPDR S&P 500")],
            cash_flows=[IbkrCashFlow("T1", "20261015;101500", 500.0, "USD", 1.0)])


class SyncTest(unittest.TestCase):
    def setUp(self):
        self.engine = create_db_engine(":memory:", 1000)
        init_db(self.engine)
        queries.upsert_daily_prices(self.engine, [
            {"symbol": "EURUSD=X", "date": date(2026, 10, d), "open": 1.12, "high": 1.13, "low": 1.11,
             "close": 1.10 + d / 1000, "volume": 0, "provider": "test"} for d in (9, 12, 13, 14, 15, 16)])

    def tearDown(self):
        self.engine.dispose()

    def rows(self, table):
        with self.engine.connect() as conn:
            return [dict(r) for r in conn.execute(select(table)).mappings()]

    def sync(self, now=FRIDAY_EVENING, transactions=()):
        return sync_accounts(self.engine, SETTINGS_SMALL, now=now,
                             trading212_client=_with_instruments(FakeTrading212(transactions)),
                             ibkr_client=FakeIbkr())

    def test_snapshots_in_usd(self):
        result = self.sync(transactions=[
            {"type": "DEPOSIT", "amount": 1000, "currency": "EUR", "dateTime": "2026-10-10T09:00:00Z", "reference": "d1"},
            {"type": "INTEREST_ON_FREE_CASH", "amount": 1.5, "currency": "EUR", "dateTime": "2026-10-11T00:00:00Z",
             "reference": "i1"}])
        self.assertEqual(result.failed, {})
        accounts = {r["broker"]: r for r in self.rows(broker_account_snapshots)}
        t212, ibkr = accounts["trading212"], accounts["ibkr"]
        self.assertEqual((t212["session"], ibkr["session"]), (date(2026, 10, 16), date(2026, 10, 16)))
        self.assertAlmostEqual(t212["fx_to_usd"], 1.116)
        self.assertAlmostEqual(t212["value_usd"], 11000 * 1.116)
        self.assertAlmostEqual(ibkr["value_usd"], 960 + 700 + 2000)
        positions = {(r["broker"], r["symbol"]): r for r in self.rows(broker_position_snapshots)}
        self.assertEqual(positions[("trading212", "META")]["in_pool"], 1)  # FB_US_EQ is META
        self.assertEqual(positions[("trading212", "BMW")]["in_pool"], 0)
        self.assertAlmostEqual(positions[("trading212", "BMW")]["value_usd"], 10 * 60 * 1.116)  # EUR price -> USD
        self.assertAlmostEqual(positions[("trading212", "MSFT")]["value_usd"], 1500.0)  # USD price
        self.assertEqual(positions[("ibkr", "BRK-B")]["in_pool"], 1)
        self.assertEqual(positions[("ibkr", "SPY")]["in_pool"], 0)  # a benchmark, not a pool stock
        flows = {r["reference"]: r for r in self.rows(broker_cash_flows)}
        self.assertEqual(set(flows), {"d1", "T1"})  # interest is a return, not a flow
        self.assertEqual(flows["d1"]["session"], date(2026, 10, 12))  # a Saturday deposit counts from Monday
        self.assertEqual(flows["T1"]["session"], date(2026, 10, 15))

    def test_a_rerun_replaces_the_snapshot_and_adds_no_flow_twice(self):
        deposit = {"type": "DEPOSIT", "amount": 100, "currency": "EUR", "dateTime": "2026-10-14T09:00:00Z",
                   "reference": "d2"}
        self.sync(transactions=[deposit])
        again = self.sync(transactions=[deposit])
        self.assertEqual(again.flows_added, 0)
        self.assertEqual(len(self.rows(broker_account_snapshots)), 2)
        self.assertEqual(len(self.rows(broker_position_snapshots)), 5)

    def test_trading212_window(self):
        self.assertEqual(trading212_session(FRIDAY_EVENING, "XNYS"), date(2026, 10, 16))
        sunday = datetime(2026, 10, 18, 20, 0, tzinfo=timezone.utc)
        self.assertEqual(trading212_session(sunday, "XNYS"), date(2026, 10, 16))  # weekend: still Friday's
        monday_morning = datetime(2026, 10, 19, 7, 30, tzinfo=timezone.utc)  # Xetra has opened
        with self.assertRaisesRegex(SyncError, "before European exchanges open"):
            trading212_session(monday_morning, "XNYS")
        result = self.sync(now=monday_morning)
        self.assertIn("trading212", result.failed)  # Trading 212 refused, and none stored before: a failure
        self.assertEqual([r["broker"] for r in self.rows(broker_account_snapshots)], ["ibkr"])  # IBKR stored
        self.sync(now=FRIDAY_EVENING)  # the evening run stores Friday's Trading 212 snapshot ...
        later = self.sync(now=monday_morning)  # ... so a late morning run is fine
        self.assertEqual(later.failed, {})
        self.assertIn("trading212 2026-10-16: already stored earlier, kept", later.notes[0])

    def test_symbols(self):
        self.assertEqual(project_symbol_trading212("FB_US_EQ", INSTRUMENTS["FB_US_EQ"]), ("META", True))
        self.assertEqual(project_symbol_trading212("BMWd_EQ", INSTRUMENTS["BMWd_EQ"]), ("BMW", False))
        self.assertEqual(project_symbol_trading212("SQ_US_EQ", None), ("XYZ", True))  # no metadata: rename table


def _with_instruments(client):
    """sync_trading212 reads the instrument list through a file cache; the fake answers directly."""
    import quant.broker.sync as sync

    original = sync._instruments
    sync._instruments = lambda c, now: c.instruments()
    unittest.addModuleCleanup(lambda: setattr(sync, "_instruments", original))
    return client


if __name__ == "__main__":
    unittest.main()
