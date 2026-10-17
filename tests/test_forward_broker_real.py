"""Forward-test run 2 design: `real` from the broker snapshots; offline, hand-made snapshots."""

import unittest
from datetime import date

import pandas as pd

import quant.broker.tables  # noqa: F401  registers the broker tables
from quant.broker.tables import broker_account_snapshots, broker_cash_flows, broker_position_snapshots
from quant.forward.broker_real import (
    BROKER_VALUE_SOURCE,
    CASH_PART,
    IN_POOL,
    OUTSIDE_POOL,
    broker_holdings,
    broker_real,
    broker_start,
)
from quant.persistence.database import create_db_engine, init_db
from settings_helper import SETTINGS

D1, D2, D3 = date(2026, 10, 19), date(2026, 10, 20), date(2026, 10, 21)
SESSIONS = pd.DatetimeIndex([D1, D2, D3])


def account(session, broker, value, cash, fx=1.0, currency="USD"):
    return {"session": session, "broker": broker, "account_id": "x", "account_currency": currency, "fx_to_usd": fx,
            "value_account_currency": value / fx, "cash_account_currency": cash / fx, "value_usd": value,
            "cash_usd": cash, "statement_date": str(session), "fetched_at": pd.Timestamp("2026-10-21 22:00")}


def position(session, broker, ticker, symbol, in_pool, quantity, value_usd, currency="USD"):
    return {"session": session, "broker": broker, "broker_ticker": ticker, "symbol": symbol, "name": symbol,
            "asset_type": "STOCK", "in_pool": in_pool, "quantity": quantity, "price": value_usd / quantity,
            "price_currency": currency, "value_usd": value_usd}


class BrokerRealTest(unittest.TestCase):
    def setUp(self):
        self.engine = create_db_engine(":memory:", 1000)
        init_db(self.engine)
        accounts = [
            account(D1, "ibkr", 1500, 500), account(D1, "trading212", 1100, 100, fx=1.1, currency="EUR"),
            # D2: MSFT +10%, BMW (EUR) +0, EUR/USD up 1.1 -> 1.2; a deposit of 500 USD lands in IBKR cash.
            account(D2, "ibkr", 2100, 1000), account(D2, "trading212", 1200, 109.09, fx=1.2, currency="EUR"),
            account(D3, "trading212", 1200, 109.09, fx=1.2, currency="EUR"),  # IBKR's D3 statement missing
        ]
        positions = [
            position(D1, "ibkr", "MSFT", "MSFT", 1, 2, 1000), position(D1, "trading212", "BMWd_EQ", "BMW", 0, 10, 1000,
                                                                          "EUR"),
            position(D2, "ibkr", "MSFT", "MSFT", 1, 2, 1100),
            position(D2, "trading212", "BMWd_EQ", "BMW", 0, 10, 1000 * 1.2 / 1.1, "EUR"),
            position(D3, "trading212", "BMWd_EQ", "BMW", 0, 10, 1000 * 1.2 / 1.1, "EUR"),
        ]
        with self.engine.begin() as conn:
            conn.execute(broker_account_snapshots.insert(), accounts)
            conn.execute(broker_position_snapshots.insert(), positions)
            conn.execute(broker_cash_flows.insert(), [{"broker": "ibkr", "reference": "t1",
                                                        "happened_at": pd.Timestamp(D2), "session": D2,
                                                        "kind": "DEPOSIT", "amount": 500, "currency": "USD",
                                                        "amount_usd": 500}])

    def tearDown(self):
        self.engine.dispose()

    def test_start_needs_both_brokers(self):
        value, source, rows = broker_start(self.engine, pd.Timestamp(D1))
        self.assertEqual((value, source), (2600.0, BROKER_VALUE_SOURCE))
        self.assertEqual({r["symbol"]: r["quantity"] for r in rows}, {"MSFT": 2.0, "BMW": 10.0, "CASH": 600.0})
        with self.assertRaisesRegex(ValueError, "no ibkr snapshot of 2026-10-21"):
            broker_start(self.engine, pd.Timestamp(D3))

    def test_deposits_are_not_returns_and_parts_add_up(self):
        real = broker_real(self.engine, SETTINGS, SESSIONS, start_value=2600.0)
        # D2: value 3300 with a 500 deposit -> (3300 - 500) / 2600 - 1
        expected = (3300 - 500) / 2600 - 1
        self.assertAlmostEqual(real.returns.loc[pd.Timestamp(D2)], expected)
        self.assertAlmostEqual(real.value.loc[pd.Timestamp(D2)], 2600 * (1 + expected))
        self.assertAlmostEqual(real.account_value.loc[pd.Timestamp(D2)], 3300)
        parts = real.contributions.loc[pd.Timestamp(D2)]
        self.assertAlmostEqual(parts[IN_POOL], 100 / 2600)  # MSFT +100
        self.assertAlmostEqual(parts[OUTSIDE_POOL], (1000 * 1.2 / 1.1 - 1000) / 2600)  # BMW in USD: only FX moved
        self.assertAlmostEqual(parts[IN_POOL] + parts[OUTSIDE_POOL] + parts[CASH_PART], expected)
        self.assertGreater(parts["FX (inside the others)"], 0)  # EUR rose against USD
        self.assertEqual(real.incomplete, [pd.Timestamp(D3)])  # IBKR carried forward on D3
        self.assertAlmostEqual(real.weights.loc[pd.Timestamp(D2), "MSFT"], 1100 / 3300)

    def test_holdings_for_the_signals(self):
        holdings = broker_holdings(self.engine, SETTINGS, pd.Timestamp(D3))
        self.assertEqual(holdings.quantities, {"MSFT": 2.0})  # IBKR's latest (D2) plus Trading 212's D3
        self.assertEqual(holdings.unmanaged, {"BMW": 10.0})
        self.assertAlmostEqual(holdings.cash, 1000 + 109.09)


if __name__ == "__main__":
    unittest.main()
