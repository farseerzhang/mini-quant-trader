import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from quant.data.store import store_market_data
from quant.persistence import queries
from quant.persistence.database import create_db_engine, init_db
from tests.settings_helper import SETTINGS


class FakeProvider:
    def __init__(self, prices: pd.DataFrame, actions: pd.DataFrame):
        self._prices = prices
        self._actions = actions

    def get_daily_data(self, symbol, start, end):
        return self._prices

    def get_corporate_actions(self, symbol, start, end):
        return self._actions


def _prices_frame(rows):
    index = pd.DatetimeIndex([r[0] for r in rows], name="Date")
    return pd.DataFrame(
        {
            "Open": [r[1] for r in rows],
            "High": [r[2] for r in rows],
            "Low": [r[3] for r in rows],
            "Close": [r[4] for r in rows],
            "Volume": [r[5] for r in rows],
        },
        index=index,
    )


def _actions_frame(rows):
    return pd.DataFrame(rows, columns=["date", "action_type", "value"])


class StoreMarketDataTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "quant.db"
        self.engine = create_db_engine(self.db_path, SETTINGS.database.busy_timeout_ms)
        init_db(self.engine)

    def tearDown(self):
        self.engine.dispose()
        self._tmp.cleanup()

    def _provider(self):
        prices = _prices_frame(
            [
                ("2026-09-29", 10.0, 12.0, 9.0, 11.0, 100),
                ("2026-09-30", 11.0, 13.0, 10.5, 12.5, 200),
            ]
        )
        actions = _actions_frame(
            [(date(2026, 9, 30), "DIVIDEND", 0.25)]
        )
        return FakeProvider(prices, actions)

    def test_stores_prices_and_actions(self):
        summary = store_market_data(
            self.engine, self._provider(), "aapl", date(2026, 9, 29), date(2026, 10, 1),
            provider_name=SETTINGS.market.provider,
        )
        self.assertEqual(summary.symbol, "AAPL")
        self.assertEqual(summary.stored_prices, 2)
        self.assertEqual(summary.stored_actions, 1)
        self.assertEqual(summary.quarantined, 0)
        self.assertEqual(queries.count_daily_prices(self.engine, "AAPL"), 2)
        self.assertEqual(queries.latest_stored_date(self.engine, "AAPL"), date(2026, 9, 30))

    def test_is_idempotent(self):
        store_market_data(
            self.engine, self._provider(), "AAPL", date(2026, 9, 29), date(2026, 10, 1),
            provider_name=SETTINGS.market.provider,
        )
        second = store_market_data(
            self.engine, self._provider(), "AAPL", date(2026, 9, 29), date(2026, 10, 1),
            provider_name=SETTINGS.market.provider,
        )
        self.assertEqual(second.stored_prices, 0)
        self.assertEqual(second.stored_actions, 0)
        self.assertEqual(queries.count_daily_prices(self.engine, "AAPL"), 2)

    def test_invalid_rows_are_quarantined_not_stored(self):
        prices = _prices_frame(
            [
                ("2026-09-29", 10.0, 12.0, 9.0, 11.0, 100),
                ("2026-09-30", 11.0, 9.0, 10.5, 12.5, 200),  # high < close -> invalid
            ]
        )
        provider = FakeProvider(prices, _actions_frame([]))
        summary = store_market_data(
            self.engine, provider, "AAPL", date(2026, 9, 29), date(2026, 10, 1),
            provider_name=SETTINGS.market.provider,
        )
        self.assertEqual(summary.stored_prices, 1)
        self.assertEqual(summary.quarantined, 1)
        self.assertEqual(queries.count_daily_prices(self.engine, "AAPL"), 1)

    def test_survives_reopen(self):
        store_market_data(
            self.engine, self._provider(), "AAPL", date(2026, 9, 29), date(2026, 10, 1),
            provider_name=SETTINGS.market.provider,
        )
        self.engine.dispose()
        reopened = create_db_engine(self.db_path, SETTINGS.database.busy_timeout_ms)
        try:
            self.assertEqual(queries.count_daily_prices(reopened, "AAPL"), 2)
        finally:
            reopened.dispose()


if __name__ == "__main__":
    unittest.main()
