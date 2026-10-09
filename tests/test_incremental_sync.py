import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from quant.data.store import sync_market_data
from quant.data.yfinance_provider import MarketDataError
from quant.persistence import queries
from quant.persistence.database import create_db_engine, init_db
from tests.settings_helper import SETTINGS


class RecordingProvider:
    def __init__(self, fail_symbols=()):
        self.calls = []
        self.fail_symbols = set(fail_symbols)

    def get_daily_data(self, symbol, start, end):
        self.calls.append((symbol, start, end))
        if symbol in self.fail_symbols:
            raise MarketDataError("simulated provider failure")
        return pd.DataFrame(
            {
                "Open": [100.0],
                "High": [102.0],
                "Low": [99.0],
                "Close": [101.0],
                "Volume": [1000],
            },
            index=pd.DatetimeIndex(["2026-10-01"], name="Date"),
        )

    def get_corporate_actions(self, symbol, start, end):
        return pd.DataFrame(columns=["date", "action_type", "value"])


class IncrementalSyncTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.engine = create_db_engine(
            Path(self.temp_dir.name) / "market.db", SETTINGS.database.busy_timeout_ms
        )
        init_db(self.engine)
        self.now = datetime(2026, 10, 2, 18, 52, tzinfo=timezone.utc)
        self.sync_settings = {
            "provider_name": SETTINGS.market.provider,
            "calendar_name": SETTINGS.market.calendar,
        }

    def tearDown(self):
        self.engine.dispose()
        self.temp_dir.cleanup()

    def test_existing_symbol_starts_at_next_session_and_uses_completed_session(self):
        queries.upsert_daily_prices(
            self.engine,
            [{
                "symbol": "AAPL", "date": date(2026, 9, 30),
                "open": 98.0, "high": 102.0, "low": 97.0, "close": 100.0,
                "volume": 900, "provider": "test",
            }],
        )
        provider = RecordingProvider()

        report = sync_market_data(
            self.engine, provider, ["AAPL"], date(2020, 1, 1), now=self.now, **self.sync_settings
        )

        self.assertEqual(report.as_of, date(2026, 10, 1))
        self.assertEqual(provider.calls, [("AAPL", date(2026, 10, 1), date(2026, 10, 2))])
        self.assertEqual(report.summaries[0].stored_prices, 1)
        self.assertEqual(queries.latest_stored_date(self.engine, "AAPL"), date(2026, 10, 1))

    def test_new_symbol_uses_configured_initial_date(self):
        provider = RecordingProvider()
        sync_market_data(
            self.engine, provider, ["MSFT"], date(2024, 1, 2), now=self.now, **self.sync_settings
        )
        self.assertEqual(provider.calls, [("MSFT", date(2024, 1, 2), date(2026, 10, 2))])

    def test_provider_failure_does_not_abort_other_symbols(self):
        provider = RecordingProvider(fail_symbols=["BAD"])
        report = sync_market_data(
            self.engine, provider, ["BAD", "AAPL"], date(2020, 1, 1), now=self.now, **self.sync_settings
        )
        self.assertEqual([failure.symbol for failure in report.failures], ["BAD"])
        self.assertEqual([summary.symbol for summary in report.summaries], ["AAPL"])


if __name__ == "__main__":
    unittest.main()
