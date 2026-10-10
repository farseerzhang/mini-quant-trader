import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import date
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from quant.cli import main
from quant.data.yfinance_provider import MarketDataError
from tests.settings_helper import SETTINGS

PROJECT_SETTINGS = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"


class RecordingProvider:
    def __init__(self, failing=()):
        self.calls = []
        self.failing = set(failing)

    def get_daily_data(self, symbol, start, end):
        self.calls.append((symbol, start, end))
        if symbol in self.failing:
            raise MarketDataError("simulated failure")
        return pd.DataFrame(
            {"Open": [10.0], "High": [11.0], "Low": [9.0], "Close": [10.5], "Volume": [100]},
            index=pd.DatetimeIndex(["2026-10-01"], name="Date"),
        )

    def get_corporate_actions(self, symbol, start, end):
        return pd.DataFrame(columns=["date", "action_type", "value"])


class UpdateCommandTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        document = yaml.safe_load(PROJECT_SETTINGS.read_text(encoding="utf-8"))
        document["database"]["url"] = f"sqlite:///{Path(self.temp.name).as_posix()}/update.db"
        self.config = Path(self.temp.name) / "settings.yaml"
        self.config.write_text(yaml.safe_dump(document), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def run_update(self, provider, *extra):
        with patch("quant.cli.YFinanceProvider", return_value=provider), patch(
            "quant.data.store.last_completed_session", return_value=date(2026, 10, 1)
        ), redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            return main(["--config", str(self.config), "update-market-data", *extra])

    def test_without_arguments_updates_the_configured_pool_from_initial_history(self):
        provider = RecordingProvider()
        self.assertEqual(self.run_update(provider), 0)
        expected = [
            (symbol, SETTINGS.market.initial_history, date(2026, 10, 2))
            for symbol in SETTINGS.market.universe
        ]
        self.assertEqual(provider.calls, expected)

    def test_initial_date_overrides_the_settings_file(self):
        provider = RecordingProvider()
        self.assertEqual(self.run_update(provider, "--initial-date", "2026-09-29"), 0)
        self.assertEqual({call[1] for call in provider.calls}, {date(2026, 9, 29)})

    def test_one_failing_symbol_does_not_abort_the_pool(self):
        failing = SETTINGS.market.universe[0]
        provider = RecordingProvider(failing=[failing])
        self.assertEqual(self.run_update(provider), 1)
        self.assertEqual(len(provider.calls), len(SETTINGS.market.universe))

    def test_initial_date_after_the_last_session_is_rejected_before_any_request(self):
        provider = RecordingProvider()
        self.assertEqual(self.run_update(provider, "--initial-date", "2026-10-05"), 1)
        self.assertEqual(provider.calls, [])


if __name__ == "__main__":
    unittest.main()
