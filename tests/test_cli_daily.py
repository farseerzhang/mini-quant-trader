import logging
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import date
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from quant import cli
from quant.config.settings import DEFAULT_CONFIG_PATH

PROJECT_SETTINGS = Path(__file__).resolve().parents[1] / DEFAULT_CONFIG_PATH


class RisingMarketProvider:
    """Business-day bars for any range: steadily rising prices and a calm VIX."""

    def get_daily_data(self, symbol, start, end):
        dates = pd.bdate_range(start, end, inclusive="left")
        if symbol == "^VIX":
            close = np.full(len(dates), 15.0)
        else:
            close = 100 + np.arange(len(dates)) * 0.2
        return pd.DataFrame(
            {"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close,
             "Volume": np.full(len(dates), 1000)},
            index=dates,
        )

    def get_corporate_actions(self, symbol, start, end):
        return pd.DataFrame(columns=["date", "action_type", "value"])


class RunDailyStageOrderTests(unittest.TestCase):
    def run_daily(self, update="ok", features=0, regime=0, signals=0, forward=0):
        calls = []

        def stage(name, result):
            def run(args, settings):
                calls.append(name)
                return result
            return run

        with patch.object(cli, "_update_market_data", stage("update", update)), patch.object(
            cli, "_run_features", stage("features", features)
        ), patch.object(cli, "_run_regime", stage("regime", regime)), patch.object(
            cli, "_run_signals", stage("signals", signals)
        ), patch.object(cli, "_run_forward", stage("forward", forward)), patch.object(  # these runs use the project settings: never log into the real database
            cli, "_record_daily", lambda args, settings, status, message: calls.append(f"logged {status}")
        ), patch.object(cli, "configure_logging", lambda settings, database: logging.NullHandler()), redirect_stdout(
            StringIO()
        ), redirect_stderr(StringIO()):
            code = cli.main(["--config", str(PROJECT_SETTINGS), "run-daily"])
        return code, calls

    def test_runs_every_stage_in_order(self):
        self.assertEqual(self.run_daily(), (0, ["update", "features", "regime", "signals", "forward", "logged OK"]))

    def test_a_failed_market_data_update_stops_the_run(self):
        self.assertEqual(self.run_daily(update="failed"), (1, ["update", "logged FAILED"]))

    def test_a_failed_feature_stage_stops_before_the_regime(self):
        self.assertEqual(self.run_daily(features=1), (1, ["update", "features", "logged FAILED"]))

    def test_failed_symbols_do_not_stop_the_run_but_are_reported(self):
        self.assertEqual(self.run_daily(update="partial"), (1, ["update", "features", "regime", "signals", "forward", "logged FAILED"]))

    def test_a_failed_regime_stage_stops_before_the_signals(self):
        self.assertEqual(self.run_daily(regime=1), (1, ["update", "features", "regime", "logged FAILED"]))

    def test_a_failed_signal_stage_fails_the_run(self):
        self.assertEqual(self.run_daily(signals=1), (1, ["update", "features", "regime", "signals", "logged FAILED"]))

    def test_a_failed_forward_stage_fails_the_run(self):
        self.assertEqual(self.run_daily(forward=1), (1, ["update", "features", "regime", "signals", "forward", "logged FAILED"]))


class RunDailyEndToEndTests(unittest.TestCase):
    def test_one_command_goes_from_prices_to_stored_signals(self):
        with tempfile.TemporaryDirectory() as directory:
            document = yaml.safe_load(PROJECT_SETTINGS.read_text(encoding="utf-8"))
            document["database"]["url"] = f"sqlite:///{Path(directory).as_posix()}/daily.db"
            document["market"]["symbols"] = {"information_technology": ["AAPL"]}
            document["positions"]["path"] = f"{Path(directory).as_posix()}/positions.csv"
            (Path(directory) / "positions.csv").write_text("symbol,quantity\nCASH,1000\n", encoding="utf-8")
            config = Path(directory) / "settings.yaml"
            config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

            output = StringIO()
            with patch("quant.cli.YFinanceProvider", return_value=RisingMarketProvider()), patch(
                "quant.data.store.last_completed_session", return_value=date(2026, 1, 30)
            ), redirect_stdout(output), redirect_stderr(StringIO()):
                code = cli.main(
                    ["--config", str(config), "run-daily", "--initial-date", "2025-01-02"]
                )

        self.assertEqual(code, 0)
        text = output.getvalue()
        for stage in ("[1/5] update-market-data", "[2/5] calculate-features", "[3/5] calculate-regime",
                      "[4/5] generate-signals", "[5/5] forward-test"):
            self.assertIn(stage, text)
        self.assertIn("Latest: BULL (score +5)", text)
        self.assertIn("BULL regime (rule_based-v1)", text)
        self.assertIn("BUY 1", text)  # all cash, AAPL in an uptrend
        self.assertIn("run-daily finished", text)


if __name__ == "__main__":
    unittest.main()
