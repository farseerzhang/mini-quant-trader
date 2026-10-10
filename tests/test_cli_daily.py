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
    def run_daily(self, update="ok", features=0, regime=0):
        calls = []

        def stage(name, result):
            def run(args, settings):
                calls.append(name)
                return result
            return run

        with patch.object(cli, "_update_market_data", stage("update", update)), patch.object(
            cli, "_run_features", stage("features", features)
        ), patch.object(cli, "_run_regime", stage("regime", regime)), redirect_stdout(
            StringIO()
        ), redirect_stderr(StringIO()):
            code = cli.main(["--config", str(PROJECT_SETTINGS), "run-daily"])
        return code, calls

    def test_runs_every_stage_in_order(self):
        self.assertEqual(self.run_daily(), (0, ["update", "features", "regime"]))

    def test_a_failed_market_data_update_stops_the_run(self):
        self.assertEqual(self.run_daily(update="failed"), (1, ["update"]))

    def test_a_failed_feature_stage_stops_before_the_regime(self):
        self.assertEqual(self.run_daily(features=1), (1, ["update", "features"]))

    def test_failed_symbols_do_not_stop_the_run_but_are_reported(self):
        self.assertEqual(self.run_daily(update="partial"), (1, ["update", "features", "regime"]))


class RunDailyEndToEndTests(unittest.TestCase):
    def test_one_command_goes_from_prices_to_a_stored_regime(self):
        with tempfile.TemporaryDirectory() as directory:
            document = yaml.safe_load(PROJECT_SETTINGS.read_text(encoding="utf-8"))
            document["database"]["url"] = f"sqlite:///{Path(directory).as_posix()}/daily.db"
            document["market"]["symbols"] = ["AAPL"]
            config = Path(directory) / "settings.yaml"
            config.write_text(yaml.safe_dump(document), encoding="utf-8")

            output = StringIO()
            with patch("quant.cli.YFinanceProvider", return_value=RisingMarketProvider()), patch(
                "quant.data.store.last_completed_session", return_value=date(2026, 1, 30)
            ), redirect_stdout(output), redirect_stderr(StringIO()):
                code = cli.main(
                    ["--config", str(config), "run-daily", "--initial-date", "2025-01-02"]
                )

        self.assertEqual(code, 0)
        text = output.getvalue()
        for stage in ("[1/3] update-market-data", "[2/3] calculate-features", "[3/3] calculate-regime"):
            self.assertIn(stage, text)
        self.assertIn("Latest: BULL (score +5)", text)
        self.assertIn("run-daily finished", text)


if __name__ == "__main__":
    unittest.main()
