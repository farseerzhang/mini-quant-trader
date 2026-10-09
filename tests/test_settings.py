import tempfile
import unittest
from datetime import date
from pathlib import Path

from quant.cli import main
from quant.config.settings import (
    DEFAULT_CONFIG_PATH,
    ConfigError,
    FeatureSettings,
    load_settings,
)


def _write(directory: str, text: str) -> Path:
    path = Path(directory) / "settings.yaml"
    path.write_text(text, encoding="utf-8")
    return path


class SettingsTests(unittest.TestCase):
    def test_project_settings_file_loads_every_proposal_section(self):
        settings = load_settings(DEFAULT_CONFIG_PATH)
        self.assertEqual(settings.database.path, "data/quant.db")
        self.assertEqual(settings.market.initial_history, date(2005, 1, 1))
        self.assertEqual(settings.market.universe, ["AAPL", "MSFT", "SPY", "QQQ", "^VIX"])
        self.assertEqual(settings.features.sma, (20, 50, 200))
        self.assertEqual(settings.regime.enter_threshold, 5)
        self.assertEqual(settings.regime.min_dwell_days, 5)
        self.assertEqual(settings.model["type"], "xgboost")
        self.assertEqual(settings.retrain["holdout_start"], date(2024, 1, 1))
        self.assertEqual(settings.backtest["fee_bps"], 1)
        self.assertEqual(settings.risk["max_position_size"], 0.10)

    def test_missing_sections_fall_back_to_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = load_settings(_write(directory, "features:\n  rsi_period: 10\n"))
        self.assertEqual(settings.features.rsi_period, 10)
        self.assertEqual(settings.market.provider, "yfinance")

    def test_feature_version_follows_config(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = load_settings(_write(directory, "features:\n  version: 2\n"))
        self.assertEqual(settings.features.feature_version, "technical-v2")
        self.assertEqual(settings.features.market_feature_version, "market-v2")

    def test_rejects_unknown_and_invalid_settings(self):
        bad_documents = {
            "unknown section": "bogus: {}\n",
            "unknown key": "market:\n  symbolz: [SPY]\n",
            "unsupported sma": "features:\n  sma: [10, 50, 200]\n",
            "bad thresholds": "regime:\n  enter_threshold: 2\n  exit_threshold: 3\n",
            "bad database url": "database:\n  url: postgresql://host/db\n",
            "symbols not a list": "market:\n  symbols: SPY\n",
            "bad date": "market:\n  initial_history: not-a-date\n",
        }
        for name, document in bad_documents.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ConfigError):
                    load_settings(_write(directory, document))

    def test_missing_file_is_a_clear_error(self):
        with self.assertRaises(ConfigError):
            load_settings("does-not-exist.yaml")

    def test_cli_reports_configuration_errors(self):
        self.assertEqual(main(["--config", "does-not-exist.yaml", "calculate-features"]), 1)

    def test_default_feature_settings_match_the_project_file(self):
        self.assertEqual(load_settings(DEFAULT_CONFIG_PATH).features, FeatureSettings())


if __name__ == "__main__":
    unittest.main()