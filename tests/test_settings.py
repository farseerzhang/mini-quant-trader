import copy
import tempfile
import unittest
from datetime import date
from pathlib import Path

import yaml

from quant.cli import main
from quant.config.settings import DEFAULT_CONFIG_PATH, ConfigError, load_settings

PROJECT_SETTINGS = Path(__file__).resolve().parents[1] / DEFAULT_CONFIG_PATH


def _variant(directory: str, mutate=lambda document: None) -> Path:
    """Write a copy of the real settings file with one change applied."""
    document = yaml.safe_load(PROJECT_SETTINGS.read_text(encoding="utf-8"))
    document = copy.deepcopy(document)
    mutate(document)
    path = Path(directory) / "settings.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


class SettingsTests(unittest.TestCase):
    def test_project_settings_file_loads_every_proposal_section(self):
        settings = load_settings(PROJECT_SETTINGS)
        self.assertEqual(settings.database.path, "data/quant.db")
        self.assertEqual(settings.market.initial_history, date(2005, 1, 1))
        self.assertGreaterEqual(len(settings.market.symbols), 30)
        self.assertEqual(settings.market.symbols[:2], ("AAPL", "MSFT"))
        self.assertEqual(settings.market.universe[-3:], ["SPY", "QQQ", "^VIX"])
        self.assertEqual(settings.features.sma, (20, 50, 200))
        self.assertEqual(settings.regime.enter_threshold, 3)
        self.assertEqual(settings.regime.min_dwell_days, 5)
        self.assertEqual(settings.regime.rule_version, "rule_based-v1")
        self.assertEqual(settings.model["type"], "xgboost")
        self.assertEqual(settings.model["target"]["kind"], "excess_vs_benchmark_vol_scaled")
        self.assertGreaterEqual(settings.model["embargo_days"], settings.model["target"]["horizon_days"] + 1)
        self.assertEqual(settings.retrain["validation"]["holdout_months"], 12)
        self.assertEqual(settings.retrain["search"]["seed"], 42)
        self.assertEqual(settings.backtest["fee_bps"], 1)
        self.assertEqual(settings.risk["max_position_size"], 0.10)

    def test_values_come_from_the_file_not_from_code(self):
        def change(document):
            document["features"]["rsi_period"] = 10
            document["features"]["version"] = 2
            document["market"]["symbols"] = ["NVDA"]
            document["market"]["request_timeout_seconds"] = 7

        with tempfile.TemporaryDirectory() as directory:
            settings = load_settings(_variant(directory, change))
        self.assertEqual(settings.features.rsi_period, 10)
        self.assertEqual(settings.features.feature_version, "technical-v2")
        self.assertEqual(settings.features.market_feature_version, "market-v2")
        self.assertEqual(settings.market.request_timeout_seconds, 7)
        self.assertEqual(settings.market.universe, ["NVDA", "SPY", "QQQ", "^VIX"])

    def test_symbols_may_be_grouped_into_sub_lists(self):
        def group(document):
            document["market"]["symbols"] = [["aapl", "MSFT"], ["XOM"], "KO", ["MSFT"]]

        with tempfile.TemporaryDirectory() as directory:
            settings = load_settings(_variant(directory, group))
        self.assertEqual(settings.market.symbols, ("AAPL", "MSFT", "XOM", "KO"))

    def test_missing_sections_and_keys_are_errors_not_silent_defaults(self):
        def drop_section(document):
            del document["regime"]

        def drop_key(document):
            del document["market"]["calendar"]

        for name, mutate in {"section": drop_section, "key": drop_key}.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ConfigError, "missing"):
                    load_settings(_variant(directory, mutate))

    def test_rejects_unknown_and_invalid_settings(self):
        def set_value(section, key, value):
            def mutate(document):
                document[section][key] = value
            return mutate

        def add_section(document):
            document["bogus"] = {}

        bad = {
            "unknown section": add_section,
            "unknown key": set_value("market", "symbolz", ["SPY"]),
            "sma not increasing": set_value("features", "sma", [50, 20, 200]),
            "sma wrong length": set_value("features", "sma", [20, 50]),
            "bad thresholds": set_value("regime", "enter_threshold", 1),
            "threshold above max score": set_value("regime", "enter_threshold", 6),
            "bad vix levels": set_value("regime", "vix_low", 35),
            "bad database url": set_value("database", "url", "postgresql://host/db"),
            "symbols not a list": set_value("market", "symbols", "SPY"),
            "bad date": set_value("market", "initial_history", "not-a-date"),
        }
        for name, mutate in bad.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ConfigError):
                    load_settings(_variant(directory, mutate))

    def test_missing_file_is_a_clear_error(self):
        with self.assertRaises(ConfigError):
            load_settings("does-not-exist.yaml")

    def test_cli_reports_configuration_errors(self):
        self.assertEqual(main(["--config", "does-not-exist.yaml", "calculate-features"]), 1)


if __name__ == "__main__":
    unittest.main()