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
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


class SettingsTests(unittest.TestCase):
    def test_project_settings_file_loads_every_proposal_section(self):
        settings = load_settings(PROJECT_SETTINGS)
        self.assertEqual(settings.database.path, "data/quant.db")
        self.assertEqual(settings.market.initial_history, date(2005, 1, 1))
        self.assertGreaterEqual(len(settings.market.stocks), 30)
        self.assertEqual(settings.market.stocks[:2], ("AAPL", "MSFT"))
        self.assertEqual(settings.market.universe[-8:], ["SPY", "QQQ", "^VIX", "TLT", "IEF", "HYG", "^TNX", "^IRX"])
        self.assertEqual(len(settings.market.symbols), 11)  # every GICS sector is represented
        self.assertEqual(settings.market.sectors["XOM"], "energy")
        self.assertEqual(settings.risk.max_sector_weight, 0.30)
        self.assertEqual(settings.strategies.regime_switch.bear_exposures[0], 0.0)
        self.assertEqual(settings.features.sma, (20, 50, 200))
        self.assertEqual(settings.regime.enter_threshold, 3)
        self.assertEqual(settings.regime.min_dwell_days, 5)
        self.assertEqual(settings.regime.rule_version, "rule_based-v1")
        self.assertEqual(settings.model.type, "xgboost")
        self.assertEqual(settings.model.target.kind, "excess_vs_benchmark_vol_scaled")
        self.assertEqual((settings.model.target.horizon_days, settings.model.embargo_days), (5, 6))
        self.assertEqual(settings.retrain.validation.holdout_months, 12)
        self.assertEqual(settings.retrain.search.seed, 42)
        self.assertEqual(settings.retrain.search.space["logistic_regression"]["C"], (0.01, 0.1, 1.0))
        self.assertEqual(settings.retrain.sample_every_sessions, 5)
        self.assertEqual(settings.backtest.fees, 0.0001)
        self.assertEqual(settings.backtest.slippage, 0.0005)
        self.assertEqual(settings.holdout_months, 12)
        self.assertEqual(settings.risk.max_position_size, 0.10)
        self.assertEqual(settings.risk.max_portfolio_exposure, 1.0)
        self.assertEqual((settings.risk.atr_stop_multiple, settings.risk.high_volatility), (2.0, 0.60))
        self.assertEqual(settings.positions.path, "data/positions.csv")
        self.assertEqual(settings.signals.strategy, "regime_switch")
        self.assertEqual(settings.signals.hold_band, settings.backtest.min_trade_weight)
        self.assertEqual((settings.logging.level, settings.logging.max_megabytes, settings.logging.backups), ("INFO", 5, 5))
        self.assertEqual(
            settings.strategies.rule_names, ("trend", "mean_reversion", "regime_switch", "regime_switch_bear50")
        )

    def test_signal_strategy_can_be_any_rule_based_strategy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _variant(directory, lambda d: d["signals"].update(strategy="regime_switch_bear50"))
            self.assertEqual(load_settings(path).signals.strategy, "regime_switch_bear50")

    def test_hold_band_that_differs_from_the_backtest_warns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _variant(directory, lambda d: d["signals"].update(hold_band=0.05))
            with self.assertWarnsRegex(UserWarning, "will not trade like the backtest"):
                self.assertEqual(load_settings(path).signals.hold_band, 0.05)

    def test_values_come_from_the_file_not_from_code(self):
        def change(document):
            document["features"]["rsi_period"] = 10
            document["features"]["version"] = 2
            document["market"]["symbols"] = {"information_technology": ["NVDA"]}
            document["market"]["request_timeout_seconds"] = 7

        with tempfile.TemporaryDirectory() as directory:
            settings = load_settings(_variant(directory, change))
        self.assertEqual(settings.features.rsi_period, 10)
        self.assertEqual(settings.features.feature_version, "technical-v2")
        self.assertEqual(settings.features.market_feature_version, "market-v2")
        self.assertEqual(settings.market.request_timeout_seconds, 7)
        self.assertEqual(settings.market.universe, ["NVDA", "SPY", "QQQ", "^VIX", "TLT", "IEF", "HYG", "^TNX", "^IRX"])

    def test_symbols_map_gics_sectors_to_stocks(self):
        def pool(document):
            document["market"]["symbols"] = {"energy": ["xom", "CVX"], "information_technology": ["AAPL"]}

        with tempfile.TemporaryDirectory() as directory:
            settings = load_settings(_variant(directory, pool))
        self.assertEqual(settings.market.stocks, ("XOM", "CVX", "AAPL"))
        self.assertEqual(settings.market.sectors, {"XOM": "energy", "CVX": "energy", "AAPL": "information_technology"})
        self.assertEqual(settings.market.universe, ["XOM", "CVX", "AAPL", "SPY", "QQQ", "^VIX", "TLT", "IEF", "HYG", "^TNX", "^IRX"])

    def test_rejects_invalid_sector_pools(self):
        def pool(value):
            def mutate(document):
                document["market"]["symbols"] = value
            return mutate

        bad = {
            "plain list": pool(["AAPL", "MSFT"]),
            "unknown sector": pool({"finacials": ["JPM"]}),
            "stock in two sectors": pool({"energy": ["XOM"], "materials": ["XOM"]}),
            "empty sector": pool({"energy": []}),
            "nested list": pool({"energy": [["XOM"]]}),
            "benchmark in the pool": pool({"financials": ["SPY"]}),
        }
        for name, mutate in bad.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ConfigError):
                    load_settings(_variant(directory, mutate))

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
            "symbols not a mapping": set_value("market", "symbols", "SPY"),
            "position cap above 1": set_value("risk", "max_position_size", 1.5),
            "sector cap below position cap": set_value("risk", "max_sector_weight", 0.05),
            "rsi exit below entry": lambda d: d["strategies"]["mean_reversion"].update(rsi_exit=20),
            "bear exposure above 1": lambda d: d["strategies"]["regime_switch"].update(bear_exposures=[1.5]),
            "no baseline exposure": lambda d: d["strategies"]["regime_switch"].update(bear_exposures=[]),
            "unknown strategy key": lambda d: d["strategies"]["mean_reversion"].update(rsi_entri=25),
            "unknown benchmark": set_value("backtest", "benchmarks", ["SPY", "QQQ_buy_and_hold"]),
            "negative slippage": set_value("backtest", "slippage_bps", -1),
            "embargo shorter than the label window": set_value("model", "embargo_days", 5),
            "unknown model type": set_value("model", "type", "svm"),
            "unknown target kind": lambda d: d["model"]["target"].update(kind="direction"),
            "threshold below round-trip costs": lambda d: d["model"]["target"].update(min_threshold=0.001),
            "unknown validation scheme": lambda d: d["retrain"]["validation"].update(scheme="random_split"),
            "zero folds": lambda d: d["retrain"]["validation"].update(min_folds=0),
            "search space for an unknown model": lambda d: d["retrain"]["search"]["space"].update(svm={"C": [1]}),
            "empty candidate list": lambda d: d["retrain"]["search"]["space"]["random_forest"].update(max_depth=[]),
            "missing sampling stride": lambda d: d["retrain"].pop("sample_every_sessions"),
            "misspelled gate": lambda d: d["retrain"]["gates"]["predictor"].update(max_calibraton_error=0.05),
            "gate with the wrong type": lambda d: d["retrain"]["gates"]["predictor"].update(leak_tests_pass="yes"),
            "misspelled regime gate": lambda d: d["retrain"]["gates"]["regime"].update(min_avg_regime_dwel_days=10),
            "regime gate with the wrong type": lambda d: d["retrain"]["gates"]["regime"].update(min_avg_regime_dwell_days="ten"),
            "bad date": set_value("market", "initial_history", "not-a-date"),
            "exposure cap above 1": set_value("risk", "max_portfolio_exposure", 1.2),
            "exposure cap of zero": set_value("risk", "max_portfolio_exposure", 0),
            "non-positive stop multiple": set_value("risk", "atr_stop_multiple", 0),
            "non-positive volatility flag": set_value("risk", "high_volatility", -0.1),
            "missing risk key": lambda d: d["risk"].pop("high_volatility"),
            "missing positions section": lambda d: d.pop("positions"),
            "empty positions path": set_value("positions", "path", " "),
            "unknown signal strategy": set_value("signals", "strategy", "momentum"),
            "ml strategy drives signals": set_value("signals", "strategy", "ml_select"),
            "hold band of 1": set_value("signals", "hold_band", 1.0),
            "unknown signals key": set_value("signals", "hold_bnad", 0.02),
            "unknown log level": set_value("logging", "level", "LOUD"),
            "zero log size": set_value("logging", "max_megabytes", 0),
            "negative backups": set_value("logging", "backups", -1),
            "missing logging section": lambda d: d.pop("logging"),
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