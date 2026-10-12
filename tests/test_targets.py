import dataclasses
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from quant.persistence import queries, tables
from quant.persistence.database import create_db_engine
from quant.regime.pipeline import RULE_METHOD
from quant.signals.targets import TARGET_COLUMNS, compute_targets, load_targets
from quant.strategies import build_strategies
from tests.backtest_fixture import build_market_db, small_settings

STOCKS = ["AAPL", "MSFT", "XOM"]  # information_technology x2, energy


def _panel(n: int, **overrides) -> dict[str, pd.DataFrame]:
    """Every stock in an uptrend with a positive MACD (trend holds all three)."""
    index = pd.bdate_range("2026-01-01", periods=n)
    base = {"price_vs_sma200": 0.1, "sma50_vs_sma200": 0.1, "macd_hist_over_price": 0.01, "rsi14": 55.0,
            "realized_volatility_20d": 0.2, "atr14_over_price": 0.02}
    panel = {}
    for column in TARGET_COLUMNS:
        value = overrides.get(column, base[column])
        panel[column] = pd.DataFrame(np.broadcast_to(value, (n, len(STOCKS))).astype(float), index=index, columns=STOCKS)
    return panel


def _regimes(labels: list[str], n: int) -> pd.Series:
    index = pd.bdate_range("2026-01-01", periods=n)
    return pd.Series((labels * n)[:n], index=index, dtype=object)


class TargetDetailTests(unittest.TestCase):
    def setUp(self):
        self.settings = small_settings()
        self.n = 10
        self.panel = _panel(self.n)
        self.closes = pd.DataFrame(100.0, index=self.panel["rsi14"].index, columns=STOCKS)

    def targets(self, settings=None, regime="BULL", panel=None):
        return compute_targets(panel or self.panel, _regimes([regime], self.n), settings or self.settings, self.closes)

    def test_bull_trend_holds_every_stock_at_the_position_cap(self):
        targets = self.targets()
        stocks = targets.stocks
        self.assertEqual((targets.session, targets.regime, targets.regime_exposure), (self.panel["rsi14"].index[-1], "BULL", 1.0))
        np.testing.assert_allclose(stocks["target_weight"], 0.10)  # 1/3 capped at 10%
        self.assertEqual(list(stocks["rule"]), ["trend"] * 3)
        self.assertTrue(stocks["position_cap"].all())
        self.assertFalse(stocks["sector_cap"].any() or stocks["exposure_cap"].any())
        self.assertEqual(list(stocks["held_sessions"]), [self.n] * 3)
        self.assertAlmostEqual(targets.cash, 0.70)
        self.assertEqual(targets.regime_source, "rule_based-v1")

    def test_sector_cap_is_attributed_to_the_capped_sector_only(self):
        settings = dataclasses.replace(self.settings, risk=dataclasses.replace(self.settings.risk, max_sector_weight=0.15))
        stocks = self.targets(settings).stocks
        np.testing.assert_allclose(stocks["target_weight"], [0.075, 0.075, 0.10])
        self.assertEqual(stocks["sector_cap"].to_dict(), {"AAPL": True, "MSFT": True, "XOM": False})

    def test_portfolio_exposure_cap_scales_every_weight(self):
        settings = dataclasses.replace(self.settings, risk=dataclasses.replace(self.settings.risk, max_portfolio_exposure=0.15))
        targets = self.targets(settings)
        self.assertAlmostEqual(targets.invested, 0.15)
        np.testing.assert_allclose(targets.weights, 0.05)
        self.assertTrue(targets.stocks["exposure_cap"].all())

    def test_bear_regime_holds_cash_but_keeps_the_rule(self):
        targets = self.targets(regime="BEAR")
        self.assertEqual(targets.regime_exposure, 0.0)
        self.assertEqual(targets.invested, 0.0)
        self.assertEqual(list(targets.stocks["rule"]), ["trend"] * 3)  # the reason can say "BEAR: cash"

    def test_risk_flags_never_change_weights(self):
        volatile = _panel(self.n, realized_volatility_20d=np.array([0.2, 0.2, 0.9]))
        calm, wild = self.targets().stocks, self.targets(panel=volatile).stocks
        self.assertEqual(wild["high_volatility"].to_dict(), {"AAPL": False, "MSFT": False, "XOM": True})
        pd.testing.assert_series_equal(calm["target_weight"], wild["target_weight"])
        np.testing.assert_allclose(wild["stop_price"], 100 * (1 - 2.0 * 0.02))  # close - 2 x ATR14

    def test_stock_without_data_at_t_is_not_held(self):
        panel = _panel(self.n)
        panel["price_vs_sma200"].iloc[-1, 0] = np.nan
        stocks = self.targets(panel=panel).stocks
        self.assertFalse(stocks.loc["AAPL", "has_data"])
        self.assertEqual(stocks.loc["AAPL", "target_weight"], 0.0)
        self.assertTrue(pd.isna(stocks.loc["AAPL", "rule"]))


class LiveEqualsBacktestTests(unittest.TestCase):
    """Section 18.1: the live target at T is the backtest's weight at T, exactly."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.settings = small_settings()
        cls.db = Path(cls.temp.name) / "targets.db"
        build_market_db(cls.db, cls.settings)
        cls.engine = create_db_engine(cls.db, 5000)
        stocks = list(cls.settings.market.stocks)
        cls.panel = queries.load_feature_panel(cls.engine, cls.settings.features.feature_version, stocks, TARGET_COLUMNS)
        cls.regimes = queries.load_regimes(cls.engine, RULE_METHOD, cls.settings.regime.rule_version)
        cls.closes = queries.load_raw_closes(cls.engine, stocks)

    @classmethod
    def tearDownClass(cls):
        cls.engine.dispose()
        cls.temp.cleanup()

    def test_every_rule_strategy_matches_its_backtest_weights_at_t(self):
        for strategy in build_strategies(self.settings):
            with self.subTest(strategy.name):
                backtest = strategy.weights(self.panel, self.regimes)
                targets = compute_targets(self.panel, self.regimes, self.settings, self.closes, strategy=strategy.name)
                pd.testing.assert_series_equal(
                    targets.weights, backtest.loc[targets.session].fillna(0.0), check_names=False
                )
                weighted = targets.stocks[targets.weights > 0]
                self.assertTrue(weighted["rule"].notna().all())  # every weight has a rule behind it

    def test_targets_at_an_earlier_session_ignore_later_data(self):
        backtest = build_strategies(self.settings)[2].weights(self.panel, self.regimes)  # regime_switch
        held_days = backtest.index[(backtest.sum(axis=1) > 0) & backtest.index.isin(self.regimes.index)]
        for cut in (held_days[len(held_days) // 3], held_days[-40]):
            with self.subTest(str(cut.date())):
                panel = {name: frame.loc[:cut] for name, frame in self.panel.items()}
                targets = compute_targets(panel, self.regimes.loc[:cut], self.settings, self.closes)
                self.assertEqual(targets.session, cut)
                pd.testing.assert_series_equal(targets.weights, backtest.loc[cut].fillna(0.0), check_names=False)

    def test_load_targets_uses_the_latest_stored_session(self):
        targets = load_targets(self.engine, self.settings)
        self.assertEqual(targets.session, self.regimes.index[-1])
        self.assertEqual(targets.strategy, "regime_switch")
        self.assertEqual(list(targets.stocks.index), list(self.settings.market.stocks))
        self.assertTrue(targets.stocks["close"].notna().all())
        self.assertLessEqual(targets.invested, 1.0)


if __name__ == "__main__":
    unittest.main()
