import dataclasses
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import select

from quant.data.store import sync_symbol_table
from quant.features.pipeline import calculate_and_store_features
from quant.persistence import queries, tables
from quant.persistence.database import create_db_engine, init_db
from quant.regime.rule_based import BEAR, BULL, SIDEWAYS
from quant.strategies import PANEL_COLUMNS, build_strategies, regime_switch_weights
from quant.strategies.rules import mean_reversion_holdings, trend_holdings
from quant.strategies.sizing import target_weights
from tests.settings_helper import SETTINGS

RISK = SETTINGS.risk  # 10% per stock, 30% per sector
REVERSION = dataclasses.replace(
    SETTINGS.strategies.mean_reversion, rsi_entry=30, rsi_exit=50, max_holding_days=3
)


def _dates(n):
    return pd.bdate_range("2026-01-05", periods=n)


def _panel(n, symbols, **columns):
    """A feature panel: each keyword is a column, given as a list of rows (one value per symbol)."""
    index = _dates(n)
    defaults = {"price_vs_sma200": 0.05, "sma50_vs_sma200": 0.03, "macd_hist_over_price": 0.001, "rsi14": 50.0}
    panel = {}
    for name in PANEL_COLUMNS:
        values = columns.get(name)
        if values is None:
            values = [[defaults[name]] * len(symbols)] * n
        panel[name] = pd.DataFrame(np.asarray(values, dtype=float), index=index, columns=symbols)
    return panel


class TrendRuleTests(unittest.TestCase):
    def test_enters_only_when_all_three_conditions_hold(self):
        panel = _panel(
            1,
            ["A", "B", "C", "D", "E"],
            price_vs_sma200=[[0.1, -0.1, 0.1, 0.1, np.nan]],
            sma50_vs_sma200=[[0.1, 0.1, -0.1, 0.1, 0.1]],
            macd_hist_over_price=[[0.01, 0.01, 0.01, -0.01, 0.01]],
        )
        self.assertEqual(list(trend_holdings(panel).iloc[0]), [True, False, False, False, False])

    def test_macd_is_an_entry_trigger_and_only_the_trend_ends_a_holding(self):
        panel = _panel(
            6,
            ["A"],
            price_vs_sma200=[[0.1], [0.1], [0.1], [0.1], [-0.1], [0.1]],
            sma50_vs_sma200=[[0.1]] * 6,
            macd_hist_over_price=[[-0.01], [0.01], [-0.01], [-0.01], [-0.01], [-0.01]],
        )
        # No entry without MACD (day 0); entered on day 1; held while MACD is negative;
        # exited when price drops below SMA200 (day 4); not re-entered without MACD (day 5).
        self.assertEqual(list(trend_holdings(panel)["A"]), [False, True, True, True, False, False])


class MeanReversionRuleTests(unittest.TestCase):
    def _run(self, rsi, histogram=None, above=None, allow_entry=None):
        n = len(rsi)
        panel = _panel(
            n,
            ["A"],
            rsi14=[[value] for value in rsi],
            macd_hist_over_price=[[value] for value in (histogram or [0.001 * (i + 1) for i in range(n)])],
            price_vs_sma200=[[value] for value in (above or [0.05] * n)],
        )
        if allow_entry is not None:
            allow_entry = pd.Series(allow_entry, index=_dates(n))
        return list(mean_reversion_holdings(panel, REVERSION, allow_entry)["A"])

    def test_enters_on_a_dip_and_exits_when_rsi_recovers(self):
        self.assertEqual(self._run([45, 25, 35, 55, 45]), [False, True, True, False, False])

    def test_cannot_enter_on_the_first_session(self):
        # "MACD histogram rising" compares with the previous session, which day 0 does not have.
        self.assertEqual(self._run([25, 45]), [False, False])

    def test_exits_after_max_holding_days(self):
        # max_holding_days = 3: held on the entry session and the next two, then exited.
        self.assertEqual(self._run([45, 25, 40, 40, 40, 40]), [False, True, True, True, False, False])

    def test_does_not_re_enter_on_the_exit_session(self):
        self.assertEqual(self._run([45, 25, 25, 25, 25, 25]), [False, True, True, True, False, True])

    def test_needs_an_uptrend_and_a_rising_macd_histogram(self):
        self.assertEqual(self._run([45, 25], above=[-0.01, -0.01]), [False, False])
        self.assertEqual(self._run([45, 25], histogram=[0.002, 0.001]), [False, False])

    def test_allow_entry_restricts_entries_but_not_exits(self):
        self.assertEqual(self._run([25, 25, 55, 25], allow_entry=[False, True, False, False]),
                         [False, True, False, False])


class SizingTests(unittest.TestCase):
    def test_equal_weight_capped_per_stock(self):
        sector_names = list(SETTINGS.market.symbols)
        sectors = {symbol: sector_names[i % len(sector_names)] for i, symbol in enumerate("ABCDEFGHIJKL")}
        holdings = pd.DataFrame([[True] * 2 + [False] * 10, [True] * 12], index=_dates(2), columns=list(sectors))
        weights = target_weights(holdings, sectors, RISK)
        self.assertEqual(list(weights.iloc[0, :2]), [0.10, 0.10])           # 1/2 capped at 10%, rest cash
        np.testing.assert_allclose(weights.iloc[1], 1 / 12)                 # 12 held: 8.3% each, fully invested

    def test_sector_cap_scales_the_sector_down_and_keeps_the_excess_in_cash(self):
        sectors = {**{f"T{i}": "information_technology" for i in range(5)}, "XOM": "energy"}
        holdings = pd.DataFrame([[True] * 6], index=_dates(1), columns=list(sectors))
        weights = target_weights(holdings, sectors, RISK).iloc[0]
        np.testing.assert_allclose(weights[[f"T{i}" for i in range(5)]], 0.06)  # 5 x 10% = 50% -> 30%
        self.assertAlmostEqual(weights["XOM"], 0.10)
        self.assertAlmostEqual(weights.sum(), 0.40)

    def test_caps_are_never_exceeded(self):
        rng = np.random.default_rng(7)
        sectors = SETTINGS.market.sectors
        holdings = pd.DataFrame(rng.random((300, len(sectors))) < 0.4, index=_dates(300), columns=list(sectors))
        weights = target_weights(holdings, sectors, RISK)
        by_sector = weights.T.groupby(pd.Series(sectors)).sum().T
        self.assertLessEqual(weights.max().max(), RISK.max_position_size + 1e-12)
        self.assertLessEqual(by_sector.max().max(), RISK.max_sector_weight + 1e-12)
        self.assertLessEqual(weights.sum(axis=1).max(), 1 + 1e-12)
        self.assertTrue((weights.where(~holdings, 0.0) == 0).all().all())

    def test_a_stock_without_a_sector_is_an_error(self):
        holdings = pd.DataFrame([[True]], index=_dates(1), columns=["ZZZ"])
        with self.assertRaisesRegex(ValueError, "ZZZ"):
            target_weights(holdings, SETTINGS.market.sectors, RISK)


class RegimeSwitchTests(unittest.TestCase):
    def setUp(self):
        self.symbols = ["AAPL", "XOM"]
        # AAPL is in a clean uptrend (trend rule holds); XOM dips on day 1 (mean-reversion entry).
        self.panel = _panel(
            4,
            self.symbols,
            rsi14=[[55, 40], [55, 25], [55, 40], [55, 40]],
            macd_hist_over_price=[[0.001, 0.002], [0.001, 0.003], [0.001, 0.004], [0.001, 0.005]],
            sma50_vs_sma200=[[0.03, -0.01]] * 4,
        )

    def weights(self, regimes, bear_exposure=0.0):
        return regime_switch_weights(self.panel, pd.Series(regimes, index=_dates(4)), SETTINGS, bear_exposure)

    def test_bull_uses_trend_and_sideways_uses_mean_reversion(self):
        bull = self.weights([BULL] * 4)
        sideways = self.weights([SIDEWAYS] * 4)
        self.assertEqual(list(bull["AAPL"]), [0.10] * 4)
        self.assertEqual(list(bull["XOM"]), [0.0] * 4)
        self.assertEqual(list(sideways["AAPL"]), [0.0] * 4)
        self.assertEqual(list(sideways["XOM"]), [0.0, 0.10, 0.10, 0.10])

    def test_bear_exposure_scales_trend_positions(self):
        self.assertEqual(list(self.weights([BEAR] * 4)["AAPL"]), [0.0] * 4)
        self.assertEqual(list(self.weights([BEAR] * 4, bear_exposure=0.5)["AAPL"]), [0.05] * 4)

    def test_no_regime_means_no_positions(self):
        weights = self.weights([np.nan, np.nan, BULL, BULL])
        self.assertEqual(list(weights["AAPL"]), [0.0, 0.0, 0.10, 0.10])

    def test_dip_trades_only_start_on_sideways_sessions(self):
        # XOM's dip signal is on day 1 (BULL); by the SIDEWAYS days its RSI no longer qualifies.
        self.assertEqual(list(self.weights([BULL, BULL, SIDEWAYS, SIDEWAYS])["XOM"]), [0.0] * 4)


class StrategyRegistryTests(unittest.TestCase):
    def test_baselines_and_pre_registered_variant(self):
        names = [strategy.name for strategy in build_strategies(SETTINGS)]
        self.assertEqual(names, ["trend", "mean_reversion", "regime_switch", "regime_switch_bear50"])

    def test_truncation_at_t_does_not_change_any_strategy_weight_up_to_t(self):
        rng = np.random.default_rng(11)
        n, symbols = 120, list(SETTINGS.market.stocks[:12])
        phase = np.arange(n)[:, None] / 9 + np.arange(len(symbols))[None, :]
        panel = _panel(
            n,
            symbols,
            price_vs_sma200=np.sin(phase) * 0.1,
            sma50_vs_sma200=np.cos(phase / 2) * 0.1,
            macd_hist_over_price=np.sin(phase * 1.3) * 0.01,
            rsi14=50 + 30 * np.sin(phase * 0.7) + rng.normal(0, 2, (n, len(symbols))),
        )
        regimes = pd.Series(np.array([BULL, SIDEWAYS, BEAR])[(np.arange(n) // 15) % 3], index=_dates(n))
        cut = _dates(n)[77]
        for strategy in build_strategies(SETTINGS):
            with self.subTest(strategy.name):
                full = strategy.weights(panel, regimes)
                truncated = strategy.weights({k: v.loc[:cut] for k, v in panel.items()}, regimes.loc[:cut])
                pd.testing.assert_frame_equal(full.loc[:cut], truncated)
                self.assertGreater(full.to_numpy().sum(), 0)


class SymbolTableTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.engine = create_db_engine(Path(self.temp_dir.name) / "s.db", SETTINGS.database.busy_timeout_ms)
        init_db(self.engine)

    def tearDown(self):
        self.engine.dispose()
        self.temp_dir.cleanup()

    def _rows(self):
        with self.engine.connect() as conn:
            rows = conn.execute(select(tables.symbols)).mappings().all()
        return {row["symbol"]: (row["asset_type"], row["sector"], row["active"]) for row in rows}

    def test_sync_writes_sectors_and_asset_types_and_ingestion_keeps_them(self):
        sync_symbol_table(self.engine, {"XOM": "energy", "AAPL": "information_technology"}, ["SPY", "^VIX"])
        queries.upsert_symbol(self.engine, "XOM")  # what ingestion does for every symbol
        self.assertEqual(
            self._rows(),
            {
                "XOM": ("STOCK", "energy", 1),
                "AAPL": ("STOCK", "information_technology", 1),
                "SPY": ("BENCHMARK", None, 1),
                "^VIX": ("BENCHMARK", None, 1),
            },
        )

    def test_symbols_removed_from_config_become_inactive(self):
        sync_symbol_table(self.engine, {"XOM": "energy", "AAPL": "information_technology"}, ["SPY"])
        sync_symbol_table(self.engine, {"AAPL": "information_technology"}, ["SPY"])
        self.assertEqual(self._rows()["XOM"], ("STOCK", "energy", 0))
        self.assertEqual(self._rows()["AAPL"][2], 1)


class FeaturePanelTests(unittest.TestCase):
    def test_panel_is_dates_by_symbols_in_the_requested_order(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = create_db_engine(Path(directory) / "p.db", SETTINGS.database.busy_timeout_ms)
            init_db(engine)
            dates = _dates(30)
            rows = []
            for symbol, base in (("SPY", 100.0), ("XOM", 50.0), ("AAPL", 80.0)):
                for i, stamp in enumerate(dates):
                    close = base + i
                    rows.append({"symbol": symbol, "date": stamp.date(), "open": close, "high": close * 1.01,
                                 "low": close * 0.99, "close": close, "volume": 1000, "provider": "test"})
            queries.upsert_daily_prices(engine, rows)
            calculate_and_store_features(engine, SETTINGS.features)
            panel = queries.load_feature_panel(engine, SETTINGS.features.feature_version, ["XOM", "AAPL"], PANEL_COLUMNS)
            engine.dispose()
        self.assertEqual(set(panel), set(PANEL_COLUMNS))
        self.assertEqual(list(panel["rsi14"].columns), ["XOM", "AAPL"])
        self.assertEqual(len(panel["rsi14"]), 30)
        self.assertEqual(panel["rsi14"].iloc[-1]["AAPL"], 100.0)  # only gains: RSI 100


if __name__ == "__main__":
    unittest.main()
