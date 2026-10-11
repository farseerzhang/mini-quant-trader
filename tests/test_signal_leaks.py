"""Point in time across the whole signal path (sections 4.5, 18.1 and 21).

A database that ends at session T must give exactly the signals (weights, shares and
reasons) that the full database gives when it is cut at T: prices, features, the regime,
the strategy, the risk layer, the valuation and the reasons never see a later session.
"""

import dataclasses
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quant.features.pipeline import calculate_and_store_features
from quant.persistence import queries, tables
from quant.persistence.database import create_db_engine
from quant.regime.pipeline import RULE_METHOD, calculate_and_store_regimes
from quant.signals.engine import build_signals, load_signals
from quant.signals.positions import read_positions, value_holdings
from quant.signals.reasons import with_reasons
from quant.signals.targets import TARGET_COLUMNS, compute_targets
from quant.strategies import build_strategies
from tests.backtest_fixture import DATES, build_market_db, small_settings


class WholePathPointInTimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        base = small_settings()
        cls.positions = root / "positions.csv"
        cls.positions.write_text("symbol,quantity,cost_basis\nAAPL,40,95\nXOM,30,\nCASH,4000,\n", encoding="utf-8")
        cls.settings = dataclasses.replace(base, positions=dataclasses.replace(base.positions, path=str(cls.positions)))
        build_market_db(root / "full.db", cls.settings)
        cls.full = create_db_engine(root / "full.db", 5000)
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls.full.dispose()
        cls.temp.cleanup()

    def cut_database(self, cut: pd.Timestamp):
        """The same market, as it would have been stored on the evening of ``cut``."""
        path = self.root / f"cut-{cut.date()}.db"
        build_market_db(path, self.settings)
        engine = create_db_engine(path, 5000)
        with engine.begin() as conn:
            conn.execute(tables.daily_prices.delete().where(tables.daily_prices.c.date > cut.date()))
            for table in (tables.features, tables.market_features, tables.market_regimes):
                conn.execute(table.delete())
        calculate_and_store_features(engine, self.settings.features)
        calculate_and_store_regimes(engine, self.settings.regime, self.settings.features)
        return engine

    def full_history_signals_at(self, cut: pd.Timestamp):
        stocks = list(self.settings.market.stocks)
        panel = queries.load_feature_panel(self.full, self.settings.features.feature_version, stocks, TARGET_COLUMNS)
        regimes = queries.load_regimes(self.full, RULE_METHOD, self.settings.regime.rule_version)
        closes = queries.load_raw_closes(self.full, stocks)
        targets = compute_targets({k: v.loc[:cut] for k, v in panel.items()}, regimes.loc[:cut], self.settings, closes)
        holdings = read_positions(self.positions, stocks, self.settings.backtest.initial_cash)
        valuation = value_holdings(holdings, closes, cut)
        return with_reasons(build_signals(targets, holdings, valuation, self.settings.signals.hold_band), self.settings)

    def test_signals_on_a_database_ending_at_t_equal_the_full_history_cut_at_t(self):
        for cut in (DATES[400], DATES[520]):
            with self.subTest(str(cut.date())):
                engine = self.cut_database(cut)
                try:
                    live = load_signals(engine, self.settings)
                finally:
                    engine.dispose()
                replay = self.full_history_signals_at(cut)
                self.assertEqual(live.session, cut)
                pd.testing.assert_frame_equal(live.rows, replay.rows)  # weights, shares and reasons
                pd.testing.assert_series_equal(live.targets.weights, replay.targets.weights)
                self.assertEqual(live.portfolio_value, replay.portfolio_value)
                self.assertGreater(len(live.rows), 0)

    def test_live_targets_equal_the_backtest_weights_on_every_cut(self):
        stocks = list(self.settings.market.stocks)
        panel = queries.load_feature_panel(self.full, self.settings.features.feature_version, stocks, TARGET_COLUMNS)
        regimes = queries.load_regimes(self.full, RULE_METHOD, self.settings.regime.rule_version)
        closes = queries.load_raw_closes(self.full, stocks)
        backtest = next(s for s in build_strategies(self.settings) if s.name == "regime_switch").weights(panel, regimes)
        for cut in regimes.index[::37]:
            targets = compute_targets({k: v.loc[:cut] for k, v in panel.items()}, regimes.loc[:cut], self.settings, closes)
            pd.testing.assert_series_equal(targets.weights, backtest.loc[cut].fillna(0.0), check_names=False)


if __name__ == "__main__":
    unittest.main()
