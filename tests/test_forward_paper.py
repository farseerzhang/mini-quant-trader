import dataclasses
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quant.backtest.data import load_prices
from quant.backtest.engine import simulate
from quant.backtest.orders import buy_and_hold, order_targets
from quant.config.settings import ForwardTestSettings
from quant.forward.freeze import freeze_session
from quant.forward.paper import paper_targets, run_paper, update_paper_equity
from quant.persistence import queries
from quant.persistence.database import create_db_engine
from quant.regime.pipeline import RULE_METHOD
from quant.strategies import PANEL_COLUMNS, build_strategies
from tests.backtest_fixture import DATES, build_market_db, small_settings
from tests.test_forward_freeze import EVENING

WINDOW = DATES[-30:]


class PaperFixture(unittest.TestCase):
    """A stored market, regime_switch's weights and a frozen live window (shared with the report tests)."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        base = small_settings(random_runs=0)
        self.settings = dataclasses.replace(
            base, forward_test=ForwardTestSettings({}),
            positions=dataclasses.replace(base.positions, path=str(root / "positions.csv")))
        build_market_db(root / "paper.db", self.settings)
        self.engine = create_db_engine(root / "paper.db", 5000)
        stocks = list(self.settings.market.stocks)
        panel = queries.load_feature_panel(self.engine, "technical-v1", stocks, PANEL_COLUMNS)
        regimes = queries.load_regimes(self.engine, RULE_METHOD, "rule_based-v1")
        regime_switch = next(s for s in build_strategies(self.settings) if s.name == "regime_switch")
        self.weights = regime_switch.weights(panel, regimes).reindex(columns=stocks).fillna(0.0)
        self.prices = load_prices(self.engine, [*stocks, "SPY"])

    def tearDown(self):
        self.engine.dispose()
        self.temp.cleanup()

    def freeze(self, sessions):
        """Write the live targets of ``sessions`` as the evening freezes would (live = backtest)."""
        queries.record_forward_start(self.engine, sessions[0].date(), 50_000.0, "test",
                                     [{"symbol": "CASH", "quantity": 50_000.0, "cost_basis": None}])
        for session in sessions:
            row = self.weights.loc[session]
            held = row[row > 0]
            queries.freeze_forward(
                self.engine,
                {"session": session.date(), "portfolio": "live", "strategy": "regime_switch", "regime": "BULL",
                 "regime_source": "rule_based-v1", "model_version": "rule_based-v1",
                 "invested": float(held.sum()), "positions": int(len(held))},
                [{"symbol": s, "target_weight": float(w), "close": None, "score": None} for s, w in held.items()],
            )

    def expected(self, weights):
        bt = dataclasses.replace(self.settings.backtest, initial_cash=50_000.0)
        return simulate(order_targets(weights, bt.min_trade_weight), self.prices, bt)



class PaperTests(PaperFixture):
    def test_paper_equals_the_backtest_engine_on_the_same_targets(self):
        self.freeze(WINDOW)
        result = run_paper(self.engine, self.settings)
        expected = self.expected(self.weights.loc[WINDOW])
        pd.testing.assert_series_equal(result.simulations["live"].value, expected.value, check_names=False, check_freq=False, check_index_type=False)
        self.assertEqual(result.simulations["live"].value.iloc[0], 50_000.0)  # cash on the start session
        self.assertEqual(result.missed["live"], [])
        bt = dataclasses.replace(self.settings.backtest, initial_cash=50_000.0)
        spy = simulate(buy_and_hold(WINDOW, "SPY"), self.prices, bt)
        pd.testing.assert_series_equal(result.simulations["SPY"].value, spy.value, check_names=False, check_freq=False, check_index_type=False)

    def test_a_missed_session_keeps_the_previous_targets(self):
        missed = WINDOW[10]
        self.freeze([s for s in WINDOW if s != missed])
        result = run_paper(self.engine, self.settings)
        self.assertEqual(result.missed["live"], [missed])
        held_over = self.weights.loc[WINDOW].copy()
        held_over.loc[missed] = held_over.loc[WINDOW[9]]
        expected = self.expected(held_over)
        pd.testing.assert_series_equal(result.simulations["live"].value, expected.value, check_names=False, check_freq=False, check_index_type=False)
        self.assertEqual(len(result.sessions), len(WINDOW))  # still valued every session

    def test_paper_targets_fill_forward(self):
        sessions = pd.bdate_range("2026-01-05", periods=4)
        frozen = pd.DataFrame({"A": [0.1, 0.2]}, index=sessions[[0, 2]])
        self.assertEqual(list(paper_targets(frozen, sessions)["A"]), [0.1, 0.1, 0.2, 0.2])

    def test_equity_is_stored_and_replaced(self):
        self.assertIsNone(update_paper_equity(self.engine, self.settings))  # before the first freeze
        self.freeze(WINDOW[:5])
        update_paper_equity(self.engine, self.settings)
        update_paper_equity(self.engine, self.settings)  # a re-run replaces, never duplicates
        equity = queries.load_forward_equity(self.engine)
        self.assertEqual(sorted(equity["portfolio"].unique()), ["SPY", "live", "real"])
        real = equity[equity["portfolio"] == "real"]
        self.assertTrue((real["value"] == 50_000.0).all())  # all cash, no trades
        self.assertEqual(equity.groupby("portfolio")["date"].count().to_dict(),
                         {name: len(DATES[DATES >= WINDOW[0]]) for name in ("SPY", "live", "real")})

    def test_the_first_freeze_records_the_start_from_the_positions_file(self):
        Path(self.settings.positions.path).write_text("symbol,quantity\nAAPL,10\nCASH,2500\n", encoding="utf-8")
        freeze_session(self.engine, self.settings, now=EVENING)
        start = queries.forward_start(self.engine)
        closes = queries.load_raw_closes(self.engine, ["AAPL"])
        self.assertEqual(start["started_on"], DATES[-1].date())
        self.assertAlmostEqual(start["start_value"], 10 * closes["AAPL"].iloc[-1] + 2500)
        self.assertFalse(queries.record_forward_start(self.engine, DATES[-1].date(), 1.0, "x"))  # never moves


if __name__ == "__main__":
    unittest.main()
