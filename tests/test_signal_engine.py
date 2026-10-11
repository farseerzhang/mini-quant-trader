import dataclasses
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from quant.persistence.database import create_db_engine
from quant.signals.engine import BUY, EXIT, HOLD, REDUCE, build_signals, classify, load_signals
from quant.signals.positions import parse_positions, value_holdings
from quant.signals.targets import Targets
from tests.backtest_fixture import build_market_db, small_settings

SESSION = pd.Timestamp("2026-10-09")
POOL = ("AAPL", "MSFT", "XOM", "CVX", "KO")


class ClassifyTests(unittest.TestCase):
    def test_mapping_table(self):
        h = 0.02
        cases = {
            "new position": (0.05, 0.0, BUY),
            "exit": (0.0, 0.05, EXIT),
            "increase by exactly the band": (0.10, 0.08, BUY),
            "reduce by exactly the band": (0.06, 0.08, REDUCE),
            "just inside the band up": (0.0999, 0.08, HOLD),
            "just inside the band down": (0.0601, 0.08, HOLD),
            "unchanged": (0.08, 0.08, HOLD),
            "neither held nor wanted": (0.0, 0.0, None),
            "tiny new position still trades": (0.001, 0.0, BUY),
            "tiny exit still trades": (0.0, 0.001, EXIT),
        }
        for name, (target, current, expected) in cases.items():
            with self.subTest(name):
                self.assertEqual(classify(target, current, h), expected)

    def test_zero_band_trades_every_change(self):
        self.assertEqual(classify(0.081, 0.08, 0.0), BUY)
        self.assertEqual(classify(0.08, 0.08, 0.0), HOLD)


def _targets(weights: dict[str, float], closes: dict[str, float], has_data: dict[str, bool] | None = None) -> Targets:
    stocks = pd.DataFrame(index=pd.Index(POOL, name="symbol"))
    stocks["target_weight"] = pd.Series(weights).reindex(POOL).fillna(0.0)
    stocks["close"] = pd.Series(closes).reindex(POOL)
    stocks["has_data"] = pd.Series(has_data or {}).reindex(POOL).fillna(True).astype(bool)
    return Targets(SESSION, "regime_switch", "BULL", "rule_based-v1", 1.0, stocks)


def _signals(csv: str, weights: dict[str, float], has_data=None, hold_band=0.02):
    closes = {"AAPL": 100.0, "MSFT": 50.0, "XOM": 200.0, "CVX": 1000.0, "KO": 10.0}
    holdings = parse_positions(csv.strip().splitlines(), POOL)
    frame = pd.DataFrame({s: [c] for s, c in closes.items()}, index=[SESSION])
    valuation = value_holdings(holdings, frame, SESSION)
    return build_signals(_targets(weights, closes, has_data), holdings, valuation, hold_band)


class BuildSignalTests(unittest.TestCase):
    def test_signals_and_suggested_shares(self):
        # Portfolio: AAPL 10 x 100 = 1000, MSFT 40 x 50 = 2000, XOM 5 x 200 = 1000, cash 6000 -> 10000.
        signals = _signals(
            "symbol,quantity\nAAPL,10\nMSFT,40\nXOM,5\nCASH,6000",
            {"AAPL": 0.15, "MSFT": 0.19, "KO": 0.10},  # XOM dropped
        )
        rows = signals.rows
        self.assertEqual(signals.portfolio_value, 10000)
        self.assertEqual(list(rows.index), ["XOM", "AAPL", "KO", "MSFT"])  # EXIT, then BUYs by weight, then HOLD
        self.assertEqual(rows["signal"].to_dict(), {"XOM": EXIT, "KO": BUY, "AAPL": BUY, "MSFT": HOLD})
        self.assertEqual(rows.loc["AAPL", "suggested_shares"], 5)    # +5 points x 10000 / 100
        self.assertEqual(rows.loc["KO", "suggested_shares"], 100)    # new: 10% x 10000 / 10
        self.assertEqual(rows.loc["XOM", "suggested_shares"], -5)    # everything
        self.assertEqual(rows.loc["MSFT", "suggested_shares"], 0)    # -1 point: inside the band
        self.assertAlmostEqual(rows.loc["AAPL", "change"], 0.05)
        self.assertEqual(signals.counts, {EXIT: 1, REDUCE: 0, BUY: 2, HOLD: 1})
        self.assertAlmostEqual(signals.estimated_cash_after, 6000 + 1000 - 500 - 1000)

    def test_reduce_rounds_down_and_never_sells_more_than_held(self):
        signals = _signals("symbol,quantity\nAAPL,10.5\nCASH,8950", {"AAPL": 0.0525})  # 1050 of 10000 -> 525
        row = signals.rows.loc["AAPL"]
        self.assertEqual((row["signal"], row["suggested_shares"]), (REDUCE, -5))  # 5.25 shares -> 5

    def test_exit_sells_a_fractional_remainder(self):
        signals = _signals("symbol,quantity\nAAPL,2.5\nCASH,100", {})
        self.assertEqual(signals.rows.loc["AAPL", "suggested_shares"], -2.5)

    def test_change_below_one_share_is_flagged(self):
        signals = _signals("symbol,quantity\nCASH,10000", {"CVX": 0.05})  # 500 < one CVX share at 1000
        row = signals.rows.loc["CVX"]
        self.assertEqual((row["signal"], row["suggested_shares"], row["below_one_share"]), (BUY, 0, True))
        self.assertTrue(any("less than one share" in w for w in signals.warnings))

    def test_held_stock_without_data_is_kept_with_a_warning(self):
        signals = _signals("symbol,quantity\nXOM,5\nCASH,0", {}, has_data={"XOM": False})
        row = signals.rows.loc["XOM"]
        self.assertEqual((row["signal"], row["suggested_shares"], row["data_gap"]), (HOLD, 0, True))
        self.assertTrue(any("XOM: held but no data" in w for w in signals.warnings))

    def test_all_cash_builds_the_whole_target_and_never_overspends(self):
        weights = {"AAPL": 0.10, "MSFT": 0.10, "XOM": 0.10, "CVX": 0.10, "KO": 0.10}
        signals = _signals("symbol,quantity\nCASH,12345", weights)
        self.assertTrue((signals.rows["signal"] == BUY).all())
        self.assertGreaterEqual(signals.estimated_cash_after, 0.5 * 12345 - 1e-9)

    def test_unmanaged_holdings_get_no_signal(self):
        holdings = parse_positions("symbol,quantity\nTSLA,3\nAAPL,10\nCASH,0".splitlines(), POOL)
        valuation = value_holdings(holdings, pd.DataFrame({"AAPL": [100.0]}, index=[SESSION]), SESSION)
        signals = build_signals(_targets({"AAPL": 1.0}, {"AAPL": 100.0}), holdings, valuation, 0.02)
        self.assertNotIn("TSLA", signals.rows.index)
        self.assertTrue(any("TSLA is not in the stock pool" in w for w in signals.warnings))


class LoadSignalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        base = small_settings()
        cls.positions = cls.root / "positions.csv"
        cls.settings = dataclasses.replace(base, positions=dataclasses.replace(base.positions, path=str(cls.positions)))
        build_market_db(cls.root / "signals.db", cls.settings)
        cls.engine = create_db_engine(cls.root / "signals.db", 5000)

    @classmethod
    def tearDownClass(cls):
        cls.engine.dispose()
        cls.temp.cleanup()

    def test_without_a_positions_file_every_target_is_a_buy(self):
        self.positions.unlink(missing_ok=True)
        signals = load_signals(self.engine, self.settings)
        self.assertEqual(signals.portfolio_value, self.settings.backtest.initial_cash)
        self.assertTrue(any("assuming all cash" in w for w in signals.warnings))
        wanted = signals.targets.weights[signals.targets.weights > 0]
        self.assertEqual(sorted(signals.rows.index), sorted(wanted.index))
        self.assertTrue((signals.rows["signal"] == BUY).all())

    def test_holdings_are_valued_and_compared_with_the_targets(self):
        self.positions.write_text("symbol,quantity,cost_basis\nAAPL,100,90\nXOM,50,\nCASH,5000,\n", encoding="utf-8")
        signals = load_signals(self.engine, self.settings)
        targets = signals.targets.weights
        self.assertAlmostEqual(signals.valuation.weights.sum() + signals.valuation.cash_weight, 1.0)
        for symbol, row in signals.rows.iterrows():
            self.assertEqual(row["signal"], classify(targets[symbol], row["current_weight"], self.settings.signals.hold_band))
        for symbol in ("AAPL", "XOM"):  # held, so each has a row whatever the target
            self.assertIn(symbol, signals.rows.index)
        np.testing.assert_allclose(signals.rows["target_weight"], targets[signals.rows.index])


if __name__ == "__main__":
    unittest.main()
