import dataclasses
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quant.persistence.database import create_db_engine
from quant.signals.engine import build_signals, load_signals
from quant.signals.positions import parse_positions, value_holdings
from quant.signals.reasons import with_reasons
from quant.signals.targets import Targets
from tests.backtest_fixture import build_market_db, small_settings
from tests.settings_helper import SETTINGS

SESSION = pd.Timestamp("2026-10-09")
POOL = ("AAPL", "MSFT", "XOM", "KO")
SECTORS = {"AAPL": "information_technology", "MSFT": "information_technology", "XOM": "energy",
           "KO": "consumer_staples"}


def _stock(**overrides) -> dict:
    stock = {"sector": None, "target_weight": 0.0, "rule": None, "held_sessions": 0, "position_cap": False,
             "sector_cap": False, "exposure_cap": False, "price_vs_sma200": 0.10, "sma50_vs_sma200": 0.05,
             "macd_hist_over_price": 0.01, "rsi14": 55.0, "realized_volatility_20d": 0.25,
             "atr14_over_price": 0.02, "macd_hist_prev": 0.005, "has_data": True, "close": 100.0,
             "stop_price": 96.0, "high_volatility": False}
    stock.update(overrides)
    return stock


def _reasons(stocks: dict[str, dict], csv: str, regime="BULL", exposure=1.0, strategy="regime_switch") -> pd.Series:
    frame = pd.DataFrame.from_dict(
        {s: {**_stock(), **stocks.get(s, {}), "sector": SECTORS[s]} for s in POOL}, orient="index"
    ).rename_axis("symbol")
    frame["rule"] = frame["rule"].astype(object).where(frame["rule"].notna(), None)
    targets = Targets(SESSION, strategy, regime, "rule_based-v1", exposure, frame)
    holdings = parse_positions(csv.strip().splitlines(), POOL)
    closes = pd.DataFrame({s: [100.0] for s in POOL}, index=[SESSION])
    signals = build_signals(targets, holdings, value_holdings(holdings, closes, SESSION), SETTINGS.signals.hold_band)
    return with_reasons(signals, SETTINGS).rows["reason"]


class ReasonTests(unittest.TestCase):
    def test_trend_buy_lists_regime_rule_limits_flags_and_action(self):
        reason = _reasons(
            {"AAPL": dict(target_weight=0.10, rule="trend", held_sessions=30, position_cap=True, sector_cap=True,
                          realized_volatility_20d=0.72, high_volatility=True)},
            "symbol,quantity\nCASH,10000",
        )["AAPL"]
        expected_in_order = [
            "Bull regime (rule_based-v1): trend rule.",
            "Trend: price +10% vs SMA200, SMA50 +5% vs SMA200; in the strategy for 30 sessions.",
            "Equal weight 1/1 capped at 10% per stock.",
            "Sector cap 30% binds (information technology).",
            "High volatility: 20-day 72% (above 60%).",
            "ATR stop 96.00 (2 x ATR below 100.00).",
            "Target 10.0% vs current 0.0% -> BUY (new position; buy 10 shares).",
        ]
        positions = [reason.index(part) for part in expected_in_order]
        self.assertEqual(positions, sorted(positions), reason)

    def test_entry_today_is_named(self):
        reason = _reasons({"KO": dict(target_weight=0.05, rule="trend", held_sessions=1)}, "symbol,quantity\nCASH,1000")["KO"]
        self.assertIn("entered today (MACD histogram positive)", reason)

    def test_exit_says_which_trend_condition_broke(self):
        reasons = _reasons(
            {"AAPL": dict(price_vs_sma200=-0.04), "XOM": dict(sma50_vs_sma200=-0.02)},
            "symbol,quantity\nAAPL,10\nXOM,10\nCASH,0",
        )
        self.assertIn("Trend broken: price -4% vs SMA200.", reasons["AAPL"])
        self.assertIn("Trend broken: SMA50 -2% vs SMA200.", reasons["XOM"])
        self.assertTrue(reasons["AAPL"].endswith("-> EXIT (sell all 10 shares)."))
        self.assertNotIn("ATR stop", reasons["AAPL"])  # flags are for positions that stay

    def test_uptrend_without_entry_trigger(self):
        reason = _reasons({}, "symbol,quantity\nKO,10\nCASH,0")["KO"]
        self.assertIn("waiting for the MACD histogram to turn positive", reason)

    def test_sideways_explains_that_only_dip_buys_are_held(self):
        reason = _reasons({}, "symbol,quantity\nXOM,10\nCASH,0", regime="SIDEWAYS")["XOM"]
        self.assertIn("Sideways regime (rule_based-v1): dip-buy rule.", reason)
        self.assertIn("Sideways regime holds dip buys only. Not a dip buy: RSI 55 (entry < 40).", reason)

    def test_dip_buy_shows_rsi_and_holding_day(self):
        reason = _reasons(
            {"MSFT": dict(target_weight=0.10, rule="mean_reversion", held_sessions=2, rsi14=36.0)},
            "symbol,quantity\nCASH,1000", regime="SIDEWAYS",
        )["MSFT"]
        self.assertIn("Dip buy: RSI 36 (entry < 40, exit > 50); day 2 of 10.", reason)

    def test_bear_holds_cash(self):
        reason = _reasons({"AAPL": dict(rule="trend", held_sessions=40)}, "symbol,quantity\nAAPL,10\nCASH,0",
                          regime="BEAR", exposure=0.0)["AAPL"]
        self.assertIn("Bear regime (rule_based-v1): trend rule at 0% exposure.", reason)
        self.assertIn("Bear exposure 0%: held in cash.", reason)
        self.assertTrue(reason.endswith("-> EXIT (sell all 10 shares)."))

    def test_hold_inside_the_band_and_data_gap(self):
        reasons = _reasons(
            {"AAPL": dict(target_weight=0.095, rule="trend", held_sessions=9), "XOM": dict(has_data=False)},
            "symbol,quantity\nAAPL,10\nXOM,5\nCASH,8500",  # 10000 in all: AAPL is 10%
        )
        self.assertTrue(reasons["AAPL"].endswith("-> HOLD (change -0.5 points, inside the 2-point band)."))
        self.assertIn("No indicator data at T.", reasons["XOM"])
        self.assertTrue(reasons["XOM"].endswith("-> HOLD (no data at T: kept, check the data)."))

    def test_single_rule_strategy_says_it_ignores_the_regime(self):
        reason = _reasons({"KO": dict(target_weight=0.05, rule="trend", held_sessions=3)},
                          "symbol,quantity\nCASH,1000", regime="BEAR", strategy="trend")["KO"]
        self.assertTrue(reason.startswith("Bear regime (rule_based-v1); trend ignores the regime."))


class LoadedReasonTests(unittest.TestCase):
    def test_every_loaded_signal_has_a_reason_ending_in_its_signal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = small_settings()
            positions = root / "positions.csv"
            positions.write_text("symbol,quantity\nAAPL,100\nXOM,50\nCASH,5000\n", encoding="utf-8")
            settings = dataclasses.replace(base, positions=dataclasses.replace(base.positions, path=str(positions)))
            build_market_db(root / "reasons.db", settings)
            engine = create_db_engine(root / "reasons.db", 5000)
            try:
                rows = load_signals(engine, settings).rows
            finally:
                engine.dispose()
        self.assertFalse(rows.empty)
        for symbol, row in rows.iterrows():
            self.assertIn(f"-> {row['signal']}", row["reason"], symbol)


if __name__ == "__main__":
    unittest.main()
