import dataclasses
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from quant.config.settings import ForwardTestSettings
from quant.forward.freeze import freeze_session
from quant.forward.real import (
    BUY, DIVIDEND, SELL, Start, Trade, TradesError, execution_costs, followed_share, holdings_at, parse_trades,
    real_portfolio, replay,
)
from quant.persistence import queries
from quant.persistence.database import create_db_engine
from quant.signals.engine import load_signals
from tests.backtest_fixture import DATES, build_market_db, small_settings
from tests.test_forward_freeze import EVENING

SESSIONS = pd.bdate_range("2026-10-09", periods=5)  # Fri .. Thu
START = Start(SESSIONS[0], {"AAPL": 10.0}, {"AAPL": 100.0}, 1_000.0)


def parse(text):
    return parse_trades(text.strip().splitlines())


class ParseTests(unittest.TestCase):
    def test_reads_buys_sells_and_dividends(self):
        trades = parse("""
            # my trades
            date,symbol,side,quantity,price,fees,amount
            2026-10-12,aapl,SELL,4,110,1.5,
            2026-10-13,MSFT,buy,2,50,,
            2026-10-14,AAPL,DIVIDEND,,,,3.25
        """.replace("            ", ""))
        self.assertEqual([(t.symbol, t.side) for t in trades], [("AAPL", SELL), ("MSFT", BUY), ("AAPL", DIVIDEND)])
        self.assertEqual([t.cash_change for t in trades], [4 * 110 - 1.5, -100.0, 3.25])

    def test_rejects_invalid_lines(self):
        bad = {
            "unknown column": ("date,symbol,side,qty\n", "unknown qty"),
            "bad date": ("date,symbol,side,quantity,price\nyesterday,AAPL,BUY,1,1", "line 2: date"),
            "cash symbol": ("date,symbol,side,quantity,price\n2026-10-12,CASH,BUY,1,1", "symbol must be a stock"),
            "bad side": ("date,symbol,side,quantity,price\n2026-10-12,AAPL,SHORT,1,1", "BUY, SELL or DIVIDEND"),
            "no price": ("date,symbol,side,quantity,price\n2026-10-12,AAPL,BUY,1,", "needs quantity and price"),
            "zero quantity": ("date,symbol,side,quantity,price\n2026-10-12,AAPL,BUY,0,5", "positive"),
            "dividend with shares": ("date,symbol,side,quantity,price,fees,amount\n2026-10-12,AAPL,DIVIDEND,1,,,2",
                                     "only amount"),
            "amount on a buy": ("date,symbol,side,quantity,price,fees,amount\n2026-10-12,AAPL,BUY,1,5,,2",
                                "only for DIVIDEND"),
        }
        for name, (text, message) in bad.items():
            with self.subTest(name), self.assertRaisesRegex(TradesError, message):
                parse_trades(text.splitlines())


class ReplayTests(unittest.TestCase):
    def trades(self):
        return [Trade(2, SESSIONS[1], "AAPL", SELL, 4, 110, 1.0), Trade(3, SESSIONS[2], "MSFT", BUY, 2, 50),
                Trade(4, SESSIONS[3], "AAPL", DIVIDEND, amount=3.0)]

    def test_holdings_and_cash_at_each_close(self):
        ledger = replay(START, self.trades(), SESSIONS)
        self.assertEqual(list(ledger.quantities["AAPL"]), [10, 6, 6, 6, 6])
        self.assertEqual(list(ledger.quantities["MSFT"]), [0, 0, 2, 2, 2])
        self.assertEqual(list(ledger.cash), [1000, 1439, 1339, 1342, 1342])

    def test_order_of_the_file_does_not_matter(self):
        forward = replay(START, self.trades(), SESSIONS)
        backward = replay(START, self.trades()[::-1], SESSIONS)
        pd.testing.assert_frame_equal(forward.quantities, backward.quantities)

    def test_invalid_histories(self):
        cases = {
            "oversell": ([Trade(2, SESSIONS[1], "AAPL", SELL, 11, 1)], "only 10 are held"),
            "on the start": ([Trade(2, SESSIONS[0], "AAPL", BUY, 1, 1)], "not after the start"),
            "in the future": ([Trade(2, SESSIONS[-1] + pd.Timedelta(days=1), "AAPL", BUY, 1, 1)], "after the latest"),
        }
        for name, (trades, message) in cases.items():
            with self.subTest(name), self.assertRaisesRegex(TradesError, message):
                replay(START, trades, SESSIONS)

    def test_negative_cash_is_a_warning(self):
        ledger = replay(START, [Trade(2, SESSIONS[1], "MSFT", BUY, 100, 50)], SESSIONS)
        self.assertIn("cash is negative", ledger.warnings[0])

    def test_followed_share_compares_with_the_previous_target(self):
        actual = pd.DataFrame({"A": [0.5, 0.5, 0.2], "B": [0.0, 0.3, 0.6]}, index=SESSIONS[:3])
        targets = pd.DataFrame({"A": [0.5, 0.2, 0.2], "B": [0.5, 0.6, 0.6]}, index=SESSIONS[:3])
        share = followed_share(actual, targets)
        self.assertEqual(list(share.index), list(SESSIONS[1:3]))
        np.testing.assert_allclose(share, [0.5 + 0.3, 0.2 + 0.6])

    def test_execution_cost_against_the_open(self):
        opens = pd.DataFrame({"AAPL": [100.0], "MSFT": [50.0]}, index=[SESSIONS[1]])
        costs = execution_costs([Trade(2, SESSIONS[1], "AAPL", BUY, 10, 100.5, 1.0),
                                 Trade(3, SESSIONS[1], "MSFT", SELL, 2, 49.9)], opens).set_index("symbol")
        self.assertAlmostEqual(costs.loc["AAPL", "cost_bps"], (0.005 + 1.0 / 1005) * 10_000)
        self.assertAlmostEqual(costs.loc["MSFT", "cost_bps"], (50 / 49.9 - 1) * 10_000)


class RealPortfolioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        base = small_settings(random_runs=0)
        self.positions, self.trades = root / "positions.csv", root / "trades.csv"
        self.settings = dataclasses.replace(
            base, forward_test=ForwardTestSettings({}),
            positions=dataclasses.replace(base.positions, path=str(self.positions), trades_path=str(self.trades)))
        build_market_db(root / "real.db", self.settings)
        self.engine = create_db_engine(root / "real.db", 5000)
        self.positions.write_text("symbol,quantity,cost_basis\nAAPL,10,90\nTSLA,2,200\nCASH,1000,\n", encoding="utf-8")

    def tearDown(self):
        self.engine.dispose()
        self.temp.cleanup()

    def test_the_start_is_stored_and_the_positions_file_is_then_ignored(self):
        freeze_session(self.engine, self.settings, now=EVENING, start=True)
        start = queries.forward_start_positions(self.engine).set_index("symbol")["quantity"].to_dict()
        self.assertEqual(start, {"AAPL": 10.0, "TSLA": 2.0, "CASH": 1000.0})
        self.positions.write_text("symbol,quantity\nCASH,1\n", encoding="utf-8")  # edited after the start
        holdings = holdings_at(self.engine, self.settings, DATES[-1])
        self.assertEqual((dict(holdings.quantities), holdings.cash, dict(holdings.unmanaged)),
                         ({"AAPL": 10.0}, 1000.0, {"TSLA": 2.0}))
        signals = load_signals(self.engine, self.settings)
        self.assertEqual(signals.holdings.cash, 1000.0)
        self.assertIn("TSLA is not in the stock pool", " | ".join(signals.warnings))

    def test_the_start_value_counts_every_holding_and_unpriced_ones_at_cost(self):
        freeze_session(self.engine, self.settings, now=EVENING, start=True)
        close = queries.load_raw_closes(self.engine, ["AAPL"])["AAPL"].iloc[-1]
        self.assertAlmostEqual(queries.forward_start(self.engine)["start_value"], 10 * close + 2 * 200 + 1000)

    def test_real_portfolio_from_trades_is_valued_every_session(self):
        freeze_session(self.engine, self.settings, now=EVENING, start=True)
        start = queries.forward_start(self.engine)
        # The fixture ends at the start; a trade log for later sessions is checked against stored data.
        self.trades.write_text(f"date,symbol,side,quantity,price\n{DATES[-1].date()},AAPL,SELL,1,100\n",
                               encoding="utf-8")
        with self.assertRaisesRegex(TradesError, "not after the start"):
            real_portfolio(self.engine, self.settings, pd.DatetimeIndex([DATES[-1]]))
        self.trades.write_text("date,symbol,side,quantity,price\n", encoding="utf-8")
        _, _, valued, warnings = real_portfolio(self.engine, self.settings, pd.DatetimeIndex([DATES[-1]]))
        self.assertAlmostEqual(valued["value"].iloc[0], start["start_value"])
        self.assertIn("TSLA: no stored close", " | ".join(warnings))


if __name__ == "__main__":
    unittest.main()
