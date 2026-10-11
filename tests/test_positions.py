import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quant.persistence import queries
from quant.persistence.database import create_db_engine, init_db
from quant.signals.positions import (
    CASH,
    MANUAL,
    PositionsError,
    parse_positions,
    read_positions,
    snapshot_rows,
    value_holdings,
)
from tests.backtest_fixture import build_market_db, small_settings

POOL = ("AAPL", "MSFT", "XOM")


def parse(text: str):
    return parse_positions(text.strip().splitlines(), POOL)


class ParseTests(unittest.TestCase):
    def test_reads_holdings_cash_and_cost_basis(self):
        holdings = parse("""
            # my account
            symbol,quantity,cost_basis
            aapl, 10, 175.20
            MSFT,2.5,
            XOM,0,
            CASH,25000,
        """.replace("            ", ""))
        self.assertEqual(dict(holdings.quantities), {"AAPL": 10.0, "MSFT": 2.5})  # XOM 0 = not held
        self.assertEqual(dict(holdings.cost_basis), {"AAPL": 175.20, "MSFT": None})
        self.assertEqual((holdings.cash, holdings.source, holdings.warnings), (25000.0, MANUAL, ()))

    def test_cost_basis_column_is_optional(self):
        self.assertEqual(parse("symbol,quantity\nAAPL,1\nCASH,0").cash, 0.0)

    def test_stock_outside_the_pool_is_not_managed(self):
        holdings = parse("symbol,quantity,cost_basis\nTSLA,3,200\nAAPL,1,\nCASH,100,")
        self.assertEqual(dict(holdings.unmanaged), {"TSLA": 3.0})
        self.assertNotIn("TSLA", holdings.quantities)
        self.assertIn("TSLA is not in the stock pool", holdings.warnings[0])

    def test_rejects_invalid_files(self):
        bad = {
            "empty": ("", "empty"),
            "missing column": ("symbol,shares\nCASH,1", "missing quantity"),
            "unknown column": ("symbol,quantity,price\nCASH,1,", "unknown column"),
            "duplicate": ("symbol,quantity\nAAPL,1\naapl,2\nCASH,0", "listed twice"),
            "negative": ("symbol,quantity\nAAPL,-1\nCASH,0", "short positions"),
            "not a number": ("symbol,quantity\nAAPL,ten\nCASH,0", "not a number"),
            "no quantity": ("symbol,quantity\nAAPL,\nCASH,0", "no quantity"),
            "no cash row": ("symbol,quantity\nAAPL,1", "no CASH row"),
            "negative cash": ("symbol,quantity\nCASH,-5", "negative"),
            "cash with cost basis": ("symbol,quantity,cost_basis\nCASH,5,1", "no cost_basis"),
            "negative cost basis": ("symbol,quantity,cost_basis\nAAPL,1,-3\nCASH,0,", "must not be negative"),
            "too many values": ("symbol,quantity\nAAPL,1,2\nCASH,0", "values for 2 columns"),
            "infinite": ("symbol,quantity\nAAPL,inf\nCASH,0", "finite"),
        }
        for name, (text, message) in bad.items():
            with self.subTest(name), self.assertRaisesRegex(PositionsError, message):
                parse_positions(text.splitlines(), POOL)

    def test_errors_name_the_file_and_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "positions.csv"
            path.write_text("symbol,quantity\nAAPL,1\nAAPL,2\nCASH,0\n", encoding="utf-8")
            with self.assertRaisesRegex(PositionsError, r"positions\.csv: line 3: AAPL is listed twice"):
                read_positions(path, POOL, 100_000)

    def test_missing_file_is_an_all_cash_notional_portfolio(self):
        holdings = read_positions(Path("does-not-exist.csv"), POOL, 100_000)
        self.assertEqual((dict(holdings.quantities), holdings.cash, holdings.path), ({}, 100_000.0, None))
        self.assertIn("assuming all cash", holdings.warnings[0])


class ValuationTests(unittest.TestCase):
    def setUp(self):
        index = pd.DatetimeIndex(["2026-10-07", "2026-10-08", "2026-10-09"])
        self.closes = pd.DataFrame({"AAPL": [100.0, 110.0, 120.0], "MSFT": [50.0, 52.0, None]}, index=index)
        self.holdings = parse("symbol,quantity\nAAPL,10\nMSFT,20\nCASH,800")

    def test_weights_are_shares_times_close_over_holdings_plus_cash(self):
        valuation = value_holdings(self.holdings, self.closes, pd.Timestamp("2026-10-08"))
        self.assertEqual(valuation.total, 10 * 110 + 20 * 52 + 800)
        self.assertAlmostEqual(valuation.weights["AAPL"], 1100 / 2940)
        self.assertAlmostEqual(valuation.weights.sum() + valuation.cash_weight, 1.0)
        self.assertEqual(valuation.warnings, ())

    def test_missing_close_on_t_uses_the_last_close_with_a_warning(self):
        valuation = value_holdings(self.holdings, self.closes, pd.Timestamp("2026-10-09"))
        self.assertEqual(valuation.closes["MSFT"], 52.0)
        self.assertIn("valued at the 2026-10-08 close", valuation.warnings[0])

    def test_later_prices_are_never_used(self):
        valuation = value_holdings(self.holdings, self.closes, pd.Timestamp("2026-10-07"))
        self.assertEqual(valuation.closes["AAPL"], 100.0)

    def test_holding_without_prices_is_an_error(self):
        holdings = parse("symbol,quantity\nXOM,1\nCASH,0")
        with self.assertRaisesRegex(PositionsError, "XOM: no stored prices"):
            value_holdings(holdings, self.closes, pd.Timestamp("2026-10-09"))

    def test_all_cash_portfolio(self):
        valuation = value_holdings(parse("symbol,quantity\nCASH,500"), self.closes, pd.Timestamp("2026-10-09"))
        self.assertEqual((valuation.total, valuation.cash_weight, len(valuation.weights)), (500.0, 1.0, 0))


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.settings = small_settings()
        self.db = Path(self.temp.name) / "positions.db"
        build_market_db(self.db, self.settings)
        self.engine = create_db_engine(self.db, 5000)
        init_db(self.engine)

    def tearDown(self):
        self.engine.dispose()
        self.temp.cleanup()

    def test_snapshot_is_replaced_on_rerun(self):
        as_of = pd.Timestamp("2026-10-09").date()
        first = parse("symbol,quantity,cost_basis\nAAPL,10,150\nTSLA,1,\nCASH,100,")
        queries.replace_positions(self.engine, as_of, MANUAL, snapshot_rows(first, as_of))
        second = parse("symbol,quantity\nMSFT,5\nCASH,50")
        queries.replace_positions(self.engine, as_of, MANUAL, snapshot_rows(second, as_of))
        stored = queries.load_positions(self.engine, as_of, MANUAL).set_index("symbol")
        self.assertEqual(sorted(stored.index), [CASH, "MSFT"])  # AAPL and TSLA are gone
        self.assertEqual(stored.loc[CASH, "quantity"], 50.0)

    def test_unmanaged_holdings_are_stored_but_flagged(self):
        as_of = pd.Timestamp("2026-10-09").date()
        holdings = parse("symbol,quantity,cost_basis\nAAPL,10,150\nTSLA,1,\nCASH,100,")
        queries.replace_positions(self.engine, as_of, MANUAL, snapshot_rows(holdings, as_of))
        stored = queries.load_positions(self.engine, as_of, MANUAL).set_index("symbol")
        self.assertEqual(stored["managed"].to_dict(), {"AAPL": 1, CASH: 1, "TSLA": 0})
        self.assertEqual(stored.loc["AAPL", "cost_basis"], 150.0)

    def test_valuation_uses_stored_as_traded_closes(self):
        closes = queries.load_raw_closes(self.engine, list(POOL))
        session = closes.index[-1]
        holdings = parse("symbol,quantity\nAAPL,3\nXOM,2\nCASH,10")
        valuation = value_holdings(holdings, closes, session)
        self.assertAlmostEqual(valuation.values["AAPL"], 3 * closes.loc[session, "AAPL"])
        self.assertAlmostEqual(valuation.weights.sum() + valuation.cash_weight, 1.0)


if __name__ == "__main__":
    unittest.main()
