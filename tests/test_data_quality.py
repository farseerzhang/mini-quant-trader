import unittest
from datetime import date

import pandas as pd

from quant.data.data_quality import validate_prices


def _frame(rows):
    index = pd.DatetimeIndex([r[0] for r in rows], name="Date")
    data = {
        "Open": [r[1] for r in rows],
        "High": [r[2] for r in rows],
        "Low": [r[3] for r in rows],
        "Close": [r[4] for r in rows],
        "Volume": [r[5] for r in rows],
    }
    return pd.DataFrame(data, index=index)


class ValidatePricesTests(unittest.TestCase):
    def test_accepts_valid_bar(self):
        prices = _frame([("2026-09-30", 10.0, 12.0, 9.0, 11.0, 100)])
        good, bad = validate_prices("AAPL", prices)
        self.assertEqual(bad, [])
        self.assertEqual(good[0]["date"], date(2026, 9, 30))
        self.assertEqual(good[0]["close"], 11.0)
        self.assertEqual(good[0]["volume"], 100)

    def test_rejects_invalid_ohlc(self):
        prices = _frame([("2026-09-30", 10.0, 9.0, 9.5, 11.0, 100)])  # high < close
        good, bad = validate_prices("AAPL", prices)
        self.assertEqual(good, [])
        self.assertIn("high is not the maximum", bad[0]["reason"])

    def test_rejects_non_positive_price(self):
        prices = _frame([("2026-09-30", 0.0, 12.0, 9.0, 11.0, 100)])
        good, bad = validate_prices("AAPL", prices)
        self.assertEqual(good, [])
        self.assertEqual(bad[0]["reason"], "non-positive price")

    def test_rejects_negative_volume(self):
        prices = _frame([("2026-09-30", 10.0, 12.0, 9.0, 11.0, -5)])
        good, bad = validate_prices("AAPL", prices)
        self.assertEqual(good, [])
        self.assertEqual(bad[0]["reason"], "negative volume")

    def test_allows_null_volume(self):
        prices = _frame([("2026-09-30", 10.0, 12.0, 9.0, 11.0, None)])
        good, bad = validate_prices("^VIX", prices)
        self.assertEqual(bad, [])
        self.assertIsNone(good[0]["volume"])


if __name__ == "__main__":
    unittest.main()
