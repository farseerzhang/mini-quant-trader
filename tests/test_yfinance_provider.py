import unittest
from datetime import date
from unittest.mock import patch

import pandas as pd

from quant.data.yfinance_provider import MarketDataError, YFinanceProvider

# python -m unittest tests.test_yfinance_provider -v
class YFinanceProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.history = pd.DataFrame(
            {
                "Open": [10.0],
                "High": [12.0],
                "Low": [9.0],
                "Close": [11.0],
                "Adj Close": [10.5],
                "Volume": [100],
                "Dividends": [0.5],
                "Stock Splits": [0.0],
            },
            index=pd.DatetimeIndex(["2026-09-30"], tz="America/New_York", name="Date"),
        )

    @patch("quant.data.yfinance_provider.yf.Ticker")
    def test_returns_unadjusted_daily_prices(self, ticker: object) -> None:
        ticker.return_value.history.return_value = self.history

        result = YFinanceProvider().get_daily_data(
            " aapl ", date(2026, 9, 30), date(2026, 10, 1)
        )

        self.assertEqual(list(result.columns), ["Open", "High", "Low", "Close", "Volume"])
        self.assertEqual(result.iloc[0]["Close"], 11.0)
        ticker.assert_called_once_with("AAPL")

    @patch("quant.data.yfinance_provider.yf.Ticker")
    def test_returns_corporate_actions_separately(self, ticker: object) -> None:
        ticker.return_value.history.return_value = self.history

        result = YFinanceProvider().get_corporate_actions(
            "AAPL", date(2026, 9, 30), date(2026, 10, 1)
        )

        self.assertEqual(result.to_dict("records"), [
            {"date": date(2026, 9, 30), "action_type": "DIVIDEND", "value": 0.5}
        ])

    @patch("quant.data.yfinance_provider.yf.Ticker")
    def test_empty_response_is_reported(self, ticker: object) -> None:
        ticker.return_value.history.return_value = self.history.iloc[0:0]

        with self.assertRaises(MarketDataError):
            YFinanceProvider().get_daily_data(
                "AAPL", date(2026, 9, 30), date(2026, 10, 1)
            )


if __name__ == "__main__":
    unittest.main()
