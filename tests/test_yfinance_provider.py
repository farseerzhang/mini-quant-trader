import unittest
from datetime import date
from unittest.mock import patch

import pandas as pd

from quant.data.yfinance_provider import MarketDataError, YFinanceProvider
from tests.settings_helper import SETTINGS

NO_SPLITS = pd.Series(dtype=float)


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

    def _provider(self) -> YFinanceProvider:
        return YFinanceProvider(SETTINGS.market.request_timeout_seconds)

    @patch("quant.data.yfinance_provider.yf.Ticker")
    def test_returns_unadjusted_daily_prices(self, ticker: object) -> None:
        ticker.return_value.history.return_value = self.history
        ticker.return_value.splits = NO_SPLITS

        result = self._provider().get_daily_data(" aapl ", date(2026, 9, 30), date(2026, 10, 1))

        self.assertEqual(list(result.columns), ["Open", "High", "Low", "Close", "Volume"])
        self.assertEqual(result.iloc[0]["Close"], 11.0)
        ticker.assert_called_once_with("AAPL")

    @patch("quant.data.yfinance_provider.yf.Ticker")
    def test_returns_corporate_actions_separately(self, ticker: object) -> None:
        ticker.return_value.history.return_value = self.history
        ticker.return_value.splits = NO_SPLITS

        result = self._provider().get_corporate_actions("AAPL", date(2026, 9, 30), date(2026, 10, 1))

        self.assertEqual(result.to_dict("records"), [
            {"date": date(2026, 9, 30), "action_type": "DIVIDEND", "value": 0.5}
        ])

    @patch("quant.data.yfinance_provider.yf.Ticker")
    def test_undoes_yahoo_split_rescaling_to_return_values_as_traded(self, ticker: object) -> None:
        # Yahoo's view of AAPL around the 2020-08-31 4:1 split, fetched after a
        # hypothetical later 2:1 split: every value is rescaled for the splits after it.
        history = pd.DataFrame(
            {
                "Open": [63.0, 63.79],
                "High": [63.2, 65.5],
                "Low": [62.0, 63.2],
                "Close": [62.4, 64.52],
                "Adj Close": [61.0, 63.0],
                "Volume": [375_260_000, 451_405_400],
                "Dividends": [0.1025, 0.0],
                "Stock Splits": [0.0, 4.0],
            },
            index=pd.DatetimeIndex(["2020-08-28", "2020-08-31"], tz="America/New_York"),
        )
        ticker.return_value.history.return_value = history
        ticker.return_value.splits = pd.Series(
            [4.0, 2.0],
            index=pd.DatetimeIndex(["2020-08-31 09:30", "2030-01-02 09:30"], tz="America/New_York"),
        )
        provider = self._provider()

        prices = provider.get_daily_data("AAPL", date(2020, 8, 28), date(2020, 9, 1))
        actions = provider.get_corporate_actions("AAPL", date(2020, 8, 28), date(2020, 9, 1))

        self.assertAlmostEqual(prices.iloc[0]["Close"], 499.2)   # x4 x2
        self.assertEqual(prices.iloc[0]["Volume"], 46_907_500)   # /4 /2
        self.assertAlmostEqual(prices.iloc[1]["Close"], 129.04)  # x2: only the later split
        records = {row["action_type"]: row["value"] for row in actions.to_dict("records")}
        self.assertAlmostEqual(records["DIVIDEND"], 0.82)        # as paid
        self.assertEqual(records["SPLIT"], 4.0)                  # ratios are not rescaled
        ticker.return_value.history.assert_called_once()         # one request for both datasets

    @patch("quant.data.yfinance_provider.yf.Ticker")
    def test_empty_response_is_reported(self, ticker: object) -> None:
        ticker.return_value.history.return_value = self.history.iloc[0:0]

        with self.assertRaises(MarketDataError):
            self._provider().get_daily_data("AAPL", date(2026, 9, 30), date(2026, 10, 1))


if __name__ == "__main__":
    unittest.main()
