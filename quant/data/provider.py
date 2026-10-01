from datetime import date
from typing import Protocol

import pandas as pd


class MarketDataProvider(Protocol):
    def get_daily_data(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Return unadjusted daily OHLCV bars for [start, end)."""

    def get_corporate_actions(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Return dividends and splits for [start, end)."""
