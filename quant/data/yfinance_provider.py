"""Yahoo Finance market data, returned as originally traded (proposal sections 6.2 and 8).

Yahoo's ``auto_adjust=False`` history is not as traded: prices, volumes and
dividends are rescaled onto the share basis of every split known at fetch time.
AAPL's 2020-08-28 close, for example, comes back as 124.81 instead of 499.23
(the 4:1 split on 2020-08-31), and its 2020-08-07 dividend as 0.205 instead
of 0.82. This provider multiplies that rescaling back out using the symbol's
full split history, so stored rows never change when a later split happens.
"""

from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf


_PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
_OHLC_COLUMNS = ["Open", "High", "Low", "Close"]
_ACTION_COLUMNS = ["date", "action_type", "value"]


class MarketDataError(RuntimeError):
    """Raised when the market data provider cannot return usable data."""


class YFinanceProvider:
    def __init__(self, request_timeout_seconds: float) -> None:
        self._timeout = request_timeout_seconds
        # get_daily_data and get_corporate_actions are called back to back for the
        # same window; keep the last response so it is requested only once.
        self._last_history: tuple[tuple[str, date, date], pd.DataFrame, pd.Series] | None = None

    def get_daily_data(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        history, factor = self._get_history(symbol, start, end)
        prices = history.loc[:, _PRICE_COLUMNS].copy()
        prices[_OHLC_COLUMNS] = prices[_OHLC_COLUMNS].mul(factor, axis=0)
        prices["Volume"] = (prices["Volume"] / factor).round()
        return prices

    def get_corporate_actions(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        history, factor = self._get_history(symbol, start, end)
        actions: list[dict[str, object]] = []
        for action_type, column in (("DIVIDEND", "Dividends"), ("SPLIT", "Stock Splits")):
            for action_date, value in history.loc[history[column] > 0, column].items():
                # Dividends are rescaled like prices; split ratios are not.
                scale = factor[action_date] if action_type == "DIVIDEND" else 1.0
                actions.append(
                    {
                        "date": action_date.date(),
                        "action_type": action_type,
                        "value": float(value) * scale,
                    }
                )
        return pd.DataFrame(actions, columns=_ACTION_COLUMNS)

    def _get_history(self, symbol: str, start: date, end: date) -> tuple[pd.DataFrame, pd.Series]:
        normalized_symbol = symbol.strip().upper()
        if not normalized_symbol:
            raise ValueError("symbol must not be empty")
        if start >= end:
            raise ValueError("start must be earlier than end")

        key = (normalized_symbol, start, end)
        if self._last_history is not None and self._last_history[0] == key:
            return self._last_history[1], self._last_history[2]

        ticker = yf.Ticker(normalized_symbol)
        try:
            history = ticker.history(
                start=start.isoformat(),
                end=end.isoformat(),
                interval="1d",
                auto_adjust=False,
                actions=True,
                timeout=self._timeout,
            )
        except yf.exceptions.YFException as exc:
            raise MarketDataError(f"Yahoo Finance request for {normalized_symbol} failed: {exc}") from exc

        if history.empty:
            raise MarketDataError(
                f"Yahoo Finance returned no daily data for {normalized_symbol} "
                f"between {start} and {end}."
            )

        try:
            splits = ticker.splits
        except yf.exceptions.YFException as exc:
            raise MarketDataError(f"Yahoo Finance split history for {normalized_symbol} failed: {exc}") from exc

        factor = _split_factor(history.index, splits)
        self._last_history = (key, history, factor)
        return history, factor


def _split_factor(index: pd.DatetimeIndex, splits: pd.Series) -> pd.Series:
    """Per bar, the product of every split ratio with an ex-date after the bar.

    Multiplying Yahoo's prices by this factor (and dividing volumes by it)
    restores the values as traded on that day.
    """
    bar_dates = np.array([timestamp.date() for timestamp in index], dtype=object)
    factor = np.ones(len(index))
    for split_timestamp, ratio in splits.items():
        ratio = float(ratio)
        if not np.isfinite(ratio) or ratio <= 0:
            continue
        factor[bar_dates < pd.Timestamp(split_timestamp).date()] *= ratio
    return pd.Series(factor, index=index)
