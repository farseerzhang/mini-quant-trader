from datetime import date

import pandas as pd
import yfinance as yf


_PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
_ACTION_COLUMNS = ["date", "action_type", "value"]


class MarketDataError(RuntimeError):
    """Raised when the market data provider cannot return usable data."""


class YFinanceProvider:
    def get_daily_data(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        history = self._get_history(symbol, start, end)
        return history.loc[:, _PRICE_COLUMNS].copy()

    def get_corporate_actions(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        history = self._get_history(symbol, start, end)
        actions: list[dict[str, object]] = []
        for action_type, column in (("DIVIDEND", "Dividends"), ("SPLIT", "Stock Splits")):
            for action_date, value in history.loc[history[column] > 0, column].items():
                actions.append(
                    {
                        "date": action_date.date(),
                        "action_type": action_type,
                        "value": float(value),
                    }
                )
        return pd.DataFrame(actions, columns=_ACTION_COLUMNS)

    @staticmethod
    def _get_history(symbol: str, start: date, end: date) -> pd.DataFrame:
        normalized_symbol = symbol.strip().upper()
        if not normalized_symbol:
            raise ValueError("symbol must not be empty")
        if start >= end:
            raise ValueError("start must be earlier than end")

        try:
            history = yf.Ticker(normalized_symbol).history(
                start=start.isoformat(),
                end=end.isoformat(),
                interval="1d",
                auto_adjust=False,
                actions=True,
                timeout=20,
            )
        except yf.exceptions.YFException as exc:
            raise MarketDataError(f"Yahoo Finance request for {normalized_symbol} failed: {exc}") from exc

        if history.empty:
            raise MarketDataError(
                f"Yahoo Finance returned no daily data for {normalized_symbol} "
                f"between {start} and {end}."
            )
        return history
