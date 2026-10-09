"""Point-in-time technical features and corporate-action adjustments."""

from __future__ import annotations

from datetime import date
from typing import Mapping

import numpy as np
import pandas as pd

from quant.config.settings import FeatureSettings

FEATURE_COLUMNS = (
    "price_vs_sma20",
    "price_vs_sma50",
    "price_vs_sma200",
    "sma50_vs_sma200",
    "return_1m",
    "return_3m",
    "return_6m",
    "return_12m",
    "rsi14",
    "macd_over_price",
    "macd_signal_over_price",
    "macd_hist_over_price",
    "atr14_over_price",
    "realized_volatility_20d",
    "volume_ratio_20d",
    "relative_return_1m_spy",
)

MARKET_FEATURE_COLUMNS = (
    "spy_trend",
    "qqq_trend",
    "vix",
    "vix_change",
    "market_volatility",
)


def prepare_adjusted_series(prices: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """Derive split-adjusted OHLCV and a dividend-aware total-return close.

    Split ratios are normalized onto the latest share basis: pre-split prices
    are divided by the split ratio and pre-split volumes are multiplied by it.
    Dividends are similarly scaled for splits occurring after their ex-date.
    The total-return index is chained from the first available adjusted close.
    """
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(prices.columns):
        raise ValueError(f"prices must include columns: {', '.join(sorted(required))}")

    frame = prices.copy()
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).normalize()
    frame = frame.sort_index()
    if frame.index.has_duplicates:
        raise ValueError("prices contain duplicate dates")
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    split_events: list[tuple[pd.Timestamp, float]] = []
    dividend_events: list[tuple[pd.Timestamp, float]] = []
    if not actions.empty:
        for row in actions.itertuples(index=False):
            action_date = pd.Timestamp(getattr(row, "ex_date")).normalize()
            action_type = str(getattr(row, "action_type")).upper()
            value = float(getattr(row, "value"))
            if action_type == "SPLIT":
                if not np.isfinite(value) or value <= 0:
                    raise ValueError(f"invalid split ratio {value} on {action_date.date()}")
                split_events.append((action_date, value))
            elif action_type == "DIVIDEND":
                if not np.isfinite(value) or value < 0:
                    raise ValueError(f"invalid dividend {value} on {action_date.date()}")
                dividend_events.append((action_date, value))
            else:
                raise ValueError(f"unsupported corporate action: {action_type}")

    price_factor = pd.Series(1.0, index=frame.index)
    for split_date, ratio in split_events:
        price_factor.loc[frame.index < split_date] /= ratio

    adjusted = frame.copy()
    for column in ("open", "high", "low", "close"):
        adjusted[column] = frame[column] * price_factor
    if "volume" in frame:
        adjusted["volume"] = pd.to_numeric(frame["volume"], errors="coerce") / price_factor

    dividend_by_date = pd.Series(0.0, index=frame.index)
    for dividend_date, amount in dividend_events:
        if dividend_date not in dividend_by_date.index:
            continue
        dividend_factor = 1.0
        for split_date, ratio in split_events:
            if split_date > dividend_date:
                dividend_factor /= ratio
        dividend_by_date.loc[dividend_date] += amount * dividend_factor

    previous_close = adjusted["close"].shift(1)
    daily_total_return = (adjusted["close"] + dividend_by_date) / previous_close - 1.0
    if len(adjusted):
        daily_total_return.iloc[0] = 0.0
    adjusted["total_return_close"] = adjusted["close"].iloc[0] * (1.0 + daily_total_return).cumprod()
    return adjusted


def _rsi(close: pd.Series, period: int) -> pd.Series:
    change = close.diff()
    average_gain = change.clip(lower=0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    average_loss = -change.clip(upper=0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    relative_strength = average_gain / average_loss
    result = 100.0 - 100.0 / (1.0 + relative_strength)
    result = result.mask((average_loss == 0) & (average_gain > 0), 100.0)
    result = result.mask((average_loss == 0) & (average_gain == 0), 50.0)
    return result


def _atr(adjusted: pd.DataFrame, period: int) -> pd.Series:
    previous_close = adjusted["close"].shift(1)
    true_range = pd.concat(
        [
            adjusted["high"] - adjusted["low"],
            (adjusted["high"] - previous_close).abs(),
            (adjusted["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def calculate_symbol_features(
    adjusted: pd.DataFrame,
    spy_total_return_close: pd.Series | None = None,
    settings: FeatureSettings | None = None,
) -> pd.DataFrame:
    """Calculate scale-free features using only current and prior rows."""
    settings = settings or FeatureSettings()
    sma_short, sma_medium, sma_long = settings.sma
    close = adjusted["total_return_close"].astype(float)
    features = pd.DataFrame(index=adjusted.index)

    sma20 = close.rolling(sma_short, min_periods=sma_short).mean()
    sma50 = close.rolling(sma_medium, min_periods=sma_medium).mean()
    sma200 = close.rolling(sma_long, min_periods=sma_long).mean()
    features["price_vs_sma20"] = close / sma20 - 1.0
    features["price_vs_sma50"] = close / sma50 - 1.0
    features["price_vs_sma200"] = close / sma200 - 1.0
    features["sma50_vs_sma200"] = sma50 / sma200 - 1.0

    for name, periods in (("return_1m", 20), ("return_3m", 63), ("return_6m", 126), ("return_12m", 252)):
        features[name] = close / close.shift(periods) - 1.0

    features["rsi14"] = _rsi(close, settings.rsi_period)
    macd = close.ewm(span=12, min_periods=12, adjust=False).mean() - close.ewm(
        span=26, min_periods=26, adjust=False
    ).mean()
    macd_signal = macd.ewm(span=9, min_periods=9, adjust=False).mean()
    features["macd_over_price"] = macd / close
    features["macd_signal_over_price"] = macd_signal / close
    features["macd_hist_over_price"] = (macd - macd_signal) / close

    features["atr14_over_price"] = _atr(adjusted, settings.atr_period) / adjusted["close"]
    daily_return = close.pct_change(fill_method=None)
    features["realized_volatility_20d"] = daily_return.rolling(20, min_periods=20).std() * np.sqrt(252)

    if "volume" in adjusted:
        volume = pd.to_numeric(adjusted["volume"], errors="coerce")
        average_volume = volume.rolling(20, min_periods=20).mean()
        features["volume_ratio_20d"] = volume / average_volume.where(average_volume != 0)
    else:
        features["volume_ratio_20d"] = np.nan

    if spy_total_return_close is None:
        features["relative_return_1m_spy"] = np.nan
    else:
        spy_return = spy_total_return_close / spy_total_return_close.shift(20) - 1.0
        symbol_return = close / close.shift(20) - 1.0
        features["relative_return_1m_spy"] = symbol_return - spy_return.reindex(features.index)

    return features.loc[:, FEATURE_COLUMNS]


def calculate_market_features(
    adjusted_by_symbol: Mapping[str, pd.DataFrame], settings: FeatureSettings | None = None
) -> pd.DataFrame:
    """Calculate date-keyed benchmark and volatility features; SPY is required."""
    settings = settings or FeatureSettings()
    trend_window = settings.sma[2]
    spy = adjusted_by_symbol.get("SPY")
    if spy is None or spy.empty:
        raise ValueError("SPY market data is required to calculate market features")

    market = pd.DataFrame(index=spy.index)
    for symbol, column in (("SPY", "spy_trend"), ("QQQ", "qqq_trend")):
        frame = adjusted_by_symbol.get(symbol)
        if frame is None or frame.empty:
            market[column] = np.nan
            continue
        close = frame["total_return_close"].astype(float)
        trend = close / close.rolling(trend_window, min_periods=trend_window).mean() - 1.0
        market[column] = trend.reindex(market.index)

    vix = adjusted_by_symbol.get("^VIX")
    if vix is None:
        vix = adjusted_by_symbol.get("VIX")
    if vix is None or vix.empty:
        market["vix"] = np.nan
        market["vix_change"] = np.nan
    else:
        vix_close = vix["close"].astype(float)
        market["vix"] = vix_close.reindex(market.index)
        market["vix_change"] = vix_close.pct_change(fill_method=None).reindex(market.index)

    spy_returns = spy["total_return_close"].astype(float).pct_change(fill_method=None)
    market["market_volatility"] = spy_returns.rolling(20, min_periods=20).std().mul(np.sqrt(252))
    return market.loc[:, MARKET_FEATURE_COLUMNS]
