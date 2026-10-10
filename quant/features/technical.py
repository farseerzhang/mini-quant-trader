"""Point-in-time technical features and corporate-action adjustments."""

from __future__ import annotations

from datetime import date
from typing import Mapping

import re

import numpy as np
import pandas as pd

from quant.config.settings import ConfigError, FeatureSettings
from quant.persistence import tables

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
    "return_5d",
    "relative_return_3m_spy",
    "relative_return_6m_spy",
    "beta_spy_60d",
    "corr_spy_60d",
    "volatility_ratio_20_60",
    "drawdown_252d",
)

MARKET_FEATURE_COLUMNS = (
    "spy_trend",
    "qqq_trend",
    "vix",
    "vix_change",
    "market_volatility",
    "spy_return_1d",
    "qqq_return_1d",
    "log_vix",
    "log_market_volatility",
)


def check_schema_compatibility(settings: FeatureSettings) -> None:
    """The features table names its SMA columns after their windows (price_vs_sma20, ...).

    The windows in settings must match those stored columns, so the schema is the
    single definition of what is supported rather than a second copy in code.
    """
    stored = tuple(
        int(match.group(1))
        for column in tables.features.c.keys()
        if (match := re.fullmatch(r"price_vs_sma(\d+)", column))
    )
    if tuple(settings.sma) != stored:
        raise ConfigError(
            f"features.sma {list(settings.sma)} does not match the stored feature columns "
            f"{[f'price_vs_sma{window}' for window in stored]}; changing the windows needs a schema change"
        )


def prepare_adjusted_series(prices: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """Derive split-adjusted OHLCV and a dividend-aware total-return close.

    Split ratios are normalized onto the latest share basis: pre-split prices
    are divided by the split ratio and pre-split volumes are multiplied by it.
    Dividends are similarly scaled for splits occurring after their ex-date.
    The total-return index is chained from the first available adjusted close.

    ``total_return_open`` puts the open on the same basis: it applies the
    close's total-return factor of that day (``total_return_close / close``) to
    the adjusted open. Backtests fill at it (signal at the close of T, fill at
    the open of T+1), so holding overnight into an ex-date earns the dividend
    and buying at the ex-date open does not. Like any total-return index built
    from closes, it assumes dividends are reinvested at the ex-date close; the error
    against reinvesting at the open is the dividend yield times that day's
    open-to-close move (about 0.005% for a 0.5% dividend and a 1% move).
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
    adjusted["total_return_open"] = adjusted["open"] * (adjusted["total_return_close"] / adjusted["close"])
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
    settings: FeatureSettings,
    spy_total_return_close: pd.Series | None = None,
) -> pd.DataFrame:
    """Calculate scale-free features using only current and prior rows."""
    check_schema_compatibility(settings)
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

    features["return_5d"] = close / close.shift(5) - 1.0
    volatility_60d = daily_return.rolling(60, min_periods=60).std()
    features["volatility_ratio_20_60"] = daily_return.rolling(20, min_periods=20).std() / volatility_60d.where(
        volatility_60d != 0
    )
    features["drawdown_252d"] = close / close.rolling(252, min_periods=252).max() - 1.0

    relative_columns = ("relative_return_1m_spy", "relative_return_3m_spy", "relative_return_6m_spy")
    if spy_total_return_close is None:
        for name in (*relative_columns, "beta_spy_60d", "corr_spy_60d"):
            features[name] = np.nan
    else:
        spy_close = spy_total_return_close.astype(float)
        for name, periods in zip(relative_columns, (20, 63, 126)):
            spy_return = (spy_close / spy_close.shift(periods) - 1.0).reindex(features.index)
            features[name] = (close / close.shift(periods) - 1.0) - spy_return

        # SPY's daily returns on its own sessions, aligned to this symbol's sessions.
        spy_daily = spy_close.pct_change(fill_method=None).reindex(features.index)
        covariance = daily_return.rolling(60, min_periods=60).cov(spy_daily)
        spy_variance = spy_daily.rolling(60, min_periods=60).var()
        features["beta_spy_60d"] = covariance / spy_variance.where(spy_variance != 0)
        features["corr_spy_60d"] = daily_return.rolling(60, min_periods=60).corr(spy_daily)

    return features.loc[:, FEATURE_COLUMNS]


def calculate_market_features(
    adjusted_by_symbol: Mapping[str, pd.DataFrame], settings: FeatureSettings
) -> pd.DataFrame:
    """Calculate date-keyed benchmark and volatility features; SPY is required."""
    check_schema_compatibility(settings)
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

    # HMM observations (section 13): daily log total returns and log levels, which are
    # far closer to Gaussian than the raw, right-skewed VIX and volatility levels.
    for symbol, column in (("SPY", "spy_return_1d"), ("QQQ", "qqq_return_1d")):
        frame = adjusted_by_symbol.get(symbol)
        if frame is None or frame.empty:
            market[column] = np.nan
            continue
        log_close = np.log(frame["total_return_close"].astype(float))
        market[column] = log_close.diff().reindex(market.index)
    market["log_vix"] = np.log(market["vix"].where(market["vix"] > 0))
    market["log_market_volatility"] = np.log(market["market_volatility"].where(market["market_volatility"] > 0))
    return market.loc[:, MARKET_FEATURE_COLUMNS]
