"""Feature research: how much each input says about the next h sessions (research only).

The rank information coefficient (IC) of a feature on session T is the Spearman
correlation, across the pool stocks, between the feature at T and the label's
excess return (open T+1 -> open T+1+h). Its mean over many sessions measures
how well the feature ranks stocks; how often each year's mean has the same sign
measures whether that holds up over time. A feature that is the same for every
stock on a day (all market features) has no IC: it cannot rank stocks at all.

``candidate_features`` builds extra stock features from the stored prices,
volume and configured sectors, using data up to and including T only. Screening
features on the development period is itself a selection: whatever is chosen
here must still pass the walk-forward and the gates, and the holdout stays unused.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from quant.backtest.data import Prices
from quant.models.dataset import Dataset


def candidate_features(
    prices: Prices,
    volume: pd.DataFrame,
    stocks: list[str],
    sectors: Mapping[str, str],
    benchmark: str = "SPY",
) -> dict[str, pd.DataFrame]:
    """Extra stock features (dates x stocks), point-in-time."""
    close = prices.close[stocks]
    open_ = prices.open[stocks]
    returns = close.pct_change(fill_method=None)
    spy = prices.close[benchmark].pct_change(fill_method=None)
    features: dict[str, pd.DataFrame] = {}

    features["return_1d"] = returns
    features["overnight_gap"] = open_ / close.shift(1) - 1.0
    features["intraday_return"] = close / open_ - 1.0
    # Classic 12-1 momentum: the last 12 months excluding the most recent month.
    features["momentum_12_1"] = close.shift(21) / close.shift(252) - 1.0
    vol_126 = returns.rolling(126, min_periods=126).std() * np.sqrt(252)
    features["momentum_6m_vol_adjusted"] = (close / close.shift(126) - 1.0) / vol_126

    # Idiosyncratic volatility: daily returns after removing their 60-session beta to SPY.
    covariance = returns.rolling(60, min_periods=60).cov(spy)
    beta = covariance.div(spy.rolling(60, min_periods=60).var(), axis=0)
    residual = returns - beta.mul(spy, axis=0)
    features["idiosyncratic_volatility_60d"] = residual.rolling(60, min_periods=60).std() * np.sqrt(252)
    features["max_return_20d"] = returns.rolling(20, min_periods=20).max()

    dollar_volume = (close * volume[stocks]).where(volume[stocks] > 0)
    features["log_dollar_volume_20d"] = np.log(dollar_volume.rolling(20, min_periods=20).mean())
    features["illiquidity_20d"] = np.log((returns.abs() / dollar_volume).rolling(20, min_periods=20).mean())

    # Relative to the stock's sector (sector mean over the pool stocks in it).
    sector_of = pd.Series({stock: sectors[stock] for stock in stocks})
    for name, periods in (("sector_relative_return_5d", 5), ("sector_relative_return_1m", 20)):
        stock_return = close / close.shift(periods) - 1.0
        sector_mean = stock_return.T.groupby(sector_of).transform("mean").T
        features[name] = stock_return - sector_mean
    return features


def rank_ic(feature: pd.DataFrame, target: pd.DataFrame) -> pd.Series:
    """Daily cross-sectional Spearman correlation between a feature and the target."""
    feature, target = feature.align(target, join="inner")
    valid = feature.notna() & target.notna()
    f = feature.where(valid).rank(axis=1)
    t = target.where(valid).rank(axis=1)
    f = f.sub(f.mean(axis=1), axis=0)
    t = t.sub(t.mean(axis=1), axis=0)
    ic = (f * t).sum(axis=1) / np.sqrt((f**2).sum(axis=1) * (t**2).sum(axis=1))
    return ic[valid.sum(axis=1) >= 10]


def ic_summary(ic: pd.Series, horizon: int) -> dict[str, float]:
    """Mean IC, a t-statistic on non-overlapping sessions, and its consistency across years."""
    sampled = ic.iloc[::horizon]
    yearly = ic.groupby(ic.index.year).mean()
    sign = np.sign(ic.mean())
    return {
        "mean_ic": float(ic.mean()),
        "t_stat": float(sampled.mean() / sampled.std() * np.sqrt(len(sampled))) if sampled.std() > 0 else float("nan"),
        "years_same_sign": float((np.sign(yearly) == sign).mean()),
        "years": int(len(yearly)),
        "sessions": int(len(ic)),
    }


# Size / liquidity features: with a pool chosen today, being small in the past predicts
# having grown since (survivorship bias), so they must not be selected for trading.
SURVIVORSHIP_BIASED = ("log_dollar_volume_20d", "illiquidity_20d")


def with_candidates(dataset: Dataset, candidates: Mapping[str, pd.DataFrame]) -> Dataset:
    """The dataset with candidate features added as columns; rows missing any are dropped."""
    frame = dataset.frame.copy()
    for name, wide in candidates.items():
        long = wide.rename_axis(index="date", columns="symbol").stack(future_stack=True).rename(name)
        frame[name] = long.reindex(frame.index)
    return Dataset(frame.dropna(subset=list(candidates)))


def ic_selector(pool: list[str], horizon: int, min_abs_t: float = 2.0):
    """Keep the pool features whose rank IC on the given rows has |t| >= ``min_abs_t``
    (at least the single strongest one)."""

    def select(rows: Dataset) -> list[str]:
        frame = rows.frame
        target = frame["excess"].unstack("symbol")
        scores = {}
        for name in pool:
            summary = ic_summary(rank_ic(frame[name].unstack("symbol"), target), horizon)
            scores[name] = abs(summary["t_stat"]) if summary["t_stat"] == summary["t_stat"] else 0.0
        chosen = [name for name in pool if scores[name] >= min_abs_t]
        return chosen or [max(scores, key=scores.get)]

    return select
