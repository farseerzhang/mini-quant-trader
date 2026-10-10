"""Prediction target: volatility-scaled excess return over the benchmark (proposal section 14).

For a stock on session T, with horizon h:

    entry          open of T+1             (the backtest's fill: signal at the close of T)
    exit           open of T+1+h
    stock_return   total-return open(exit) / open(entry) - 1
    bench_return   the same window for the benchmark
    excess         stock_return - bench_return
    sigma_T        std of the stock's daily total-return close-to-close returns over the
                   last volatility_window_days sessions up to and including T
    threshold      max(k * sigma_T * sqrt(h), min_threshold)
    label          1 if excess > threshold else 0

The label is the only quantity in the system that looks past T; it is used to train
and score models, never as an input. The last h + 1 sessions have no label yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant.config.settings import TargetSettings


@dataclass(frozen=True)
class Labels:
    """Dates x stocks. ``label`` is 1.0 / 0.0, NaN where it cannot be known yet."""

    label: pd.DataFrame
    excess: pd.DataFrame
    threshold: pd.DataFrame


def future_window_return(total_return_open: pd.DataFrame | pd.Series, horizon: int):
    """Return from the open of T+1 to the open of T+1+horizon, indexed by T."""
    entry = total_return_open.shift(-1)
    exit_ = total_return_open.shift(-(1 + horizon))
    return exit_ / entry - 1.0


def volatility_threshold(
    total_return_close: pd.DataFrame, target: TargetSettings
) -> pd.DataFrame:
    """max(k * sigma_T * sqrt(h), min_threshold), using returns up to and including T."""
    window = target.volatility_window_days
    sigma = total_return_close.pct_change(fill_method=None).rolling(window, min_periods=window).std()
    scaled = target.k * sigma * np.sqrt(target.horizon_days)
    return scaled.clip(lower=target.min_threshold).where(sigma.notna())


def make_labels(
    stock_open: pd.DataFrame,
    stock_close: pd.DataFrame,
    benchmark_open: pd.Series,
    target: TargetSettings,
) -> Labels:
    """Labels for every stock and session; inputs are total-return prices, dates x stocks."""
    if target.kind != "excess_vs_benchmark_vol_scaled":
        raise NotImplementedError(f"target kind {target.kind!r} is not implemented yet")
    horizon = target.horizon_days
    stock_return = future_window_return(stock_open, horizon)
    benchmark_return = future_window_return(benchmark_open.reindex(stock_open.index), horizon)
    excess = stock_return.sub(benchmark_return, axis=0)
    threshold = volatility_threshold(stock_close, target)
    label = (excess > threshold).astype(float).where(excess.notna() & threshold.notna())
    return Labels(label=label, excess=excess, threshold=threshold)
