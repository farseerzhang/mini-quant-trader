"""From target weights to the orders the backtest places (proposal sections 4.5, 18 and 20)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def order_targets(weights: pd.DataFrame, min_trade_weight: float) -> pd.DataFrame:
    """Target weights to place as orders on each date (NaN = no order for that stock).

    Entries (0 -> w) and exits (w -> 0) always trade. A held position is resized
    only when its target moves at least ``min_trade_weight`` away from the last
    target that was ordered, which ignores the small reweighting that happens
    whenever the number of holdings changes (section 18). The comparison is
    against the last ordered target, not the drifted market weight, so the order
    list depends on the weights alone.
    """
    targets = weights.fillna(0.0).to_numpy(dtype=float)
    out = np.full(targets.shape, np.nan)
    current = np.zeros(targets.shape[1])
    for t in range(targets.shape[0]):
        target = targets[t]
        held, wanted = current > 0, target > 0
        resize = held & wanted & (np.abs(target - current) >= min_trade_weight)
        trade = (~held & wanted) | (held & ~wanted) | resize
        out[t, trade] = target[trade]
        current[trade] = target[trade]
    return pd.DataFrame(out, index=weights.index, columns=weights.columns)


def execution_sizes(orders: pd.DataFrame) -> pd.DataFrame:
    """Shift orders decided at the close of T onto T+1, where they fill at the open."""
    return orders.shift(1)


def buy_and_hold(index: pd.DatetimeIndex, symbol: str) -> pd.DataFrame:
    """Order targets for 100% in one symbol, bought once at the start."""
    out = pd.DataFrame(np.nan, index=index, columns=[symbol])
    out.iloc[0] = 1.0
    return out


def equal_weight_monthly(index: pd.DatetimeIndex, symbols: list[str]) -> pd.DataFrame:
    """Order targets for 1/N in every symbol, set at the start and rebalanced on the
    first session of each month."""
    out = pd.DataFrame(np.nan, index=index, columns=symbols)
    month = index.to_period("M")
    first_of_month = np.r_[True, month[1:] != month[:-1]]
    out.loc[first_of_month] = 1.0 / len(symbols)
    return out


def permuted_weights(weights: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """The same weights assigned to a random permutation of the stocks.

    Timing, number held, exposure and turnover are unchanged; only which stock
    receives each weight series is random. Sector caps are not re-applied.
    """
    shuffled = weights.iloc[:, rng.permutation(weights.shape[1])]
    shuffled.columns = weights.columns
    return shuffled
