"""Volatility targeting on `regime_switch` (proposal section 17.1.5.3, trial C).

An overlay: `regime_switch`'s target weights are kept and scaled each session T by

    m_T = min(1, 0.15 / sigma_T)

where sigma_T is SPY's realized volatility over the last 20 sessions, annualized: the sample
standard deviation of the 20 daily log returns of SPY's total-return close up to T, times
sqrt(252). No leverage: the overlay only ever reduces exposure. Small daily changes of the
targets are absorbed by backtest.min_trade_weight, as for every strategy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_VOLATILITY = 0.15
VOLATILITY_SESSIONS = 20
SESSIONS_PER_YEAR = 252


def realized_volatility(closes: pd.Series) -> pd.Series:
    """Annualized volatility of the last 20 daily log returns, per session (NaN until 20 returns exist)."""
    returns = np.log(closes.dropna()).diff()
    return returns.rolling(VOLATILITY_SESSIONS, min_periods=VOLATILITY_SESSIONS).std() * np.sqrt(SESSIONS_PER_YEAR)


def volatility_multiplier(volatility: pd.Series) -> pd.Series:
    """min(1, 0.15 / sigma); NaN where sigma is unknown."""
    return (TARGET_VOLATILITY / volatility).clip(upper=1.0)


def volatility_target_weights(weights: pd.DataFrame, multiplier: pd.Series) -> pd.DataFrame:
    """The strategy's weights scaled by the multiplier of each session. A session in ``weights`` without
    a multiplier raises: the trial window starts years after SPY's first 20 sessions."""
    scale = multiplier.reindex(weights.index)
    if scale.isna().any():
        missing = scale.index[scale.isna()]
        raise ValueError(f"no SPY volatility on {len(missing)} session(s), first {missing[0].date()}")
    return weights.mul(scale, axis=0)
