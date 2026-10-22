"""Dual momentum (proposal section 17.1.5.1, trial A): stocks while they beat cash, else bonds if
bonds beat cash, else cash.

On each rebalance session T (the window's first session, then the first session of each month):

- ``r_stocks`` = SPY's total return over the last 252 sessions; ``r_bonds`` = IEF's;
- ``r_cash`` = the average of ^IRX (the 3-month T-bill yield, in percent) over those 252 sessions / 100;
- 100% SPY if ``r_stocks > r_cash``; else 100% IEF if ``r_bonds > r_cash``; else 100% cash.

The choice is held until the next rebalance session. SPY and IEF are diversified funds, so the
single-stock caps (risk.max_position_size, max_sector_weight) do not apply (confirmed with the
registration). Only data up to the close of T is used; the backtest fills at the open of T+1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STOCKS = "SPY"
BONDS = "IEF"
CASH = "CASH"
CASH_YIELD_SYMBOL = "^IRX"
LOOKBACK_SESSIONS = 252
FUNDS = (STOCKS, BONDS)


def rebalance_sessions(sessions: pd.DatetimeIndex, first: pd.Timestamp) -> pd.DatetimeIndex:
    """``first`` (the window's first session) and the first session of every later month."""
    sessions = pd.DatetimeIndex(sessions).sort_values()
    later = sessions[sessions >= first]
    if len(later) == 0:
        return later
    month = later.to_period("M")
    first_of_month = np.r_[False, month[1:] != month[:-1]]
    return later[:1].append(later[first_of_month])


def dual_momentum_choices(closes: pd.DataFrame, cash_yield: pd.Series, rebalances: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per rebalance session: r_stocks, r_bonds, r_cash and the choice (SPY, IEF or CASH).

    ``closes`` are total-return closes of SPY and IEF on their common sessions; ``cash_yield`` is
    ^IRX in percent (a missing session is left out of the average). A rebalance session with fewer
    than 252 earlier sessions, or without a cash yield in its window, raises: the window always
    starts years after 2005, so this means missing data, not a warm-up.
    """
    closes = closes.loc[:, list(FUNDS)].dropna()
    sessions = closes.index
    yields = cash_yield.reindex(sessions)
    rows = []
    for session in rebalances:
        position = sessions.get_loc(session)
        if position < LOOKBACK_SESSIONS:
            raise ValueError(f"dual momentum on {session.date()}: fewer than {LOOKBACK_SESSIONS} earlier sessions")
        start = sessions[position - LOOKBACK_SESSIONS]
        returns = closes.loc[session] / closes.loc[start] - 1.0
        window = yields.iloc[position - LOOKBACK_SESSIONS + 1:position + 1].dropna()
        if window.empty:
            raise ValueError(f"dual momentum on {session.date()}: no {CASH_YIELD_SYMBOL} in the last "
                             f"{LOOKBACK_SESSIONS} sessions")
        r_stocks, r_bonds, r_cash = float(returns[STOCKS]), float(returns[BONDS]), float(window.mean() / 100.0)
        choice = STOCKS if r_stocks > r_cash else BONDS if r_bonds > r_cash else CASH
        rows.append({"session": session, "r_stocks": r_stocks, "r_bonds": r_bonds, "r_cash": r_cash,
                     "choice": choice})
    return pd.DataFrame(rows, columns=["session", "r_stocks", "r_bonds", "r_cash", "choice"]).set_index("session")


def dual_momentum_weights(choices: pd.Series, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Target weights (sessions x SPY, IEF): 1.0 in the chosen fund from its rebalance session until
    the next one; 0 in both while the choice is CASH or before the first rebalance."""
    chosen = choices.reindex(index).ffill()
    return pd.DataFrame({fund: (chosen == fund).astype(float) for fund in FUNDS}, index=index)
