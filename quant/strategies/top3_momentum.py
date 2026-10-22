"""The three strongest stocks of the last 12 months, quarterly (proposal section 17.1.6, trial E).

On each rebalance session (the window's first session, then the first session of January, April,
July and October): rank the pool stocks by total return over the last 252 sessions up to the close of
T (no skipped month; a stock without a close at T or 252 sessions back is not ranked; ties to the
earlier stock in the pool order) and hold the top 3, one third each, until the next rebalance. No keep
band, no regime timing. The single-stock and sector caps do not apply (confirmed with the
registration). Between rebalances the weights drift with prices; on each rebalance session the three
holdings are set back to one third each.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.strategies.ml import top_n_holdings

LOOKBACK_SESSIONS = 252
TOP_N = 3
QUARTER_START_MONTHS = (1, 4, 7, 10)


def quarter_rebalance_sessions(sessions: pd.DatetimeIndex, first: pd.Timestamp) -> pd.DatetimeIndex:
    """``first`` (the window's first session) and the first session of every later quarter."""
    sessions = pd.DatetimeIndex(sessions).sort_values()
    later = sessions[sessions >= first]
    if len(later) == 0:
        return later
    quarter = later.to_period("Q")
    first_of_quarter = np.r_[False, quarter[1:] != quarter[:-1]]
    return later[:1].append(later[first_of_quarter])


def twelve_month_scores(closes: pd.DataFrame) -> pd.DataFrame:
    """Total return over the last 252 sessions (sessions x stocks); NaN when unrankable."""
    score = closes / closes.shift(LOOKBACK_SESSIONS) - 1.0
    return score.where(closes.notna())


def top3_holdings(closes: pd.DataFrame, rebalances: pd.DatetimeIndex) -> pd.DataFrame:
    """Which stocks are held on every session: the top 3 on each rebalance session, held in between;
    nothing before the first rebalance."""
    scores = twelve_month_scores(closes)
    chosen = top_n_holdings(scores.loc[rebalances], TOP_N, every=1)
    return chosen.reindex(scores.index).ffill().fillna(False).astype(bool)


def top3_order_targets(holdings: pd.DataFrame, rebalances: pd.DatetimeIndex) -> pd.DataFrame:
    """Order targets for the backtest: on each rebalance session one third for each held stock and 0
    for every other stock (so a dropped stock is sold and a kept one is reset to one third); no order
    on other sessions, so the weights drift with prices (NaN = no order)."""
    targets = pd.DataFrame(np.nan, index=holdings.index, columns=holdings.columns)
    dates = rebalances.intersection(holdings.index)
    targets.loc[dates] = holdings.loc[dates].astype(float) / TOP_N
    return targets
