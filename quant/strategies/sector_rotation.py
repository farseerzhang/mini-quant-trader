"""Sector rotation with sector ETFs (proposal section 17.1.5.4, trial D).

Universe: the nine original SPDR sector ETFs (all trading since 1998). On each rebalance session
(the window's first session, then the first session of each month): rank them by total return from
252 to 21 sessions before T (the 12-1 score of trial B) and hold the top 3, one third each, until the
next rebalance. The weights are multiplied by the regime's exposure every session (BULL and SIDEWAYS
1, BEAR 0, no regime 0). Sector funds are diversified, so the single-stock caps do not apply
(confirmed with the registration).
"""

from __future__ import annotations

import pandas as pd

from quant.config.settings import Settings
from quant.strategies.ml import top_n_holdings
from quant.strategies.momentum_regime import momentum_scores, regime_exposure

SECTOR_FUNDS = ("XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLU")
TOP_SECTORS = 3


def sector_holdings(closes: pd.DataFrame, rebalances: pd.DatetimeIndex) -> pd.DataFrame:
    """Which funds are held on every session: the top 3 by the 12-1 score on the rebalance sessions
    (no keep band; ties to the earlier fund in SECTOR_FUNDS), held in between; nothing before the first."""
    scores = momentum_scores(closes.loc[:, list(SECTOR_FUNDS)])
    chosen = top_n_holdings(scores.loc[rebalances], TOP_SECTORS, every=1)
    return chosen.reindex(scores.index).ffill().fillna(False).astype(bool)


def sector_rotation_weights(holdings: pd.DataFrame, regimes: pd.Series | None, settings: Settings) -> pd.DataFrame:
    """One third per held fund, times the regime's exposure; ``regimes=None`` gives the always-invested
    variant (descriptive)."""
    weights = holdings.astype(float) / TOP_SECTORS
    if regimes is None:
        return weights
    return weights.mul(regime_exposure(regimes, holdings.index, settings), axis=0)
