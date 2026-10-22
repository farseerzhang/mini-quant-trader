"""12-1 momentum inside the regime timing (proposal section 17.1.5.2, trial B).

`regime_switch`'s market timing is kept and its stock choice is replaced:

- on each rebalance session (the window's first session, then the first session of each month),
  rank the pool stocks by total return from 252 to 21 sessions before T; a stock without a close
  252 sessions back, 21 sessions back or at T is not ranked;
- hold the top 20; a held stock is kept while it ranks within the top 30 (the keep band of
  ``top_n_holdings``, shared with ml_select);
- the selection is held until the next rebalance session;
- sizing as `regime_switch`: equal weight, risk.max_position_size and max_sector_weight, times the
  regime's exposure every session (BULL and SIDEWAYS 1, BEAR the baseline bear exposure 0, no regime 0).
"""

from __future__ import annotations

import pandas as pd

from quant.config.settings import Settings
from quant.regime.rule_based import BEAR, BULL, SIDEWAYS
from quant.strategies.ml import top_n_holdings
from quant.strategies.sizing import target_weights

LOOKBACK_SESSIONS = 252
SKIP_SESSIONS = 21
TOP_N = 20
KEEP_RANK = 30


def momentum_scores(closes: pd.DataFrame) -> pd.DataFrame:
    """Total return from 252 to 21 sessions before each session (sessions x stocks); NaN when unrankable."""
    score = closes.shift(SKIP_SESSIONS) / closes.shift(LOOKBACK_SESSIONS) - 1.0
    return score.where(closes.notna())


def momentum_holdings(scores: pd.DataFrame, rebalances: pd.DatetimeIndex) -> pd.DataFrame:
    """Which stocks are held on every session: chosen on the rebalance sessions, held in between;
    nothing before the first rebalance."""
    chosen = top_n_holdings(scores.loc[rebalances], TOP_N, every=1, keep_rank=KEEP_RANK)
    held = chosen.reindex(scores.index).ffill()
    return held.fillna(False).astype(bool)


def regime_exposure(regimes: pd.Series, index: pd.DatetimeIndex, settings: Settings) -> pd.Series:
    """As regime_switch and ml_select: BULL and SIDEWAYS 1, BEAR the baseline bear exposure, no regime 0."""
    regime = regimes.reindex(index)
    exposure = pd.Series(0.0, index=index)
    exposure[(regime == BULL) | (regime == SIDEWAYS)] = 1.0
    exposure[regime == BEAR] = settings.strategies.regime_switch.bear_exposures[0]
    return exposure


def momentum_regime_weights(holdings: pd.DataFrame, regimes: pd.Series | None, settings: Settings) -> pd.DataFrame:
    """Target weights at the close of T; ``regimes=None`` gives the always-invested variant (descriptive)."""
    exposure = 1.0 if regimes is None else regime_exposure(regimes, holdings.index, settings)
    return target_weights(holdings, settings.market.sectors, settings.risk, exposure)
