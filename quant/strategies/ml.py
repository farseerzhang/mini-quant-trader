"""`ml_select`: the regime times the market, the PREDICTOR picks the stocks (proposal sections 14, 17).

Market exposure follows the rule-based regime exactly as in `regime_switch`
(BULL / SIDEWAYS: invested; BEAR: the baseline bear exposure; no regime:
nothing), applied every session. Which stocks are held is refreshed every
``rebalance_every_sessions`` sessions: the ``top_n`` stocks with the highest
calibrated probability at that session's close. Sizing uses the usual position
and sector caps. The probability is never a reason to buy on its own; the regime
and the risk limits decide exposure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.config.settings import Settings
from quant.regime.rule_based import BEAR, BULL, SIDEWAYS
from quant.strategies.sizing import target_weights


def top_n_holdings(probabilities: pd.DataFrame, top_n: int, every: int) -> pd.DataFrame:
    """Hold the ``top_n`` highest probabilities, re-chosen every ``every`` sessions.

    Selections are made on sessions 0, every, 2*every, ... of ``probabilities``
    and held until the next one. Missing probabilities are never selected; ties
    go to the earlier column.
    """
    values = probabilities.to_numpy(dtype=float)
    out = np.zeros(values.shape, dtype=bool)
    current = np.zeros(values.shape[1], dtype=bool)
    for t in range(values.shape[0]):
        if t % every == 0:
            row = values[t]
            available = np.flatnonzero(~np.isnan(row))
            ranked = available[np.argsort(-row[available], kind="stable")]
            current = np.zeros(values.shape[1], dtype=bool)
            current[ranked[:top_n]] = True
        out[t] = current
    return pd.DataFrame(out, index=probabilities.index, columns=probabilities.columns)


def ml_select_weights(probabilities: pd.DataFrame, regimes: pd.Series, settings: Settings) -> pd.DataFrame:
    """Target weights at the close of T from probabilities and the regime at T."""
    config = settings.strategies.ml_select
    holdings = top_n_holdings(probabilities, config.top_n, config.rebalance_every_sessions)
    regime = regimes.reindex(probabilities.index)
    exposure = pd.Series(0.0, index=probabilities.index)
    exposure[(regime == BULL) | (regime == SIDEWAYS)] = 1.0
    exposure[regime == BEAR] = settings.strategies.regime_switch.bear_exposures[0]
    return target_weights(holdings, settings.market.sectors, settings.risk, exposure)
