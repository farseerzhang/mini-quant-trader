"""`ml_select`: the regime times the market, the PREDICTOR picks the stocks (proposal sections 14, 17).

Market exposure follows the rule-based regime exactly as in `regime_switch`
(BULL / SIDEWAYS: invested; BEAR: the baseline bear exposure; no regime:
nothing), applied every session. Which stocks are held is refreshed every
``rebalance_every_sessions`` sessions: ``top_n`` stocks by calibrated probability
at that session's close, keeping a held stock while it ranks within ``keep_rank``. Sizing uses the usual position
and sector caps. The probability is never a reason to buy on its own; the regime
and the risk limits decide exposure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.config.settings import Settings
from quant.regime.rule_based import BEAR, BULL, SIDEWAYS
from quant.strategies.sizing import target_weights


def top_n_holdings(probabilities: pd.DataFrame, top_n: int, every: int, keep_rank: int | None = None) -> pd.DataFrame:
    """Hold ``top_n`` stocks by probability, re-chosen every ``every`` sessions.

    On sessions 0, every, 2*every, ...: a held stock that still ranks within
    ``keep_rank`` stays; the free places are filled with the highest-ranked
    stocks not held. With ``keep_rank = top_n`` (the default) this is simply the
    current top ``top_n``. A keep band above ``top_n`` stops a stock being sold
    and bought back as its rank wobbles around the cut-off. Missing
    probabilities are never held; ties go to the earlier column.
    """
    keep_rank = top_n if keep_rank is None else keep_rank
    values = probabilities.to_numpy(dtype=float)
    out = np.zeros(values.shape, dtype=bool)
    current = np.zeros(values.shape[1], dtype=bool)
    for t in range(values.shape[0]):
        if t % every == 0:
            row = values[t]
            available = np.flatnonzero(~np.isnan(row))
            ranked = available[np.argsort(-row[available], kind="stable")]
            kept = [column for column in ranked[:keep_rank] if current[column]]
            chosen = kept + [column for column in ranked if not current[column]][: top_n - len(kept)]
            current = np.zeros(values.shape[1], dtype=bool)
            current[chosen] = True
        out[t] = current
    return pd.DataFrame(out, index=probabilities.index, columns=probabilities.columns)


def ml_select_parts(probabilities: pd.DataFrame, regimes: pd.Series, settings: Settings):
    """The selection and the regime exposure per session, as StrategyParts (rule ``ml_select``)."""
    from quant.strategies import StrategyParts  # the package imports the rule modules, not this one

    config = settings.strategies.ml_select
    holdings = top_n_holdings(probabilities, config.top_n, config.rebalance_every_sessions, config.keep_rank)
    regime = regimes.reindex(probabilities.index)
    exposure = pd.Series(0.0, index=probabilities.index)
    exposure[(regime == BULL) | (regime == SIDEWAYS)] = 1.0
    exposure[regime == BEAR] = settings.strategies.regime_switch.bear_exposures[0]
    return StrategyParts({"ml_select": holdings}, exposure)


def ml_select_weights(probabilities: pd.DataFrame, regimes: pd.Series, settings: Settings) -> pd.DataFrame:
    """Target weights at the close of T from probabilities and the regime at T."""
    parts = ml_select_parts(probabilities, regimes, settings)
    return target_weights(parts.holdings, settings.market.sectors, settings.risk, parts.exposure)
