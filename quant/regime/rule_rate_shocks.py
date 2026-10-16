"""rule_rate_shocks-v1: the rule-based regime with two interest-rate danger inputs (proposal section 12.7).

Research only; nothing here is stored or used by run-daily, and rule_based-v1 stays ACTIVE.

- Input 6, short_rate_drop_20d: -1 when the 3-month Treasury yield (Yahoo ^IRX, percent) fell more
  than 0.50 points over 20 sessions (emergency easing priced within a month), else 0.
- Input 7, long_rate_jump_60d: -1 when the 10-year yield (^TNX) rose more than 0.75 points over
  60 sessions (a rate shock), else 0.
- Neither input is ever +1. The score runs from -7 to +5; E, X, the dwell and the VIX thresholds
  are v1's, unchanged.
- A session without a yield takes the previous session's value, at most one session (bond-market
  holidays, Yahoo's gaps).
- The recursion starts on the first session with all seven inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import Engine

from quant.config.settings import RegimeSettings, Settings
from quant.persistence import queries
from quant.regime.rule_based import classify_regimes, score_inputs

NAME = "rule_rate_shocks-v1"
SHORT_RATE, LONG_RATE = "^IRX", "^TNX"
YIELD_SYMBOLS = (SHORT_RATE, LONG_RATE)
SHORT_RATE_DROP, LONG_RATE_JUMP = "short_rate_drop_20d", "long_rate_jump_60d"
RATE_INPUTS = (SHORT_RATE_DROP, LONG_RATE_JUMP)
SHORT_WINDOW, SHORT_DROP_POINTS = 20, 0.50  # fixed in section 12.7
LONG_WINDOW, LONG_JUMP_POINTS = 60, 0.75
FILL_SESSIONS = 1


def rate_shock_inputs(yields: pd.DataFrame) -> pd.DataFrame:
    """Inputs 6 and 7 (-1 or 0; NaN during warm-up) from yields in percent on the sessions of ``yields``."""
    filled = yields.ffill(limit=FILL_SESSIONS)
    short_change = filled[SHORT_RATE] - filled[SHORT_RATE].shift(SHORT_WINDOW)
    long_change = filled[LONG_RATE] - filled[LONG_RATE].shift(LONG_WINDOW)
    return pd.DataFrame({
        SHORT_RATE_DROP: np.where(short_change < -SHORT_DROP_POINTS, -1.0, 0.0),
        LONG_RATE_JUMP: np.where(long_change > LONG_JUMP_POINTS, -1.0, 0.0),
    }, index=yields.index).where(pd.DataFrame({SHORT_RATE_DROP: short_change.notna(),
                                               LONG_RATE_JUMP: long_change.notna()}))


def rate_shock_scores(inputs: pd.DataFrame, settings: RegimeSettings,
                      extra: tuple[str, ...] = RATE_INPUTS) -> pd.Series:
    """The five-input score plus the `extra` rate inputs (each 0 or -1); NaN where any input is missing."""
    values = inputs.loc[:, list(extra)].astype(float)
    return (score_inputs(inputs, settings) + values.sum(axis=1)).where(values.notna().all(axis=1))


def load_rate_shock_inputs(engine: Engine, settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(the five v1 inputs and the two rate inputs per session, the yields on those sessions)."""
    missing = [s for s in YIELD_SYMBOLS if s not in settings.market.market_symbols]
    if missing:
        raise ValueError(f"add {', '.join(missing)} to market.market_symbols and run update-market-data")
    v1 = queries.load_regime_inputs(engine, settings.features.feature_version, settings.features.market_feature_version)
    yields = queries.load_raw_closes(engine, list(YIELD_SYMBOLS))
    yields.index = pd.DatetimeIndex(yields.index)
    yields = yields.reindex(v1.index)
    return v1.join(rate_shock_inputs(yields)), yields


def rule_rate_shocks_regime(inputs: pd.DataFrame, settings: RegimeSettings) -> pd.Series:
    """The candidate regime from the first session with all seven inputs."""
    complete = inputs.dropna()
    if complete.empty:
        raise ValueError(f"no session has all seven {NAME} inputs")
    inputs = inputs.loc[complete.index[0]:]
    return classify_regimes(rate_shock_scores(inputs, settings), settings)["regime"]
