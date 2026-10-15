"""rule_bonds-v1: the rule-based regime with two bond-market inputs (proposal section 12.5).

Tested as "rule_based-v2" in the section 12.5 trial and holdout check, and renamed on
2026-10-15 so that it cannot be confused with a future `regime.version: 2` of the ACTIVE rules
(whose name is `rule_based-v<regime.version>`). The rules are unchanged:

- Inputs 6 and 7 at the close of T, from total-return closes over 20 sessions:
  credit_appetite_20d = HYG's return minus IEF's; stock_vs_bond_20d = SPY's minus TLT's.
  Each scores its sign, risk-on +1 (the direction of inputs 1-5 of section 12.1).
- Seven inputs (-7 .. +7) with E = 4, X = 1; dwell and VIX thresholds as configured.
- The recursion starts on the first session with all seven inputs (2007-05-09 on the stored data).

Not stored: the forward-test shadow `shadow_rule_bonds` computes it each evening.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
from sqlalchemy import Engine

from quant.backtest.data import load_prices
from quant.config.settings import RegimeSettings, Settings
from quant.persistence import queries
from quant.regime.rule_based import classify_regimes, score_inputs

NAME = "rule_bonds-v1"
WINDOW = 20  # sessions, as in CNN's components
BOND_SYMBOLS = ("TLT", "IEF", "HYG")
CREDIT, SAFE_HAVEN = "credit_appetite_20d", "stock_vs_bond_20d"
BOND_INPUTS = (CREDIT, SAFE_HAVEN)
ENTER, EXIT = 4, 1  # fixed in section 12.5 for both the seven- and the six-input scores


def bond_inputs(close: pd.DataFrame) -> pd.DataFrame:
    """Inputs 6 and 7 per session from total-return closes (SPY, TLT, IEF, HYG) up to T."""
    change = close / close.shift(WINDOW) - 1.0
    return pd.DataFrame({CREDIT: change["HYG"] - change["IEF"], SAFE_HAVEN: change["SPY"] - change["TLT"]},
                        index=close.index)


def bond_scores(inputs: pd.DataFrame, settings: RegimeSettings, extra: tuple[str, ...] = BOND_INPUTS) -> pd.Series:
    """The five-input score plus the sign of each `extra` input; NaN where any input is missing."""
    score = score_inputs(inputs, settings)
    values = inputs.loc[:, list(extra)].astype(float)
    return (score + np.sign(values).sum(axis=1)).where(values.notna().all(axis=1))


def bond_settings(settings: RegimeSettings) -> RegimeSettings:
    """The configured regime settings with this rule's thresholds (E = 4, X = 1)."""
    return dataclasses.replace(settings, enter_threshold=ENTER, exit_threshold=EXIT)


def load_bond_inputs(engine: Engine, settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(the seven inputs per session, the total-return closes of SPY / TLT / IEF / HYG)."""
    missing = [s for s in BOND_SYMBOLS if s not in settings.market.market_symbols]
    if missing:
        raise ValueError(f"add {', '.join(missing)} to market.market_symbols and run update-market-data")
    v1 = queries.load_regime_inputs(engine, settings.features.feature_version, settings.features.market_feature_version)
    close = load_prices(engine, ["SPY", *BOND_SYMBOLS]).close
    return v1.join(bond_inputs(close.reindex(v1.index))), close


def rule_bonds_regime(engine: Engine, settings: Settings) -> pd.Series:
    """rule_bonds-v1 per session up to the latest stored one, from the first session with all
    seven inputs: the regime of the forward-test shadow (section 12.5.3)."""
    inputs, _ = load_bond_inputs(engine, settings)
    complete = inputs.dropna()
    if complete.empty:
        raise ValueError(f"no session has all seven {NAME} inputs")
    inputs = inputs.loc[complete.index[0]:]
    return classify_regimes(bond_scores(inputs, settings.regime), bond_settings(settings.regime))["regime"]
