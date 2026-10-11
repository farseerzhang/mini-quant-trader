"""Turn holdings into target portfolio weights under the risk limits (proposal section 19)."""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from quant.config.settings import RiskSettings


def target_weights(
    holdings: pd.DataFrame,
    sectors: Mapping[str, str],
    risk: RiskSettings,
    exposure: pd.Series | float = 1.0,
) -> pd.DataFrame:
    """Equal-weight the held stocks, then apply the position cap, the sector cap and exposure.

    1. Each held stock gets 1 / (number held), capped at risk.max_position_size.
    2. A sector whose total exceeds risk.max_sector_weight is scaled down
       proportionally to the cap. The excess stays in cash; it is not moved to
       other sectors, so a cap never forces extra buying.
    3. The result is multiplied by ``exposure`` (per date, or one number), e.g. the
       regime's exposure in regime_switch.
    4. If the invested total is above risk.max_portfolio_exposure, every weight is
       scaled down proportionally to it (section 19.1; 1.0 never binds).

    Weights are fractions of portfolio value; whatever does not add up to 1 is cash.
    """
    held = holdings.fillna(False).astype(bool)
    sector_of = pd.Series({symbol: sectors.get(symbol) for symbol in held.columns})
    unknown = sorted(sector_of[sector_of.isna()].index)
    if unknown:
        raise ValueError(f"no sector configured for: {', '.join(unknown)}")

    count = held.sum(axis=1)
    per_stock = (1.0 / count.where(count > 0)).clip(upper=risk.max_position_size).fillna(0.0)
    weights = held.astype(float).mul(per_stock, axis=0)

    sector_totals = weights.T.groupby(sector_of.to_numpy()).sum().T
    sector_scale = (risk.max_sector_weight / sector_totals).clip(upper=1.0)
    weights = weights * sector_scale.loc[:, sector_of.to_numpy()].to_numpy()

    if isinstance(exposure, pd.Series):
        weights = weights.mul(exposure.reindex(weights.index).fillna(0.0), axis=0)
    else:
        weights = weights * float(exposure)

    invested = weights.sum(axis=1)
    exposure_scale = (risk.max_portfolio_exposure / invested.where(invested > 0)).clip(upper=1.0).fillna(1.0)
    return weights.mul(exposure_scale, axis=0)
