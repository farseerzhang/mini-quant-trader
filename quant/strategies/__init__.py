"""Rule-based strategies: target weights at the close of T (proposal section 17).

Each strategy maps a feature panel (one dates x stocks DataFrame per feature)
and the rule-based regime (one value per date) to target weights per stock.
These are the baselines that ML strategies must beat after costs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from quant.config.settings import Settings
from quant.regime.rule_based import BEAR, BULL, SIDEWAYS
from quant.strategies.rules import PANEL_COLUMNS, Panel, mean_reversion_holdings, trend_holdings
from quant.strategies.sizing import target_weights

__all__ = ["PANEL_COLUMNS", "Strategy", "build_strategies", "regime_switch_weights"]


@dataclass(frozen=True)
class Strategy:
    name: str
    description: str
    weights: Callable[[Panel, pd.Series], pd.DataFrame]


def regime_switch_weights(panel: Panel, regimes: pd.Series, settings: Settings, bear_exposure: float) -> pd.DataFrame:
    """BULL -> trend, SIDEWAYS -> mean reversion, BEAR -> trend scaled by bear_exposure.

    Sessions without a regime (warm-up) hold nothing. Mean-reversion trades can
    only start on SIDEWAYS sessions, so the strategy never picks up a dip trade
    that began while another regime was in charge.
    """
    reference = panel["price_vs_sma200"]
    regime = regimes.reindex(reference.index)
    bull, sideways, bear = (regime == BULL), (regime == SIDEWAYS), (regime == BEAR)

    trend = trend_holdings(panel)
    reversion = mean_reversion_holdings(panel, settings.strategies.mean_reversion, allow_entry=sideways)
    trend_days = (bull | bear).to_numpy()[:, None]
    holdings = (trend & trend_days) | (reversion & sideways.to_numpy()[:, None])

    exposure = pd.Series(0.0, index=reference.index)
    exposure[bull | sideways] = 1.0
    exposure[bear] = bear_exposure
    return target_weights(holdings, settings.market.sectors, settings.risk, exposure)


def _bear_variant_name(bear_exposure: float) -> str:
    return f"regime_switch_bear{round(bear_exposure * 100):d}"


def build_strategies(settings: Settings) -> list[Strategy]:
    """The configured baselines, in report order. The first bear exposure is the
    `regime_switch` baseline; the others are the pre-registered variants."""
    sectors, risk = settings.market.sectors, settings.risk
    reversion = settings.strategies.mean_reversion
    strategies = [
        Strategy(
            "trend",
            "enter: price > SMA200, SMA50 > SMA200, MACD > signal; exit when either SMA condition fails; "
            "ignores the regime",
            lambda panel, regimes: target_weights(trend_holdings(panel), sectors, risk),
        ),
        Strategy(
            "mean_reversion",
            f"price > SMA200, RSI < {reversion.rsi_entry:g}, MACD histogram rising; "
            f"exit RSI > {reversion.rsi_exit:g} or after {reversion.max_holding_days} sessions",
            lambda panel, regimes: target_weights(mean_reversion_holdings(panel, reversion), sectors, risk),
        ),
    ]
    for position, bear_exposure in enumerate(settings.strategies.regime_switch.bear_exposures):
        name = "regime_switch" if position == 0 else _bear_variant_name(bear_exposure)
        strategies.append(
            Strategy(
                name,
                f"BULL -> trend, SIDEWAYS -> mean reversion, BEAR -> trend x {bear_exposure:g}",
                lambda panel, regimes, b=bear_exposure: regime_switch_weights(panel, regimes, settings, b),
            )
        )
    return strategies
