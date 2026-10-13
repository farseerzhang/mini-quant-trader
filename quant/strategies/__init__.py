"""Rule-based strategies: target weights at the close of T (proposal section 17).

Each strategy maps a feature panel (one dates x stocks DataFrame per feature)
and the rule-based regime (one value per date) to target weights per stock.
These are the baselines that ML strategies must beat after costs.

WHICH STRATEGIES EXIST, AND WHERE EACH ONE IS BACKTESTED
--------------------------------------------------------
Rule-based (``build_strategies``; all run by ``quant backtest`` with the rule-based regime,
2005 .. holdout, against SPY buy-and-hold, equal weight and a random baseline):

    trend                 uptrend rule only (price > SMA200, SMA50 > SMA200, MACD trigger); ignores the regime
    mean_reversion        dip-buy rule only (RSI entry / exit, max holding days); ignores the regime
    regime_switch         BULL -> trend, SIDEWAYS -> mean_reversion, BEAR -> cash.  <- the strategy that ships
    regime_switch_bear50  as regime_switch, BEAR -> trend at 50% (from strategies.regime_switch.bear_exposures)

Elsewhere (not in ``quant backtest``):

    ml_select             the PREDICTOR picks the stocks (quant/strategies/ml.py). Backtested only by
                          ``quant retrain --role predictor --model logistic_regression|random_forest|xgboost``,
                          walk-forward 2014 .. 2025, against regime_switch (rule-based regime);
                          ``--model all`` runs the three and prints one comparison table.
    HMM regime            ``quant retrain --role regime`` runs regime_switch and regime_switch_bear50 with the
                          walk-forward HMM regime and with the rule-based one, 2014 .. 2025.
    forward test          live = signals.strategy (regime_switch), plus the shadows in forward_test.shadows
                          (regime_switch with the HMM regime, ml_select with the logistic regression).

There is no single run that crosses every strategy with every regime and predictor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from quant.config.settings import Settings, bear_variant_name
from quant.regime.rule_based import BEAR, BULL, SIDEWAYS
from quant.strategies.rules import PANEL_COLUMNS, Panel, mean_reversion_holdings, trend_holdings
from quant.strategies.sizing import target_weights

__all__ = [
    "PANEL_COLUMNS", "Strategy", "StrategyParts", "build_strategies", "parts_weights",
    "regime_switch_parts", "regime_switch_weights", "strategy_parts",
]


@dataclass(frozen=True)
class Strategy:
    name: str
    description: str
    weights: Callable[[Panel, pd.Series], pd.DataFrame]


@dataclass(frozen=True)
class StrategyParts:
    """What a rule-based strategy holds at each close, which rule holds it, and the exposure.

    ``rules`` maps a rule name (trend, mean_reversion) to the stocks that rule holds
    (boolean, dates x stocks); a stock is held by at most one rule on a date.
    """

    rules: dict[str, pd.DataFrame]
    exposure: pd.Series  # regime exposure per date (1.0 for strategies that ignore the regime)

    @property
    def holdings(self) -> pd.DataFrame:
        frames = list(self.rules.values())
        held = frames[0].copy()
        for frame in frames[1:]:
            held |= frame
        return held


def regime_switch_parts(panel: Panel, regimes: pd.Series, settings: Settings, bear_exposure: float) -> StrategyParts:
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

    exposure = pd.Series(0.0, index=reference.index)
    exposure[bull | sideways] = 1.0
    exposure[bear] = bear_exposure
    return StrategyParts({"trend": trend & trend_days, "mean_reversion": reversion & sideways.to_numpy()[:, None]},
                         exposure)


def regime_switch_weights(panel: Panel, regimes: pd.Series, settings: Settings, bear_exposure: float) -> pd.DataFrame:
    return parts_weights(regime_switch_parts(panel, regimes, settings, bear_exposure), settings)


def parts_weights(parts: StrategyParts, settings: Settings) -> pd.DataFrame:
    """Target weights from the parts: equal weight, position and sector caps, exposure (section 17.2)."""
    return target_weights(parts.holdings, settings.market.sectors, settings.risk, parts.exposure)


def strategy_parts(name: str, panel: Panel, regimes: pd.Series, settings: Settings) -> StrategyParts:
    """The parts of the rule-based strategy ``name`` (one of settings.strategies.rule_names)."""
    reference = panel["price_vs_sma200"]
    full = pd.Series(1.0, index=reference.index)
    if name == "trend":
        return StrategyParts({"trend": trend_holdings(panel)}, full)
    if name == "mean_reversion":
        return StrategyParts({"mean_reversion": mean_reversion_holdings(panel, settings.strategies.mean_reversion)}, full)
    for position, bear_exposure in enumerate(settings.strategies.regime_switch.bear_exposures):
        if name == ("regime_switch" if position == 0 else bear_variant_name(bear_exposure)):
            return regime_switch_parts(panel, regimes, settings, bear_exposure)
    raise ValueError(f"unknown rule-based strategy {name!r}; use one of {', '.join(settings.strategies.rule_names)}")


def build_strategies(settings: Settings) -> list[Strategy]:
    """The configured baselines, in report order. The first bear exposure is the
    `regime_switch` baseline; the others are the pre-registered variants."""
    reversion = settings.strategies.mean_reversion
    descriptions = {
        "trend": "enter: price > SMA200, SMA50 > SMA200, MACD > signal; exit when either SMA condition fails; "
                 "ignores the regime",
        "mean_reversion": f"price > SMA200, RSI < {reversion.rsi_entry:g}, MACD histogram rising; "
                          f"exit RSI > {reversion.rsi_exit:g} or after {reversion.max_holding_days} sessions",
    }
    for position, bear_exposure in enumerate(settings.strategies.regime_switch.bear_exposures):
        name = "regime_switch" if position == 0 else bear_variant_name(bear_exposure)
        descriptions[name] = f"BULL -> trend, SIDEWAYS -> mean reversion, BEAR -> trend x {bear_exposure:g}"
    return [
        Strategy(name, description,
                 lambda panel, regimes, n=name: parts_weights(strategy_parts(n, panel, regimes, settings), settings))
        for name, description in descriptions.items()
    ]
