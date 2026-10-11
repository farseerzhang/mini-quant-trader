"""Target weights for the latest session T and why each stock has its weight (sections 18.1, 19.1).

The weights are the backtest's: the configured strategy is run over the full stored history
(trend holdings persist from their entry, mean reversion counts its holding days) and the
row for T is taken. Nothing here sizes differently from the backtest. Alongside the weight,
each stock gets what the reason needs: the rule that holds it, how long it has been held,
which risk limit bound, its indicators at T and the informational risk flags.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy import Engine

from quant.config.settings import Settings
from quant.persistence import queries
from quant.regime.pipeline import RULE_METHOD
from quant.strategies import PANEL_COLUMNS, StrategyParts, parts_weights, strategy_parts
from quant.strategies.rules import Panel

# Indicators reported per stock (the strategy inputs plus the two risk-flag inputs).
RISK_COLUMNS = ("realized_volatility_20d", "atr14_over_price")
TARGET_COLUMNS = tuple(dict.fromkeys((*PANEL_COLUMNS, *RISK_COLUMNS)))
REGIME = "REGIME"


@dataclass(frozen=True)
class Targets:
    session: pd.Timestamp  # T: the close the targets are computed at
    strategy: str
    regime: str | None
    regime_source: str  # model_version of the regime used, e.g. rule_based-v1
    regime_exposure: float
    stocks: pd.DataFrame  # one row per pool stock, see stock_details()

    @property
    def weights(self) -> pd.Series:
        return self.stocks["target_weight"]

    @property
    def invested(self) -> float:
        return float(self.weights.sum())

    @property
    def cash(self) -> float:
        return 1.0 - self.invested


def _held_sessions(held: pd.DataFrame) -> pd.Series:
    """Consecutive sessions each stock has been held, up to and including the last row."""
    values = held.to_numpy()
    count = np.zeros(values.shape[1], dtype=int)
    for row in values:
        count = np.where(row, count + 1, 0)
    return pd.Series(count, index=held.columns)


def stock_details(
    parts: StrategyParts, weights: pd.DataFrame, panel: Panel, settings: Settings,
    session: pd.Timestamp, raw_close: pd.Series,
) -> pd.DataFrame:
    """Per stock at T: target weight, the rule holding it, binding limits, indicators, flags.

    Limits are attributed in the order they are applied (section 19.1): the position cap
    binds when 1 / (number held) is above it; the sector cap when the capped stocks of a
    sector add up to more than it; the exposure cap when the invested total after the regime
    exposure is above it.
    """
    risk = settings.risk
    stocks = list(weights.columns)
    sector = pd.Series(settings.market.sectors).reindex(stocks)
    rule = pd.Series(None, index=stocks, dtype=object)
    for name, frame in parts.rules.items():
        rule[frame.loc[session].astype(bool).to_numpy()] = name
    held = rule.notna()

    count = int(held.sum())
    equal = 1.0 / count if count else 0.0
    capped = held * min(equal, risk.max_position_size)
    sector_total = capped.groupby(sector).transform("sum")
    after_sector = capped * (risk.max_sector_weight / sector_total.where(sector_total > 0)).clip(upper=1.0).fillna(1.0)
    exposure = float(parts.exposure.loc[session])

    frame = pd.DataFrame(index=pd.Index(stocks, name="symbol"))
    frame["sector"] = sector
    frame["target_weight"] = weights.loc[session].fillna(0.0)
    frame["rule"] = rule
    frame["held_sessions"] = _held_sessions(parts.holdings.loc[:session]).where(held, 0)
    frame["position_cap"] = held & (equal > risk.max_position_size)
    frame["sector_cap"] = held & (sector_total > risk.max_sector_weight + 1e-12)
    frame["exposure_cap"] = held & ((after_sector.sum() * exposure) > risk.max_portfolio_exposure + 1e-12)
    for column in TARGET_COLUMNS:
        frame[column] = panel[column].loc[session].reindex(stocks) if session in panel[column].index else np.nan
    frame["macd_hist_prev"] = panel["macd_hist_over_price"].loc[:session].shift(1).iloc[-1].reindex(stocks)
    frame["has_data"] = frame["price_vs_sma200"].notna()
    frame["close"] = raw_close.reindex(stocks)
    frame["stop_price"] = frame["close"] * (1 - risk.atr_stop_multiple * frame["atr14_over_price"])
    frame["high_volatility"] = frame["realized_volatility_20d"] > risk.high_volatility
    return frame


def compute_targets(
    panel: Panel, regimes: pd.Series, settings: Settings, raw_closes: pd.DataFrame,
    strategy: str | None = None, regime_source: str | None = None,
) -> Targets:
    """Targets at the last session with a regime (T), from data at or before T only."""
    if regimes.empty:
        raise ValueError("no stored regimes; run calculate-features and calculate-regime first")
    strategy = strategy or settings.signals.strategy
    session = regimes.index[-1]
    reference = panel["price_vs_sma200"]
    if session not in reference.index:
        raise ValueError(f"no features for {session.date()}, the latest regime; run calculate-features first")
    panel = {name: frame.loc[:session] for name, frame in panel.items()}
    regimes = regimes.loc[:session]
    parts = strategy_parts(strategy, panel, regimes, settings)
    weights = parts_weights(parts, settings)
    raw_close = raw_closes.loc[:session].ffill().iloc[-1] if not raw_closes.empty else pd.Series(dtype=float)
    return Targets(
        session=session,
        strategy=strategy,
        regime=regimes.iloc[-1],
        regime_source=regime_source or settings.regime.rule_version,
        regime_exposure=float(parts.exposure.loc[session]),
        stocks=stock_details(parts, weights, panel, settings, session, raw_close),
    )


def load_targets(engine: Engine, settings: Settings) -> Targets:
    """Targets from the stored features and regime. The regime is the ACTIVE REGIME model's;
    with none ACTIVE (the case until `quant promote` exists) it is the rule-based regime."""
    active = queries.active_model(engine, REGIME)
    if active is not None:
        raise NotImplementedError(
            f"an ACTIVE regime model ({active['model_id']}) is not supported by the signal engine yet; "
            "signals use the rule-based regime"
        )
    stocks = list(settings.market.stocks)
    panel = queries.load_feature_panel(engine, settings.features.feature_version, stocks, TARGET_COLUMNS)
    regimes = queries.load_regimes(engine, RULE_METHOD, settings.regime.rule_version)
    since = None if regimes.empty else (regimes.index[-1] - pd.Timedelta(days=30)).date()
    raw_closes = queries.load_raw_closes(engine, stocks, since=since)
    return compute_targets(panel, regimes, settings, raw_closes)
