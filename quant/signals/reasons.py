"""Human-readable reason per signal (proposal sections 6.9 and 18.1).

Built only from the factors that decided the weight, in a fixed order:

1. the regime and where it came from;
2. the strategy rule that holds the stock (with its indicator values), or why it is not held;
3. each risk limit that bound;
4. the informational risk flags (volatility, ATR stop);
5. ``target x% vs current y% -> SIGNAL`` with the suggested shares.
"""

from __future__ import annotations

import pandas as pd

from quant.config.settings import Settings
from quant.regime.rule_based import BEAR, BULL, SIDEWAYS
from quant.signals.engine import BUY, EXIT, HOLD, REDUCE, Signals
from quant.signals.targets import Targets

REGIME_NAMES = {BULL: "Bull", SIDEWAYS: "Sideways", BEAR: "Bear"}
TREND, MEAN_REVERSION = "trend", "mean_reversion"


def _pct(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}%}"


def _signed(value: float) -> str:
    return f"{value:+.0%}"


def active_rule(targets: Targets) -> str | None:
    """Which rule decides membership at T: trend, mean_reversion, or None (no regime)."""
    strategy, regime = targets.strategy, targets.regime
    if strategy == TREND:
        return TREND
    if strategy == MEAN_REVERSION:
        return MEAN_REVERSION
    if regime in (BULL, BEAR):
        return TREND
    if regime == SIDEWAYS:
        return MEAN_REVERSION
    return None


def regime_sentence(targets: Targets) -> str:
    name = REGIME_NAMES.get(targets.regime, "No")
    source = f"({targets.regime_source})"
    if targets.strategy in (TREND, MEAN_REVERSION):
        return f"{name} regime {source}; {targets.strategy} ignores the regime."
    rule = {BULL: "trend rule", SIDEWAYS: "dip-buy rule", BEAR: "trend rule"}.get(targets.regime)
    if targets.regime == BEAR:
        return f"{name} regime {source}: {rule} at {_pct(targets.regime_exposure, 0)} exposure."
    if rule is None:
        return f"No regime at {targets.session.date()}: no positions."
    return f"{name} regime {source}: {rule}."


def rule_sentence(stock: pd.Series, targets: Targets, settings: Settings) -> str:
    """Why the strategy holds the stock, or why it does not."""
    reversion = settings.strategies.mean_reversion
    price, sma = stock["price_vs_sma200"], stock["sma50_vs_sma200"]
    if not bool(stock["has_data"]):
        return "No indicator data at T."
    if stock["rule"] == TREND:
        held = int(stock["held_sessions"])
        since = "entered today (MACD histogram positive)" if held == 1 else f"in the strategy for {held} sessions"
        return f"Trend: price {_signed(price)} vs SMA200, SMA50 {_signed(sma)} vs SMA200; {since}."
    if stock["rule"] == MEAN_REVERSION:
        held = int(stock["held_sessions"])
        return (f"Dip buy: RSI {stock['rsi14']:.0f} (entry < {reversion.rsi_entry:g}, exit > {reversion.rsi_exit:g}); "
                f"day {held} of {reversion.max_holding_days}.")
    rule = active_rule(targets)
    if rule == TREND:
        if price <= 0:
            return f"Trend broken: price {_signed(price)} vs SMA200."
        if sma <= 0:
            return f"Trend broken: SMA50 {_signed(sma)} vs SMA200."
        return "In an uptrend but not entered: waiting for the MACD histogram to turn positive."
    if rule == MEAN_REVERSION:
        prefix = "Sideways regime holds dip buys only. " if targets.strategy not in (TREND, MEAN_REVERSION) else ""
        if price <= 0:
            return f"{prefix}Not a dip buy: price {_signed(price)} vs SMA200 (needs an uptrend)."
        if stock["rsi14"] >= reversion.rsi_entry:
            return f"{prefix}Not a dip buy: RSI {stock['rsi14']:.0f} (entry < {reversion.rsi_entry:g})."
        return f"{prefix}Not a dip buy: RSI {stock['rsi14']:.0f} but the MACD histogram is not rising."
    return "No regime: not held."


def limit_sentences(stock: pd.Series, targets: Targets, settings: Settings) -> list[str]:
    risk = settings.risk
    if pd.isna(stock["rule"]) or stock["target_weight"] <= 0:
        return []
    held = int(targets.stocks["rule"].notna().sum())
    out = []
    if bool(stock["position_cap"]):
        out.append(f"Equal weight 1/{held} capped at {_pct(risk.max_position_size, 0)} per stock.")
    else:
        out.append(f"Equal weight 1/{held}.")
    if bool(stock["sector_cap"]):
        sector = str(stock["sector"]).replace("_", " ")
        out.append(f"Sector cap {_pct(risk.max_sector_weight, 0)} binds ({sector}).")
    if bool(stock["exposure_cap"]):
        out.append(f"Portfolio exposure cap {_pct(risk.max_portfolio_exposure, 0)} binds.")
    return out


def flag_sentences(stock: pd.Series, settings: Settings, signal: str) -> list[str]:
    risk = settings.risk
    if signal == EXIT or not bool(stock["has_data"]):
        return []
    out = []
    volatility = stock["realized_volatility_20d"]
    if pd.notna(volatility):
        if bool(stock["high_volatility"]):
            out.append(f"High volatility: 20-day {_pct(volatility, 0)} (above {_pct(risk.high_volatility, 0)}).")
        else:
            out.append(f"Volatility {_pct(volatility, 0)} within limit.")
    if pd.notna(stock["stop_price"]):
        out.append(f"ATR stop {stock['stop_price']:.2f} ({risk.atr_stop_multiple:g} x ATR below {stock['close']:.2f}).")
    return out


def action_sentence(row: pd.Series, hold_band: float) -> str:
    signal, shares = row["signal"], row["suggested_shares"]
    head = f"Target {_pct(row['target_weight'])} vs current {_pct(row['current_weight'])} -> {signal}"
    if row["data_gap"]:
        return f"{head} (no data at T: kept, check the data)."
    if signal == HOLD:
        return f"{head} (change {row['change'] * 100:+.1f} points, inside the {hold_band * 100:g}-point band)."
    count = f"{abs(shares):g} share{'s' if abs(shares) != 1 else ''}"
    if row["below_one_share"]:
        return f"{head} (less than one share at this portfolio value; 0 suggested)."
    if signal == BUY:
        kind = "new position" if row["current_weight"] <= 0 else "increase"
        return f"{head} ({kind}; buy {count})."
    if signal == REDUCE:
        return f"{head} (sell {count})."
    return f"{head} (sell all {count})."


def signal_reason(row: pd.Series, stock: pd.Series, targets: Targets, settings: Settings, hold_band: float) -> str:
    parts = [regime_sentence(targets), rule_sentence(stock, targets, settings)]
    if stock["rule"] in (TREND, MEAN_REVERSION) and stock["target_weight"] <= 0 and targets.regime_exposure == 0:
        parts.append("Bear exposure 0%: held in cash.")
    parts += limit_sentences(stock, targets, settings)
    parts += flag_sentences(stock, settings, row["signal"])
    parts.append(action_sentence(row, hold_band))
    return " ".join(parts)


def with_reasons(signals: Signals, settings: Settings) -> Signals:
    """The same signals with a ``reason`` column."""
    stocks = signals.targets.stocks
    reasons = [
        signal_reason(row, stocks.loc[symbol], signals.targets, settings, signals.hold_band)
        for symbol, row in signals.rows.iterrows()
    ]
    rows = signals.rows.assign(reason=pd.Series(reasons, index=signals.rows.index, dtype=object))
    return Signals(signals.session, signals.targets, signals.holdings, signals.valuation, signals.hold_band,
                   rows, signals.warnings)
