"""Target vs current weight -> BUY / HOLD / REDUCE / EXIT (proposal section 18.1).

With hold band h (signals.hold_band):

    current 0, target > 0                  -> BUY (new position; always trades)
    current > 0, target 0                  -> EXIT (always trades)
    both > 0, target - current >= h        -> BUY (increase)
    both > 0, current - target >= h        -> REDUCE (to target)
    both > 0, |target - current| < h       -> HOLD
    current 0, target 0                    -> no signal

A held stock without data at T gets HOLD with a warning instead of EXIT: a missing input
never counts as a signal, and a gap in today's data is more likely a data problem than a
trend break. Suggested shares are an estimate at the close of T; the order fills at the
open of T+1 (section 4.5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
from sqlalchemy import Engine

from quant.config.settings import Settings
from quant.persistence import queries
from quant.signals.positions import Holdings, Valuation, read_positions, value_holdings
from quant.signals.targets import Targets, load_targets

BUY, HOLD, REDUCE, EXIT = "BUY", "HOLD", "REDUCE", "EXIT"
SIGNAL_ORDER = (EXIT, REDUCE, BUY, HOLD)  # sells first: they fund the buys
TOLERANCE = 1e-9  # weights are sums of floats; 0.10 - 0.08 must count as a 2-point change


def classify(target: float, current: float, hold_band: float) -> str | None:
    """The signal for one stock, or None when it is neither held nor wanted."""
    held, wanted = current > TOLERANCE, target > TOLERANCE
    if not held:
        return BUY if wanted else None
    if not wanted:
        return EXIT
    change = target - current
    threshold = max(hold_band - TOLERANCE, TOLERANCE)  # a zero band still holds an unchanged weight
    if change >= threshold:
        return BUY
    if -change >= threshold:
        return REDUCE
    return HOLD


@dataclass(frozen=True)
class Signals:
    session: pd.Timestamp
    targets: Targets
    holdings: Holdings
    valuation: Valuation
    hold_band: float
    rows: pd.DataFrame  # one row per stock with a signal, in SIGNAL_ORDER then by weight
    warnings: tuple[str, ...]

    @property
    def portfolio_value(self) -> float:
        return self.valuation.total

    @property
    def counts(self) -> dict[str, int]:
        return {signal: int((self.rows["signal"] == signal).sum()) for signal in SIGNAL_ORDER}

    @property
    def estimated_cash_after(self) -> float:
        """Cash after the suggested trades at the close of T (before costs)."""
        return self.valuation.cash - float(self.rows["trade_value"].sum())


def _shares(signal: str, change: float, held: float, value: float, close: float) -> float:
    if signal == EXIT:
        return -held  # sell everything, including a fractional remainder
    if signal == HOLD or not close or not math.isfinite(close):
        return 0.0
    shares = math.floor(abs(change) * value / close + TOLERANCE)  # 0.15 - 0.10 is 0.0499...
    return float(shares if signal == BUY else -min(shares, held))


def build_signals(targets: Targets, holdings: Holdings, valuation: Valuation, hold_band: float) -> Signals:
    stocks = targets.stocks
    current = valuation.weights.reindex(stocks.index).fillna(0.0)
    quantity = pd.Series(dict(holdings.quantities), dtype=float).reindex(stocks.index).fillna(0.0)
    value = valuation.total
    warnings = list(holdings.warnings) + list(valuation.warnings)

    records = []
    for symbol, row in stocks.iterrows():
        target = float(row["target_weight"])
        signal = classify(target, float(current[symbol]), hold_band)
        if signal is None:
            continue
        no_data = signal == EXIT and not bool(row["has_data"])
        if no_data:
            signal = HOLD
            warnings.append(f"{symbol}: held but no data at {targets.session.date()}; kept (HOLD) - check the data")
        close = float(valuation.closes.get(symbol, row["close"]))
        shares = _shares(signal, target - current[symbol], quantity[symbol], value, close)
        records.append({
            "symbol": symbol,
            "signal": signal,
            "target_weight": target,
            "current_weight": float(current[symbol]),
            "change": target - float(current[symbol]),
            "quantity": float(quantity[symbol]),
            "close": close,
            "suggested_shares": shares,
            "trade_value": shares * close if math.isfinite(close) else 0.0,
            "below_one_share": signal in (BUY, REDUCE) and shares == 0,
            "data_gap": no_data,
        })
    rows = pd.DataFrame(records, columns=[
        "symbol", "signal", "target_weight", "current_weight", "change", "quantity", "close",
        "suggested_shares", "trade_value", "below_one_share", "data_gap",
    ])
    if not rows.empty:
        rank = rows["signal"].map({signal: i for i, signal in enumerate(SIGNAL_ORDER)})
        rows = rows.assign(_rank=rank).sort_values(["_rank", "target_weight", "symbol"], ascending=[True, False, True])
        rows = rows.drop(columns="_rank")
    for symbol in rows.loc[rows["below_one_share"], "symbol"]:
        warnings.append(f"{symbol}: the change is less than one share at this portfolio value; no shares suggested")
    return Signals(targets.session, targets, holdings, valuation, hold_band, rows.set_index("symbol"), tuple(warnings))


def load_signals(engine: Engine, settings: Settings) -> Signals:
    """Targets at the latest session T against the positions file, valued at the close of T,
    each signal with its reason."""
    from quant.signals.reasons import with_reasons  # reasons builds on this module

    targets = load_targets(engine, settings)
    holdings = read_positions(settings.positions.path, settings.market.stocks, settings.backtest.initial_cash)
    since = (targets.session - pd.Timedelta(days=30)).date()
    closes = queries.load_raw_closes(engine, list(holdings.quantities), since=since) if holdings.quantities else pd.DataFrame()
    valuation = value_holdings(holdings, closes, targets.session)
    return with_reasons(build_signals(targets, holdings, valuation, settings.signals.hold_band), settings)
