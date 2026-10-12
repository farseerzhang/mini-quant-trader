"""Paper portfolios of the forward test (Phase 10 pre-registration).

Each portfolio (``live``, every shadow, and ``SPY``) starts from the recorded start value in cash
on the start session and trades only its frozen targets, with the backtest engine: a target
frozen at the close of T fills at the open of T+1, with the backtest's fees, slippage and
minimum trade size. A missed session (not frozen) keeps the previous targets, so nothing is
traded that day; the portfolio is still valued every session. Values use the same total-return
prices as the backtest. Derived data: recomputed and replaced on every run.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import pandas as pd
from sqlalchemy import Engine

from quant.backtest.data import load_prices
from quant.backtest.engine import Simulation, simulate
from quant.backtest.orders import buy_and_hold, order_targets
from quant.config.settings import Settings
from quant.persistence import queries

BENCHMARK = "SPY"


@dataclass(frozen=True)
class PaperResult:
    started_on: pd.Timestamp
    start_value: float
    sessions: pd.DatetimeIndex  # every session from the start to the latest stored one
    simulations: dict[str, Simulation]
    missed: dict[str, list[pd.Timestamp]]  # sessions without a freeze, per portfolio


def paper_targets(frozen: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Frozen targets (frozen sessions x symbols) on every session; a missed one keeps the last."""
    return frozen.reindex(sessions).ffill().fillna(0.0)


def run_paper(engine: Engine, settings: Settings) -> PaperResult | None:
    """Simulate every portfolio from the start; None before the first freeze."""
    start = queries.forward_start(engine)
    if start is None:
        return None
    started_on = pd.Timestamp(start["started_on"])
    portfolios = ["live", *settings.forward_test.shadows]
    frozen = {name: queries.load_forward_targets(engine, name) for name in portfolios}
    symbols = sorted({s for frame in frozen.values() for s in frame.columns} | {BENCHMARK})
    prices = load_prices(engine, symbols)
    sessions = prices.close.index[prices.close.index >= started_on]
    bt = dataclasses.replace(settings.backtest, initial_cash=float(start["start_value"]))

    simulations, missed = {}, {}
    for name, targets in frozen.items():
        weights = paper_targets(targets, sessions).reindex(columns=symbols, fill_value=0.0)
        simulations[name] = simulate(order_targets(weights, bt.min_trade_weight), prices, bt)
        missed[name] = [s for s in sessions if s not in targets.index]
    simulations[BENCHMARK] = simulate(buy_and_hold(sessions, BENCHMARK), prices, bt)
    return PaperResult(started_on, float(start["start_value"]), sessions, simulations, missed)


def update_paper_equity(engine: Engine, settings: Settings) -> PaperResult | None:
    result = run_paper(engine, settings)
    if result is None:
        return None
    rows = [
        {"date": date.date(), "portfolio": name, "value": float(value),
         "invested": float(simulation.invested.loc[date])}
        for name, simulation in result.simulations.items()
        for date, value in simulation.value.items()
    ]
    from quant.forward.real import real_portfolio

    real = real_portfolio(engine, settings, result.sessions)
    if real is not None:
        _, _, valued, _ = real
        rows += [{"date": date.date(), "portfolio": "real", "value": float(r["value"]),
                  "invested": None if pd.isna(r["invested"]) else float(r["invested"])}
                 for date, r in valued.iterrows()]
    queries.replace_forward_equity(engine, rows)
    return result
