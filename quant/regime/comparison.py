"""Trade the same strategies with different regimes (proposal sections 13 and 16.1).

The REGIME role is judged by what it does for the strategies downstream: each
regime-dependent strategy is run once per regime (the strategy code, its
settings and the costs are identical; only the regime series differs) over the
same window, from fresh cash. Benchmarks that do not use a regime run once.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant.backtest.data import Prices
from quant.backtest.engine import Simulation, simulate
from quant.backtest.metrics import METRIC_NAMES, period_metrics
from quant.backtest.orders import buy_and_hold, equal_weight_monthly, order_targets
from quant.config.settings import Settings
from quant.models.folds import Fold
from quant.strategies import build_strategies
from quant.strategies.rules import Panel

REGIME_STRATEGIES = ("regime_switch", "regime_switch_bear50")


@dataclass(frozen=True)
class RegimeComparison:
    start: pd.Timestamp
    end: pd.Timestamp
    simulations: dict[str, Simulation]
    overall: pd.DataFrame  # portfolio x metrics
    by_fold: pd.DataFrame  # (fold, portfolio) -> sharpe, total_return, max_drawdown

    def versus(self, candidate: str, baseline: str) -> dict[str, float]:
        sharpe = self.by_fold["sharpe"].unstack()
        return {
            "sharpe": float(self.overall.loc[candidate, "sharpe"]),
            "baseline_sharpe": float(self.overall.loc[baseline, "sharpe"]),
            "max_drawdown": float(self.overall.loc[candidate, "max_drawdown"]),
            "baseline_max_drawdown": float(self.overall.loc[baseline, "max_drawdown"]),
            "folds": int(len(sharpe)),
            "folds_higher_sharpe": int((sharpe[candidate] > sharpe[baseline]).sum()),
        }


def portfolio_name(strategy: str, regime_name: str) -> str:
    return f"{strategy} [{regime_name}]"


def compare_regimes(
    regimes: dict[str, pd.Series],
    folds: list[Fold],
    prices: Prices,
    panel: Panel,
    settings: Settings,
) -> RegimeComparison:
    """``regimes`` maps a name (e.g. 'rule', 'hmm') to a regime per session covering the window."""
    start, end = folds[0].validation_start, folds[-1].validation_end
    index = prices.close.index[(prices.close.index >= start) & (prices.close.index <= end)]
    bt = settings.backtest
    strategies = {s.name: s for s in build_strategies(settings) if s.name in REGIME_STRATEGIES}

    simulations: dict[str, Simulation] = {}
    for strategy_name, strategy in strategies.items():
        for regime_name, regime in regimes.items():
            missing = index.difference(regime.dropna().index)
            if len(missing):
                raise ValueError(f"regime {regime_name!r} has no value on {len(missing)} session(s) of the window")
            weights = strategy.weights(panel, regime).reindex(index).fillna(0.0)
            simulations[portfolio_name(strategy_name, regime_name)] = simulate(
                order_targets(weights, bt.min_trade_weight), prices, bt
            )
    stocks = list(settings.market.stocks)
    simulations["SPY"] = simulate(buy_and_hold(index, "SPY"), prices, bt)
    simulations["equal_weight"] = simulate(equal_weight_monthly(index, stocks), prices, bt)

    def metrics(sim: Simulation, first, last):
        return period_metrics(sim.value, sim.invested, sim.orders, sim.trades, first, last)

    overall = pd.DataFrame({name: metrics(sim, start, end) for name, sim in simulations.items()}).T
    rows = []
    for fold in folds:
        for name, sim in simulations.items():
            m = metrics(sim, fold.validation_start, fold.validation_end)
            rows.append({"fold": fold.number, "portfolio": name, **{k: m[k] for k in ("sharpe", "total_return", "max_drawdown")}})
    return RegimeComparison(
        start=start,
        end=end,
        simulations=simulations,
        overall=overall.loc[:, list(METRIC_NAMES)],
        by_fold=pd.DataFrame(rows).set_index(["fold", "portfolio"]),
    )
