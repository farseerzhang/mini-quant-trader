"""Portfolio backtests of the rule-based strategies against the baselines (proposal section 20).

Execution: target weights decided at the close of T are placed as orders at the
open of T+1 (total-return open), with fees and slippage per side, cash shared
across stocks and sells before buys. The portfolio is valued at the close.

Every portfolio is simulated over the whole history once. Metrics are then
reported separately for the development period and for the holdout (the most
recent ``retrain.validation.holdout_months``); the holdout is never used to
choose rules or parameters (section 15.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import vectorbt as vbt
from sqlalchemy import Engine

from quant.backtest.data import Prices, load_prices
from quant.backtest.metrics import METRIC_NAMES, period_metrics, regime_returns, yearly_returns
from quant.backtest.orders import (
    buy_and_hold,
    equal_weight_monthly,
    execution_sizes,
    order_targets,
    permuted_weights,
)
from quant.config.settings import BacktestSettings, Settings
from quant.persistence import queries
from quant.regime.pipeline import RULE_METHOD
from quant.strategies import PANEL_COLUMNS, Strategy, build_strategies

STRATEGY = "STRATEGY"
BENCHMARK = "BENCHMARK"
_RANDOM_BATCH = 25  # random-baseline portfolios simulated per VectorBT call
_EMPTY_ORDERS = pd.DataFrame(columns=["Timestamp", "Size", "Price", "Fees"])
_EMPTY_TRADES = pd.DataFrame(columns=["Exit Timestamp", "PnL", "Status"])


@dataclass(frozen=True)
class Periods:
    start: pd.Timestamp
    development_end: pd.Timestamp
    holdout_start: pd.Timestamp
    end: pd.Timestamp


@dataclass(frozen=True)
class Simulation:
    value: pd.Series
    invested: pd.Series
    orders: pd.DataFrame
    trades: pd.DataFrame


@dataclass(frozen=True)
class PortfolioResult:
    name: str
    kind: str
    description: str
    value: pd.Series
    invested: pd.Series
    development: dict[str, float]
    holdout: dict[str, float]
    yearly: pd.Series
    by_regime: pd.DataFrame
    # Development-period metrics of each random-baseline run (strategies only).
    random_baseline: pd.DataFrame | None = None


@dataclass(frozen=True)
class BacktestReport:
    periods: Periods
    results: list[PortfolioResult] = field(default_factory=list)

    def table(self, period: str) -> pd.DataFrame:
        """One row per portfolio, metrics as columns; period is 'development' or 'holdout'."""
        if period not in ("development", "holdout"):
            raise ValueError("period must be 'development' or 'holdout'")
        return pd.DataFrame(
            {result.name: getattr(result, period) for result in self.results}
        ).T.loc[:, list(METRIC_NAMES)]

    def random_baseline_summary(self) -> pd.DataFrame:
        """Per strategy: its development Sharpe and CAGR against the random runs."""
        rows = {}
        for result in self.results:
            runs = result.random_baseline
            if runs is None or runs.empty:
                continue
            rows[result.name] = {
                "sharpe": result.development["sharpe"],
                "random_sharpe_median": float(runs["sharpe"].median()),
                "sharpe_percentile": float((runs["sharpe"] < result.development["sharpe"]).mean()),
                "cagr": result.development["cagr"],
                "random_cagr_median": float(runs["cagr"].median()),
                "cagr_percentile": float((runs["cagr"] < result.development["cagr"]).mean()),
                "runs": int(len(runs)),
            }
        return pd.DataFrame(rows).T


def split_periods(index: pd.DatetimeIndex, holdout_months: int) -> Periods:
    """Development = everything before the holdout; holdout = sessions in the last
    ``holdout_months`` months up to the last session."""
    end = index[-1]
    holdout = index[index > end - pd.DateOffset(months=holdout_months)]
    development = index[index < holdout[0]]
    if len(development) == 0:
        raise ValueError("the history is shorter than the holdout; nothing left for development")
    return Periods(start=index[0], development_end=development[-1], holdout_start=holdout[0], end=end)


def _from_orders(targets: pd.DataFrame, prices: Prices, settings: BacktestSettings, group_by) -> vbt.Portfolio:
    symbols = targets.columns.get_level_values(-1)
    close = prices.close.loc[targets.index, symbols].to_numpy()
    fill = prices.open.loc[targets.index, symbols].to_numpy()
    as_frame = lambda values: pd.DataFrame(values, index=targets.index, columns=targets.columns)  # noqa: E731
    return vbt.Portfolio.from_orders(
        close=as_frame(close),
        size=execution_sizes(targets),
        size_type="targetpercent",
        price=as_frame(fill),
        fees=settings.fees,
        slippage=settings.slippage,
        init_cash=settings.initial_cash,
        cash_sharing=True,
        group_by=group_by,
        call_seq="auto",
    )


def simulate(targets: pd.DataFrame, prices: Prices, settings: BacktestSettings) -> Simulation:
    """Run one portfolio from order targets (dates x symbols, NaN = no order)."""
    portfolio = _from_orders(targets, prices, settings, group_by=True)
    value = portfolio.value()
    invested = 1.0 - portfolio.cash() / value
    orders = portfolio.orders.records_readable
    trades = portfolio.trades.records_readable
    return Simulation(
        value=value,
        invested=invested,
        orders=orders if len(orders) else _EMPTY_ORDERS,
        trades=trades if len(trades) else _EMPTY_TRADES,
    )


def random_baseline(
    weights: pd.DataFrame, prices: Prices, settings: BacktestSettings, periods: Periods
) -> pd.DataFrame:
    """Development metrics of ``random_baseline_runs`` portfolios that keep the strategy's
    timing and sizing but assign its weights to randomly permuted stocks."""
    rng = np.random.default_rng(settings.random_seed)
    runs = settings.random_baseline_runs
    rows = []
    for first in range(0, runs, _RANDOM_BATCH):
        batch = range(first, min(first + _RANDOM_BATCH, runs))
        frames = {
            run: order_targets(permuted_weights(weights, rng), settings.min_trade_weight) for run in batch
        }
        targets = pd.concat(frames, axis=1, names=["run", "symbol"])
        values = _from_orders(targets, prices, settings, group_by=targets.columns.get_level_values("run")).value()
        for run in batch:
            metrics = period_metrics(
                values[run], pd.Series(np.nan, index=values.index), _EMPTY_ORDERS, _EMPTY_TRADES,
                periods.start, periods.development_end,
            )
            rows.append({"run": run, **{k: metrics[k] for k in ("total_return", "cagr", "sharpe", "max_drawdown")}})
    return pd.DataFrame(rows).set_index("run") if rows else pd.DataFrame()


def _result(name, kind, description, simulation: Simulation, periods: Periods, regimes, random=None):
    common = (simulation.value, simulation.invested, simulation.orders, simulation.trades)
    return PortfolioResult(
        name=name,
        kind=kind,
        description=description,
        value=simulation.value,
        invested=simulation.invested,
        development=period_metrics(*common, periods.start, periods.development_end),
        holdout=period_metrics(*common, periods.holdout_start, periods.end),
        yearly=yearly_returns(simulation.value, periods.start, periods.development_end),
        by_regime=regime_returns(simulation.value, regimes, periods.start, periods.development_end),
        random_baseline=random,
    )


def run_backtests(
    engine: Engine, settings: Settings, strategies: list[Strategy] | None = None
) -> BacktestReport:
    """Backtest the strategies and the configured benchmarks over the stored history.

    By default the strategies are ``build_strategies(settings)``: trend, mean_reversion,
    regime_switch and regime_switch_bear50, all with the **rule-based** regime. ML strategies
    (ml_select) and the HMM regime are backtested by ``quant retrain`` instead; see the
    overview in quant/strategies/__init__.py.

    The common start is the first session with a stored regime, which is also the
    first session every strategy input (SMA200-based) is available.
    """
    stocks = list(settings.market.stocks)
    panel = queries.load_feature_panel(engine, settings.features.feature_version, stocks, PANEL_COLUMNS)
    regimes = queries.load_regimes(engine, RULE_METHOD, settings.regime.rule_version)
    if regimes.empty:
        raise ValueError("no stored regimes; run calculate-features and calculate-regime first")
    prices = load_prices(engine, [*stocks, "SPY"])
    index = prices.close.index[prices.close.index >= regimes.index[0]]
    periods = split_periods(index, settings.holdout_months)
    bt = settings.backtest

    results = []
    for strategy in strategies if strategies is not None else build_strategies(settings):
        weights = strategy.weights(panel, regimes).reindex(index).fillna(0.0)
        simulation = simulate(order_targets(weights, bt.min_trade_weight), prices, bt)
        random = random_baseline(weights, prices, bt, periods) if bt.random_baseline_runs else None
        results.append(_result(strategy.name, STRATEGY, strategy.description, simulation, periods, regimes, random))

    if "SPY" in bt.benchmarks:
        simulation = simulate(buy_and_hold(index, "SPY"), prices, bt)
        results.append(_result("SPY", BENCHMARK, "SPY buy-and-hold", simulation, periods, regimes))
    if "equal_weight" in bt.benchmarks:
        simulation = simulate(equal_weight_monthly(index, stocks), prices, bt)
        results.append(
            _result("equal_weight", BENCHMARK, f"all {len(stocks)} pool stocks, 1/N, rebalanced monthly",
                    simulation, periods, regimes)
        )
    return BacktestReport(periods=periods, results=results)
