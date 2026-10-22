"""What the strategy trials of proposal section 17.1.5 share: the window, the baseline, the
comparison and the four pass criteria.

- Window: the walk-forward validation window from the fold rule at run time (holdout untouched).
- Baseline: `regime_switch` (bear exposure 0) with the stored rule_based-v1 regime on the pool.
- Every portfolio starts from fresh cash on the window's first session, with the same costs.
- Pass criteria (all must hold): Sharpe after costs higher than the baseline; max drawdown no
  deeper; a higher Sharpe in at least 6 of the validation years; targets point in time.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sqlalchemy import Engine

from quant.backtest.data import Prices
from quant.backtest.engine import Simulation, simulate, split_periods
from quant.backtest.metrics import METRIC_NAMES, period_metrics
from quant.backtest.orders import order_targets
from quant.config.settings import Settings
from quant.features.technical import prepare_adjusted_series
from quant.models.folds import Fold, walk_forward_folds
from quant.models.gates import FAIL, PASS, GateReport, GateResult
from quant.models.report import percent
from quant.persistence import queries
from quant.regime.comparison import RegimeComparison, compare_regimes, portfolio_name
from quant.regime.pipeline import RULE_METHOD
from quant.strategies.rules import Panel

_NO_ACTIONS = pd.DataFrame(columns=["ex_date", "action_type", "value"])
BASELINE_REGIME = "rule_based-v1"
BASELINE = portfolio_name("regime_switch", BASELINE_REGIME)
MIN_YEARS_HIGHER = 6


def total_return_closes(stored: dict[str, pd.DataFrame], actions: dict[str, pd.DataFrame], symbols: list[str],
                        until: pd.Timestamp | None = None) -> pd.DataFrame:
    """Total-return closes (sessions x symbols) from the stored rows, optionally from data up to ``until``
    only (prices and corporate actions alike), as they would have been computed on that evening. Used by
    the point-in-time checks."""
    closes = {}
    for symbol in symbols:
        if symbol not in stored:
            raise ValueError(f"no stored prices for {symbol}; update market data first")
        frame = stored[symbol]
        symbol_actions = actions.get(symbol, _NO_ACTIONS)
        if until is not None:
            frame = frame.loc[:until]
            if not symbol_actions.empty:
                symbol_actions = symbol_actions[pd.to_datetime(symbol_actions["ex_date"]) <= until]
        if frame.empty:  # listed after ``until``: no closes yet, as on that evening
            closes[symbol] = pd.Series(dtype=float)
            continue
        closes[symbol] = prepare_adjusted_series(frame, symbol_actions)["total_return_close"]
    return pd.DataFrame(closes)


def trial_folds(engine: Engine, settings: Settings) -> list[Fold]:
    """The yearly validation folds from the fold rule, as in the regime trials (sections 12.5, 12.7)."""
    from quant.models.dataset import load_dataset

    dataset = load_dataset(engine, settings)
    holdout_start = split_periods(dataset.sessions, settings.holdout_months).holdout_start
    return walk_forward_folds(dataset.labelled().sessions, holdout_start, settings.retrain.validation,
                              settings.model.embargo_days)


def baseline_regime(engine: Engine, settings: Settings) -> pd.Series:
    """The stored rule_based-v1 regime, which the live signals use."""
    if settings.regime.rule_version != BASELINE_REGIME:
        raise ValueError(f"the registered baseline is {BASELINE_REGIME}, but regime.rule_version is "
                         f"{settings.regime.rule_version}")
    regime = queries.load_regimes(engine, RULE_METHOD, BASELINE_REGIME)
    if regime.empty:
        raise ValueError(f"no stored {BASELINE_REGIME} regime; run `quant calculate-regime`")
    return regime


def _metrics(simulation: Simulation, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    return period_metrics(simulation.value, simulation.invested, simulation.orders, simulation.trades, start, end)


def compare_with_baseline(candidates: dict[str, pd.DataFrame], folds: list[Fold], prices: Prices, panel: Panel,
                          regime: pd.Series, settings: Settings) -> RegimeComparison:
    """The baseline (and bear50, SPY, equal weight, as `compare_regimes`) plus each candidate,
    given as target weights (sessions x symbols), over the same window from fresh cash."""
    base = compare_regimes({BASELINE_REGIME: regime}, folds, prices, panel, settings)
    index = prices.close.index[(prices.close.index >= base.start) & (prices.close.index <= base.end)]
    bt = settings.backtest
    simulations = dict(base.simulations)
    overall = base.overall.copy()
    rows = [base.by_fold.reset_index()]
    for name, weights in candidates.items():
        targets = order_targets(weights.reindex(index).fillna(0.0), bt.min_trade_weight)
        simulation = simulate(targets, prices, bt)
        simulations[name] = simulation
        overall.loc[name] = pd.Series(_metrics(simulation, base.start, base.end)).reindex(overall.columns)
        for fold in folds:
            m = _metrics(simulation, fold.validation_start, fold.validation_end)
            rows.append(pd.DataFrame([{"fold": fold.number, "portfolio": name,
                                       **{k: m[k] for k in ("sharpe", "total_return", "max_drawdown")}}]))
    by_fold = pd.concat(rows, ignore_index=True).set_index(["fold", "portfolio"])
    return RegimeComparison(base.start, base.end, simulations, overall.loc[:, list(METRIC_NAMES)], by_fold)


@dataclass(frozen=True)
class PointInTimeCheck:
    cut: pd.Timestamp
    decisions: int  # decisions up to the cut that were compared
    identical: bool


def strategy_gates(comparison: RegimeComparison, candidate: str, checks: list[PointInTimeCheck]) -> GateReport:
    v = comparison.versus(candidate, BASELINE)
    return GateReport([
        GateResult("sharpe_higher", PASS if v["sharpe"] > v["baseline_sharpe"] else FAIL,
                   f"Sharpe {v['sharpe']:.4f} vs {v['baseline_sharpe']:.4f} for {BASELINE}", "higher after costs"),
        GateResult("max_drawdown_no_deeper", PASS if v["max_drawdown"] >= v["baseline_max_drawdown"] else FAIL,
                   f"{percent(v['max_drawdown'])} vs {percent(v['baseline_max_drawdown'])}", "no deeper"),
        GateResult("years_higher", PASS if v["folds_higher_sharpe"] >= MIN_YEARS_HIGHER else FAIL,
                   f"{v['folds_higher_sharpe']} of {v['folds']} validation years", f">= {MIN_YEARS_HIGHER}"),
        GateResult("point_in_time", PASS if checks and all(c.identical for c in checks) else FAIL,
                   f"{sum(c.identical for c in checks)}/{len(checks)} cuts identical "
                   f"({', '.join(str(c.cut.date()) for c in checks)})",
                   "targets unchanged when later sessions are added"),
    ])


def overall_table(comparison: RegimeComparison) -> pd.DataFrame:
    """The portfolios' metrics over the window, formatted for the report."""
    columns = ["cagr", "sharpe", "sortino", "max_drawdown", "volatility", "avg_invested", "turnover"]
    table = comparison.overall.loc[:, [k for k in columns if k in comparison.overall.columns]].copy()
    for key in ("cagr", "max_drawdown", "volatility", "avg_invested"):
        if key in table:
            table[key] = table[key].map(percent)
    for key in ("sharpe", "sortino"):
        if key in table:
            table[key] = table[key].map(lambda v: f"{v:.4f}")
    if "turnover" in table:
        table["turnover"] = table["turnover"].map(lambda v: "-" if pd.isna(v) else f"{v:.1f}x")
    return table.rename_axis("portfolio")


def yearly_sharpe_table(comparison: RegimeComparison, folds: list[Fold], portfolios: list[str]) -> pd.DataFrame:
    """Sharpe per validation year for the given portfolios."""
    sharpe = comparison.by_fold["sharpe"].unstack()
    years = sharpe.loc[:, portfolios].map(lambda x: f"{x:.2f}")
    years.index = [f"{f.validation_start.date()} .. {f.validation_end.date()}" for f in folds]
    years.index.name = "validation year"
    return years
