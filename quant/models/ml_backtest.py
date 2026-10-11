"""Walk-forward backtest of the ML strategy against the rule-based baseline (proposal section 15.3).

Each session of the evaluation window is traded with a probability from the fold
whose validation window contains it, i.e. from a model trained only on sessions
ending at least ``embargo_days`` before that window. The predictions of all folds
are stitched into one out-of-sample probability panel and traded as
`ml_select`. Every portfolio (ML strategies, `regime_switch`, SPY, equal weight)
is simulated from fresh cash over the same window: the first validation session
to the last one. Nothing after the last validation session, and so nothing in
the holdout, is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant.backtest.data import Prices
from quant.backtest.engine import Periods, Simulation, random_baseline, simulate
from quant.backtest.metrics import METRIC_NAMES, period_metrics
from quant.backtest.orders import buy_and_hold, equal_weight_monthly, order_targets
from quant.config.settings import Settings
from quant.models.folds import Fold
from quant.models.walk_forward import WalkForwardResult
from quant.strategies import build_strategies
from quant.strategies.ml import ml_select_weights
from quant.strategies.rules import Panel

BASELINE = "regime_switch"


@dataclass(frozen=True)
class MLBacktest:
    start: pd.Timestamp
    end: pd.Timestamp
    simulations: dict[str, Simulation]
    # Portfolio x metrics over the whole evaluation window.
    overall: pd.DataFrame
    # (fold, portfolio) rows: each validation window's sharpe, total_return, max_drawdown.
    by_fold: pd.DataFrame
    # ML strategy -> development-window random-baseline summary.
    random: pd.DataFrame

    def versus_baseline(self, name: str) -> dict[str, float]:
        """How an ML strategy compares with `regime_switch`, fold by fold and overall."""
        sharpe = self.by_fold["sharpe"].unstack()
        returns = self.by_fold["total_return"].unstack()
        return {
            "folds": int(len(sharpe)),
            "folds_higher_sharpe": int((sharpe[name] > sharpe[BASELINE]).sum()),
            "fraction_higher_sharpe": float((sharpe[name] > sharpe[BASELINE]).mean()),
            "folds_higher_return": int((returns[name] > returns[BASELINE]).sum()),
            "sharpe": float(self.overall.loc[name, "sharpe"]),
            "baseline_sharpe": float(self.overall.loc[BASELINE, "sharpe"]),
            "max_drawdown": float(self.overall.loc[name, "max_drawdown"]),
            "baseline_max_drawdown": float(self.overall.loc[BASELINE, "max_drawdown"]),
        }


def probability_panel(predictions: pd.DataFrame) -> pd.DataFrame:
    """Out-of-sample predictions (date, symbol) -> calibrated probabilities, dates x symbols."""
    if predictions.index.duplicated().any():
        raise ValueError("a (date, symbol) was predicted more than once")
    return predictions["probability"].unstack("symbol").sort_index()


def check_out_of_sample(predictions: pd.DataFrame, folds: list[Fold], sessions: pd.DatetimeIndex, embargo: int) -> None:
    """Fail if any prediction is not from a model trained before its session minus the embargo."""
    by_number = {fold.number: fold for fold in folds}
    dates = predictions.index.get_level_values("date")
    for number, group_dates in pd.Series(dates, index=predictions.index).groupby(predictions["fold"].to_numpy()):
        fold = by_number[int(number)]
        if not pd.DatetimeIndex(group_dates).isin(fold.validation).all():
            raise ValueError(f"fold {number} predicted sessions outside its validation window")
        gap = sessions.get_loc(fold.validation_start) - sessions.get_loc(fold.train_end) - 1
        if gap < embargo:
            raise ValueError(f"fold {number} trained {gap} session(s) before its validation window (< {embargo})")


def run_ml_backtest(
    walk_forward: dict[str, WalkForwardResult],
    folds: list[Fold],
    prices: Prices,
    panel: Panel,
    regimes: pd.Series,
    settings: Settings,
) -> MLBacktest:
    """Backtest ``ml_select`` for each walk-forward result against the baseline and benchmarks.

    ``walk_forward`` maps a portfolio name (e.g. ``ml_logistic_regression``) to its result.
    """
    start, end = folds[0].validation_start, folds[-1].validation_end
    index = prices.close.index[(prices.close.index >= start) & (prices.close.index <= end)]
    bt = settings.backtest
    stocks = list(settings.market.stocks)

    weights: dict[str, pd.DataFrame] = {}
    for name, result in walk_forward.items():
        check_out_of_sample(result.predictions, folds, prices.close.index, settings.model.embargo_days)
        probabilities = probability_panel(result.predictions).reindex(index=index, columns=stocks)
        weights[name] = ml_select_weights(probabilities, regimes, settings)
    baseline = next(strategy for strategy in build_strategies(settings) if strategy.name == BASELINE)
    weights[BASELINE] = baseline.weights(panel, regimes)

    simulations = {
        name: simulate(order_targets(frame.reindex(index).fillna(0.0), bt.min_trade_weight), prices, bt)
        for name, frame in weights.items()
    }
    simulations["SPY"] = simulate(buy_and_hold(index, "SPY"), prices, bt)
    simulations["equal_weight"] = simulate(equal_weight_monthly(index, stocks), prices, bt)

    def metrics(simulation: Simulation, first, last) -> dict[str, float]:
        return period_metrics(simulation.value, simulation.invested, simulation.orders, simulation.trades, first, last)

    overall = pd.DataFrame({name: metrics(sim, start, end) for name, sim in simulations.items()}).T
    rows = []
    for fold in folds:
        for name, sim in simulations.items():
            m = metrics(sim, fold.validation_start, fold.validation_end)
            rows.append({"fold": fold.number, "portfolio": name, **{k: m[k] for k in ("sharpe", "total_return", "max_drawdown")}})
    by_fold = pd.DataFrame(rows).set_index(["fold", "portfolio"])

    window = Periods(start=start, development_end=end, holdout_start=end, end=end)
    random_rows = {}
    if bt.random_baseline_runs:
        for name in walk_forward:
            runs = random_baseline(weights[name].reindex(index).fillna(0.0), prices, bt, window)
            own = overall.loc[name]
            random_rows[name] = {
                "sharpe": own["sharpe"],
                "random_sharpe_median": float(runs["sharpe"].median()),
                "sharpe_percentile": float((runs["sharpe"] < own["sharpe"]).mean()),
                "cagr": own["cagr"],
                "random_cagr_median": float(runs["cagr"].median()),
                "cagr_percentile": float((runs["cagr"] < own["cagr"]).mean()),
                "runs": int(len(runs)),
            }
    return MLBacktest(
        start=start,
        end=end,
        simulations=simulations,
        overall=overall.loc[:, list(METRIC_NAMES)],
        by_fold=by_fold,
        random=pd.DataFrame(random_rows).T,
    )
