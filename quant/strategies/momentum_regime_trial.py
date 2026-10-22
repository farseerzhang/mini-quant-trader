"""Pre-registered trial B of proposal section 17.1.5: 12-1 momentum inside the regime timing; one run,
a report, nothing stored.

`momentum_regime` against `regime_switch` with rule_based-v1 over the walk-forward validation window
from the fold rule, from fresh cash, with the same costs. Both use the same stored regime, so the
difference is the stock choice. The four pass criteria of section 17.1.5 must all hold; the
always-invested momentum and everything else in the report are descriptive.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import Engine

from quant.backtest.data import load_prices
from quant.config.settings import Settings
from quant.models.folds import Fold
from quant.models.gates import GateReport
from quant.models.report import markdown_table, percent
from quant.persistence import queries
from quant.regime.comparison import RegimeComparison
from quant.regime.gates import default_cuts
from quant.strategies.dual_momentum import rebalance_sessions
from quant.strategies.momentum_regime import (
    KEEP_RANK,
    TOP_N,
    momentum_holdings,
    momentum_regime_weights,
    momentum_scores,
)
from quant.strategies.strategy_trial import (
    BASELINE,
    PointInTimeCheck,
    baseline_regime,
    compare_with_baseline,
    overall_table,
    strategy_gates,
    total_return_closes,
    trial_folds,
    yearly_sharpe_table,
)

NAME = "momentum_regime"
ALWAYS_INVESTED = "momentum_12_1 (always invested)"


def trial_weights(closes: pd.DataFrame, regime: pd.Series | None, first: pd.Timestamp, last: pd.Timestamp,
                  settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(holdings, target weights) on every session of ``closes``, chosen on the rebalance sessions
    from ``first`` up to ``last``."""
    rebalances = rebalance_sessions(closes.index, first)
    holdings = momentum_holdings(momentum_scores(closes), rebalances[rebalances <= last])
    return holdings, momentum_regime_weights(holdings, regime, settings)


def point_in_time_checks(stored: dict[str, pd.DataFrame], actions: dict[str, pd.DataFrame], stocks: list[str],
                         regime: pd.Series, weights: pd.DataFrame, first: pd.Timestamp, settings: Settings,
                         cuts: list[pd.Timestamp]) -> list[PointInTimeCheck]:
    """Recompute the targets from prices, corporate actions and regimes up to each cut only; none from
    the window's first session up to the cut may change."""
    checks = []
    for cut in cuts:
        closes = total_return_closes(stored, actions, stocks, until=cut)
        _, again = trial_weights(closes, regime.loc[:cut], first, cut, settings)
        span = weights.loc[first:cut]
        again = again.reindex(index=span.index, columns=span.columns)
        same = bool(again.notna().all().all() and np.allclose(again.to_numpy(), span.to_numpy(), atol=1e-12))
        checks.append(PointInTimeCheck(pd.Timestamp(cut), int(len(span)), same))
    return checks


@dataclass(frozen=True)
class MomentumRegimeTrialOutcome:
    folds: list[Fold]
    holdings: pd.DataFrame  # sessions of the window x stocks
    comparison: RegimeComparison
    gates: GateReport
    checks: list[PointInTimeCheck]


def run_momentum_regime_trial(engine: Engine, settings: Settings) -> MomentumRegimeTrialOutcome:
    from quant.strategies import PANEL_COLUMNS

    folds = trial_folds(engine, settings)
    first, last = folds[0].validation_start, folds[-1].validation_end
    stocks = list(settings.market.stocks)
    regime = baseline_regime(engine, settings)
    prices = load_prices(engine, [*stocks, "SPY"])
    panel = queries.load_feature_panel(engine, settings.features.feature_version, stocks, PANEL_COLUMNS)
    closes = prices.close.loc[:, stocks]
    holdings, weights = trial_weights(closes, regime, first, last, settings)
    _, always = trial_weights(closes, None, first, last, settings)
    comparison = compare_with_baseline({NAME: weights, ALWAYS_INVESTED: always}, folds, prices, panel, regime,
                                       settings)
    stored, actions = queries.load_market_data(engine, stocks)
    checks = point_in_time_checks(stored, actions, stocks, regime, weights, first, settings, default_cuts(folds))
    window = holdings.loc[first:last]
    return MomentumRegimeTrialOutcome(folds, window, comparison, strategy_gates(comparison, NAME, checks), checks)


def holdings_table(holdings: pd.DataFrame, rows: int = 15) -> pd.DataFrame:
    """The stocks held most often over the window (share of sessions held), descriptive."""
    share = holdings.mean().sort_values(ascending=False)
    share = share[share > 0].head(rows)
    return pd.DataFrame({"share of sessions held": share.map(percent)}).rename_axis("stock")


def selection_summary(holdings: pd.DataFrame) -> pd.DataFrame:
    """Stocks held per session and how many were replaced per month, descriptive."""
    count = holdings.sum(axis=1)
    month_end = holdings.groupby(holdings.index.to_period("M")).tail(1)
    replaced = (month_end.astype(int).diff().clip(lower=0).sum(axis=1)).iloc[1:]
    return pd.DataFrame({"value": {
        "stocks held (average)": f"{count.mean():.1f}",
        "stocks replaced per month (average)": f"{replaced.mean():.1f}" if len(replaced) else "-",
        "different stocks held over the window": str(int(holdings.any().sum())),
    }}).rename_axis("measure")


def write_momentum_regime_report(outcome: MomentumRegimeTrialOutcome, reports_dir: str | Path, now: datetime) -> Path:
    c = outcome.comparison
    verdict = "PASSED" if outcome.gates.passed else "FAILED"
    lines = [
        f"# Section 17.1.5 trial B: 12-1 momentum inside the regime timing ({now:%Y-%m-%d %H:%M} UTC)",
        "",
        f"Pre-registered 2026-10-21 (proposal.md section 17.1.5.2). Window {c.start.date()} .. {c.end.date()} "
        f"({len(outcome.folds)} validation years from the fold rule, holdout untouched), from fresh cash, same "
        f"costs. {NAME}: on the window's first session and the first session of each month, the top {TOP_N} pool "
        f"stocks by total return from 252 to 21 sessions back (keep band: top {KEEP_RANK}), sized as "
        f"regime_switch and times the same rule_based-v1 exposure (BEAR: cash). Baseline {BASELINE}. The pool is "
        "today's large caps, chosen partly for strong past returns, which favours momentum. One of four trials "
        "registered together: a pass is weaker evidence than a single pre-registered pass, and a Sharpe margin "
        "below about 0.1 is within noise.",
        "",
        f"## Gates: {verdict}",
        "",
        markdown_table(outcome.gates.table()),
        "",
        f"## Trading comparison (descriptive except {NAME} vs {BASELINE})",
        "",
        markdown_table(overall_table(c)),
        "",
        "## Sharpe per validation year",
        "",
        markdown_table(yearly_sharpe_table(c, outcome.folds, [NAME, BASELINE, ALWAYS_INVESTED, "SPY"])),
        "",
        "## Selection",
        "",
        markdown_table(selection_summary(outcome.holdings)),
        "",
        "## Stocks held most often",
        "",
        markdown_table(holdings_table(outcome.holdings)),
        "",
        "Research output, not advice.",
    ]
    directory = Path(reports_dir) / "strategies"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"momentum-regime-trial-{now:%Y%m%dT%H%M%SZ}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
