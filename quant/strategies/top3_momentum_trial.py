"""Pre-registered trial E of proposal section 17.1.6: the three strongest stocks of the last 12 months,
quarterly; one run, a report, nothing stored.

`top3_momentum_12m_quarterly` against `regime_switch` with rule_based-v1 over the walk-forward
validation window from the fold rule, from fresh cash, with the same costs. The four pass criteria of
section 17.1.5 must all hold; everything else in the report is descriptive.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine

from quant.backtest.data import load_prices
from quant.config.settings import Settings
from quant.models.folds import Fold
from quant.models.gates import GateReport
from quant.models.report import markdown_table
from quant.persistence import queries
from quant.regime.comparison import RegimeComparison
from quant.regime.gates import default_cuts
from quant.strategies.momentum_regime_trial import holdings_table
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
from quant.strategies.top3_momentum import quarter_rebalance_sessions, top3_holdings, top3_order_targets

NAME = "top3_momentum_12m_quarterly"


def trial_holdings(closes: pd.DataFrame, first: pd.Timestamp,
                   last: pd.Timestamp) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """(holdings on every session of ``closes``, the rebalance sessions from ``first`` up to ``last``)."""
    rebalances = quarter_rebalance_sessions(closes.index, first)
    rebalances = rebalances[rebalances <= last]
    return top3_holdings(closes, rebalances), rebalances


def point_in_time_checks(stored: dict[str, pd.DataFrame], actions: dict[str, pd.DataFrame], stocks: list[str],
                         holdings: pd.DataFrame, first: pd.Timestamp,
                         cuts: list[pd.Timestamp]) -> list[PointInTimeCheck]:
    """Recompute the holdings from prices and corporate actions up to each cut only; none from the
    window's first session to the cut may change."""
    checks = []
    for cut in cuts:
        closes = total_return_closes(stored, actions, stocks, until=cut)
        again, _ = trial_holdings(closes, first, cut)
        span = holdings.loc[first:cut]
        again = again.reindex(index=span.index, columns=span.columns)
        same = bool(again.notna().all().all() and again.astype(bool).equals(span.astype(bool)))
        checks.append(PointInTimeCheck(pd.Timestamp(cut), int(len(span)), same))
    return checks


@dataclass(frozen=True)
class Top3MomentumTrialOutcome:
    folds: list[Fold]
    holdings: pd.DataFrame  # sessions of the window x stocks
    rebalances: pd.DatetimeIndex
    comparison: RegimeComparison
    gates: GateReport
    checks: list[PointInTimeCheck]


def run_top3_momentum_trial(engine: Engine, settings: Settings) -> Top3MomentumTrialOutcome:
    from quant.strategies import PANEL_COLUMNS

    folds = trial_folds(engine, settings)
    first, last = folds[0].validation_start, folds[-1].validation_end
    stocks = list(settings.market.stocks)
    regime = baseline_regime(engine, settings)
    prices = load_prices(engine, [*stocks, "SPY"])
    panel = queries.load_feature_panel(engine, settings.features.feature_version, stocks, PANEL_COLUMNS)
    holdings, rebalances = trial_holdings(prices.close.loc[:, stocks], first, last)
    orders = top3_order_targets(holdings, rebalances)
    comparison = compare_with_baseline({}, folds, prices, panel, regime, settings, ready_orders={NAME: orders})
    stored, actions = queries.load_market_data(engine, stocks)
    checks = point_in_time_checks(stored, actions, stocks, holdings, first, default_cuts(folds))
    return Top3MomentumTrialOutcome(folds, holdings.loc[first:last], rebalances, comparison,
                                    strategy_gates(comparison, NAME, checks), checks)


def picks_table(holdings: pd.DataFrame, rebalances: pd.DatetimeIndex) -> pd.DataFrame:
    """The three stocks chosen on each rebalance session (descriptive)."""
    dates = rebalances.intersection(holdings.index)
    rows = {d.date(): ", ".join(holdings.columns[holdings.loc[d]]) for d in dates}
    return pd.DataFrame({"held until the next rebalance": rows}).rename_axis("rebalance")


def write_top3_momentum_report(outcome: Top3MomentumTrialOutcome, reports_dir: str | Path, now: datetime) -> Path:
    c = outcome.comparison
    verdict = "PASSED" if outcome.gates.passed else "FAILED"
    distinct = int(outcome.holdings.any().sum())
    lines = [
        f"# Section 17.1.6 trial E: the three strongest stocks of the last 12 months, quarterly "
        f"({now:%Y-%m-%d %H:%M} UTC)",
        "",
        f"Pre-registered 2026-10-22 (proposal.md section 17.1.6). Window {c.start.date()} .. {c.end.date()} "
        f"({len(outcome.folds)} validation years from the fold rule, holdout untouched), from fresh cash, same "
        f"costs. {NAME}: on the window's first session and the first session of each quarter, the 3 pool stocks "
        "with the highest total return over the last 252 sessions, one third each, always invested, no "
        f"single-stock or sector cap. Baseline {BASELINE}. The pool is today's large caps, chosen partly for "
        "strong past returns, which favours this strategy most; momentum on this pool had been seen (trial B). "
        "A pass would be weak evidence.",
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
        markdown_table(yearly_sharpe_table(c, outcome.folds, [NAME, BASELINE, "equal_weight", "SPY"])),
        "",
        f"## Stocks held most often ({distinct} different stocks over the window)",
        "",
        markdown_table(holdings_table(outcome.holdings)),
        "",
        "## Picks per quarter",
        "",
        markdown_table(picks_table(outcome.holdings, outcome.rebalances)),
        "",
        "Research output, not advice.",
    ]
    directory = Path(reports_dir) / "strategies"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"top3-momentum-trial-{now:%Y%m%dT%H%M%SZ}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
