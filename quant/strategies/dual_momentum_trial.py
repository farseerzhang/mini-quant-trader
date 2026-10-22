"""Pre-registered trial A of proposal section 17.1.5: dual momentum; one run, a report, nothing stored.

`dual_momentum` (SPY / IEF / cash, monthly) against `regime_switch` with rule_based-v1 over the
walk-forward validation window from the fold rule, from fresh cash, with the same costs. The four
pass criteria of section 17.1.5 must all hold; everything else in the report is descriptive.
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
from quant.features.technical import prepare_adjusted_series
from quant.models.folds import Fold
from quant.models.gates import GateReport
from quant.models.report import markdown_table, percent
from quant.persistence import queries
from quant.regime.comparison import RegimeComparison
from quant.regime.gates import default_cuts
from quant.strategies.dual_momentum import (
    BONDS,
    CASH,
    CASH_YIELD_SYMBOL,
    FUNDS,
    STOCKS,
    dual_momentum_choices,
    dual_momentum_weights,
    rebalance_sessions,
)
from quant.strategies.strategy_trial import (
    BASELINE,
    PointInTimeCheck,
    baseline_regime,
    compare_with_baseline,
    overall_table,
    strategy_gates,
    trial_folds,
    yearly_sharpe_table,
)

NAME = "dual_momentum"
_NO_ACTIONS = pd.DataFrame(columns=["ex_date", "action_type", "value"])


def fund_closes(stored: dict[str, pd.DataFrame], actions: dict[str, pd.DataFrame],
                until: pd.Timestamp | None = None) -> pd.DataFrame:
    """Total-return closes of SPY and IEF from the stored rows, optionally only from data up to ``until``
    (prices and corporate actions alike), as they would have been computed on that evening."""
    closes = {}
    for fund in FUNDS:
        if fund not in stored:
            raise ValueError(f"no stored prices for {fund}; update market data first")
        frame = stored[fund]
        fund_actions = actions.get(fund, _NO_ACTIONS)
        if until is not None:
            frame = frame.loc[:until]
            if not fund_actions.empty:
                fund_actions = fund_actions[pd.to_datetime(fund_actions["ex_date"]) <= until]
        closes[fund] = prepare_adjusted_series(frame, fund_actions)["total_return_close"]
    return pd.DataFrame(closes).dropna()


def trial_choices(closes: pd.DataFrame, cash_yield: pd.Series, first: pd.Timestamp,
                  last: pd.Timestamp) -> pd.DataFrame:
    """The monthly choices from the window's first session up to ``last``."""
    rebalances = rebalance_sessions(closes.index, first)
    return dual_momentum_choices(closes, cash_yield, rebalances[rebalances <= last])


def point_in_time_checks(stored: dict[str, pd.DataFrame], actions: dict[str, pd.DataFrame], cash_yield: pd.Series,
                         choices: pd.DataFrame, first: pd.Timestamp, cuts: list[pd.Timestamp]) -> list[PointInTimeCheck]:
    """Recompute the choices from data ending at each cut; none up to the cut may change."""
    checks = []
    for cut in cuts:
        again = trial_choices(fund_closes(stored, actions, until=cut), cash_yield.loc[:cut], first, cut)
        before = choices.loc[:cut]
        same = (again.index.equals(before.index) and again["choice"].equals(before["choice"])
                and np.allclose(again[["r_stocks", "r_bonds", "r_cash"]].to_numpy(),
                                before[["r_stocks", "r_bonds", "r_cash"]].to_numpy(), rtol=1e-9, atol=1e-12))
        checks.append(PointInTimeCheck(pd.Timestamp(cut), len(again), bool(same)))
    return checks


@dataclass(frozen=True)
class DualMomentumTrialOutcome:
    folds: list[Fold]
    choices: pd.DataFrame
    comparison: RegimeComparison
    gates: GateReport
    checks: list[PointInTimeCheck]


def run_dual_momentum_trial(engine: Engine, settings: Settings) -> DualMomentumTrialOutcome:
    from quant.strategies import PANEL_COLUMNS

    folds = trial_folds(engine, settings)
    first, last = folds[0].validation_start, folds[-1].validation_end
    stored, actions = queries.load_market_data(engine, list(FUNDS))
    cash = queries.load_raw_closes(engine, [CASH_YIELD_SYMBOL])
    if CASH_YIELD_SYMBOL not in cash:
        raise ValueError(f"no stored {CASH_YIELD_SYMBOL}; update market data first")
    cash_yield = cash[CASH_YIELD_SYMBOL]
    choices = trial_choices(fund_closes(stored, actions), cash_yield, first, last)

    stocks = list(settings.market.stocks)
    prices = load_prices(engine, [*stocks, *FUNDS])
    panel = queries.load_feature_panel(engine, settings.features.feature_version, stocks, PANEL_COLUMNS)
    index = prices.close.index[(prices.close.index >= first) & (prices.close.index <= last)]
    weights = dual_momentum_weights(choices["choice"], index)
    comparison = compare_with_baseline({NAME: weights}, folds, prices, panel, baseline_regime(engine, settings),
                                       settings)
    checks = point_in_time_checks(stored, actions, cash_yield, choices, first, default_cuts(folds))
    return DualMomentumTrialOutcome(folds, choices, comparison, strategy_gates(comparison, NAME, checks), checks)


def allocation_table(choices: pd.DataFrame) -> pd.DataFrame:
    """How many monthly decisions went to each choice (descriptive)."""
    counts = choices["choice"].value_counts().reindex([STOCKS, BONDS, CASH], fill_value=0)
    return pd.DataFrame({"decisions": counts, "share": (counts / max(len(choices), 1)).map(percent)}).rename_axis("holds")


def switches_table(choices: pd.DataFrame) -> pd.DataFrame:
    """Every rebalance session whose choice differs from the previous one, with the returns behind it."""
    changed = choices["choice"].ne(choices["choice"].shift())
    table = choices.loc[changed].copy()
    for key in ("r_stocks", "r_bonds", "r_cash"):
        table[key] = table[key].map(percent)
    table.index = [d.date() for d in table.index]
    return table.rename_axis("session")


def write_dual_momentum_report(outcome: DualMomentumTrialOutcome, reports_dir: str | Path, now: datetime) -> Path:
    c = outcome.comparison
    verdict = "PASSED" if outcome.gates.passed else "FAILED"
    lines = [
        f"# Section 17.1.5 trial A: dual momentum ({now:%Y-%m-%d %H:%M} UTC)",
        "",
        f"Pre-registered 2026-10-21 (proposal.md section 17.1.5.1). Window {c.start.date()} .. {c.end.date()} "
        f"({len(outcome.folds)} validation years from the fold rule, holdout untouched), from fresh cash, same "
        f"costs. {NAME}: on the window's first session and the first session of each month, 100% {STOCKS} if its "
        f"252-session total return beats cash (the average {CASH_YIELD_SYMBOL} / 100), else 100% {BONDS} if that "
        f"beats cash, else cash. Baseline {BASELINE}. One of four trials registered together: a pass is weaker "
        "evidence than a single pre-registered pass, and a Sharpe margin below about 0.1 is within noise.",
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
        markdown_table(yearly_sharpe_table(c, outcome.folds, [NAME, BASELINE, "SPY"])),
        "",
        "## Monthly decisions",
        "",
        markdown_table(allocation_table(outcome.choices)),
        "",
        "## Switches (first decision and every change)",
        "",
        markdown_table(switches_table(outcome.choices)),
        "",
        "Research output, not advice.",
    ]
    directory = Path(reports_dir) / "strategies"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"dual-momentum-trial-{now:%Y%m%dT%H%M%SZ}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
