"""Pre-registered trial D of proposal section 17.1.5: sector rotation with sector ETFs; one run, a report,
nothing stored.

`sector_rotation` (top 3 of the nine SPDR sector ETFs by 12-1 momentum, monthly, times the regime's
exposure) against `regime_switch` with rule_based-v1 over the walk-forward validation window from the
fold rule, from fresh cash, with the same costs. The four pass criteria of section 17.1.5 must all
hold; the always-invested rotation and everything else in the report are descriptive.
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
from quant.strategies.sector_rotation import SECTOR_FUNDS, TOP_SECTORS, sector_holdings, sector_rotation_weights
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

NAME = "sector_rotation"
ALWAYS_INVESTED = "sector_rotation (always invested)"


def trial_weights(closes: pd.DataFrame, regime: pd.Series | None, first: pd.Timestamp, last: pd.Timestamp,
                  settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(holdings, target weights) on every session of ``closes``, chosen on the rebalance sessions
    from ``first`` up to ``last``."""
    funds = closes.loc[:, list(SECTOR_FUNDS)].dropna(how="all")
    rebalances = rebalance_sessions(funds.index, first)
    holdings = sector_holdings(funds, rebalances[rebalances <= last])
    return holdings, sector_rotation_weights(holdings, regime, settings)


def point_in_time_checks(stored: dict[str, pd.DataFrame], actions: dict[str, pd.DataFrame], regime: pd.Series,
                         weights: pd.DataFrame, first: pd.Timestamp, settings: Settings,
                         cuts: list[pd.Timestamp]) -> list[PointInTimeCheck]:
    """Recompute the targets from the funds' prices and corporate actions and the regime up to each cut
    only; none from the window's first session to the cut may change."""
    checks = []
    for cut in cuts:
        closes = total_return_closes(stored, actions, list(SECTOR_FUNDS), until=cut)
        _, again = trial_weights(closes, regime.loc[:cut], first, cut, settings)
        span = weights.loc[first:cut]
        again = again.reindex(index=span.index, columns=span.columns)
        same = bool(again.notna().all().all() and np.allclose(again.to_numpy(), span.to_numpy(), atol=1e-12))
        checks.append(PointInTimeCheck(pd.Timestamp(cut), int(len(span)), same))
    return checks


@dataclass(frozen=True)
class SectorRotationTrialOutcome:
    folds: list[Fold]
    holdings: pd.DataFrame  # sessions of the window x funds
    comparison: RegimeComparison
    gates: GateReport
    checks: list[PointInTimeCheck]


def run_sector_rotation_trial(engine: Engine, settings: Settings) -> SectorRotationTrialOutcome:
    from quant.strategies import PANEL_COLUMNS

    missing = [fund for fund in SECTOR_FUNDS if fund not in settings.market.market_symbols]
    if missing:
        raise ValueError(f"add {', '.join(missing)} to market.market_symbols and run update-market-data")
    folds = trial_folds(engine, settings)
    first, last = folds[0].validation_start, folds[-1].validation_end
    stocks = list(settings.market.stocks)
    regime = baseline_regime(engine, settings)
    prices = load_prices(engine, [*stocks, "SPY", *SECTOR_FUNDS])
    panel = queries.load_feature_panel(engine, settings.features.feature_version, stocks, PANEL_COLUMNS)
    closes = prices.close.loc[:, list(SECTOR_FUNDS)]
    holdings, weights = trial_weights(closes, regime, first, last, settings)
    _, always = trial_weights(closes, None, first, last, settings)
    comparison = compare_with_baseline({NAME: weights, ALWAYS_INVESTED: always}, folds, prices, panel, regime,
                                       settings)
    stored, actions = queries.load_market_data(engine, list(SECTOR_FUNDS))
    checks = point_in_time_checks(stored, actions, regime, weights, first, settings, default_cuts(folds))
    return SectorRotationTrialOutcome(folds, holdings.loc[first:last], comparison,
                                      strategy_gates(comparison, NAME, checks), checks)


def fund_table(holdings: pd.DataFrame) -> pd.DataFrame:
    """Share of the window's sessions each fund was held (descriptive)."""
    share = holdings.mean().reindex(list(SECTOR_FUNDS)).sort_values(ascending=False)
    return pd.DataFrame({"share of sessions held": share.map(percent)}).rename_axis("fund")


def changes_per_year(holdings: pd.DataFrame) -> pd.DataFrame:
    """Funds bought per calendar year (a fund entering the top 3), descriptive."""
    entries = (holdings.astype(int).diff().clip(lower=0).sum(axis=1)).iloc[1:]
    return pd.DataFrame({"funds bought": entries.groupby(entries.index.year).sum().astype(int)}).rename_axis("year")


def write_sector_rotation_report(outcome: SectorRotationTrialOutcome, reports_dir: str | Path, now: datetime) -> Path:
    c = outcome.comparison
    verdict = "PASSED" if outcome.gates.passed else "FAILED"
    lines = [
        f"# Section 17.1.5 trial D: sector rotation with sector ETFs ({now:%Y-%m-%d %H:%M} UTC)",
        "",
        f"Pre-registered 2026-10-21 (proposal.md section 17.1.5.4). Window {c.start.date()} .. {c.end.date()} "
        f"({len(outcome.folds)} validation years from the fold rule, holdout untouched), from fresh cash, same "
        f"costs. {NAME}: on the window's first session and the first session of each month, the top "
        f"{TOP_SECTORS} of {', '.join(SECTOR_FUNDS)} by total return from 252 to 21 sessions back, one third "
        f"each, times the rule_based-v1 exposure (BEAR: cash). Baseline {BASELINE}. One of four trials registered "
        "together: a pass is weaker evidence than a single pre-registered pass, and a Sharpe margin below about "
        "0.1 is within noise.",
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
        "## Funds held",
        "",
        markdown_table(fund_table(outcome.holdings)),
        "",
        markdown_table(changes_per_year(outcome.holdings)),
        "",
        "Research output, not advice.",
    ]
    directory = Path(reports_dir) / "strategies"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"sector-rotation-trial-{now:%Y%m%dT%H%M%SZ}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
