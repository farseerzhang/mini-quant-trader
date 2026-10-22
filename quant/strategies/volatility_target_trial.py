"""Pre-registered trial C of proposal section 17.1.5: volatility targeting on `regime_switch`; one run,
a report, nothing stored.

`regime_switch_voltarget` (regime_switch's weights x min(1, 0.15 / SPY's 20-session volatility))
against `regime_switch` with rule_based-v1 over the walk-forward validation window from the fold
rule, from fresh cash, with the same costs. The four pass criteria of section 17.1.5 must all hold;
everything else in the report is descriptive.
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
from quant.strategies.volatility_target import (
    TARGET_VOLATILITY,
    VOLATILITY_SESSIONS,
    realized_volatility,
    volatility_multiplier,
    volatility_target_weights,
)

NAME = "regime_switch_voltarget"
MARKET = "SPY"


def regime_switch_weights_over(panel, regime: pd.Series, index: pd.DatetimeIndex, settings: Settings) -> pd.DataFrame:
    """regime_switch's target weights (the baseline strategy) on the given sessions."""
    from quant.strategies import build_strategies

    strategy = next(s for s in build_strategies(settings) if s.name == "regime_switch")
    return strategy.weights(panel, regime).reindex(index).fillna(0.0)


def trial_weights(panel, regime: pd.Series, spy_closes: pd.Series, index: pd.DatetimeIndex,
                  settings: Settings) -> tuple[pd.DataFrame, pd.Series]:
    """(target weights, multiplier) on the sessions of ``index``."""
    multiplier = volatility_multiplier(realized_volatility(spy_closes))
    base = regime_switch_weights_over(panel, regime, index, settings)
    return volatility_target_weights(base, multiplier), multiplier.reindex(index)


def point_in_time_checks(panel, regime: pd.Series, stored: dict[str, pd.DataFrame], actions: dict[str, pd.DataFrame],
                         weights: pd.DataFrame, settings: Settings, cuts: list[pd.Timestamp]) -> list[PointInTimeCheck]:
    """Recompute the targets from features, regimes and SPY's prices and corporate actions up to each cut
    only; none from the window's first session to the cut may change."""
    checks = []
    for cut in cuts:
        span = weights.loc[:cut]
        spy = total_return_closes(stored, actions, [MARKET], until=cut)[MARKET]
        cut_panel = {name: frame.loc[:cut] for name, frame in panel.items()}
        again, _ = trial_weights(cut_panel, regime.loc[:cut], spy, span.index, settings)
        again = again.reindex(columns=span.columns)
        same = bool(again.notna().all().all() and np.allclose(again.to_numpy(), span.to_numpy(), atol=1e-12))
        checks.append(PointInTimeCheck(pd.Timestamp(cut), int(len(span)), same))
    return checks


@dataclass(frozen=True)
class VolatilityTargetTrialOutcome:
    folds: list[Fold]
    multiplier: pd.Series  # sessions of the window
    regime: pd.Series  # sessions of the window
    comparison: RegimeComparison
    gates: GateReport
    checks: list[PointInTimeCheck]


def run_volatility_target_trial(engine: Engine, settings: Settings) -> VolatilityTargetTrialOutcome:
    from quant.strategies import PANEL_COLUMNS

    folds = trial_folds(engine, settings)
    first, last = folds[0].validation_start, folds[-1].validation_end
    stocks = list(settings.market.stocks)
    regime = baseline_regime(engine, settings)
    prices = load_prices(engine, [*stocks, MARKET])
    panel = queries.load_feature_panel(engine, settings.features.feature_version, stocks, PANEL_COLUMNS)
    index = prices.close.index[(prices.close.index >= first) & (prices.close.index <= last)]
    weights, multiplier = trial_weights(panel, regime, prices.close[MARKET], index, settings)
    comparison = compare_with_baseline({NAME: weights}, folds, prices, panel, regime, settings)
    stored, actions = queries.load_market_data(engine, [MARKET])
    checks = point_in_time_checks(panel, regime, stored, actions, weights, settings, default_cuts(folds))
    return VolatilityTargetTrialOutcome(folds, multiplier, regime.reindex(index), comparison,
                                        strategy_gates(comparison, NAME, checks), checks)


def multiplier_table(multiplier: pd.Series, regime: pd.Series) -> pd.DataFrame:
    """How often the overlay binds and how much, overall and per regime (descriptive)."""
    rows = {}
    for name, mask in [("all sessions", pd.Series(True, index=multiplier.index)),
                       *[(f"regime {r}", regime == r) for r in ("BULL", "SIDEWAYS", "BEAR")]]:
        values = multiplier[mask.reindex(multiplier.index).fillna(False)]
        if values.empty:
            continue
        rows[name] = {"sessions": len(values), "binds (m < 1)": percent((values < 1).mean()),
                      "average m": f"{values.mean():.2f}", "lowest m": f"{values.min():.2f}"}
    return pd.DataFrame(rows).T.rename_axis("sessions of the window")


def multiplier_by_year(multiplier: pd.Series) -> pd.DataFrame:
    """Share of sessions with m < 1 and the average m, per calendar year (descriptive)."""
    grouped = multiplier.groupby(multiplier.index.year)
    return pd.DataFrame({"binds (m < 1)": grouped.apply(lambda v: percent((v < 1).mean())),
                         "average m": grouped.mean().map(lambda v: f"{v:.2f}")}).rename_axis("year")


def write_volatility_target_report(outcome: VolatilityTargetTrialOutcome, reports_dir: str | Path,
                                   now: datetime) -> Path:
    c = outcome.comparison
    verdict = "PASSED" if outcome.gates.passed else "FAILED"
    lines = [
        f"# Section 17.1.5 trial C: volatility targeting on regime_switch ({now:%Y-%m-%d %H:%M} UTC)",
        "",
        f"Pre-registered 2026-10-21 (proposal.md section 17.1.5.3). Window {c.start.date()} .. {c.end.date()} "
        f"({len(outcome.folds)} validation years from the fold rule, holdout untouched), from fresh cash, same "
        f"costs. {NAME}: regime_switch's weights x min(1, {TARGET_VOLATILITY:.0%} / SPY's {VOLATILITY_SESSIONS}-session "
        f"annualized volatility), every session; no leverage; a held position is resized only when its target "
        f"moves at least backtest.min_trade_weight. Baseline {BASELINE}. One of four trials registered together: "
        "a pass is weaker evidence than a single pre-registered pass, and a Sharpe margin below about 0.1 is "
        "within noise.",
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
        "## The multiplier m over the window",
        "",
        markdown_table(multiplier_table(outcome.multiplier, outcome.regime)),
        "",
        markdown_table(multiplier_by_year(outcome.multiplier)),
        "",
        "Research output, not advice.",
    ]
    directory = Path(reports_dir) / "strategies"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"volatility-target-trial-{now:%Y%m%dT%H%M%SZ}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
