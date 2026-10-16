"""Pre-registered trial of rule_rate_shocks-v1 (proposal section 12.7); one run, a report, nothing stored.

`regime_switch` (bear exposure 0) is traded with rule_based-v1 and with the candidate over the
walk-forward validation window from the fold rule (holdout untouched), from fresh cash. All five
pass criteria of section 13.3 must hold. Each rate input alone, bear50, the regimes, the sell-offs
and how often each input fires are descriptive only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import Engine

from quant.backtest.data import load_prices
from quant.backtest.engine import split_periods
from quant.config.settings import RegimeSettings, Settings
from quant.models.folds import Fold, walk_forward_folds
from quant.models.gates import FAIL, PASS, GateReport, GateResult
from quant.models.report import markdown_table, percent
from quant.persistence import queries
from quant.regime.bond_trial import regime_profile
from quant.regime.combo_trial import sell_off_spells
from quant.regime.comparison import RegimeComparison, compare_regimes, portfolio_name
from quant.regime.gates import default_cuts
from quant.regime.rule_based import average_dwell, classify_regimes, score_inputs
from quant.regime.rule_rate_shocks import (
    LONG_RATE_JUMP,
    NAME,
    RATE_INPUTS,
    SHORT_RATE_DROP,
    load_rate_shock_inputs,
    rate_shock_inputs,
    rate_shock_scores,
)

BASELINE = "rule_based-v1"
SINGLE_INPUT_VARIANTS = {f"v1 + {SHORT_RATE_DROP}": (SHORT_RATE_DROP,), f"v1 + {LONG_RATE_JUMP}": (LONG_RATE_JUMP,)}
MIN_AVERAGE_DWELL = 10
MIN_YEARS_HIGHER = 6


def trial_regimes(inputs: pd.DataFrame, settings: RegimeSettings) -> dict[str, pd.Series]:
    """v1, the candidate and each rate input alone, all from the first session with all seven inputs."""
    complete = inputs.dropna()
    if complete.empty:
        raise ValueError(f"no session has all seven {NAME} inputs")
    inputs = inputs.loc[complete.index[0]:]
    regimes = {BASELINE: classify_regimes(score_inputs(inputs, settings), settings)["regime"],
               NAME: classify_regimes(rate_shock_scores(inputs, settings), settings)["regime"]}
    for name, extra in SINGLE_INPUT_VARIANTS.items():
        regimes[name] = classify_regimes(rate_shock_scores(inputs, settings, extra), settings)["regime"]
    return regimes


@dataclass(frozen=True)
class PointInTimeCheck:
    cut: pd.Timestamp
    sessions: int
    identical: bool


def point_in_time_checks(inputs: pd.DataFrame, yields: pd.DataFrame, settings: RegimeSettings,
                         regimes: dict[str, pd.Series], cuts: list[pd.Timestamp]) -> list[PointInTimeCheck]:
    """Recompute the rate inputs and every regime from data ending at each cut; nothing up to it may change."""
    checks = []
    for cut in cuts:
        v1_inputs = inputs.loc[:cut, [c for c in inputs.columns if c not in RATE_INPUTS]]
        cut_inputs = v1_inputs.join(rate_shock_inputs(yields.loc[:cut]))
        same_inputs = cut_inputs[list(RATE_INPUTS)].equals(inputs.loc[:cut, list(RATE_INPUTS)])
        again = trial_regimes(cut_inputs, settings)
        same = same_inputs and all(again[name].equals(regimes[name].loc[:cut]) for name in regimes)
        checks.append(PointInTimeCheck(pd.Timestamp(cut), len(again[NAME]), bool(same)))
    return checks


def evaluate(comparison: RegimeComparison, candidate: pd.Series, checks: list[PointInTimeCheck]) -> GateReport:
    v = comparison.versus(portfolio_name("regime_switch", NAME), portfolio_name("regime_switch", BASELINE))
    dwell = average_dwell(candidate.loc[comparison.start:comparison.end])
    return GateReport([
        GateResult("sharpe_higher", PASS if v["sharpe"] > v["baseline_sharpe"] else FAIL,
                   f"regime_switch Sharpe {v['sharpe']:.4f} vs {v['baseline_sharpe']:.4f} with {BASELINE}",
                   "higher after costs"),
        GateResult("max_drawdown_no_deeper", PASS if v["max_drawdown"] >= v["baseline_max_drawdown"] else FAIL,
                   f"{percent(v['max_drawdown'])} vs {percent(v['baseline_max_drawdown'])}", "no deeper"),
        GateResult("years_higher", PASS if v["folds_higher_sharpe"] >= MIN_YEARS_HIGHER else FAIL,
                   f"{v['folds_higher_sharpe']} of {v['folds']} validation years", f">= {MIN_YEARS_HIGHER}"),
        GateResult("min_avg_regime_dwell_days", PASS if dwell >= MIN_AVERAGE_DWELL else FAIL,
                   f"{dwell:.1f} sessions", f">= {MIN_AVERAGE_DWELL}"),
        GateResult("point_in_time", PASS if checks and all(c.identical for c in checks) else FAIL,
                   f"{sum(c.identical for c in checks)}/{len(checks)} cuts identical "
                   f"({', '.join(str(c.cut.date()) for c in checks)})",
                   "inputs and regimes unchanged when later sessions are added"),
    ])


@dataclass(frozen=True)
class RateShocksTrialOutcome:
    folds: list[Fold]
    first_session: pd.Timestamp
    regimes: dict[str, pd.Series]
    comparison: RegimeComparison
    gates: GateReport
    inputs: pd.DataFrame
    spy: pd.Series


def run_rate_shocks_trial(engine: Engine, settings: Settings) -> RateShocksTrialOutcome:
    from quant.models.dataset import load_dataset
    from quant.strategies import PANEL_COLUMNS

    stocks = list(settings.market.stocks)
    dataset = load_dataset(engine, settings)
    holdout_start = split_periods(dataset.sessions, settings.holdout_months).holdout_start
    folds = walk_forward_folds(dataset.labelled().sessions, holdout_start, settings.retrain.validation,
                               settings.model.embargo_days)
    inputs, yields = load_rate_shock_inputs(engine, settings)
    regimes = trial_regimes(inputs, settings.regime)
    prices = load_prices(engine, [*stocks, "SPY"])
    panel = queries.load_feature_panel(engine, settings.features.feature_version, stocks, PANEL_COLUMNS)
    comparison = compare_regimes(regimes, folds, prices, panel, settings)
    checks = point_in_time_checks(inputs, yields, settings.regime, regimes, default_cuts(folds))
    return RateShocksTrialOutcome(folds, regimes[NAME].index[0], regimes, comparison,
                                  evaluate(comparison, regimes[NAME], checks), inputs, prices.close["SPY"])


def input_firing_by_year(inputs: pd.DataFrame, start, end) -> pd.DataFrame:
    """Share of sessions on which each rate input is -1, per calendar year of the window (descriptive)."""
    window = inputs.loc[start:end, list(RATE_INPUTS)]
    shares = (window == -1).groupby(window.index.year).mean()
    return shares.map(lambda v: f"{v:.0%}").rename_axis("year")


def write_rate_shocks_report(outcome: RateShocksTrialOutcome, reports_dir: str | Path, now: datetime) -> Path:
    c = outcome.comparison
    columns = ["cagr", "sharpe", "sortino", "max_drawdown", "volatility", "avg_invested", "turnover"]
    table = c.overall.loc[:, [k for k in columns if k in c.overall.columns]].copy()
    for key in ("cagr", "max_drawdown", "volatility", "avg_invested"):
        if key in table:
            table[key] = table[key].map(percent)
    for key in ("sharpe", "sortino"):
        if key in table:
            table[key] = table[key].map(lambda v: f"{v:.4f}")
    if "turnover" in table:
        table["turnover"] = table["turnover"].map(lambda v: "-" if pd.isna(v) else f"{v:.1f}x")
    sharpe = c.by_fold["sharpe"].unstack()
    years = pd.DataFrame({name: sharpe[portfolio_name("regime_switch", name)] for name in outcome.regimes}).map(
        lambda x: f"{x:.2f}")
    years.index = [f"{f.validation_start.date()} .. {f.validation_end.date()}" for f in outcome.folds]
    years.index.name = "validation year"
    window = {name: r.loc[c.start:c.end] for name, r in outcome.regimes.items()}
    verdict = "PASSED" if outcome.gates.passed else "FAILED"
    lines = [
        f"# Section 12.7 trial: interest-rate shock inputs ({now:%Y-%m-%d %H:%M} UTC)",
        "",
        f"Pre-registered 2026-10-16 (proposal.md section 12.7). Window {c.start.date()} .. {c.end.date()} "
        f"({len(outcome.folds)} validation years from the fold rule, holdout untouched); every recursion starts "
        f"{outcome.first_session.date()}. Candidate {NAME}: v1's five inputs plus {SHORT_RATE_DROP} and "
        f"{LONG_RATE_JUMP} (each 0 or -1), v1's thresholds. The thresholds were chosen with known episodes in "
        "mind, so a pass is partly in sample. Research only: rule_based-v1 stays ACTIVE.",
        "",
        f"## Gates: {verdict}",
        "",
        markdown_table(outcome.gates.table()),
        "",
        "## Trading comparison (descriptive except regime_switch [candidate] vs [rule_based-v1])",
        "",
        markdown_table(table.rename_axis("portfolio")),
        "",
        "## regime_switch Sharpe per validation year",
        "",
        markdown_table(years),
        "",
        "## Regimes over the window",
        "",
        markdown_table(regime_profile(outcome.regimes, outcome.spy, c.start, c.end)),
        "",
        "## BEAR in each sell-off: entry (from the peak) and exit (off the low)",
        "",
        markdown_table(sell_off_spells(window, outcome.spy)),
        "",
        "## How often each rate input is -1 (share of sessions per year)",
        "",
        markdown_table(input_firing_by_year(outcome.inputs, c.start, c.end)),
        "",
        "Research output, not advice.",
    ]
    directory = Path(reports_dir) / "regime"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"rate-shocks-trial-{now:%Y%m%dT%H%M%SZ}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
