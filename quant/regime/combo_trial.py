"""Pre-registered trial: the rule-based regime with an HMM BEAR trigger (proposal section 13.3).

Research only: nothing here is stored or used by run-daily; the ACTIVE regime does not change.

- Combined regime at T: BEAR if rule_based-v1 or the walk-forward HMM says BEAR at T,
  otherwise rule_based-v1's regime. The HMM's SIDEWAYS is ignored; no dwell on top.
- rule_based-v1: the ACTIVE rules (section 12) over the whole stored history.
- HMM: the walk-forward regime of section 13 (yearly fits on the development folds,
  filtered, the configured dwell): out of sample on every validation session.
- Descriptive: "both must say BEAR" (BEAR only when both say BEAR; a rule-based BEAR
  without the HMM's becomes SIDEWAYS).
- One trial on the walk-forward validation window; the holdout is not used.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine

from quant.backtest.data import load_prices
from quant.backtest.engine import split_periods
from quant.config.settings import Settings
from quant.models.folds import Fold, walk_forward_folds
from quant.models.gates import FAIL, PASS, GateReport, GateResult
from quant.models.report import markdown_table, percent
from quant.persistence import queries
from quant.regime.bond_trial import regime_profile
from quant.regime.comparison import RegimeComparison, compare_regimes, portfolio_name
from quant.regime.gates import default_cuts, point_in_time_checks
from quant.regime.hmm import load_observations
from quant.regime.hmm_walk_forward import HMMWalkForward, run_hmm_walk_forward
from quant.regime.report import SELL_OFFS
from quant.regime.rule_based import BEAR, SIDEWAYS, average_dwell, calculate_rule_regimes

V1, HMM, COMBO, BOTH = "rule_based-v1", "hmm (walk-forward)", "combined", "both must say BEAR"
MIN_DWELL = 10
MIN_YEARS = 6  # of the 11 validation years with a higher Sharpe


def combine(rule: pd.Series, hmm: pd.Series) -> pd.Series:
    """BEAR if either says BEAR, otherwise the rule-based regime; on the HMM's sessions."""
    rule = rule.reindex(hmm.index)
    if rule.isna().any():
        raise ValueError(f"the rule-based regime is missing on {int(rule.isna().sum())} HMM session(s)")
    return rule.where((rule != BEAR) & (hmm != BEAR), BEAR).rename("regime")


def both_bear(rule: pd.Series, hmm: pd.Series) -> pd.Series:
    """Descriptive variant: BEAR only when both say BEAR; a rule-based BEAR alone becomes SIDEWAYS."""
    rule = rule.reindex(hmm.index)
    out = rule.where(rule != BEAR, SIDEWAYS)
    return out.where(~((rule == BEAR) & (hmm == BEAR)), BEAR).rename("regime")


@dataclass(frozen=True)
class PointInTime:
    cut: pd.Timestamp
    sessions: int
    identical: bool


def rule_point_in_time(inputs: pd.DataFrame, settings: Settings, rule: pd.Series, hmm: pd.Series,
                       combined: pd.Series, cuts: list[pd.Timestamp]) -> list[PointInTime]:
    """The rule-based regime and the combination recomputed from inputs ending at each cut.

    (The HMM part has its own check, point_in_time_checks, which refits up to each cut.)
    """
    checks = []
    for cut in cuts:
        truncated = calculate_rule_regimes(inputs.loc[:cut], settings.regime)["regime"]
        same_rule = truncated.equals(rule.loc[:cut])
        again = combine(truncated, hmm.loc[:cut])
        checks.append(PointInTime(pd.Timestamp(cut), len(again), bool(same_rule and again.equals(combined.loc[:cut]))))
    return checks


@dataclass(frozen=True)
class ComboTrialOutcome:
    folds: list[Fold]
    walk_forward: HMMWalkForward
    regimes: dict[str, pd.Series]  # V1, COMBO, BOTH (and the HMM alone, for the tables)
    comparison: RegimeComparison
    gates: GateReport
    spy: pd.Series


def evaluate(comparison: RegimeComparison, combined: pd.Series, hmm_checks, rule_checks: list[PointInTime]) -> GateReport:
    v = comparison.versus(portfolio_name("regime_switch", COMBO), portfolio_name("regime_switch", V1))
    dwell = average_dwell(combined.loc[comparison.start:comparison.end])
    checks_ok = bool(hmm_checks) and bool(rule_checks) and all(c.passed for c in hmm_checks) and all(
        c.identical for c in rule_checks)
    return GateReport([
        GateResult("sharpe_higher", PASS if v["sharpe"] > v["baseline_sharpe"] else FAIL,
                   f"regime_switch Sharpe {v['sharpe']:.4f} vs {v['baseline_sharpe']:.4f} with {V1}",
                   "higher after costs"),
        GateResult("max_drawdown_no_deeper", PASS if v["max_drawdown"] >= v["baseline_max_drawdown"] else FAIL,
                   f"{percent(v['max_drawdown'])} vs {percent(v['baseline_max_drawdown'])}", "no deeper"),
        GateResult("years_higher", PASS if v["folds_higher_sharpe"] >= MIN_YEARS else FAIL,
                   f"{v['folds_higher_sharpe']} of {v['folds']} validation years", f">= {MIN_YEARS}"),
        GateResult("min_avg_regime_dwell_days", PASS if dwell >= MIN_DWELL else FAIL, f"{dwell:.1f} sessions",
                   f">= {MIN_DWELL}"),
        GateResult("point_in_time", PASS if checks_ok else FAIL,
                   f"HMM {sum(c.passed for c in hmm_checks)}/{len(hmm_checks)}, rule-based and combined "
                   f"{sum(c.identical for c in rule_checks)}/{len(rule_checks)} cuts identical "
                   f"({', '.join(str(c.cut.date()) for c in rule_checks)})",
                   "unchanged when later sessions are added"),
    ])


def run_combo_trial(engine: Engine, settings: Settings) -> ComboTrialOutcome:
    from quant.models.dataset import load_dataset
    from quant.strategies import PANEL_COLUMNS

    stocks = list(settings.market.stocks)
    dataset = load_dataset(engine, settings)
    holdout_start = split_periods(dataset.sessions, settings.holdout_months).holdout_start
    folds = walk_forward_folds(dataset.labelled().sessions, holdout_start, settings.retrain.validation,
                               settings.model.embargo_days)
    observations = load_observations(engine, settings)
    walk_forward = run_hmm_walk_forward(observations, folds, settings)
    inputs = queries.load_regime_inputs(engine, settings.features.feature_version,
                                        settings.features.market_feature_version)
    rule = calculate_rule_regimes(inputs, settings.regime)["regime"]
    hmm = walk_forward.regime
    combined = combine(rule, hmm)
    regimes = {V1: rule, COMBO: combined, BOTH: both_bear(rule, hmm)}
    prices = load_prices(engine, [*stocks, "SPY"])
    panel = queries.load_feature_panel(engine, settings.features.feature_version, stocks, PANEL_COLUMNS)
    comparison = compare_regimes(regimes, folds, prices, panel, settings)
    cuts = default_cuts(folds)
    hmm_checks = point_in_time_checks(observations, folds, settings, walk_forward, cuts)
    rule_checks = rule_point_in_time(inputs, settings, rule, hmm, combined, cuts)
    gates = evaluate(comparison, combined, hmm_checks, rule_checks)
    return ComboTrialOutcome(folds, walk_forward, {**regimes, HMM: hmm}, comparison, gates, prices.close["SPY"])


# Descriptive tables (section 13.3: reported, decide nothing).

def sell_off_spells(regimes: dict[str, pd.Series], spy: pd.Series) -> pd.DataFrame:
    """Per sell-off: the first BEAR session and the first session after it that is no longer BEAR,
    with SPY's distance from its peak (entry) and from its low (exit) by then."""
    rows = {}
    for name, start, end in SELL_OFFS:
        cells = {}
        for label, regime in regimes.items():
            window = regime.loc[start:end]
            bear = window[window == BEAR]
            if window.empty:
                cells[label] = "outside the window"
                continue
            if bear.empty:
                cells[label] = "no BEAR"
                continue
            first = bear.index[0]
            entry = spy.loc[first] / spy.loc[start:first].max() - 1
            after = regime.loc[first:]
            out = after[after != BEAR]
            if out.empty:
                cells[label] = f"in {first.date()} ({entry:+.0%}), still BEAR"
                continue
            leave = out.index[0]
            rebound = spy.loc[leave] / spy.loc[first:leave].min() - 1
            cells[label] = f"in {first.date()} ({entry:+.0%}), out {leave.date()} ({rebound:+.0%} off the low)"
        rows[name] = cells
    return pd.DataFrame(rows).T.rename_axis("sell-off")


def write_combo_report(outcome: ComboTrialOutcome, reports_dir: str | Path, now: datetime) -> Path:
    c = outcome.comparison
    columns = ["cagr", "sharpe", "sortino", "max_drawdown", "volatility", "avg_invested", "turnover"]
    table = c.overall.loc[:, [k for k in columns if k in c.overall.columns]].copy()
    for k in ("cagr", "max_drawdown", "volatility", "avg_invested"):
        if k in table:
            table[k] = table[k].map(percent)
    for k in ("sharpe", "sortino"):
        if k in table:
            table[k] = table[k].map(lambda v: f"{v:.4f}")
    if "turnover" in table:
        table["turnover"] = table["turnover"].map(lambda v: "-" if pd.isna(v) else f"{v:.1f}x")
    traded = [V1, COMBO, BOTH]
    sharpe = c.by_fold["sharpe"].unstack()
    years = pd.DataFrame({name: sharpe[portfolio_name("regime_switch", name)] for name in traded}).map(
        lambda x: f"{x:.2f}")
    years.index = [f"{f.validation_start.date()} .. {f.validation_end.date()}" for f in outcome.folds]
    years.index.name = "validation year"
    window = {name: r.loc[c.start:c.end] for name, r in outcome.regimes.items()}
    verdict = "PASSED" if outcome.gates.passed else "FAILED"
    lines = [
        f"# Section 13.3 trial: rule-based regime with an HMM BEAR trigger ({now:%Y-%m-%d %H:%M} UTC)",
        "",
        f"Pre-registered 2026-10-15 (proposal.md section 13.3). Window {c.start.date()} .. {c.end.date()} "
        f"({len(outcome.folds)} validation years from the fold rule, holdout untouched). Combined: BEAR if "
        f"{V1} or the walk-forward HMM says BEAR, otherwise {V1}. The idea came from sell-offs inside this "
        "window, so a pass is partly in sample; only a forward shadow can follow from it.",
        "",
        f"## Gates: {verdict}",
        "",
        markdown_table(outcome.gates.table()),
        "",
        "## Trading comparison (descriptive except regime_switch [combined] vs [rule_based-v1])",
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
        "Research output, not advice.",
    ]
    directory = Path(reports_dir) / "regime"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"combo-trial-{now:%Y%m%dT%H%M%SZ}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
