"""Pre-registered trial: bond-market fear inputs in the rule score (proposal section 12.5).

Research only: nothing here is stored or used by run-daily, and rule_based-v1 stays ACTIVE.

- Inputs 6 and 7, at the close of T from total-return closes over 20 sessions:
  credit_appetite_20d = HYG's return minus IEF's; stock_vs_bond_20d = SPY's minus TLT's.
  Each scores its sign, risk-on +1 (the direction of inputs 1-5).
- Candidate rule_based-v2: the seven inputs (-7 .. +7) with E = 4, X = 1; dwell and VIX
  thresholds unchanged. Descriptive: each bond input alone (six inputs, E = 4, X = 1).
- v1 and every candidate start their recursion on the same first session with all seven inputs.
- One trial on the walk-forward validation window; the holdout is not used.

The inputs are computed here from stored prices rather than stored as market_features
columns, so the table the running forward test reads is left unchanged.
"""

from __future__ import annotations

import dataclasses
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
from quant.regime.comparison import RegimeComparison, compare_regimes, portfolio_name
from quant.regime.gates import default_cuts
from quant.regime.report import sell_off_reactions
from quant.regime.rule_based import BEAR, BULL, SIDEWAYS, average_dwell, classify_regimes, score_inputs

WINDOW = 20  # sessions, as in CNN's components
BOND_SYMBOLS = ("TLT", "IEF", "HYG")
CREDIT, SAFE_HAVEN = "credit_appetite_20d", "stock_vs_bond_20d"
BOND_INPUTS = (CREDIT, SAFE_HAVEN)
ENTER, EXIT = 4, 1  # fixed in section 12.5 for both the seven- and the six-input scores
V1, V2 = "rule_based-v1", "rule_based-v2"
SINGLES = {f"v1 + {CREDIT}": (CREDIT,), f"v1 + {SAFE_HAVEN}": (SAFE_HAVEN,)}  # descriptive
MIN_DWELL = 10  # retrain.gates.regime.min_avg_regime_dwell_days at registration
FORWARD = 21  # sessions: SPY's next-month return for the descriptive rank correlation


def bond_inputs(close: pd.DataFrame) -> pd.DataFrame:
    """Inputs 6 and 7 per session from total-return closes (SPY, TLT, IEF, HYG) up to T."""
    change = close / close.shift(WINDOW) - 1.0
    return pd.DataFrame({CREDIT: change["HYG"] - change["IEF"], SAFE_HAVEN: change["SPY"] - change["TLT"]},
                        index=close.index)


def candidate_scores(inputs: pd.DataFrame, settings: RegimeSettings, extra: tuple[str, ...]) -> pd.Series:
    """The five-input score plus the sign of each `extra` input; NaN where any input is missing."""
    score = score_inputs(inputs, settings)
    values = inputs.loc[:, list(extra)].astype(float)
    return (score + np.sign(values).sum(axis=1)).where(values.notna().all(axis=1))


def candidate_settings(settings: RegimeSettings) -> RegimeSettings:
    return dataclasses.replace(settings, enter_threshold=ENTER, exit_threshold=EXIT)


def trial_regimes(inputs: pd.DataFrame, settings: RegimeSettings) -> dict[str, pd.Series]:
    """v1, the candidate and the two single-input variants, from the same first session.

    ``inputs``: the five v1 inputs and the two bond inputs per session.
    """
    complete = inputs.dropna()
    if complete.empty:
        raise ValueError("no session has all seven inputs")
    start = complete.index[0]
    inputs = inputs.loc[start:]
    candidate = candidate_settings(settings)
    regimes = {V1: classify_regimes(score_inputs(inputs, settings), settings)["regime"],
               V2: classify_regimes(candidate_scores(inputs, settings, BOND_INPUTS), candidate)["regime"]}
    for name, extra in SINGLES.items():
        regimes[name] = classify_regimes(candidate_scores(inputs, settings, extra), candidate)["regime"]
    return regimes


def rule_v2_regime(engine: Engine, settings: Settings) -> pd.Series:
    """rule_based-v2 per session up to the latest stored one, from the same first session as in the
    trial (the first with all seven inputs): the forward-test shadow's regime (section 12.5.3)."""
    inputs, _ = load_trial_inputs(engine, settings)
    complete = inputs.dropna()
    if complete.empty:
        raise ValueError("no session has all seven rule_based-v2 inputs")
    inputs = inputs.loc[complete.index[0]:]
    return classify_regimes(candidate_scores(inputs, settings.regime, BOND_INPUTS),
                            candidate_settings(settings.regime))["regime"]


def load_trial_inputs(engine: Engine, settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(the seven inputs per session, the total-return closes of SPY / TLT / IEF / HYG)."""
    missing = [s for s in BOND_SYMBOLS if s not in settings.market.market_symbols]
    if missing:
        raise ValueError(f"add {', '.join(missing)} to market.market_symbols and run update-market-data")
    v1 = queries.load_regime_inputs(engine, settings.features.feature_version, settings.features.market_feature_version)
    close = load_prices(engine, ["SPY", *BOND_SYMBOLS]).close
    bonds = bond_inputs(close.reindex(v1.index))
    return v1.join(bonds), close


@dataclass(frozen=True)
class PointInTimeCheck:
    cut: pd.Timestamp
    sessions: int
    identical: bool


def point_in_time(inputs: pd.DataFrame, close: pd.DataFrame, settings: RegimeSettings,
                  regimes: dict[str, pd.Series], cuts: list[pd.Timestamp]) -> list[PointInTimeCheck]:
    """Recompute everything from data ending at each cut; inputs and regimes up to it must match."""
    checks = []
    for cut in cuts:
        cut_close = close.loc[:cut]
        cut_inputs = inputs.loc[:cut, [c for c in inputs.columns if c not in BOND_INPUTS]].join(
            bond_inputs(cut_close.reindex(inputs.loc[:cut].index)))
        same_inputs = np.allclose(cut_inputs[list(BOND_INPUTS)].to_numpy(dtype=float),
                                  inputs.loc[:cut, list(BOND_INPUTS)].to_numpy(dtype=float), equal_nan=True, rtol=0,
                                  atol=1e-12)
        truncated = trial_regimes(cut_inputs, settings)
        same = same_inputs and all(truncated[name].equals(regimes[name].loc[:cut]) for name in regimes)
        checks.append(PointInTimeCheck(pd.Timestamp(cut), len(truncated[V2]), bool(same)))
    return checks


@dataclass(frozen=True)
class BondTrialOutcome:
    folds: list[Fold]
    first_session: pd.Timestamp
    regimes: dict[str, pd.Series]
    comparison: RegimeComparison
    checks: list[PointInTimeCheck]
    gates: GateReport
    spy: pd.Series
    inputs: pd.DataFrame


def evaluate(comparison: RegimeComparison, regimes: dict[str, pd.Series], checks: list[PointInTimeCheck]) -> GateReport:
    window = regimes[V2].loc[comparison.start:comparison.end]
    v = comparison.versus(portfolio_name("regime_switch", V2), portfolio_name("regime_switch", V1))
    dwell = average_dwell(window)
    return GateReport([
        GateResult("beats_rule_based_regime_after_costs", PASS if v["sharpe"] > v["baseline_sharpe"] else FAIL,
                   f"regime_switch Sharpe {v['sharpe']:.4f} vs {v['baseline_sharpe']:.4f} with {V1}",
                   "higher Sharpe after costs"),
        GateResult("min_avg_regime_dwell_days", PASS if dwell >= MIN_DWELL else FAIL, f"{dwell:.1f} sessions",
                   f">= {MIN_DWELL}"),
        GateResult("point_in_time", PASS if checks and all(c.identical for c in checks) else FAIL,
                   f"{sum(c.identical for c in checks)}/{len(checks)} cuts identical "
                   f"({', '.join(f'{c.cut.date()}: {c.sessions:,} sessions' for c in checks)})",
                   "inputs and regimes unchanged when later sessions are added"),
    ])


def run_bond_trial(engine: Engine, settings: Settings) -> BondTrialOutcome:
    from quant.models.dataset import load_dataset
    from quant.strategies import PANEL_COLUMNS

    stocks = list(settings.market.stocks)
    dataset = load_dataset(engine, settings)
    holdout_start = split_periods(dataset.sessions, settings.holdout_months).holdout_start
    folds = walk_forward_folds(dataset.labelled().sessions, holdout_start, settings.retrain.validation,
                               settings.model.embargo_days)
    inputs, close = load_trial_inputs(engine, settings)
    regimes = trial_regimes(inputs, settings.regime)
    prices = load_prices(engine, [*stocks, "SPY"])
    panel = queries.load_feature_panel(engine, settings.features.feature_version, stocks, PANEL_COLUMNS)
    comparison = compare_regimes(regimes, folds, prices, panel, settings)
    checks = point_in_time(inputs, close, settings.regime, regimes, default_cuts(folds))
    return BondTrialOutcome(folds, regimes[V2].index[0], regimes, comparison, checks,
                            evaluate(comparison, regimes, checks), prices.close["SPY"], inputs)


# Descriptive tables (section 12.5: reported, decide nothing).

def regime_profile(regimes: dict[str, pd.Series], spy: pd.Series, start, end) -> pd.DataFrame:
    """Per regime series over the window: share per state, changes, average length, SPY a year per state
    (a session's SPY return is attributed to the regime known at the previous close)."""
    returns = spy.pct_change()
    rows = {}
    for name, regime in regimes.items():
        known = regime.shift(1).loc[start:end]
        window = regime.loc[start:end]
        row = {f"{s} share": percent((window == s).mean(), 0) for s in (BULL, SIDEWAYS, BEAR)}
        row["changes"] = int((window != window.shift()).sum() - 1)
        row["avg length"] = f"{average_dwell(window):.0f}"
        for s in (BULL, SIDEWAYS, BEAR):
            r = returns.reindex(known.index)[known == s]
            row[f"SPY/yr in {s}"] = "-" if r.empty else f"{(1 + r.mean()) ** 252 - 1:+.0%}"
        rows[name] = row
    return pd.DataFrame(rows).T.rename_axis("regime")


def input_correlations(inputs: pd.DataFrame, spy: pd.Series, start, end) -> pd.DataFrame:
    """Rank correlation of each raw bond input at T with SPY's next-21-session return, every 21st session."""
    forward = spy.shift(-FORWARD) / spy - 1.0
    sessions = inputs.loc[start:end].index[::FORWARD]
    rows = {}
    for name in BOND_INPUTS:
        pair = pd.DataFrame({"x": inputs.loc[sessions, name], "y": forward.reindex(sessions)}).dropna()
        rho = pair["x"].rank().corr(pair["y"].rank())
        t = rho * np.sqrt((len(pair) - 2) / (1 - rho ** 2)) if len(pair) > 2 and abs(rho) < 1 else np.nan
        rows[name] = {"rank correlation": f"{rho:+.3f}", "t": f"{t:+.2f}", "sessions": len(pair)}
    return pd.DataFrame(rows).T.rename_axis("input")


# Holdout check of rule_based-v2 (section 12.5.3; holdout evaluation #3).

HOLDOUT_START = pd.Timestamp("2025-10-10")  # the fixed holdout start
HOLDOUT_END = pd.Timestamp("2026-10-09")  # the last session before forward-test run 1 started
SHADOW, NO_SHADOW = "add shadow_rule_v2", "no shadow; v2 is closed"


@dataclass(frozen=True)
class HoldoutCheck:
    start: pd.Timestamp
    end: pd.Timestamp
    carried: pd.DataFrame  # portfolio x metrics on the window, holdings carried in (decides)
    fresh: pd.DataFrame  # the same, started from cash on the window's first session (descriptive)
    regime_days: pd.DataFrame  # regime x {v1, v2}: sessions in the window
    differing: pd.DataFrame  # sessions where v1 and v2 differ
    decision: str
    sharpe_ok: bool
    drawdown_ok: bool


def holdout_decision(sharpe: float, baseline_sharpe: float, drawdown: float, baseline_drawdown: float):
    """The table of section 12.5.3: only 'both worse' closes v2."""
    sharpe_ok, drawdown_ok = sharpe >= baseline_sharpe, drawdown >= baseline_drawdown
    return (SHADOW if sharpe_ok or drawdown_ok else NO_SHADOW), sharpe_ok, drawdown_ok


def run_holdout_check(engine: Engine, settings: Settings) -> HoldoutCheck:
    from quant.backtest.engine import simulate
    from quant.backtest.metrics import period_metrics
    from quant.backtest.orders import order_targets
    from quant.strategies import PANEL_COLUMNS, build_strategies

    stocks = list(settings.market.stocks)
    inputs, _ = load_trial_inputs(engine, settings)
    full = trial_regimes(inputs, settings.regime)
    regimes = {V1: full[V1], V2: full[V2]}
    prices = load_prices(engine, [*stocks, "SPY"])
    panel = queries.load_feature_panel(engine, settings.features.feature_version, stocks, PANEL_COLUMNS)
    sessions = prices.close.index
    history = sessions[(sessions >= regimes[V2].index[0]) & (sessions <= HOLDOUT_END)]
    window = history[history >= HOLDOUT_START]
    if len(window) == 0 or window[-1] != HOLDOUT_END:
        raise ValueError(f"prices must cover {HOLDOUT_START.date()} .. {HOLDOUT_END.date()}")
    bt = settings.backtest
    strategies = {s.name: s for s in build_strategies(settings) if s.name in ("regime_switch", "regime_switch_bear50")}

    def metrics(index):
        out = {}
        for strategy_name, strategy in strategies.items():
            for regime_name, regime in regimes.items():
                weights = strategy.weights(panel, regime).reindex(index).fillna(0.0)
                sim = simulate(order_targets(weights, bt.min_trade_weight), prices, bt)
                out[portfolio_name(strategy_name, regime_name)] = period_metrics(
                    sim.value, sim.invested, sim.orders, sim.trades, window[0], window[-1])
        return pd.DataFrame(out).T

    carried, fresh = metrics(history), metrics(window)
    v1, v2 = regimes[V1].reindex(window), regimes[V2].reindex(window)
    days = pd.DataFrame({V1: v1.value_counts(), V2: v2.value_counts()}).reindex([BULL, SIDEWAYS, BEAR]).fillna(0)
    differ = pd.DataFrame({V1: v1, V2: v2})[v1 != v2].rename_axis("session")
    a, b = portfolio_name("regime_switch", V2), portfolio_name("regime_switch", V1)
    decision, sharpe_ok, drawdown_ok = holdout_decision(carried.loc[a, "sharpe"], carried.loc[b, "sharpe"],
                                                        carried.loc[a, "max_drawdown"], carried.loc[b, "max_drawdown"])
    return HoldoutCheck(window[0], window[-1], carried, fresh, days.astype(int), differ, decision, sharpe_ok,
                        drawdown_ok)


def _metric_table(frame: pd.DataFrame) -> pd.DataFrame:
    shown = pd.DataFrame(index=frame.index)
    for key in ("cagr", "max_drawdown", "avg_invested"):
        if key in frame:
            shown[key] = frame[key].map(percent)
    if "sharpe" in frame:
        shown["sharpe"] = frame["sharpe"].map(lambda v: f"{v:.2f}")
    if "turnover" in frame:
        shown["turnover"] = frame["turnover"].map(lambda v: "-" if pd.isna(v) else f"{v:.1f}x")
    return shown.rename_axis("portfolio")


def write_holdout_report(check: HoldoutCheck, reports_dir: str | Path, now: datetime) -> Path:
    a, b = portfolio_name("regime_switch", V2), portfolio_name("regime_switch", V1)
    c = check.carried
    lines = [
        f"# Section 12.5.3: holdout check of {V2} ({now:%Y-%m-%d %H:%M} UTC)",
        "",
        f"Holdout evaluation #3, pre-registered 2026-10-15. Window {check.start.date()} .. {check.end.date()} "
        f"(the fixed holdout start to the last session before forward-test run 1); {len(check.differing)} of "
        f"{int(check.regime_days[V1].sum())} sessions where the two regimes differ. Both recursions start "
        "2007-05-09; nothing refit or changed. One year is a short test (Sharpe +/-1): the check can stop v2, "
        "not confirm it.",
        "",
        f"## Decision: {check.decision}",
        "",
        f"regime_switch, holdings carried in: Sharpe {c.loc[a, 'sharpe']:.2f} with {V2} vs {c.loc[b, 'sharpe']:.2f} "
        f"with {V1} ({'at least as high' if check.sharpe_ok else 'lower'}); max drawdown "
        f"{percent(c.loc[a, 'max_drawdown'])} vs {percent(c.loc[b, 'max_drawdown'])} "
        f"({'no deeper' if check.drawdown_ok else 'deeper'}).",
        "",
        "## Holdings carried in (decides for regime_switch; bear50 descriptive)",
        "",
        markdown_table(_metric_table(check.carried)),
        "",
        "## Started from cash on the window's first session (descriptive)",
        "",
        markdown_table(_metric_table(check.fresh)),
        "",
        "## Regime days in the window",
        "",
        markdown_table(check.regime_days.rename_axis("regime")),
        "",
        "## Sessions where v1 and v2 differ",
        "",
        markdown_table(check.differing.set_axis([d.date() for d in check.differing.index]).rename_axis("session"))
        if len(check.differing) else "None.",
        "",
        "Research output, not advice.",
    ]
    directory = Path(reports_dir) / "regime"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"rule-v2-holdout-{now:%Y%m%dT%H%M%SZ}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_trial_report(outcome: BondTrialOutcome, reports_dir: str | Path, now: datetime) -> Path:
    c = outcome.comparison
    overall = c.overall
    columns = ["cagr", "sharpe", "sortino", "max_drawdown", "volatility", "avg_invested", "turnover"]
    table = overall.loc[:, [k for k in columns if k in overall.columns]].copy()
    for k in ("cagr", "max_drawdown", "volatility", "avg_invested"):
        if k in table:
            table[k] = table[k].map(percent)
    for k in ("sharpe", "sortino"):
        if k in table:
            table[k] = table[k].map(lambda v: f"{v:.4f}")
    if "turnover" in table:
        table["turnover"] = table["turnover"].map(lambda v: "-" if pd.isna(v) else f"{v:.1f}x")
    v = c.versus(portfolio_name("regime_switch", V2), portfolio_name("regime_switch", V1))
    sharpe = c.by_fold["sharpe"].unstack()
    years = pd.DataFrame({name: sharpe[portfolio_name("regime_switch", name)] for name in outcome.regimes}).map(
        lambda x: f"{x:.2f}")
    years.index = [f"{f.validation_start.date()} .. {f.validation_end.date()}" for f in outcome.folds]
    years.index.name = "validation year"
    verdict = "PASSED" if outcome.gates.passed else "FAILED"
    gates = outcome.gates.table()
    lines = [
        f"# Section 12.5 trial: bond-market fear inputs ({now:%Y-%m-%d %H:%M} UTC)",
        "",
        f"Pre-registered 2026-10-15 (proposal.md section 12.5). Window {c.start.date()} .. {c.end.date()} "
        f"({len(outcome.folds)} validation years, holdout untouched); every regime's recursion starts "
        f"{outcome.first_session.date()} (first session with all seven inputs). Candidate {V2}: seven inputs, "
        f"E = {ENTER}, X = {EXIT}. Research only: {V1} stays ACTIVE.",
        "",
        f"## Gates: {verdict}",
        "",
        markdown_table(gates),
        "",
        f"regime_switch with {V2} is higher in {v['folds_higher_sharpe']} of {v['folds']} validation years; max "
        f"drawdown {percent(v['max_drawdown'])} vs {percent(v['baseline_max_drawdown'])}. An 11-year Sharpe "
        "difference below about 0.1 is within noise (section 12.5 (d)).",
        "",
        "## Trading comparison (descriptive except regime_switch [v2] vs [v1])",
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
        "## First BEAR session per sell-off",
        "",
        markdown_table(sell_off_reactions({k: r.loc[c.start:c.end] for k, r in outcome.regimes.items()},
                                          outcome.spy)),
        "",
        "## Each bond input against SPY's next 21 sessions",
        "",
        markdown_table(input_correlations(outcome.inputs, outcome.spy, c.start, c.end)),
        "",
        "Research output, not advice.",
    ]
    directory = Path(reports_dir) / "regime"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"bond-trial-{now:%Y%m%dT%H%M%SZ}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
