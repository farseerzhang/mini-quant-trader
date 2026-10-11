"""Validation report for a REGIME candidate (proposal section 16.2), as Markdown."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from quant.models.gates import GateReport
from quant.models.report import markdown_table, percent
from quant.regime.comparison import REGIME_STRATEGIES, RegimeComparison, portfolio_name
from quant.regime.gates import RegimeEvidence
from quant.regime.hmm import LABELS_BY_MEAN_RETURN, FittedHMM
from quant.regime.hmm_walk_forward import HMMWalkForward
from quant.regime.rule_based import average_dwell

# Sell-offs to describe when they fall inside the window: (name, look from, look until).
SELL_OFFS = (
    ("2018 Q4", "2018-10-01", "2019-03-29"),
    ("2020 COVID", "2020-02-14", "2020-08-31"),
    ("2022", "2021-12-27", "2023-03-31"),
    ("2025 tariffs", "2025-02-14", "2025-07-31"),
)


def describe_states(fit: FittedHMM) -> pd.DataFrame:
    d = fit.describe()
    return pd.DataFrame({
        "daily return": d["spy_return_1d mean"].map(lambda v: f"{v:+.3%}"),
        "per year": d["spy_return_1d mean"].map(lambda v: f"{(1 + v) ** 252 - 1:+.0%}"),
        "market vol": np.exp(d["log_market_volatility mean"]).map(lambda v: f"{v:.0%}"),
        "VIX": np.exp(d["log_vix mean"]).map(lambda v: f"{v:.1f}"),
        "expected length": d["expected sessions"].map(lambda v: f"{v:.0f} sessions"),
    }).rename_axis("state")


def sell_off_reactions(regimes: dict[str, pd.Series], spy: pd.Series) -> pd.DataFrame:
    rows = {}
    for name, start, end in SELL_OFFS:
        cells = {}
        for label, regime in regimes.items():
            window = regime.loc[start:end]
            bear = window[window == "BEAR"]
            if window.empty:
                cells[label] = "outside the window"
            elif bear.empty:
                cells[label] = "no BEAR"
            else:
                first = bear.index[0]
                drop = spy.loc[first] / spy.loc[start:first].max() - 1
                cells[label] = f"{first.date()} ({drop:+.0%} from peak)"
        rows[name] = cells
    return pd.DataFrame(rows).T.rename_axis("sell-off")


def regime_report(
    model_id: str,
    final: FittedHMM,
    walk_forward: HMMWalkForward,
    rule: pd.Series,
    comparison: RegimeComparison,
    evidence: RegimeEvidence,
    gates: GateReport,
    spy: pd.Series,
    created_at: datetime,
    chart_files: dict[str, str] | None = None,
    current: dict | None = None,
) -> str:
    hmm = walk_forward.regime
    rule = rule.reindex(hmm.index)
    overall = comparison.overall
    portfolios = [portfolio_name(s, r) for s in REGIME_STRATEGIES for r in ("rule", "hmm")] + ["SPY", "equal_weight"]
    timeline = pd.DataFrame({
        label: {**{name: percent((series == name).mean(), 0) for name in LABELS_BY_MEAN_RETURN},
                "average length": f"{average_dwell(series):.0f} sessions",
                "changes": int((series != series.shift()).sum() - 1)}
        for label, series in (("HMM", hmm), ("rule-based", rule))
    }).T.rename_axis("regime")
    lines = [
        f"# Validation report: {model_id}",
        "",
        f"- **Role:** REGIME · **status:** CANDIDATE · **gates:** {'PASSED' if gates.passed else 'FAILED'}",
        f"- **Model:** 3-state Gaussian HMM on {', '.join(final.columns)} (filtered probabilities only)",
        f"- **Final fit:** {final.train_start.date()} .. {final.train_end.date()}",
        f"- **Created:** {created_at:%Y-%m-%d %H:%M} UTC",
        "",
        "Research output, not advice. A candidate changes nothing until it is promoted; promotion refuses a "
        "candidate whose gates failed unless overridden with a logged reason.",
        "",
        "## Gates",
        "",
        markdown_table(gates.table()),
        "",
    ]
    if current:
        probabilities = ", ".join(f"{label} {percent(current['probabilities'][label], 0)}" for label in LABELS_BY_MEAN_RETURN)
        forward = current["final_fit_from"]
        lines += [
            "## Current regime",
            "",
            f"- **HMM (final fit), {current['date'].date()}:** {current['hmm']} (filtered probabilities: {probabilities})",
            f"- **Rule-based, {current['rule_date'].date()}:** {current['rule']}",
            "",
            f"The final fit was trained on sessions up to {final.train_end.date()}"
            + (f"; from {forward.date()} on it is applied forward, as it would run live" if forward is not None else "")
            + ". Those sessions only get regime labels here; no gate or trading result uses them.",
            "",
        ]
    if chart_files:
        lines += [
            "## Regime timeline",
            "",
            f"![SPY (log scale) with the HMM and rule-based regimes]({chart_files['svg']})",
            "",
            f"Interactive version with hover: [{chart_files['html']}]({chart_files['html']}). The HMM ribbon is "
            f"out of sample from {hmm.index[0].date()}; before that it is the final fit's filtered regime, "
            "which has seen those years (in sample, drawn faded). The rule-based regime has no fitted "
            "parameters and is point in time throughout. After the walk-forward window (dashed line) the HMM "
            "ribbon is the final fit applied forward, up to the latest session.",
            "",
        ]
    lines += [
        "## States of the final fit",
        "",
        markdown_table(describe_states(final)),
        "",
        f"## Walk-forward regime vs the rule-based regime ({hmm.index[0].date()} .. {hmm.index[-1].date()})",
        "",
        markdown_table(timeline),
        "",
        f"Same label on {percent((hmm == rule).mean(), 0)} of sessions.",
        "",
        "First BEAR session in each sell-off:",
        "",
        markdown_table(sell_off_reactions({"HMM": hmm, "rule-based": rule}, spy)),
        "",
        "## Trading with each regime (after costs)",
        "",
        markdown_table(pd.DataFrame({
            "CAGR": overall.loc[portfolios, "cagr"].map(percent),
            "Sharpe": overall.loc[portfolios, "sharpe"].map(lambda v: f"{v:.4f}"),
            "max drawdown": overall.loc[portfolios, "max_drawdown"].map(percent),
            "invested": overall.loc[portfolios, "avg_invested"].map(lambda v: percent(v, 0)),
            "turnover/yr": overall.loc[portfolios, "turnover"].map(lambda v: "-" if pd.isna(v) else f"{v:.1f}x"),
        }).rename_axis("portfolio")),
        "",
        "## Stability across refits",
        "",
        markdown_table(evidence.agreement.map(lambda v: percent(v)).rename("agreement with the previous fit")
               .to_frame().rename_axis("fold")),
        "",
        "## Point-in-time checks",
        "",
    ]
    lines += [f"- {'PASS' if c.passed else 'FAIL'}: {c.name} ({c.detail})" for c in evidence.point_in_time_checks]
    lines += [
        "",
        "## Caveats",
        "",
        "- The HMM's BEAR is a high-volatility state; volatility stays high through rebounds, so it can stay "
        "BEAR while prices recover.",
        "- The stock pool is chosen today (survivorship bias); cash earns nothing; the holdout was not used "
        "for any gate or trading result (the chart and the current regime only label its sessions).",
        "",
    ]
    return "\n".join(lines)
