"""Validation report for a registered candidate (proposal section 16.2), as Markdown."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from quant.models.final import FinalModel, explain
from quant.models.folds import Fold
from quant.models.gates import GateReport, PredictorEvidence
from quant.models.ml_backtest import BASELINE, MLBacktest
from quant.models.search import SearchResult


def _table(frame: pd.DataFrame) -> str:
    """A plain Markdown table (no extra dependency)."""
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join([frame.index.name or ""] + columns) + " |", "|" + "---|" * (len(columns) + 1)]
    for index, row in frame.iterrows():
        lines.append("| " + " | ".join([str(index)] + [str(value) for value in row]) + " |")
    return "\n".join(lines)


def _pct(value: float, digits: int = 1) -> str:
    return "-" if pd.isna(value) else f"{value:.{digits}%}"


def validation_report(
    model_id: str,
    final: FinalModel,
    search: SearchResult,
    folds: list[Fold],
    backtest: MLBacktest,
    portfolio: str,
    evidence: PredictorEvidence,
    gates: GateReport,
    target_description: str,
    created_at: datetime,
) -> str:
    walk = search.best.folds
    verdict = "PASSED" if gates.passed else "FAILED"
    lines = [
        f"# Validation report: {model_id}",
        "",
        f"- **Role:** PREDICTOR · **status:** CANDIDATE · **gates:** {verdict}",
        f"- **Model:** {final.model_type} {final.parameters}",
        f"- **Target:** {target_description}",
        f"- **Final refit:** {final.training_start.date()} .. {final.training_end.date()} "
        f"({final.training_rows:,} rows); calibrated on the walk-forward out-of-sample predictions",
        f"- **Trials in this run:** {search.trial_count} (earlier research trials are listed in proposal section 15.4)",
        f"- **Created:** {created_at:%Y-%m-%d %H:%M} UTC",
        "",
        "Research output, not advice. A candidate changes nothing until it is promoted, and promotion refuses "
        "a candidate whose gates failed unless overridden with a logged reason.",
        "",
        "## Gates",
        "",
        _table(gates.table()),
        "",
        "## Walk-forward ML backtest (after costs)",
        "",
        f"Window {backtest.start.date()} .. {backtest.end.date()}; every session traded with a model trained "
        f"only before it. `{portfolio}` trades the calibrated probabilities as `ml_select`; "
        f"`{BASELINE}` is the rule-based baseline.",
        "",
    ]
    overall = backtest.overall
    lines.append(_table(pd.DataFrame({
        "CAGR": overall["cagr"].map(_pct),
        "Sharpe": overall["sharpe"].map(lambda v: f"{v:.2f}"),
        "max drawdown": overall["max_drawdown"].map(_pct),
        "turnover/yr": overall["turnover"].map(lambda v: "-" if pd.isna(v) else f"{v:.1f}x"),
        "fees": overall["fees_paid"].map(lambda v: f"{v:,.0f}"),
    }).rename_axis("portfolio")))
    sharpe = backtest.by_fold["sharpe"].unstack()
    sharpe.index = [f"{f.validation_start.date()} .. {f.validation_end.date()}" for f in folds]
    lines += [
        "",
        "Sharpe by validation year:",
        "",
        _table(sharpe[[portfolio, BASELINE, "SPY", "equal_weight"]].map(lambda v: f"{v:.2f}").rename_axis("fold")),
        "",
    ]
    if portfolio in backtest.random.index:
        r = backtest.random.loc[portfolio]
        lines += [
            f"Random baseline: Sharpe {r['sharpe']:.2f} beats {r['sharpe_percentile']:.0%} of {int(r['runs'])} runs "
            f"with the same timing and sizes but randomly permuted stocks (median {r['random_sharpe_median']:.2f}).",
            "",
        ]
    lines += [
        "## Probability quality per fold",
        "",
        "Skill compares with always predicting the fold's training base rate (0 = no better).",
        "",
        _table(pd.DataFrame({
            "validation": walk["validation_start"].astype(str) + " .. " + walk["validation_end"].astype(str),
            "rows": walk["rows"],
            "AUC": walk["auc"].map(lambda v: f"{v:.3f}"),
            "log-loss skill": walk["log_loss_skill"].map(lambda v: f"{v:+.2%}"),
            "Brier skill": walk["brier_skill"].map(lambda v: f"{v:+.2%}"),
            "calibration error": walk["calibration_error"].map(lambda v: f"{v:.3f}"),
        })),
        "",
        "## Hyperparameter search",
        "",
        "Selection: highest mean minus one standard deviation of the per-fold log-loss skill.",
        "",
    ]
    trials = search.trials
    params = [column for column in trials.columns if column.startswith("param_")]
    view = trials[params].rename(columns=lambda c: c[len("param_"):])
    view["score"] = trials["score"].map(lambda v: f"{v:+.3%}")
    view["log-loss skill"] = trials["log_loss_skill_mean"].map(lambda v: f"{v:+.2%}")
    view["AUC"] = trials["auc_mean"].map(lambda v: f"{v:.3f}")
    lines += [_table(view), "", "## What the model leans on", ""]
    explanation = explain(final).head(15)
    heading = "coefficient (per standard deviation of the input)" if explanation.name == "coefficient" else "importance"
    lines += [
        _table(explanation.map(lambda v: f"{v:+.3f}" if heading.startswith("coefficient") else f"{v:.1%}")
               .rename(heading).to_frame().rename_axis("input")),
        "",
        "Correlated inputs share their effect, so single coefficients should not be read alone.",
        "",
        "## Leak checks",
        "",
    ]
    lines += [f"- {'PASS' if c.passed else 'FAIL'}: {c.name} ({c.detail})" for c in evidence.leak_checks]
    lines += [
        "",
        "## Caveats",
        "",
        "- The stock pool and its sectors are chosen today (survivorship bias); cash earns nothing; the history is "
        "mostly a bull market.",
        "- The holdout was " + ("evaluated." if evidence.holdout_sharpe is not None else "not evaluated."),
        "",
    ]
    return "\n".join(lines)
