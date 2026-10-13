"""`quant retrain --role predictor --model all`: retrain every model type, then compare them.

Each model type is retrained exactly as by `--model <type>` (its own search, walk-forward ML
backtest, gates, registered CANDIDATE and report). All share the same folds, so their
walk-forward results sit on the same sessions and compare directly with each other and with
`regime_switch` (rule-based regime), SPY and equal weight.

The comparison is descriptive: picking the best of several after looking flatters the winner,
and no gate changes because of it. The holdout is not used (it is checked once, for one chosen
candidate, with `--model <type> --evaluate-holdout`).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quant.models.ml_backtest import BASELINE
from quant.models.report import markdown_table, percent
from quant.models.retrain import RetrainOutcome, portfolio_name

REFERENCE = (BASELINE, "SPY", "equal_weight")


def comparison_table(outcomes: list[RetrainOutcome]) -> pd.DataFrame:
    """One row per model type, then the reference portfolios of the same walk-forward window."""
    rows = {}
    for outcome in outcomes:
        overall = outcome.backtest.overall.loc[portfolio_name(outcome.model_type)]
        evidence = outcome.evidence
        failed = [r.name for r in outcome.gates.results if r.status == "FAIL"]
        rows[outcome.model_type] = {
            "CAGR": overall["cagr"], "Sharpe": overall["sharpe"], "max drawdown": overall["max_drawdown"],
            "turnover/yr": overall["turnover"],
            "years beating regime_switch": f"{evidence.folds_higher_sharpe}/{evidence.folds}",
            "AUC": float(outcome.search.best.summary()["auc_mean"]),
            "calibration error": evidence.calibration_error,
            "gates": "PASSED" if outcome.gates.passed else f"FAILED ({', '.join(failed)})",
            "candidate": outcome.registration.model_id,
        }
    reference = outcomes[0].backtest.overall
    for name in REFERENCE:
        if name in reference.index:
            r = reference.loc[name]
            rows[name] = {"CAGR": r["cagr"], "Sharpe": r["sharpe"], "max drawdown": r["max_drawdown"],
                          "turnover/yr": r["turnover"], "years beating regime_switch": "",
                          "AUC": float("nan"), "calibration error": float("nan"),
                          "gates": "baseline" if name == BASELINE else "benchmark", "candidate": ""}
    return pd.DataFrame(rows).T.rename_axis("portfolio")


def format_comparison(table: pd.DataFrame) -> pd.DataFrame:
    def number(fmt):
        return lambda v: "-" if v == "" or pd.isna(v) else fmt(v)

    return table.assign(**{
        "CAGR": table["CAGR"].map(number(lambda v: percent(v))),
        "Sharpe": table["Sharpe"].map(number(lambda v: f"{v:.2f}")),
        "max drawdown": table["max drawdown"].map(number(lambda v: percent(v))),
        "turnover/yr": table["turnover/yr"].map(number(lambda v: f"{v:.1f}x")),
        "AUC": table["AUC"].map(number(lambda v: f"{v:.3f}")),
        "calibration error": table["calibration error"].map(number(lambda v: f"{v:.3f}")),
    })


def write_comparison(outcomes: list[RetrainOutcome], table: pd.DataFrame, reports_dir: str | Path) -> Path:
    now = datetime.now(timezone.utc)
    folds = outcomes[0].folds
    path = Path(reports_dir) / "comparison" / f"predictors-{now:%Y%m%dT%H%M%SZ}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Predictor comparison {now:%Y-%m-%d %H:%M} UTC",
        "",
        f"Walk-forward {folds[0].validation_start.date()} .. {folds[-1].validation_end.date()} "
        f"({len(folds)} yearly folds), after costs; `ml_select` with each model against `{BASELINE}` "
        "(rule-based regime). The holdout was not used.",
        "",
        markdown_table(format_comparison(table)),
        "",
        "Descriptive only: choosing the best of several after looking flatters it, and the gates are unchanged. "
        "Each model's own report is in its folder under the reports directory.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
