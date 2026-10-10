"""Plain-text tables for `quant backtest`."""

from __future__ import annotations

import pandas as pd

from quant.backtest.engine import BacktestReport

_SURVIVORSHIP_NOTE = (
    "Caveats: the stock pool and its sectors are chosen today (survivorship bias), cash earns nothing,\n"
    "and the history is mostly a bull market. Win rate and profit factor only mean something for strategies."
)


def _pct(value: float, digits: int = 1) -> str:
    return "-" if pd.isna(value) else f"{value:.{digits}%}"


def _num(value: float, digits: int = 2) -> str:
    return "-" if pd.isna(value) else f"{value:.{digits}f}"


def metrics_table(table: pd.DataFrame) -> str:
    formatted = pd.DataFrame(
        {
            "CAGR": table["cagr"].map(_pct),
            "total": table["total_return"].map(lambda v: _pct(v, 0)),
            "vol": table["volatility"].map(_pct),
            "Sharpe": table["sharpe"].map(_num),
            "Sortino": table["sortino"].map(_num),
            "max DD": table["max_drawdown"].map(_pct),
            "invested": table["avg_invested"].map(lambda v: _pct(v, 0)),
            "turnover/yr": table["turnover"].map(lambda v: "-" if pd.isna(v) else f"{v:.1f}x"),
            "orders": table["num_trades"].map(lambda v: "-" if pd.isna(v) else f"{int(v)}"),
            "win rate": table["win_rate"].map(lambda v: _pct(v, 0)),
            "profit f.": table["profit_factor"].map(_num),
        }
    )
    return formatted.to_string()


def random_table(summary: pd.DataFrame) -> str:
    formatted = pd.DataFrame(
        {
            "Sharpe": summary["sharpe"].map(_num),
            "random median": summary["random_sharpe_median"].map(_num),
            "beats % of runs": summary["sharpe_percentile"].map(lambda v: _pct(v, 0)),
            "CAGR": summary["cagr"].map(_pct),
            "random median ": summary["random_cagr_median"].map(_pct),
            "beats % of runs ": summary["cagr_percentile"].map(lambda v: _pct(v, 0)),
        }
    )
    return formatted.to_string()


def development_report(report: BacktestReport) -> str:
    periods = report.periods
    lines = [
        f"DEVELOPMENT {periods.start.date()} .. {periods.development_end.date()}",
        metrics_table(report.table("development")),
    ]
    summary = report.random_baseline_summary()
    if not summary.empty:
        runs = int(summary["runs"].iloc[0])
        lines += [
            "",
            f"Random baseline (development; {runs} runs with the same timing and sizes, stocks permuted):",
            random_table(summary),
        ]
    lines += ["", _SURVIVORSHIP_NOTE]
    return "\n".join(lines)


def holdout_report(report: BacktestReport, evaluation_number: int) -> str:
    periods = report.periods
    return "\n".join(
        [
            f"HOLDOUT {periods.holdout_start.date()} .. {periods.end.date()} "
            f"(evaluation #{evaluation_number}; logged in pipeline_runs)",
            metrics_table(report.table("holdout")),
            "If this result changes a rule or setting, the holdout is consumed: treat it as development "
            "data from now on (section 15.2).",
        ]
    )
