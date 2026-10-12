"""`quant forward-report`: how every forward-test portfolio has done since the start (Phase 10).

Read-only: the paper portfolios are re-simulated from the frozen targets and the real account is
replayed from its trade log, in memory. Scores follow the pre-registration: total return,
annualised volatility, Sharpe (daily, no risk-free rate), max drawdown, sessions covered and
missed; for the real account also the followed share and the cost of execution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy import Engine

from quant.config.settings import Settings
from quant.forward.paper import BENCHMARK, run_paper
from quant.forward.real import execution_costs, followed_share, real_portfolio
from quant.persistence import queries

FIRST_LOOK, VERDICT = 126, 252  # forward sessions (about 6 and 12 months)


def score(value: pd.Series) -> dict[str, float]:
    """Total return, annualised volatility and Sharpe of daily returns, max drawdown."""
    value = value.dropna()
    returns = value.pct_change().dropna()
    sd = returns.std(ddof=1) if len(returns) > 1 else float("nan")
    return {
        "total_return": float(value.iloc[-1] / value.iloc[0] - 1) if len(value) else float("nan"),
        "volatility": float(sd * math.sqrt(252)) if sd == sd else float("nan"),
        "sharpe": float(returns.mean() / sd * math.sqrt(252)) if sd and sd == sd else float("nan"),
        "max_drawdown": float((value / value.cummax() - 1).min()) if len(value) else float("nan"),
    }


@dataclass(frozen=True)
class ForwardReport:
    started_on: pd.Timestamp | None
    start_value: float | None
    last_session: pd.Timestamp | None
    sessions: int  # forward sessions after the start (returns counted)
    scores: pd.DataFrame  # portfolio x metrics (+ missed)
    values: pd.DataFrame  # dates x portfolio
    followed: pd.Series  # real account, per session
    costs: pd.DataFrame  # real account, per trade
    assumed_cost_bps: float
    warnings: list[str]

    @property
    def started(self) -> bool:
        return self.started_on is not None

    @property
    def checkpoint(self) -> str:
        if self.sessions < FIRST_LOOK:
            return (f"{self.sessions} forward sessions: before the 6-month first look "
                    f"({FIRST_LOOK - self.sessions} to go). No decision on this evidence yet.")
        if self.sessions < VERDICT:
            return (f"{self.sessions} forward sessions: past the 6-month first look; the 12-month verdict is "
                    f"{VERDICT - self.sessions} sessions away.")
        return f"{self.sessions} forward sessions: past the 12-month verdict point."


def build_report(engine: Engine, settings: Settings) -> ForwardReport:
    assumed = settings.backtest.fee_bps + settings.backtest.slippage_bps
    paper = run_paper(engine, settings)
    if paper is None:
        empty = pd.DataFrame()
        return ForwardReport(None, None, None, 0, empty, empty, pd.Series(dtype=float), empty, assumed, [])
    values = pd.DataFrame({name: sim.value for name, sim in paper.simulations.items()})
    warnings: list[str] = []
    followed, costs = pd.Series(dtype=float), pd.DataFrame()
    real = real_portfolio(engine, settings, paper.sessions)
    if real is not None:
        _, ledger, valued, real_warnings = real
        warnings += real_warnings
        values["real"] = valued["value"]
        weights = valued.filter(like="value:").div(valued["value"], axis=0)
        weights.columns = [c.removeprefix("value:") for c in weights.columns]
        followed = followed_share(weights, queries.load_forward_targets(engine, "live"))
        symbols = sorted({t.symbol for t in ledger.trades})
        opens = queries.load_raw_opens(engine, symbols, since=paper.started_on.date()) if symbols else pd.DataFrame()
        costs = execution_costs(ledger.trades, opens)
    order = ["live", *settings.forward_test.shadows, "real", BENCHMARK]
    values = values[[c for c in order if c in values.columns]]
    scores = pd.DataFrame({name: score(values[name]) for name in values.columns}).T
    scores["missed"] = [len(paper.missed.get(name, [])) if name in paper.missed else np.nan for name in scores.index]
    return ForwardReport(paper.started_on, paper.start_value, paper.sessions[-1], len(paper.sessions) - 1,
                         scores, values, followed, costs, assumed, warnings)


def format_report(report: ForwardReport) -> str:
    if not report.started:
        return ("The forward test has not started: the first `quant run-daily` after the close, before the next "
                "open, freezes its first session.")
    lines = [
        f"Forward test since {report.started_on.date()} (start value {report.start_value:,.2f}), "
        f"through {report.last_session.date()}",
        report.checkpoint,
        "",
    ]
    table = pd.DataFrame({
        "return": report.scores["total_return"].map(lambda v: f"{v:+.2%}"),
        "volatility": report.scores["volatility"].map(lambda v: "-" if v != v else f"{v:.1%}"),
        "Sharpe": report.scores["sharpe"].map(lambda v: "-" if v != v else f"{v:.2f}"),
        "max DD": report.scores["max_drawdown"].map(lambda v: f"{v:.1%}"),
        "missed": report.scores["missed"].map(lambda v: "-" if v != v else f"{int(v)}"),
    }).rename_axis("portfolio")
    lines.append(table.to_string())
    if not report.followed.empty:
        lines += ["", f"Followed share of the live targets: {report.followed.mean():.0%} on average, "
                      f"{report.followed.iloc[-1]:.0%} at the latest close"]
    if not report.costs.empty:
        weighted = (report.costs["cost_bps"] * report.costs["notional"]).sum() / report.costs["notional"].sum()
        lines.append(f"Execution: {len(report.costs)} trades, {weighted:+.1f} bp against the open on average "
                     f"(fees included; the backtest assumes {report.assumed_cost_bps:g} bp)")
    elif "real" in report.scores.index:
        lines.append("Execution: no trades recorded yet")
    lines += [f"warning: {w}" for w in report.warnings]
    lines += ["", "Paper portfolios trade their frozen targets at the next open with the backtest's costs; the real "
                  "account is the trade log at as-traded closes (dividends only if recorded). One year's Sharpe has "
                  "an uncertainty of about +/-1: research output, not advice."]
    return "\n".join(lines)
