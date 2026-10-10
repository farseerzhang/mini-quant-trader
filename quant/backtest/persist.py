"""Save backtest results so they are reproducible and the dashboard can show them (section 6.11).

One `quant backtest` invocation is a batch. Each portfolio gets a DEVELOPMENT
row, and a HOLDOUT row only when the holdout was explicitly evaluated; the
stored equity curve likewise stops at the end of development unless it was.
Every row records the costs, parameters, a hash of the full configuration and
the git commit, so a result can be traced back to the code and settings that
produced it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine

from quant.backtest.engine import STRATEGY, BacktestReport, PortfolioResult
from quant.config.settings import Settings
from quant.persistence import queries

DEVELOPMENT = "DEVELOPMENT"
HOLDOUT = "HOLDOUT"
EXECUTION_CONVENTION = "signal at close T, fill at open T+1"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def config_hash(settings: Settings) -> str:
    """SHA-256 of the full loaded configuration (canonical JSON)."""
    canonical = json.dumps(dataclasses.asdict(settings), sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def git_commit() -> str | None:
    """The current commit, with '+dirty' if the working tree has uncommitted changes."""
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=_PROJECT_ROOT, capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        return commit + ("+dirty" if git("status", "--porcelain") else "")
    except (OSError, subprocess.SubprocessError):
        return None


def new_batch_id(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return f"{now:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"


def _clean(value):
    """NaN and numpy scalars -> plain Python (None for NaN)."""
    if value is None:
        return None
    if isinstance(value, float) and value != value:
        return None
    return value.item() if hasattr(value, "item") else value


def _equity(result: PortfolioResult, start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    value = result.value.loc[start:end]
    invested = result.invested.loc[start:end]
    return [
        {"date": stamp.date(), "value": float(amount), "invested": _clean(float(invested.loc[stamp]))}
        for stamp, amount in value.items()
    ]


def save_backtest(engine: Engine, settings: Settings, report: BacktestReport, include_holdout: bool) -> str:
    """Store one batch; returns its batch_id."""
    batch_id = new_batch_id()
    now = datetime.now(timezone.utc)
    periods = report.periods
    shared = {
        "batch_id": batch_id,
        "universe": json.dumps(list(settings.market.stocks)),
        "initial_cash": settings.backtest.initial_cash,
        "fee_bps": settings.backtest.fee_bps,
        "slippage_bps": settings.backtest.slippage_bps,
        "min_trade_weight": settings.backtest.min_trade_weight,
        "execution_convention": EXECUTION_CONVENTION,
        "feature_version": settings.features.feature_version,
        "parameters": json.dumps(
            {"strategies": dataclasses.asdict(settings.strategies), "risk": dataclasses.asdict(settings.risk)},
            sort_keys=True,
        ),
        "config_hash": config_hash(settings),
        "git_commit": git_commit(),
        "created_at": now,
    }
    spy = next((result for result in report.results if result.name == "SPY"), None)
    random = report.random_baseline_summary()

    windows = [(DEVELOPMENT, "development", periods.start, periods.development_end)]
    if include_holdout:
        windows.append((HOLDOUT, "holdout", periods.holdout_start, periods.end))

    runs = []
    for result in report.results:
        is_strategy = result.kind == STRATEGY
        for period, attribute, start, end in windows:
            metrics = getattr(result, attribute)
            row = {
                **shared,
                "strategy": result.name,
                "kind": result.kind,
                "strategy_version": f"strategies-v{settings.strategies.version}" if is_strategy else None,
                "description": result.description,
                "period": period,
                "start_date": start.date(),
                "end_date": end.date(),
                "model_version": settings.regime.rule_version if is_strategy else None,
                "benchmark": "SPY" if spy is not None else None,
                "benchmark_total_return": _clean(getattr(spy, attribute)["total_return"]) if spy else None,
                "benchmark_sharpe": _clean(getattr(spy, attribute)["sharpe"]) if spy else None,
                **{name: _clean(value) for name, value in metrics.items()},
            }
            # The random baseline is a development-period comparison only.
            if period == DEVELOPMENT and result.name in random.index:
                ranking = random.loc[result.name]
                row.update(
                    random_runs=int(ranking["runs"]),
                    random_sharpe_median=_clean(float(ranking["random_sharpe_median"])),
                    sharpe_percentile=_clean(float(ranking["sharpe_percentile"])),
                    random_cagr_median=_clean(float(ranking["random_cagr_median"])),
                    cagr_percentile=_clean(float(ranking["cagr_percentile"])),
                )
            runs.append((row, _equity(result, start, end)))
    queries.save_backtest_runs(engine, runs)
    return batch_id
