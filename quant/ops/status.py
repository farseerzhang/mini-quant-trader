"""`quant status`: last runs, data freshness, active model per role, warnings (section 24).

Read-only. Every check reads what the commands stored; nothing is recalculated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine

from quant.config.settings import Settings
from quant.data.calendar import last_completed_session
from quant.persistence import queries, tables
from quant.regime.pipeline import RULE_METHOD
from quant.signals.positions import MANUAL

ROLES = ("REGIME", "PREDICTOR")
FALLBACK = {"REGIME": "rule-based regime", "PREDICTOR": "rule-based signals only"}
DAILY_STAGES = (  # (label, command, stage) in run-daily order
    ("market data", "update-market-data", "ingest"),
    ("features", "calculate-features", "features"),
    ("regime", "calculate-regime", "regime"),
    ("signals", "generate-signals", "signals"),
)


@dataclass(frozen=True)
class StageStatus:
    label: str
    last_ok: datetime | None
    data_as_of: date | None
    failed_since: datetime | None  # a FAILED run after the last OK one
    message: str | None


@dataclass(frozen=True)
class StatusReport:
    checked_at: datetime
    expected_session: date | None  # last completed NYSE session
    prices_as_of: date | None  # SPY's last stored session
    stale_symbols: dict[str, date]
    missing_symbols: list[str]
    quarantined_last_week: int
    quarantined_total: int
    features_as_of: date | None
    regime_as_of: date | None
    regime: str | None
    regime_score: float | None
    signals_as_of: date | None
    signal_counts: dict[str, int]
    signal_strategy: str | None
    positions_as_of: date | None
    positions_file: Path
    positions_file_changed: bool  # edited after the last stored snapshot
    stages: list[StageStatus]
    last_run_daily: datetime | None
    active: dict[str, dict | None]
    candidates: dict[str, tuple[int, int]]  # role -> (candidates, of which gates passed)
    holdout_evaluations: int
    warnings: list[str] = field(default_factory=list)
    forward: ForwardStatus | None = None
    strategy_in_use: str = ""  # ml_select with an ACTIVE predictor, else signals.strategy
    regime_in_use: str = ""  # the ACTIVE regime model id, else rule_based-v<N>


def _stage(runs: pd.DataFrame, label: str, command: str, stage: str) -> StageStatus:
    mine = runs[(runs["command"] == command) & (runs["stage"] == stage)]
    ok = mine[mine["status"] == "OK"]
    failed = mine[mine["status"] == "FAILED"]
    last_ok = ok.iloc[-1] if not ok.empty else None
    last_failed = failed.iloc[-1] if not failed.empty else None
    failed_since = None
    if last_failed is not None and (last_ok is None or last_failed["run_id"] > last_ok["run_id"]):
        failed_since = last_failed["finished_at"]
    return StageStatus(
        label=label,
        last_ok=None if last_ok is None else last_ok["finished_at"],
        data_as_of=None if last_ok is None else last_ok["data_as_of_date"],
        failed_since=failed_since,
        message=None if last_ok is None else last_ok["message"],
    )


def collect_status(engine: Engine, settings: Settings, now: datetime | None = None) -> StatusReport:
    now = now or datetime.now(timezone.utc)
    warnings: list[str] = []
    try:
        expected = last_completed_session(now, settings.market.calendar)
    except ValueError:
        expected = None

    last_dates = queries.latest_price_dates(engine)
    prices_as_of = last_dates.get("SPY") or (max(last_dates.values()) if last_dates else None)
    universe = settings.market.universe
    missing = [s for s in universe if s not in last_dates]
    stale = {s: last_dates[s] for s in universe if s in last_dates and prices_as_of and last_dates[s] < prices_as_of}

    features_as_of = queries.latest_date(engine, tables.features.c.date,
                                         tables.features.c.feature_version == settings.features.feature_version)
    regimes = queries.load_regimes(engine, RULE_METHOD, settings.regime.rule_version)
    regime_as_of = regimes.index[-1].date() if not regimes.empty else None
    score = queries.latest_date(engine, tables.market_regimes.c.regime_score,
                                tables.market_regimes.c.date == regime_as_of,
                                tables.market_regimes.c.regime_method == RULE_METHOD,
                                tables.market_regimes.c.model_version == settings.regime.rule_version) if regime_as_of else None

    active = {role: (dict(row) if (row := queries.active_model(engine, role)) is not None else None) for role in ROLES}
    strategy_in_use = "ml_select" if active["PREDICTOR"] is not None else settings.signals.strategy
    regime_in_use = active["REGIME"]["model_id"] if active["REGIME"] is not None else settings.regime.rule_version
    signals_as_of = queries.latest_date(engine, tables.signals.c.date)
    signal_counts, signal_strategy = {}, None
    if signals_as_of is not None:
        stored = queries.load_signals(engine, signals_as_of, strategy_in_use)
        signal_counts = stored["signal"].value_counts().to_dict()
        signal_strategy = strategy_in_use if not stored.empty else None

    positions_as_of = queries.latest_date(engine, tables.positions.c.as_of_date, tables.positions.c.source == MANUAL)
    snapshot_created = queries.latest_date(engine, tables.positions.c.created_at, tables.positions.c.source == MANUAL)
    positions_file = Path(settings.positions.path)
    file_changed = False
    if positions_file.exists() and snapshot_created is not None:
        modified = datetime.fromtimestamp(positions_file.stat().st_mtime, timezone.utc)
        created = snapshot_created if snapshot_created.tzinfo else snapshot_created.replace(tzinfo=timezone.utc)
        file_changed = modified > created

    runs = queries.latest_pipeline_runs(engine)
    stages = [_stage(runs, *spec) for spec in DAILY_STAGES]
    daily = _stage(runs, "run-daily", "run-daily", "run-daily")

    counts = queries.model_counts(engine)
    candidates = {}
    for role in ROLES:
        mine = counts[(counts["role"] == role) & (counts["status"] == "CANDIDATE")]
        candidates[role] = (int(mine["models"].sum()), int(mine["gates_passed"].fillna(0).sum()))

    # Warnings, most important first.
    if expected is None:
        warnings.append(f"could not determine the last completed {settings.market.calendar} session; freshness unchecked")
    if prices_as_of is None:
        warnings.append("no stored prices: run `quant run-daily`")
    elif expected is not None and prices_as_of < expected:
        warnings.append(f"prices end {prices_as_of}, the last completed session is {expected}: run `quant run-daily`")
    for name, as_of in (("features", features_as_of), ("regime", regime_as_of), ("signals", signals_as_of)):
        if prices_as_of is not None and (as_of is None or as_of < prices_as_of):
            warnings.append(f"{name} end {as_of or 'nowhere'}, behind the prices ({prices_as_of})")
    for stage in stages + [daily]:
        if stage.failed_since is not None:
            warnings.append(f"{stage.label}: the last run FAILED ({stage.failed_since:%Y-%m-%d %H:%M})")
    if active["REGIME"] is not None and regime_as_of is not None:
        hmm_as_of = queries.latest_date(engine, tables.market_regimes.c.date,
                                        tables.market_regimes.c.regime_method == "HMM",
                                        tables.market_regimes.c.model_version == regime_in_use)
        if hmm_as_of is None or hmm_as_of < regime_as_of:
            warnings.append(f"the ACTIVE regime {regime_in_use} ends {hmm_as_of or 'nowhere'}, before {regime_as_of}: "
                            "run `quant calculate-regime`")
    if active["PREDICTOR"] is not None and regime_as_of is not None:
        model_id = active["PREDICTOR"]["model_id"]
        predicted = queries.latest_date(engine, tables.predictions.c.date, tables.predictions.c.model_version == model_id)
        if predicted is None or predicted < regime_as_of:
            warnings.append(f"predictions of {model_id} end {predicted or 'nowhere'}, before {regime_as_of}: "
                            "run `quant generate-signals`")
    if signals_as_of is not None and not signal_counts:
        warnings.append(f"no {strategy_in_use} signals for {signals_as_of}: run `quant generate-signals`")
    if missing:
        warnings.append(f"no prices stored for: {', '.join(missing)}")
    if stale:
        warnings.append("behind the others: " + ", ".join(f"{s} ({d})" for s, d in sorted(stale.items())))
    week = queries.count_quarantined(engine, since=now - timedelta(days=7))
    if week:
        warnings.append(f"{week} rows quarantined in the last 7 days (see data_quarantine)")
    forward = forward_status(engine, settings, now, prices_as_of)
    if forward.started_on is None:
        if not positions_file.exists():
            warnings.append(f"no positions file at {positions_file}: signals assume an all-cash portfolio")
        elif file_changed:
            warnings.append(f"{positions_file} changed after the last signals: run `quant generate-signals`")
    warnings += forward.warnings
    predictor = active["PREDICTOR"]
    if predictor is not None:
        regime = active["REGIME"]
        regime_version = regime["model_id"] if regime is not None else settings.regime.rule_version
        if predictor.get("depends_on") and predictor["depends_on"] != regime_version:
            warnings.append(
                f"ACTIVE predictor {predictor['model_id']} was trained on regime {predictor['depends_on']}, "
                f"but the regime in use is {regime_version}: retrain the predictor"
            )

    return StatusReport(
        checked_at=now, expected_session=expected, prices_as_of=prices_as_of, stale_symbols=stale,
        missing_symbols=missing, quarantined_last_week=week, quarantined_total=queries.count_quarantined(engine),
        features_as_of=features_as_of, regime_as_of=regime_as_of,
        regime=None if regimes.empty else regimes.iloc[-1], regime_score=score,
        signals_as_of=signals_as_of, signal_counts=signal_counts, signal_strategy=signal_strategy,
        positions_as_of=positions_as_of, positions_file=positions_file, positions_file_changed=file_changed,
        stages=stages, last_run_daily=daily.last_ok, active=active, candidates=candidates,
        holdout_evaluations=queries.count_holdout_evaluations(engine), warnings=warnings,
        strategy_in_use=strategy_in_use, regime_in_use=regime_in_use, forward=forward,
    )


@dataclass(frozen=True)
class ForwardStatus:
    started_on: date | None
    frozen: int  # sessions frozen for live
    missed: list[date]
    warnings: list[str]


def forward_status(engine: Engine, settings: Settings, now: datetime, prices_as_of: date | None) -> ForwardStatus:
    """The forward test: missed sessions, tonight's freeze, the trade log (Phase 10)."""
    from quant.data.calendar import next_open_after, sessions_between
    from quant.forward.real import TradesError, load_start, read_trades, replay

    start = queries.forward_start(engine)
    if start is None or prices_as_of is None:
        return ForwardStatus(None, 0, [], [])
    started_on = start["started_on"]
    frozen = set(queries.frozen_sessions(engine, "live")["session"])
    sessions = sessions_between(started_on, prices_as_of, settings.market.calendar)
    latest = sessions[-1] if sessions else None
    tonight = latest is not None and latest not in frozen and now < next_open_after(latest, settings.market.calendar)
    missed = [d for d in sessions if d not in frozen and not (tonight and d == latest)]
    warnings = []
    if tonight:
        deadline = next_open_after(latest, settings.market.calendar)
        warnings.append(f"forward test: {latest} is not frozen yet; run `quant run-daily` before "
                        f"{deadline:%Y-%m-%d %H:%M} UTC or it is missed")
    if missed:
        shown = ", ".join(str(d) for d in missed[-5:]) + (" ..." if len(missed) > 5 else "")
        warnings.append(f"forward test: {len(missed)} missed session(s) since {started_on} ({shown})")
    from quant.forward.broker_real import is_broker_run

    if is_broker_run(start):  # real comes from the broker snapshots: no trade log, no positions file
        return ForwardStatus(started_on, len(frozen), missed, warnings)
    try:
        ledger_start = load_start(engine)
        if sessions:
            trades = read_trades(settings.positions.trades_path)
            replay(ledger_start, trades, pd.DatetimeIndex([pd.Timestamp(d) for d in sessions]))
    except TradesError as exc:
        warnings.append(f"trade log: {exc}")
    positions = Path(settings.positions.path)
    created = start["created_at"]
    if positions.exists() and created is not None:
        created = created if created.tzinfo else created.replace(tzinfo=timezone.utc)
        if datetime.fromtimestamp(positions.stat().st_mtime, timezone.utc) > created:
            warnings.append(f"{positions} changed after the forward test started: it is no longer read; "
                            f"record trades in {settings.positions.trades_path}")
    return ForwardStatus(started_on, len(frozen), missed, warnings)


def _when(value: datetime | None) -> str:
    return "never" if value is None else f"{value:%Y-%m-%d %H:%M} UTC"


def format_status(report: StatusReport) -> str:
    lines = [f"Status at {report.checked_at:%Y-%m-%d %H:%M} UTC", "", "Data"]
    lines.append(f"  last completed session  {report.expected_session or 'unknown'}")
    lines.append(f"  prices                  {report.prices_as_of or 'none'}")
    lines.append(f"  features                {report.features_as_of or 'none'}")
    regime = f"{report.regime} (score {report.regime_score:+.0f})" if report.regime and report.regime_score is not None \
        else (report.regime or "none")
    lines.append(f"  regime                  {report.regime_as_of or 'none'}: {regime}")
    counts = ", ".join(f"{k} {v}" for k, v in sorted(report.signal_counts.items())) or "none"
    lines.append(f"  signals                 {report.signals_as_of or 'none'}: {counts}")
    lines.append(f"  positions snapshot      {report.positions_as_of or 'none'}")
    lines.append(f"  quarantined rows        {report.quarantined_total} in total, {report.quarantined_last_week} this week")
    forward = report.forward
    if forward is None or forward.started_on is None:
        lines.append("  forward test            not started (`quant forward-test --start`)")
    else:
        lines.append(f"  forward test            since {forward.started_on}: {forward.frozen} frozen, "
                     f"{len(forward.missed)} missed")
    lines += ["", "Last successful runs"]
    lines.append(f"  run-daily               {_when(report.last_run_daily)}")
    for stage in report.stages:
        data = f" (data to {stage.data_as_of})" if stage.data_as_of else ""
        lines.append(f"  {stage.label:<23} {_when(stage.last_ok)}{data}")
    lines += ["", "Models"]
    for role in ROLES:
        model = report.active[role]
        candidates, passed = report.candidates[role]
        in_use = f"ACTIVE {model['model_id']}" if model else f"none ACTIVE: {FALLBACK[role]}"
        lines.append(f"  {role:<23} {in_use}; {candidates} candidates, {passed} passed their gates")
    lines.append(f"  signals use             {report.strategy_in_use} with the {report.regime_in_use} regime")
    lines.append(f"  holdout evaluations     {report.holdout_evaluations}")
    lines += [""]
    if report.warnings:
        lines.append(f"Warnings ({len(report.warnings)})")
        lines += [f"  - {warning}" for warning in report.warnings]
    else:
        lines.append("No warnings.")
    return "\n".join(lines)
