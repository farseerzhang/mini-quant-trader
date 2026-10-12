"""Freeze each portfolio's targets for the latest session, once (Phase 10 pre-registration).

Every evening the live strategy and each shadow (``forward_test.shadows``) are computed from
the same stored data as the signals and written to ``forward_sessions`` / ``forward_targets``.
A session is frozen only before the next market open, i.e. while its signals can still be
acted on; after that it is missed for good (no backfill). A frozen session is never changed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import Engine, select

from quant.config.settings import Settings, ShadowSettings
from quant.data.calendar import next_open_after
from quant.models.live import model_predictions
from quant.persistence import queries, tables
from quant.regime.active import ActiveModelError, active_hmm_regime, load_active_hmm
from quant.regime.pipeline import RULE_METHOD
from quant.regime.hmm import load_observations
from quant.signals.targets import TARGET_COLUMNS, Targets, compute_targets, load_targets

log = logging.getLogger(__name__)
LIVE = "live"
COMMAND = "forward-test"


@dataclass(frozen=True)
class FreezeResult:
    session: pd.Timestamp
    frozen: list[str]  # portfolios written now
    already: list[str]  # portfolios frozen earlier for this session (left as they were)
    deadline: datetime  # the next open: the latest moment this session can be frozen
    missed: bool  # True when the deadline had passed: nothing was written


def _model_row(engine: Engine, model_id: str):
    models = tables.model_versions
    with engine.connect() as conn:
        row = conn.execute(select(models).where(models.c.model_id == model_id)).mappings().first()
    if row is None:
        raise ActiveModelError(f"forward_test shadow model {model_id!r} is not in model_versions")
    return row


def shadow_targets(engine: Engine, settings: Settings, shadow: ShadowSettings) -> Targets:
    """A shadow's targets at the latest session, from the same stored data as the live signals."""
    stocks = list(settings.market.stocks)
    panel = queries.load_feature_panel(engine, settings.features.feature_version, stocks, TARGET_COLUMNS)
    if shadow.regime_model is not None:
        fit = load_active_hmm(_model_row(engine, shadow.regime_model), settings)
        regimes, _ = active_hmm_regime(fit, load_observations(engine, settings), settings.regime.min_dwell_days)
        source = shadow.regime_model
    else:  # a shadow names its regime model; without one it uses the rule-based regime
        regimes = queries.load_regimes(engine, RULE_METHOD, settings.regime.rule_version)
        source = settings.regime.rule_version
    since = (regimes.index[-1] - pd.Timedelta(days=30)).date()
    raw_closes = queries.load_raw_closes(engine, stocks, since=since)
    if shadow.strategy == "ml_select":
        _, frame = model_predictions(engine, settings, _model_row(engine, shadow.predictor))
        probabilities = frame["probability"].unstack("symbol").sort_index()
        return compute_targets(panel, regimes, settings, raw_closes, regime_source=source,
                               probabilities=probabilities, predictor=shadow.predictor)
    return compute_targets(panel, regimes, settings, raw_closes, strategy=shadow.strategy, regime_source=source)


def _rows(name: str, targets: Targets) -> tuple[dict, list[dict]]:
    stocks = targets.stocks
    held = stocks[stocks["target_weight"] > 0]
    header = {
        "session": targets.session.date(), "portfolio": name, "strategy": targets.strategy,
        "regime": targets.regime, "regime_source": targets.regime_source,
        "model_version": targets.model_version, "invested": float(targets.invested), "positions": int(len(held)),
    }
    rows = [
        {"symbol": symbol, "target_weight": float(r["target_weight"]),
         "close": None if pd.isna(r["close"]) else float(r["close"]),
         "score": float(r["probability"]) if "probability" in held.columns and pd.notna(r["probability"]) else None}
        for symbol, r in held.iterrows()
    ]
    return header, rows


def start_value(engine: Engine, settings: Settings, session: pd.Timestamp) -> tuple[float, str, list[dict]]:
    """The real account at the close of ``session`` from the positions file (every holding, in the
    pool or not, at as-traded closes, plus cash), its source and its rows for the stored start.
    Without a file: the notional ``backtest.initial_cash``, all cash."""
    from quant.forward.real import Ledger, Start, start_rows, value_ledger
    from quant.signals.positions import read_positions

    holdings = read_positions(settings.positions.path, settings.market.stocks, settings.backtest.initial_cash)
    rows = start_rows(holdings)
    quantities = {**holdings.quantities, **holdings.unmanaged}
    start = Start(session, quantities, dict(holdings.cost_basis), holdings.cash)
    index = pd.DatetimeIndex([session])
    ledger = Ledger(pd.DataFrame([quantities], index=index, columns=list(quantities)).fillna(0.0),
                    pd.Series([holdings.cash], index=index), [], ())
    closes = queries.load_raw_closes(engine, list(quantities)) if quantities else pd.DataFrame()
    valued, _ = value_ledger(ledger, closes.loc[:session], start)
    source = str(holdings.path) if holdings.path is not None else "notional backtest.initial_cash (no positions file)"
    return float(valued["value"].iloc[0]), source, rows


def freeze_session(engine: Engine, settings: Settings, now: datetime | None = None) -> FreezeResult:
    """Freeze live and every shadow for the latest stored session, if still before the next open."""
    now = now or datetime.now(timezone.utc)
    live = load_targets(engine, settings)
    session = live.session
    deadline = next_open_after(session.date(), settings.market.calendar)
    portfolios = {LIVE: lambda: live}
    for name, shadow in settings.forward_test.shadows.items():
        portfolios[name] = lambda shadow=shadow: shadow_targets(engine, settings, shadow)
    stored = queries.frozen_sessions(engine)
    done = set(stored.loc[stored["session"] == session.date(), "portfolio"])
    if now >= deadline:
        missing = [name for name in portfolios if name not in done]
        if missing:
            queries.record_pipeline_run(
                engine, command=COMMAND, stage="freeze", status="SKIPPED", data_as_of_date=session.date(),
                message=f"{session.date()} not frozen for {', '.join(missing)}: the next open "
                        f"({deadline:%Y-%m-%d %H:%M} UTC) has passed (no backfill)",
            )
        return FreezeResult(session, [], sorted(done), deadline, missed=bool(missing))

    if queries.forward_start(engine) is None:
        value, source, rows = start_value(engine, settings, session)
        queries.record_forward_start(engine, session.date(), value, source, rows)
        log.info("forward test starts %s with %.2f from %s", session.date(), value, source)

    frozen, already = [], []
    for name, compute in portfolios.items():
        if name in done:
            already.append(name)
            continue
        targets = compute()
        if targets.session != session:
            raise ActiveModelError(f"{name}: its data ends {targets.session.date()}, not {session.date()}")
        header, rows = _rows(name, targets)
        (frozen if queries.freeze_forward(engine, header, rows) else already).append(name)
    if frozen:
        queries.record_pipeline_run(
            engine, command=COMMAND, stage="freeze", status="OK", data_as_of_date=session.date(),
            message=f"froze {session.date()} for {', '.join(frozen)}"
                    + (f"; already frozen: {', '.join(already)}" if already else ""),
        )
    return FreezeResult(session, frozen, already, deadline, missed=False)
