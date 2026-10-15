"""`quant sentiment summarize`: one short summary per stock and trading day (sentiment_proposal.md 8.3).

Written by sentiment.summary's backend from that day's VERIFIED opinions only (never from the
transcripts), one call per day for all of its stocks plus a market-mood paragraph. A summary
may cite only opinion ids of its own stock; one that cites any other id is dropped, so every
sentence traces back to quotes that matched a transcript. Summaries are display only: they
never enter the scorecard or the features.

A day is summarized again when its opinions change (a late video, a re-extraction): each
summary stores a hash of the opinion ids and the extractor it was written from.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

import pandas as pd
from sqlalchemy import Engine, and_, delete, select

from quant.config.settings import Settings
from quant.sentiment import tables
from quant.sentiment.backends import Backend, BackendError, UsageLimitError
from quant.sentiment.prompts import SUMMARY_SCHEMA, summary_prompt
from quant.sentiment.store import now_utc

Progress = Callable[[str], None]
OPINION_COLUMNS = ["opinion_id", "effective_date", "published_at", "item_id", "channel", "title", "is_generated",
                   "ticker", "name_as_said", "asset_type", "stance", "horizon", "condition", "conviction", "action",
                   "quote", "start_seconds", "reasoning", "verification", "model"]


def report_extractions(engine: Engine, settings: Settings) -> pd.DataFrame:
    """video_id -> model whose opinions the reports use: the first of extraction.report_models
    with a successful extraction of that video (backend and prompt version as configured)."""
    config, x = settings.sentiment.extraction, tables.sentiment_extractions
    with engine.connect() as conn:
        rows = conn.execute(select(x.c.video_id, x.c.model).where(and_(
            x.c.backend == config.backend, x.c.prompt_version == config.prompt_version, x.c.status == "OK",
            x.c.model.in_(config.report_models)))).all()
    frame = pd.DataFrame(rows, columns=["video_id", "model"])
    frame["rank"] = frame["model"].map({m: i for i, m in enumerate(config.report_models)})
    return frame.sort_values("rank").drop_duplicates("video_id")[["video_id", "model"]].reset_index(drop=True)


def load_opinions(engine: Engine, settings: Settings, start: date | None = None, end: date | None = None,
                  verified_only: bool = False) -> pd.DataFrame:
    """Opinions of the configured extractor, with creator and video: per video those of the first
    model in extraction.report_models that extracted it (report_extractions)."""
    config = settings.sentiment.extraction
    o, v, c, t = (tables.sentiment_opinions, tables.sentiment_videos, tables.sentiment_channels,
                  tables.sentiment_transcripts)
    stmt = (select(o.c.opinion_id, o.c.effective_date, o.c.published_at, o.c.item_id, c.c.name.label("channel"),
                   v.c.title, t.c.is_generated, o.c.ticker, o.c.name_as_said, o.c.asset_type, o.c.stance,
                   o.c.horizon, o.c.condition, o.c.conviction, o.c.action, o.c.quote, o.c.start_seconds,
                   o.c.reasoning, o.c.verification, o.c.model)
            .join(v, v.c.video_id == o.c.item_id).join(c, c.c.channel_id == o.c.channel_id)
            .outerjoin(t, t.c.video_id == o.c.item_id)
            .where(and_(o.c.source == "youtube", o.c.backend == config.backend,
                        o.c.model.in_(config.report_models), o.c.prompt_version == config.prompt_version))
            .order_by(o.c.effective_date, o.c.ticker, c.c.name, o.c.horizon))
    if start:
        stmt = stmt.where(o.c.effective_date >= start)
    if end:
        stmt = stmt.where(o.c.effective_date <= end)
    if verified_only:
        stmt = stmt.where(o.c.verification == "VERIFIED")
    with engine.connect() as conn:
        frame = pd.DataFrame(conn.execute(stmt).mappings().all(), columns=OPINION_COLUMNS)
    chosen = report_extractions(engine, settings).rename(columns={"video_id": "item_id"})
    frame = frame.merge(chosen, on=["item_id", "model"], how="inner")  # keeps the order above
    frame["condition"] = frame["condition"].fillna("")
    return frame


def inputs_hash(day: pd.DataFrame, settings: Settings) -> str:
    config = settings.sentiment.extraction
    key = f"{config.backend}/{config.model}/{config.prompt_version}:" + ",".join(map(str, sorted(day["opinion_id"])))
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def summary_input(day_date: date, day: pd.DataFrame) -> str:
    lines = [f"Trading day: {day_date:%Y-%m-%d}", ""]
    for ticker, group in day.groupby("ticker", sort=False):
        lines.append(f"## {ticker}")
        for o in group.itertuples(index=False):
            parts = [f"id={o.opinion_id}", o.channel, f"{o.stance} {o.horizon}", f"conviction {o.conviction}",
                     f"action {o.action}"]
            if o.condition:
                parts.append(f"condition: {o.condition}")
            parts += [f"reason: {o.reasoning}", f"quote: {o.quote}"]
            lines.append("- " + " | ".join(parts))
        lines.append("")
    return "\n".join(lines)


@dataclass
class DaySummary:
    day: date
    rows: list[dict] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)  # tickers whose summary cited foreign ids


def validate_summary(payload: dict, day_date: date, day: pd.DataFrame) -> DaySummary:
    ids = {t: set(g["opinion_id"]) for t, g in day.groupby("ticker")}
    every = set(day["opinion_id"])
    result = DaySummary(day_date)
    for item in payload.get("stocks") or []:
        ticker = str(item.get("ticker", "")).strip().upper()
        cited = {int(i) for i in item.get("opinion_ids") or []}
        if ticker not in ids or not cited or not cited <= ids[ticker] or not str(item.get("summary", "")).strip():
            result.dropped.append(ticker or "?")
            continue
        result.rows.append({"ticker": ticker, "summary": item["summary"].strip(),
                            "disagreement": int(bool(item.get("disagreement"))), "opinion_ids": sorted(cited)})
    market = payload.get("market") or {}
    cited = {int(i) for i in market.get("opinion_ids") or []}
    if str(market.get("summary", "")).strip() and cited <= every:
        result.rows.append({"ticker": tables.MARKET, "summary": market["summary"].strip(), "disagreement": 0,
                            "opinion_ids": sorted(cited)})
    else:
        result.dropped.append(tables.MARKET)
    return result


def current_hashes(engine: Engine, settings: Settings) -> dict[date, str]:
    """The inputs hash each day's stored summaries were written from (current summary config)."""
    config, s = settings.sentiment.summary, tables.sentiment_summaries
    with engine.connect() as conn:
        rows = conn.execute(select(s.c.date, s.c.inputs_hash).where(and_(
            s.c.backend == config.backend, s.c.model == config.model,
            s.c.prompt_version == config.prompt_version)).distinct())
        return {r.date: r.inputs_hash for r in rows}


def days_to_summarize(engine: Engine, settings: Settings, opinions: pd.DataFrame, force: bool = False) -> list[date]:
    stored = current_hashes(engine, settings)
    days = []
    for day_date, day in opinions.groupby("effective_date"):
        if force or stored.get(day_date) != inputs_hash(day, settings):
            days.append(day_date)
    return days


def save_day(engine: Engine, settings: Settings, summary: DaySummary, digest: str) -> None:
    config, s = settings.sentiment.summary, tables.sentiment_summaries
    now = now_utc()
    with engine.begin() as conn:
        conn.execute(delete(s).where(and_(s.c.date == summary.day, s.c.backend == config.backend,
                                          s.c.model == config.model, s.c.prompt_version == config.prompt_version)))
        for row in summary.rows:
            conn.execute(s.insert().values(
                date=summary.day, ticker=row["ticker"], summary=row["summary"], disagreement=row["disagreement"],
                opinion_ids_json=json.dumps(row["opinion_ids"]), inputs_hash=digest, backend=config.backend,
                model=config.model, prompt_version=config.prompt_version, created_at=now))


@dataclass
class SummarizeResult:
    planned: int = 0
    done: list[DaySummary] = field(default_factory=list)
    failed: list[tuple[date, str]] = field(default_factory=list)
    stopped: str | None = None


def run_summarize(engine: Engine, settings: Settings, backend: Backend, opinions: pd.DataFrame, days: list[date],
                  progress: Progress | None = None) -> SummarizeResult:
    progress = progress or (lambda line: None)
    system = summary_prompt(settings.sentiment.summary.prompt_version)
    result = SummarizeResult(planned=len(days))
    for number, day_date in enumerate(days, start=1):
        day = opinions[opinions["effective_date"] == day_date]
        tickers = day["ticker"].nunique()
        error = None
        for _attempt in (1, 2):  # one retry
            try:
                reply = backend.complete(system, summary_input(day_date, day), SUMMARY_SCHEMA)
            except UsageLimitError as exc:
                result.stopped = f"usage limit reached ({exc}); the remaining days wait for the next run"
                progress(f"  STOPPED: {result.stopped}")
                return result
            except BackendError as exc:
                error = str(exc)
                continue
            try:
                summary = validate_summary(reply.payload, day_date, day)
            except (TypeError, ValueError, AttributeError) as exc:  # e.g. an id that is not a number
                error = f"malformed reply: {exc}"
                continue
            save_day(engine, settings, summary, inputs_hash(day, settings))
            result.done.append(summary)
            stocks = sum(1 for row in summary.rows if row["ticker"] != tables.MARKET)
            progress(f"[{number}/{len(days)}] {day_date}: {stocks} of {tickers} stock summaries"
                     + (f"; dropped (cited other stocks' opinions): {', '.join(summary.dropped)}"
                        if summary.dropped else ""))
            break
        else:
            result.failed.append((day_date, error or "failed"))
            progress(f"[{number}/{len(days)}] {day_date}: FAILED ({error}); tried again next run")
    return result


def load_summaries(engine: Engine, settings: Settings, day_date: date, digest: str) -> dict[str, dict]:
    """A day's stored summaries, only if they were written from exactly these opinions."""
    config, s = settings.sentiment.summary, tables.sentiment_summaries
    with engine.connect() as conn:
        rows = conn.execute(select(s).where(and_(
            s.c.date == day_date, s.c.backend == config.backend, s.c.model == config.model,
            s.c.prompt_version == config.prompt_version, s.c.inputs_hash == digest))).mappings().all()
    return {r["ticker"]: {**dict(r), "opinion_ids": json.loads(r["opinion_ids_json"])} for r in rows}
