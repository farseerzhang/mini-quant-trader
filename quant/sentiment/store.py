"""Data access for the sentiment tables: plain functions over SQLAlchemy Core."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import pandas as pd
from sqlalchemy import Engine, and_, case, or_, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from quant.config.settings import ChannelSettings
from quant.sentiment import tables


def now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def sync_channels(engine: Engine, channels: Sequence[ChannelSettings]) -> None:
    """Make sentiment_channels match config: upsert these as active, mark every other inactive."""
    now = now_utc()
    with engine.begin() as conn:
        for channel in channels:
            values = {"channel_id": channel.channel_id, "handle": channel.handle, "name": channel.name,
                      "language": channel.language, "style_note": channel.style, "active": 1,
                      "added_at": now, "updated_at": now}
            stmt = sqlite_insert(tables.sentiment_channels).values(**values)
            update = {k: stmt.excluded[k] for k in values if k not in ("channel_id", "added_at")}
            conn.execute(stmt.on_conflict_do_update(index_elements=["channel_id"], set_=update))
        conn.execute(
            tables.sentiment_channels.update()
            .where(tables.sentiment_channels.c.channel_id.not_in([c.channel_id for c in channels]))
            .values(active=0, updated_at=now)
        )


def known_videos(engine: Engine, video_ids: Sequence[str]) -> dict[str, datetime | None]:
    """Stored video ids among these -> their published_at (None if never read)."""
    if not video_ids:
        return {}
    v = tables.sentiment_videos
    with engine.connect() as conn:
        rows = conn.execute(select(v.c.video_id, v.c.published_at).where(v.c.video_id.in_(list(video_ids))))
        return {row.video_id: row.published_at for row in rows}


def insert_video(engine: Engine, values: Mapping[str, Any]) -> bool:
    """Store a newly listed video once; a re-run never overwrites it. True if inserted."""
    now = now_utc()
    row = {"fetched_at": now, "updated_at": now, "attempts": 0, **dict(values)}
    stmt = sqlite_insert(tables.sentiment_videos).values(**row).on_conflict_do_nothing(index_elements=["video_id"])
    with engine.begin() as conn:
        return bool(conn.execute(stmt).rowcount)


def videos_needing_transcripts(engine: Engine, channel_ids: Sequence[str], since: datetime,
                               retry_no_captions_since: datetime) -> pd.DataFrame:
    """PENDING and FAILED videos published since `since`, oldest first; then NO_TRANSCRIPT videos
    published since `retry_no_captions_since` (captions often appear hours after publishing),
    so retries never crowd out videos not tried yet."""
    v, c = tables.sentiment_videos, tables.sentiment_channels
    retry = v.c.transcript_status == "NO_TRANSCRIPT"
    stmt = (
        select(v.c.video_id, v.c.channel_id, v.c.title, v.c.published_at, v.c.attempts, v.c.transcript_status,
               c.c.language, c.c.handle)
        .join(c, c.c.channel_id == v.c.channel_id)
        .where(and_(
            v.c.channel_id.in_(list(channel_ids)),
            or_(
                and_(v.c.transcript_status.in_(["PENDING", "FAILED"]), v.c.published_at >= since),
                and_(retry, v.c.published_at >= max(since, retry_no_captions_since)),
            ),
        ))
        .order_by(case((retry, 1), else_=0), v.c.published_at)
    )
    with engine.connect() as conn:
        return pd.DataFrame(conn.execute(stmt).mappings().all(),
                            columns=["video_id", "channel_id", "title", "published_at", "attempts",
                                     "transcript_status", "language", "handle"])


def set_transcript_status(engine: Engine, video_id: str, status: str, note: str | None, attempts: int) -> None:
    """Record the outcome of `attempts` caption requests."""
    v = tables.sentiment_videos
    with engine.begin() as conn:
        conn.execute(v.update().where(v.c.video_id == video_id).values(
            transcript_status=status, note=note, attempts=v.c.attempts + attempts, updated_at=now_utc()))


def save_transcript(engine: Engine, video_id: str, language: str, is_generated: bool, segments: list[dict],
                    simplified: list[dict] | None, attempts: int) -> None:
    """Store the captions and mark the video OK, in one transaction."""
    now = now_utc()
    row = {
        "video_id": video_id, "language": language, "is_generated": int(is_generated),
        "segments_json": json.dumps(segments, ensure_ascii=False),
        "segments_simplified_json": json.dumps(simplified, ensure_ascii=False) if simplified is not None else None,
        "char_count": sum(len(s["text"]) for s in segments), "fetched_at": now,
    }
    stmt = sqlite_insert(tables.sentiment_transcripts).values(**row)
    stmt = stmt.on_conflict_do_update(index_elements=["video_id"],
                                      set_={k: stmt.excluded[k] for k in row if k != "video_id"})
    v = tables.sentiment_videos
    with engine.begin() as conn:
        conn.execute(stmt)
        conn.execute(v.update().where(v.c.video_id == video_id).values(
            transcript_status="OK", note=None, attempts=v.c.attempts + attempts, updated_at=now))

