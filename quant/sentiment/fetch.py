"""`quant sentiment fetch`: list new videos, then fetch their captions (sentiment_proposal.md steps 1-2).

Idempotent: a video is stored once, by video_id, and never overwritten; captions are fetched
only for PENDING and FAILED videos. A run after a network failure therefore fills the gaps:
videos that could not be read are listed again, and FAILED captions are retried.

Per video and channel:
- members-only videos and Shorts are recorded as SKIPPED from the listing alone;
- every other new video's page is read once for its exact publish time; live-stream
  replays are SKIPPED unless sentiment.include_live, and live or upcoming streams are left
  for a later run;
- the listing is read newest first and stops at the first video older than the lookback.

Captions: at most sentiment.max_captions_per_run per run, oldest first (the rest waits for
the next run); one retry after a failure, then FAILED (retried on the next run). A video
without captions is asked again on later runs while younger than
sentiment.no_captions_retry_days, because captions often appear hours after publishing; these
retries come after every video not tried yet. If YouTube blocks the IP, the run stops at once;
the remaining videos stay PENDING.

Backfill (sentiment_proposal.md 12.6): `backfill` lists each channel back to
sentiment.evaluation.backfill_from instead of the lookback. Captions of backfill videos (published
from backfill_from up to the lookback) are fetched on every run, but only after every forward video
and retry, so the S6 forward data never waits behind the backfill.

Caption pacing (sentiment_proposal.md 12.5): the caption step runs only
sentiment.min_hours_between_caption_fetches after the last fetch that requested captions, and
sentiment.cooldown_hours_after_block after a blocked one; the times come from the
`sentiment-fetch` rows of pipeline_runs. The listing always runs. `force_captions` ignores both waits.
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

import pandas as pd
from sqlalchemy import Engine, select

from quant.config.settings import ChannelSettings, SentimentSettings, Settings
from quant.persistence import tables as core_tables
from quant.sentiment import store
from quant.sentiment.sources import MembersOnlyError, SourceError, VideoSource
from quant.sentiment.transcripts import (
    LANGUAGES,
    NoTranscriptError,
    TranscriptBlockedError,
    TranscriptError,
    TranscriptSource,
    simplified_segments,
)

PUBLIC = (None, "public", "unlisted")
NOT_YET = ("is_live", "is_upcoming", "post_live")  # not finished: read again on a later run

# Markers in the stored message of a `sentiment-fetch` run, written by `run_message` and read by
# `next_caption_fetch`.
FETCH_COMMAND = "sentiment-fetch"
BLOCKED_MARKER = "stopped: YouTube refused caption requests"
CAPTIONS_REQUESTED_MARKER = "captions requested"
CAPTIONS_FORCED_MARKER = "captions forced"
# Runs stored before 12.5 have no marker; one with "N captions" (N > 0) requested captions too.
_CAPTIONS_FETCHED_BEFORE_MARKERS = re.compile(r"\b[1-9]\d* captions\b")


@dataclass
class ChannelResult:
    handle: str
    listed: int = 0
    new: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class FetchResult:
    since: datetime
    channels: list[ChannelResult] = field(default_factory=list)
    transcripts_ok: int = 0
    no_transcript: int = 0
    transcripts_failed: int = 0
    captions_left: int = 0  # beyond sentiment.max_captions_per_run: fetched by the next run
    blocked: str | None = None  # why the run stopped early
    captions_waiting_until: datetime | None = None  # caption step skipped: next allowed fetch (naive UTC)
    captions_wait_reason: str | None = None
    captions_forced: bool = False

    @property
    def errors(self) -> list[str]:
        return [error for channel in self.channels for error in channel.errors]

    @property
    def ok(self) -> bool:
        return not self.errors and not self.transcripts_failed and self.blocked is None

    @property
    def captions_requested(self) -> bool:
        return bool(self.transcripts_ok or self.no_transcript or self.transcripts_failed or self.blocked)


def run_message(result: FetchResult) -> str:
    """The message stored with the run in pipeline_runs; `next_caption_fetch` reads its markers."""
    parts = [f"{sum(c.new for c in result.channels)} new videos, {result.transcripts_ok} captions, "
             f"{result.transcripts_failed} failed, {len(result.errors)} listing errors"]
    if result.captions_requested:
        parts.append(CAPTIONS_REQUESTED_MARKER)
    if result.captions_forced:
        parts.append(CAPTIONS_FORCED_MARKER)
    if result.captions_waiting_until is not None:
        parts.append(f"captions skipped until {result.captions_waiting_until:%Y-%m-%d %H:%M} UTC "
                     f"({result.captions_wait_reason})")
    if result.blocked:
        parts.append(f"stopped: {result.blocked}")
    return "; ".join(parts)


def _requested_captions(message: str) -> bool:
    return (CAPTIONS_REQUESTED_MARKER in message or BLOCKED_MARKER in message
            or _CAPTIONS_FETCHED_BEFORE_MARKERS.search(message) is not None)


def next_caption_fetch(engine: Engine, config: SentimentSettings, now: datetime) -> tuple[datetime, str] | None:
    """(earliest next caption fetch, why) if the caption step must wait at `now` (naive UTC), else None."""
    gap = timedelta(hours=config.min_hours_between_caption_fetches)
    cooldown = timedelta(hours=config.cooldown_hours_after_block)
    runs = core_tables.pipeline_runs
    with engine.connect() as conn:
        rows = conn.execute(select(runs.c.started_at, runs.c.message).where(
            runs.c.command == FETCH_COMMAND, runs.c.started_at >= now - max(gap, cooldown))).all()
    waits = []
    for started_at, message in rows:
        message = message or ""
        if BLOCKED_MARKER in message:
            waits.append((started_at + cooldown, "cooldown after a block"))
        if _requested_captions(message):
            waits.append((started_at + gap, f"{config.min_hours_between_caption_fetches:g} h since the last "
                                            "caption fetch"))
    until, reason = max(waits, default=(None, None))
    return (until, reason) if until is not None and until > now else None


def scan_limit(days: int) -> int:
    """How many listing entries to read at most: a few videos a day plus Shorts and members-only."""
    return 3 * days + 10


class _Pacer:
    """A random wait from sentiment.request_delay_seconds before every request but the first."""

    def __init__(self, delay: tuple[float, float], sleep: Callable[[float], None], rng: random.Random) -> None:
        self.delay, self.sleep, self.rng, self.started = delay, sleep, rng, False

    def wait(self) -> None:
        if self.started:
            self.sleep(self.rng.uniform(*self.delay))
        self.started = True


Progress = Callable[[str], None]


def _short(title: str | None, width: int = 50) -> str:
    title = (title or "").strip()
    return title if len(title) <= width else title[: width - 1] + "…"


def _list_channel(engine: Engine, channel: ChannelSettings, since: datetime, min_duration: int,
                  include_live: bool, source: VideoSource, pacer: _Pacer, limit: int,
                  progress: Progress) -> ChannelResult:
    result = ChannelResult(channel.handle)
    pacer.wait()
    try:
        listed = source.list_channel(channel.handle, limit)
    except SourceError as exc:
        result.errors.append(str(exc))
        progress(f"    listing failed: {exc}")
        return result
    result.listed = len(listed)
    known = store.known_videos(engine, [video.video_id for video in listed])
    progress(f"    {len(listed)} listed, {len(known)} already stored")

    def record(video_id: str, title: str | None, status: str, note: str | None = None, **extra) -> None:
        if store.insert_video(engine, {"video_id": video_id, "channel_id": channel.channel_id, "title": title,
                                       "transcript_status": status, "note": note, **extra}):
            result.new += 1
            result.skipped += status == "SKIPPED"
            published = extra.get("published_at")
            when = f"{published:%Y-%m-%d}" if published else "          "
            progress(f"    {when}  {'new' if status == 'PENDING' else 'skipped (' + str(note) + ')'}: "
                     f"{_short(title)}")

    for video in listed:
        if video.video_id in known:
            published = known[video.video_id]
            if published is not None and published < since:
                break
            continue
        if video.availability not in PUBLIC:
            record(video.video_id, video.title, "SKIPPED", "members-only" if video.availability == "subscriber_only"
                   else f"availability {video.availability}", availability=video.availability,
                   duration_seconds=video.duration_seconds)
            continue
        if video.duration_seconds is not None and video.duration_seconds < min_duration:
            record(video.video_id, video.title, "SKIPPED", "short", duration_seconds=video.duration_seconds)
            continue
        pacer.wait()
        try:
            details = source.details(video.video_id)
        except MembersOnlyError:
            record(video.video_id, video.title, "SKIPPED", "members-only", availability="subscriber_only")
            continue
        except SourceError as exc:
            result.errors.append(str(exc))  # not stored: listed and read again next run
            progress(f"    cannot read {video.video_id} (tried again next run): {exc}")
            continue
        if details.published_at < since:
            progress(f"    reached {details.published_at:%Y-%m-%d}, before the lookback: done")
            break  # the listing is newest first
        if details.live_status in NOT_YET:
            progress(f"    {details.live_status.replace('_', ' ')}, later run: {_short(video.title)}")
            continue
        info = {"published_at": details.published_at, "duration_seconds": details.duration_seconds,
                "live_status": details.live_status, "availability": details.availability}
        title = details.title or video.title
        if details.live_status == "was_live" and not include_live:
            record(video.video_id, title, "SKIPPED", "live replay", **info)
        elif details.duration_seconds is not None and details.duration_seconds < min_duration:
            record(video.video_id, title, "SKIPPED", "short", **info)
        else:
            record(video.video_id, title, "PENDING", **info)
    if result.new == 0 and not result.errors:
        progress("    no new videos")
    return result


def _fetch_transcripts(engine: Engine, channels: list[ChannelSettings], since: datetime, retry_since: datetime,
                       source: TranscriptSource, pacer: _Pacer, result: FetchResult, progress: Progress,
                       limit: int, backfill_from: datetime | None = None) -> None:
    waiting = store.videos_needing_transcripts(engine, [c.channel_id for c in channels], since, retry_since,
                                               backfill_from)
    pending = waiting.head(limit)  # forward first, oldest first; then retries; then the backfill
    result.captions_left = len(waiting) - len(pending)
    retries = int((waiting["transcript_status"] == "NO_TRANSCRIPT").sum())
    backfill = int((pd.to_datetime(waiting["published_at"]) < since).sum()) if len(waiting) else 0  # retries are recent
    progress(f"Captions: {len(waiting)} video(s) waiting"
             + (f" ({retries} recent without captions, asked again)" if retries else "")
             + (f" ({backfill} from the backfill, after the forward videos)" if backfill > 0 else "")
             + f"; fetching {len(pending)} this run"
             + (f" (sentiment.max_captions_per_run), {result.captions_left} left for the next run"
                if result.captions_left else ""))
    for number, video in enumerate(pending.itertuples(index=False), start=1):
        languages = LANGUAGES[video.language]
        label = f"  [{number}/{len(pending)}] {video.handle} {video.published_at:%Y-%m-%d} {_short(video.title, 40)}"
        error = None
        for attempt in (1, 2):  # one retry, then FAILED
            pacer.wait()
            try:
                transcript = source.fetch(video.video_id, languages)
            except NoTranscriptError as exc:
                store.set_transcript_status(engine, video.video_id, "NO_TRANSCRIPT", str(exc), attempt)
                result.no_transcript += 1
                progress(f"{label}: no captions ({exc})")
                break
            except TranscriptBlockedError as exc:
                result.blocked = (f"YouTube refused caption requests ({exc}) after {number - 1} of {len(pending)}; "
                                  "the remaining videos stay PENDING")
                progress(f"{label}: BLOCKED by YouTube, stopping")
                return
            except TranscriptError as exc:
                error = str(exc)
                if attempt == 1:
                    progress(f"{label}: failed ({exc}), retrying")
                continue
            simplified = simplified_segments(transcript.segments) if video.language == "zh" else None
            store.save_transcript(engine, video.video_id, transcript.language, transcript.is_generated,
                                  transcript.segments, simplified, attempt)
            result.transcripts_ok += 1
            chars = sum(len(s["text"]) for s in transcript.segments)
            progress(f"{label}: OK ({transcript.language}{', auto' if transcript.is_generated else ''}, "
                     f"{chars:,} characters)")
            break
        else:
            store.set_transcript_status(engine, video.video_id, "FAILED", error, 2)
            result.transcripts_failed += 1
            progress(f"{label}: FAILED ({error}); tried again next run")


def fetch(engine: Engine, settings: Settings, video_source: VideoSource, transcript_source: TranscriptSource,
          now: datetime | None = None, days: int | None = None, handle: str | None = None,
          sleep: Callable[[float], None] | None = None, rng: random.Random | None = None,
          progress: Progress | None = None, force_captions: bool = False, backfill: bool = False) -> FetchResult:
    """List every configured channel (or one), store new videos, then fetch missing captions
    unless the caption pacing says to wait (`force_captions` ignores it).

    `progress` receives one line per step (the CLI prints them as they happen)."""
    config = settings.sentiment
    progress = progress or (lambda line: None)
    days = days or config.lookback_days
    now = now or store.now_utc()
    since = now - timedelta(days=days)
    retry_since = now - timedelta(days=config.no_captions_retry_days)
    backfill_from = datetime.combine(config.evaluation.backfill_from, datetime.min.time())
    listed_since = min(since, backfill_from) if backfill else since
    listed_days = max(days, (now - backfill_from).days + 1) if backfill else days
    channels = [config.channel(handle)] if handle else list(config.channels)
    store.sync_channels(engine, config.channels)
    pacer = _Pacer(config.request_delay_seconds, sleep or time.sleep, rng or random.Random())
    result = FetchResult(since=since)
    low, high = config.request_delay_seconds
    progress(f"Videos published since {listed_since:%Y-%m-%d %H:%M} UTC"
             + (" (backfill listing)" if backfill else "") + f" ({low:g}-{high:g} s wait between requests)")
    for number, channel in enumerate(channels, start=1):
        progress(f"[{number}/{len(channels)}] {channel.handle} ({channel.name})")
        result.channels.append(_list_channel(engine, channel, listed_since, config.min_duration_seconds,
                                             config.include_live, video_source, pacer, scan_limit(listed_days),
                                             progress))
    wait = next_caption_fetch(engine, config, now)
    if wait is not None and not force_captions:
        result.captions_waiting_until, result.captions_wait_reason = wait
        progress(f"Captions: skipped, next fetch after {wait[0]:%Y-%m-%d %H:%M} UTC ({wait[1]}); "
                 "--force-captions fetches now")
        return result
    if wait is not None:
        result.captions_forced = True
        progress(f"Captions: forced (otherwise not before {wait[0]:%Y-%m-%d %H:%M} UTC, {wait[1]})")
    _fetch_transcripts(engine, channels, since, retry_since, transcript_source, pacer, result, progress,
                       config.max_captions_per_run, backfill_from)
    return result
