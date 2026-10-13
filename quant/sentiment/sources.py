"""Video lists from YouTube (sentiment_proposal.md section 3.1).

`VideoSource` is the interface; `YtDlpSource` implements it with yt-dlp, which needs no key
but is unofficial and can break, so it can later be replaced (e.g. by the YouTube Data API)
without touching anything downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol


@dataclass(frozen=True)
class ListedVideo:
    """One entry of a channel's /videos page, newest first; no publish time yet."""

    video_id: str
    title: str | None
    duration_seconds: int | None
    availability: str | None  # None or "public" on the listing for ordinary videos


@dataclass(frozen=True)
class VideoDetails:
    video_id: str
    channel_id: str | None
    title: str | None
    published_at: datetime  # naive UTC: the premiere time if there was one, else the upload time
    duration_seconds: int | None
    live_status: str | None  # not_live | was_live | is_live | is_upcoming | post_live
    availability: str | None


class SourceError(Exception):
    """The listing or a video page could not be read; worth retrying on a later run."""


class MembersOnlyError(SourceError):
    """The video is for channel members only and cannot be read."""


class VideoSource(Protocol):
    def list_channel(self, handle: str, limit: int) -> list[ListedVideo]:
        """The newest `limit` long-form videos of the channel, newest first."""

    def details(self, video_id: str) -> VideoDetails:
        """One video's metadata, including its exact publish time."""


def _utc(timestamp: float) -> datetime:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).replace(tzinfo=None)


class YtDlpSource:
    """yt-dlp without downloading anything: the flat channel listing, then each new video's page."""

    def __init__(self) -> None:
        import yt_dlp  # imported here, so only sentiment commands need it

        self._yt_dlp = yt_dlp
        # The video page's metadata is enough; formats (which need a JS runtime) are not read.
        self._quiet = {"quiet": True, "no_warnings": True, "skip_download": True, "ignore_no_formats_error": True}

    def list_channel(self, handle: str, limit: int) -> list[ListedVideo]:
        options = {**self._quiet, "extract_flat": "in_playlist", "playlistend": limit}
        try:
            with self._yt_dlp.YoutubeDL(options) as ydl:
                info = ydl.extract_info(f"https://www.youtube.com/{handle}/videos", download=False)
        except self._yt_dlp.utils.YoutubeDLError as exc:
            raise SourceError(f"cannot list {handle}: {exc}") from exc
        return [
            ListedVideo(
                video_id=entry["id"],
                title=entry.get("title"),
                duration_seconds=int(entry["duration"]) if entry.get("duration") is not None else None,
                availability=entry.get("availability"),
            )
            for entry in info.get("entries") or []
            if entry and entry.get("id")
        ]

    def details(self, video_id: str) -> VideoDetails:
        try:
            with self._yt_dlp.YoutubeDL(self._quiet) as ydl:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False, process=False)
        except self._yt_dlp.utils.YoutubeDLError as exc:
            if "members-only" in str(exc).lower() or "join this channel" in str(exc).lower():
                raise MembersOnlyError(str(exc)) from exc
            raise SourceError(f"cannot read video {video_id}: {exc}") from exc
        published = info.get("release_timestamp") or info.get("timestamp")
        if published is None:
            raise SourceError(f"video {video_id} has no publish time")
        return VideoDetails(
            video_id=video_id,
            channel_id=info.get("channel_id"),
            title=info.get("title"),
            published_at=_utc(published),
            duration_seconds=int(info["duration"]) if info.get("duration") is not None else None,
            live_status=info.get("live_status"),
            availability=info.get("availability"),
        )
