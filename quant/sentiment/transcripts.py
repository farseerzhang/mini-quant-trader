"""Captions (sentiment_proposal.md section 3.2).

Manual captions are preferred over auto-generated ones, in the language order of the
channel's language. Chinese captions are also stored in Simplified Chinese, so one alias
table covers both scripts. `TranscriptSource` is the interface; `YouTubeTranscriptSource`
implements it with youtube-transcript-api.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

# Caption language codes tried per channel language, in order. Nick 美股咖啡館 uses zh-TW.
LANGUAGES = {
    "en": ("en", "en-US", "en-GB"),
    "zh": ("zh-Hans", "zh-CN", "zh-SG", "zh", "zh-Hant", "zh-TW", "zh-HK"),
}


@dataclass(frozen=True)
class Transcript:
    language: str
    is_generated: bool
    segments: list[dict]  # [{start, duration, text}]


class NoTranscriptError(Exception):
    """The video has no captions in any accepted language (or captions are disabled)."""


class TranscriptBlockedError(Exception):
    """YouTube is refusing requests from this IP; stop the run and try again later."""


class TranscriptError(Exception):
    """A temporary failure; worth retrying."""


class TranscriptSource(Protocol):
    def fetch(self, video_id: str, languages: tuple[str, ...]) -> Transcript:
        ...


class YouTubeTranscriptSource:
    def __init__(self) -> None:
        import youtube_transcript_api as api  # imported here, so only sentiment commands need it

        self._api = api
        self._client = api.YouTubeTranscriptApi()

    def fetch(self, video_id: str, languages: tuple[str, ...]) -> Transcript:
        api = self._api
        try:
            available = self._client.list(video_id)
            try:
                chosen = available.find_manually_created_transcript(languages)
            except api.NoTranscriptFound:
                chosen = available.find_generated_transcript(languages)
            fetched = chosen.fetch()
        except (api.TranscriptsDisabled, api.NoTranscriptFound) as exc:
            raise NoTranscriptError(type(exc).__name__) from exc
        except api.RequestBlocked as exc:  # includes IpBlocked
            raise TranscriptBlockedError(type(exc).__name__) from exc
        except api.CouldNotRetrieveTranscript as exc:
            raise TranscriptError(f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}") from exc
        except OSError as exc:  # network errors (requests' exceptions are OSErrors)
            raise TranscriptError(f"{type(exc).__name__}: {exc}") from exc
        segments = [{"start": s.start, "duration": s.duration, "text": s.text} for s in fetched.snippets]
        return Transcript(language=chosen.language_code, is_generated=bool(chosen.is_generated), segments=segments)


_converter = None


def to_simplified(text: str) -> str:
    """Traditional -> Simplified Chinese (opencc); Simplified text passes through unchanged."""
    global _converter
    if _converter is None:
        from opencc import OpenCC  # imported on first use

        _converter = OpenCC("t2s")
    return _converter.convert(text)


def simplified_segments(segments: list[dict]) -> list[dict]:
    return [{**segment, "text": to_simplified(segment["text"])} for segment in segments]
