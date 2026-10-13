"""`quant sentiment extract`: opinions from stored transcripts (sentiment_proposal.md section 4).

Per video: the transcript (Simplified for Chinese) is cut into pieces of
sentiment.extraction.chunk_minutes, each starting chunk_overlap_seconds before the previous
one ends; every piece goes to the backend on its own. Each returned opinion is then

- validated (allowed values; a ticker that looks like one),
- checked: the ticker must be known (alias table or US listing), else UNRESOLVED; the quote
  must match its piece at least quote_match_threshold, else UNVERIFIED (kept, but left out of
  reports); start_seconds is taken from where the quote matched,
- merged within the video (section 4.5): one opinion per stock and horizon; the last stated
  stance wins, conviction is the highest, an action is kept if any piece states one, and the
  earlier quotes are kept in other_quotes_json.

Every video gets one sentiment_extractions row per extractor (backend, model, prompt
version), even with no opinions, so it is not sent again. A failed call is retried once;
then the video is FAILED and retried on the next run. A usage limit stops the run.

Videos of the gold set / AI reference whose status is not done yet are held back, so no
extraction output is seen before they are labeled (section 4.6).
"""

from __future__ import annotations

import csv
import difflib
import json
import unicodedata
from dataclasses import dataclass, field
from datetime import timezone
from pathlib import Path
from typing import Callable

import pandas as pd
from sqlalchemy import Engine, and_, delete, select

from quant.config.settings import Settings
from quant.data.calendar import first_session_opening_after
from quant.sentiment import tables
from quant.sentiment.aliases import TICKER, TickerBook, normalize_ticker
from quant.sentiment.backends import Backend, BackendError, UsageLimitError
from quant.sentiment.gold import CSV_ENCODING
from quant.sentiment.prompts import ACTIONS, ASSET_TYPES, HORIZONS, OPINION_SCHEMA, STANCES, system_prompt, user_message
from quant.sentiment.store import now_utc

LINE_SECONDS = 30  # one prompt line per ~30 s, each with its [seconds] mark
UNKNOWN = "UNKNOWN"
Progress = Callable[[str], None]


# --- transcript pieces ------------------------------------------------------------------

@dataclass(frozen=True)
class Line:
    start: int
    text: str


def transcript_lines(segments: list[dict]) -> list[Line]:
    lines, start, words = [], None, []
    for segment in segments:
        if start is not None and segment["start"] - start >= LINE_SECONDS:
            lines.append(Line(int(start), " ".join(words)))
            start, words = None, []
        if start is None:
            start = segment["start"]
        words.append(" ".join(segment["text"].split()))
    if words:
        lines.append(Line(int(start), " ".join(words)))
    return lines


def chunks(lines: list[Line], minutes: float, overlap_seconds: float) -> list[list[Line]]:
    """Pieces of about `minutes`, each starting `overlap_seconds` before the previous one ends."""
    if not lines:
        return []
    length, pieces, begin = minutes * 60, [], lines[0].start
    end_of_video = lines[-1].start
    while True:
        piece = [line for line in lines if begin <= line.start < begin + length]
        if piece:
            pieces.append(piece)
        if begin + length > end_of_video:
            return pieces
        begin = begin + length - overlap_seconds


def estimate_tokens(text: str) -> int:
    """Rough: about 4 characters per token for Latin text, about 1 token per CJK character."""
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    return cjk + (len(text) - cjk) // 4


# --- quote check --------------------------------------------------------------------------

def _normal(text: str) -> str:
    """Compare on letters and digits only: NFKC (full-width -> half-width), case-folded."""
    return "".join(ch for ch in unicodedata.normalize("NFKC", text).casefold() if ch.isalnum())


def match_quote(quote: str, piece: list[Line]) -> tuple[float, int | None]:
    """How well the quote matches the piece (0..1), and the start second of the best line."""
    target = _normal(quote)
    if not target:
        return 0.0, None
    texts = [_normal(line.text) for line in piece]
    joined, owners = "", []
    for index, text in enumerate(texts):
        joined += text
        owners.extend([index] * len(text))
    if not joined:
        return 0.0, None
    found = joined.find(target)
    if found >= 0:
        return 1.0, piece[owners[found]].start
    size, step, best, best_at = len(target), max(1, len(target) // 5), 0.0, 0
    for at in range(0, max(1, len(joined) - size + 1), step):
        window = joined[at:at + size]
        matcher = difflib.SequenceMatcher(None, target, window, autojunk=False)
        if matcher.quick_ratio() <= best:
            continue
        score = matcher.ratio()
        if score > best:
            best, best_at = score, at
    return best, piece[owners[min(best_at, len(owners) - 1)]].start


# --- validation and merging ---------------------------------------------------------------

@dataclass
class Opinion:
    ticker: str
    name_as_said: str
    asset_type: str
    stance: str
    horizon: str
    condition: str
    conviction: int
    action: str
    quote: str
    start_seconds: int | None
    reasoning: str
    quote_score: float
    verification: str
    other_quotes: list[dict] = field(default_factory=list)


class InvalidReply(BackendError):
    """The reply does not have the expected fields or values."""


def parse_opinions(payload: dict) -> list[dict]:
    items = payload.get("opinions")
    if not isinstance(items, list):
        raise InvalidReply("reply has no 'opinions' list")
    parsed = []
    for item in items:
        if not isinstance(item, dict):
            raise InvalidReply("an opinion is not an object")
        try:
            row = {
                "ticker": str(item["ticker"]).strip(), "name_as_said": str(item.get("name_as_said") or "").strip(),
                "asset_type": str(item["asset_type"]).upper(), "stance": str(item["stance"]).upper(),
                "horizon": str(item["horizon"]).upper(), "condition": str(item.get("condition") or "").strip(),
                "conviction": int(item["conviction"]), "action": str(item["action"]).upper(),
                "quote": str(item["quote"]).strip(), "reasoning": str(item.get("reasoning") or "").strip(),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidReply(f"opinion missing or bad field: {exc}") from exc
        if (row["stance"] not in STANCES or row["horizon"] not in HORIZONS or row["action"] not in ACTIONS
                or row["asset_type"] not in ASSET_TYPES or not 1 <= row["conviction"] <= 3 or not row["quote"]):
            raise InvalidReply(f"opinion with values outside the schema: {item}")
        if row["stance"] != "CONDITIONAL":
            row["condition"] = ""
        parsed.append(row)
    return parsed


def check_opinion(row: dict, piece: list[Line], book: TickerBook, threshold: float) -> Opinion:
    ticker = normalize_ticker(row["ticker"])
    if ticker != UNKNOWN and not TICKER.match(ticker):
        ticker = UNKNOWN
    score, start = match_quote(row["quote"], piece)
    if ticker == UNKNOWN or not book.known(ticker):
        verification = "UNRESOLVED"
    elif score < threshold:
        verification = "UNVERIFIED"
    else:
        verification = "VERIFIED"
    asset_type = "ETF" if book.is_etf(ticker) else row["asset_type"]
    return Opinion(ticker=ticker, name_as_said=row["name_as_said"] or ticker, asset_type=asset_type,
                   stance=row["stance"], horizon=row["horizon"], condition=row["condition"],
                   conviction=row["conviction"], action=row["action"], quote=row["quote"], start_seconds=start,
                   reasoning=row["reasoning"], quote_score=round(score, 3), verification=verification)


def merge(opinions: list[Opinion]) -> list[Opinion]:
    """Section 4.5: one opinion per stock and horizon (per name for UNKNOWN tickers)."""
    groups: dict[tuple, list[Opinion]] = {}
    for opinion in opinions:
        key = (opinion.ticker, opinion.horizon, opinion.name_as_said if opinion.ticker == UNKNOWN else "")
        groups.setdefault(key, []).append(opinion)
    merged = []
    for group in groups.values():
        group.sort(key=lambda o: (o.start_seconds is None, o.start_seconds or 0))
        verified = [o for o in group if o.verification == "VERIFIED"]
        final = (verified or group)[-1]
        seen, others = {_normal(final.quote)}, []
        for o in group:
            if o is not final and _normal(o.quote) not in seen:
                seen.add(_normal(o.quote))
                others.append({"stance": o.stance, "quote": o.quote, "start_seconds": o.start_seconds})
        actions = [o.action for o in group if o.action != "NONE"]
        merged.append(Opinion(**{**vars(final), "conviction": max(o.conviction for o in group),
                                 "action": actions[-1] if actions else "NONE", "other_quotes": others}))
    return merged


# --- the run ------------------------------------------------------------------------------

@dataclass
class VideoOutcome:
    video_id: str
    status: str  # OK | FAILED
    opinions: list[Opinion] = field(default_factory=list)
    error: str | None = None


@dataclass
class ExtractResult:
    planned: int = 0
    done: list[VideoOutcome] = field(default_factory=list)
    held_back: int = 0
    stopped: str | None = None  # usage limit

    @property
    def failed(self) -> int:
        return sum(1 for v in self.done if v.status == "FAILED")


def held_back_videos(labels_dir: str | Path) -> set[str]:
    """Videos in the gold set / AI reference that are not labeled yet."""
    path = Path(labels_dir) / "videos.csv"
    if not path.exists():
        return set()
    with path.open(encoding=CSV_ENCODING, newline="") as source:
        return {row["video_id"] for row in csv.DictReader(source)
                if row.get("video_id") and row.get("status", "").strip() not in ("done", "done-none")}


def videos_to_extract(engine: Engine, backend: str, model: str, prompt_version: int,
                      video_ids: list[str] | None = None, reextract: bool = False) -> pd.DataFrame:
    v, t, c, x = (tables.sentiment_videos, tables.sentiment_transcripts, tables.sentiment_channels,
                  tables.sentiment_extractions)
    done = select(x.c.video_id).where(and_(x.c.backend == backend, x.c.model == model,
                                           x.c.prompt_version == prompt_version, x.c.status == "OK"))
    stmt = (select(v.c.video_id, v.c.channel_id, v.c.title, v.c.published_at, c.c.name, c.c.language,
                   t.c.segments_json, t.c.segments_simplified_json)
            .join(t, t.c.video_id == v.c.video_id).join(c, c.c.channel_id == v.c.channel_id)
            .where(v.c.transcript_status == "OK").order_by(v.c.published_at))
    if not reextract:
        stmt = stmt.where(v.c.video_id.not_in(done))
    if video_ids:
        stmt = stmt.where(v.c.video_id.in_(video_ids))
    with engine.connect() as conn:
        return pd.DataFrame(conn.execute(stmt).mappings().all(), columns=list(stmt.selected_columns.keys()))


def video_pieces(video, settings: Settings) -> list[list[Line]]:
    raw = video.segments_simplified_json if isinstance(video.segments_simplified_json, str) else video.segments_json
    config = settings.sentiment.extraction
    return chunks(transcript_lines(json.loads(raw)), config.chunk_minutes, config.chunk_overlap_seconds)


def plan(videos: pd.DataFrame, settings: Settings, system: str) -> tuple[int, int]:
    """(pieces, estimated input tokens) for these videos."""
    pieces = tokens = 0
    overhead = estimate_tokens(system) + 600  # schema and message framing
    for video in videos.itertuples(index=False):
        for piece in video_pieces(video, settings):
            pieces += 1
            tokens += overhead + estimate_tokens(" ".join(line.text for line in piece))
    return pieces, tokens


def _save(engine: Engine, video, backend: Backend, prompt_version: int, outcome: VideoOutcome, chunks_done: int,
          tokens: tuple[int, int], resolved: str | None, calendar: str) -> None:
    x, o = tables.sentiment_extractions, tables.sentiment_opinions
    now = now_utc()
    published = video.published_at.to_pydatetime() if hasattr(video.published_at, "to_pydatetime") else video.published_at
    effective = first_session_opening_after(published.replace(tzinfo=timezone.utc), calendar)
    key = {"video_id": video.video_id, "backend": backend.name, "model": backend.model,
           "prompt_version": prompt_version}
    with engine.begin() as conn:
        same = and_(*(x.c[k] == val for k, val in key.items()))
        if outcome.status == "FAILED" and conn.execute(select(x.c.status).where(same)).scalar() == "OK":
            return  # a failed --reextract keeps the earlier good result
        conn.execute(delete(x).where(same))
        conn.execute(delete(o).where(and_(o.c.source == "youtube", o.c.item_id == video.video_id,
                                          o.c.backend == backend.name, o.c.model == backend.model,
                                          o.c.prompt_version == prompt_version)))
        conn.execute(x.insert().values(**key, status=outcome.status, model_resolved=resolved, chunks=chunks_done,
                                       opinions=len(outcome.opinions), input_tokens=tokens[0],
                                       output_tokens=tokens[1], error=outcome.error, extracted_at=now))
        for op in outcome.opinions:
            conn.execute(o.insert().values(
                source="youtube", item_id=video.video_id, channel_id=video.channel_id, author=None,
                published_at=published, effective_date=effective, ticker=op.ticker, name_as_said=op.name_as_said,
                asset_type=op.asset_type, stance=op.stance, horizon=op.horizon, condition=op.condition or None,
                conviction=op.conviction, action=op.action, quote=op.quote, start_seconds=op.start_seconds,
                quote_score=op.quote_score, reasoning=op.reasoning,
                other_quotes_json=json.dumps(op.other_quotes, ensure_ascii=False) if op.other_quotes else None,
                verification=op.verification, backend=backend.name, model=backend.model,
                prompt_version=prompt_version, extracted_at=now))


def extract_video(video, settings: Settings, backend: Backend, system: str, book: TickerBook,
                  progress: Progress) -> tuple[VideoOutcome, int, tuple[int, int], str | None]:
    """Raises UsageLimitError (nothing saved for this video)."""
    pieces = video_pieces(video, settings)
    found: list[Opinion] = []
    tokens_in = tokens_out = 0
    resolved = None
    published = f"{video.published_at:%Y-%m-%d}"
    for number, piece in enumerate(pieces, start=1):
        message = user_message(video.name, video.title, published, number, len(pieces),
                               [f"[{line.start}] {line.text}" for line in piece])
        error = None
        for _attempt in (1, 2):  # one retry
            try:
                reply = backend.complete(system, message, OPINION_SCHEMA)
                rows = parse_opinions(reply.payload)
            except UsageLimitError:
                raise
            except BackendError as exc:
                error = str(exc)
                continue
            tokens_in += reply.input_tokens
            tokens_out += reply.output_tokens
            resolved = reply.model
            threshold = settings.sentiment.extraction.quote_match_threshold
            found.extend(check_opinion(row, piece, book, threshold) for row in rows)
            progress(f"    piece {number}/{len(pieces)}: {len(rows)} opinion(s)")
            break
        else:
            progress(f"    piece {number}/{len(pieces)}: FAILED ({error})")
            return VideoOutcome(video.video_id, "FAILED", error=error), number - 1, (tokens_in, tokens_out), resolved
    return VideoOutcome(video.video_id, "OK", merge(found)), len(pieces), (tokens_in, tokens_out), resolved


def run_extract(engine: Engine, settings: Settings, backend: Backend, book: TickerBook, videos: pd.DataFrame,
                progress: Progress | None = None) -> ExtractResult:
    progress = progress or (lambda line: None)
    config = settings.sentiment.extraction
    system = system_prompt(config.prompt_version, book.prompt_table())
    result = ExtractResult(planned=len(videos))
    for number, video in enumerate(videos.itertuples(index=False), start=1):
        progress(f"[{number}/{len(videos)}] {video.name} {video.published_at:%Y-%m-%d} {str(video.title)[:50]}")
        try:
            outcome, done, tokens, resolved = extract_video(video, settings, backend, system, book, progress)
        except UsageLimitError as exc:
            result.stopped = f"usage limit reached ({exc}); the remaining videos wait for the next run"
            progress(f"  STOPPED: {result.stopped}")
            return result
        _save(engine, video, backend, config.prompt_version, outcome, done, tokens, resolved,
              settings.market.calendar)
        counts = pd.Series([o.verification for o in outcome.opinions]).value_counts().to_dict()
        progress(f"  {outcome.status}: {len(outcome.opinions)} opinion(s) "
                 + (", ".join(f"{k.lower()} {v}" for k, v in counts.items()) if counts else ""))
        result.done.append(outcome)
    return result
