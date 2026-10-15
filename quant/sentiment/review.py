"""`quant sentiment review-export`: the disagreements between an extractor and the labels.

The labels in sentiment.gold.labels_dir are an AI reference (decision 12), so agreement with
them is not accuracy; a person decides the disagreements (sentiment_proposal.md section 4.6).

Opinions are matched on video and stock. Cases:

- only_reference   the reference has the stock, the extractor does not
- only_extractor   the extractor has the stock, the reference does not
- direction        both have it, but no extractor opinion points the same way
                   (bullish / bearish / other) as the reference

Each case gets the extractor's quote, a link to the second in the video and the transcript
lines around it. The file holds transcript text, so it is written next to the reading copies
(git-ignored), not with the labels.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, and_, select

from quant.config.settings import Settings
from quant.sentiment import tables
from quant.sentiment.extract import transcript_lines
from quant.sentiment.gold import CSV_ENCODING

COLUMNS = ["case", "kind", "video_id", "channel", "ticker", "reference", "reference_time", "reference_note",
           "extractor", "extractor_quote", "extractor_time", "link", "transcript_around", "verdict", "comment"]
AROUND_SECONDS = 45


def direction(stance: str) -> str:
    return {"BULLISH": "+", "BEARISH": "-"}.get(stance, "0")


def _seconds(text: str) -> int | None:
    try:
        parts = [int(p) for p in str(text).split(":")]
    except ValueError:
        return None
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return None


def _clock(seconds: int | None) -> str:
    if seconds is None:
        return ""
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes // 60}:{minutes % 60:02d}:{secs:02d}" if minutes >= 60 else f"{minutes:02d}:{secs:02d}"


@dataclass
class Agreement:
    reference_pairs: int
    extractor_pairs: int
    both: int
    same_direction: int

    @property
    def recall(self) -> float:
        return self.both / self.reference_pairs if self.reference_pairs else float("nan")

    @property
    def precision(self) -> float:
        return self.both / self.extractor_pairs if self.extractor_pairs else float("nan")


def _read_labels(labels_dir: Path) -> tuple[list[str], pd.DataFrame]:
    with (labels_dir / "videos.csv").open(encoding=CSV_ENCODING, newline="") as source:
        videos = [r["video_id"] for r in csv.DictReader(source) if r.get("status", "").strip() in ("done", "done-none")]
    with (labels_dir / "opinions.csv").open(encoding=CSV_ENCODING, newline="") as source:
        opinions = pd.DataFrame(list(csv.DictReader(source)))
    return videos, opinions[opinions["video_id"].isin(videos)] if len(opinions) else opinions


def _extracted(engine: Engine, videos: list[str], backend: str, model: str, prompt_version: int) -> pd.DataFrame:
    o, x = tables.sentiment_opinions, tables.sentiment_extractions
    with engine.connect() as conn:
        done = {r.video_id for r in conn.execute(select(x.c.video_id).where(and_(
            x.c.backend == backend, x.c.model == model, x.c.prompt_version == prompt_version, x.c.status == "OK",
            x.c.video_id.in_(videos))))}
        rows = conn.execute(select(o).where(and_(o.c.item_id.in_(videos), o.c.backend == backend,
                                                 o.c.model == model, o.c.prompt_version == prompt_version)))
        frame = pd.DataFrame(rows.mappings().all(), columns=[c.name for c in o.columns])
    missing = sorted(set(videos) - done)
    if missing:
        raise ValueError(f"{len(missing)} labeled video(s) are not extracted with {backend}/{model}/v{prompt_version} "
                         f"yet (e.g. {missing[0]}); run `quant sentiment extract --model {model}` first")
    return frame


def _transcripts(engine: Engine, videos: list[str]) -> dict[str, list]:
    t, v, c = tables.sentiment_transcripts, tables.sentiment_videos, tables.sentiment_channels
    with engine.connect() as conn:
        rows = conn.execute(select(t.c.video_id, t.c.segments_json, t.c.segments_simplified_json, c.c.handle)
                            .join(v, v.c.video_id == t.c.video_id).join(c, c.c.channel_id == v.c.channel_id)
                            .where(t.c.video_id.in_(videos))).all()
    return {r.video_id: (transcript_lines(json.loads(r.segments_simplified_json or r.segments_json)), r.handle)
            for r in rows}


def _around(lines: list, seconds: int | None) -> str:
    if seconds is None:
        return ""
    near = [line for line in lines if seconds - AROUND_SECONDS <= line.start <= seconds + AROUND_SECONDS]
    return "\n".join(f"[{_clock(line.start)}] {line.text}" for line in near)


def _describe(row) -> str:
    text = f"{row['stance']} {row['horizon']}, conviction {row['conviction']}, action {row['action']}"
    condition = row.get("condition")
    return text + (f", if: {condition}" if isinstance(condition, str) and condition.strip() else "")  # not NaN


def build_review(engine: Engine, settings: Settings, model: str | None = None) -> tuple[pd.DataFrame, Agreement]:
    config = settings.sentiment.extraction
    model = model or config.model
    videos, reference = _read_labels(Path(settings.sentiment.gold.labels_dir))
    extracted = _extracted(engine, videos, config.backend, model, config.prompt_version)
    transcripts = _transcripts(engine, videos)
    ref_pairs = {(r["video_id"], r["ticker"]): r for r in reference.to_dict("records")}
    ext_pairs: dict[tuple, list[dict]] = {}
    for r in extracted.to_dict("records"):
        ext_pairs.setdefault((r["item_id"], r["ticker"]), []).append(r)
    cases, same = [], 0
    for key in sorted(set(ref_pairs) | set(ext_pairs), key=lambda k: (videos.index(k[0]), k[1])):
        ref, ext = ref_pairs.get(key), ext_pairs.get(key, [])
        if ref is not None and ext:
            if direction(ref["stance"]) in {direction(e["stance"]) for e in ext}:
                same += 1
                continue
            kind = "direction"
        else:
            kind = "only_reference" if ref is not None else "only_extractor"
        lines, handle = transcripts.get(key[0], ([], ""))
        first = ext[0] if ext else None
        at = first["start_seconds"] if first else _seconds(ref["time"])
        cases.append({
            "kind": kind, "video_id": key[0], "channel": handle, "ticker": key[1],
            "reference": _describe(ref) if ref is not None else "",
            "reference_time": ref["time"] if ref is not None else "",
            "reference_note": ref.get("note", "") if ref is not None else "",
            "extractor": " | ".join(_describe(e) for e in ext),
            "extractor_quote": " | ".join(e["quote"] for e in ext),
            "extractor_time": " | ".join(_clock(e["start_seconds"]) for e in ext),
            "link": f"https://youtu.be/{key[0]}" + (f"?t={int(at)}" if at is not None else ""),
            "transcript_around": _around(lines, at), "verdict": "", "comment": "",
        })
    frame = pd.DataFrame(cases, columns=COLUMNS[1:])
    frame.insert(0, "case", range(1, len(frame) + 1))
    both = sum(1 for key in ref_pairs if key in ext_pairs)
    return frame, Agreement(len(ref_pairs), len(ext_pairs), both, same)


def write_review(frame: pd.DataFrame, settings: Settings, model: str) -> Path:
    config = settings.sentiment.extraction
    directory = Path(settings.sentiment.gold.transcripts_dir).parent / "reviews"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"review-{config.backend}-{model}-v{config.prompt_version}.csv"
    if path.exists():
        raise ValueError(f"{path} already exists (it may hold your verdicts); move or delete it first")
    frame.to_csv(path, index=False, encoding=CSV_ENCODING)
    return path
