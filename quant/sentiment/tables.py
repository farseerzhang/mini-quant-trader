"""Tables of the opinion factor (sentiment_proposal.md section 6), on the shared metadata.

quant/persistence/tables.py does not change: these tables are registered when this module
is imported, so `init_db` creates them for the sentiment commands. S1 has the fetch tables;
opinions, summaries and out-of-pool prices follow with S2 .. S4.

Times are stored as naive UTC.
"""

from sqlalchemy import CheckConstraint, Column, DateTime, ForeignKey, Integer, String, Table, Text

from quant.persistence.tables import metadata

TRANSCRIPT_STATUSES = ("PENDING", "OK", "NO_TRANSCRIPT", "FAILED", "SKIPPED")


# Synchronized from config (sentiment.channels), which is the source of truth, like `symbols`.
sentiment_channels = Table(
    "sentiment_channels",
    metadata,
    Column("channel_id", String, primary_key=True),
    Column("handle", String, nullable=False),
    Column("name", String, nullable=False),
    Column("language", String, nullable=False),  # en | zh
    Column("style_note", String),
    Column("active", Integer, nullable=False, server_default="1"),
    Column("added_at", DateTime, nullable=False),
    Column("updated_at", DateTime, nullable=False),
)


sentiment_videos = Table(
    "sentiment_videos",
    metadata,
    Column("video_id", String, primary_key=True),
    Column("channel_id", String, ForeignKey("sentiment_channels.channel_id"), nullable=False),
    Column("title", String),
    # When viewers could first see it (premiere time if any); NULL for videos skipped from
    # the channel listing alone (members-only, Shorts), whose page is never read.
    Column("published_at", DateTime),
    Column("duration_seconds", Integer),
    Column("live_status", String),  # yt-dlp: not_live | was_live | ...
    Column("availability", String),  # yt-dlp: public | subscriber_only | ...
    Column("transcript_status", String, nullable=False),
    Column("note", String),  # why SKIPPED / NO_TRANSCRIPT / FAILED
    Column("attempts", Integer, nullable=False, server_default="0"),  # caption fetch attempts
    Column("fetched_at", DateTime, nullable=False),  # first listed
    Column("updated_at", DateTime, nullable=False),
    CheckConstraint(
        "transcript_status IN (" + ", ".join(f"'{s}'" for s in TRANSCRIPT_STATUSES) + ")",
        name="ck_sentiment_videos_status",
    ),
)


# The raw captions are the source of truth (proposal 4.3): extraction can be re-run from them.
sentiment_transcripts = Table(
    "sentiment_transcripts",
    metadata,
    Column("video_id", String, ForeignKey("sentiment_videos.video_id"), primary_key=True),
    Column("language", String, nullable=False),  # caption language code, e.g. en, zh-Hans, zh-TW
    Column("is_generated", Integer, nullable=False),  # 1 = YouTube auto captions
    Column("segments_json", Text, nullable=False),  # [{start, duration, text}]
    Column("segments_simplified_json", Text),  # Chinese channels: the same in Simplified Chinese
    Column("char_count", Integer, nullable=False),
    Column("fetched_at", DateTime, nullable=False),
)
