"""`quant sentiment report`: one Markdown report per trading day (sentiment_proposal.md 8.1).

reports/sentiment/<date>.md, for the opinions whose effective date is that session (they
were published after the previous session's open and before this one's):

- market mood (the day's summary paragraph),
- stocks and ETFs: who is bullish, bearish, conditional or neutral, the net count and the
  creators (each creator counts at most once per column and stock), pool stocks marked,
- one summary per stock, written from VERIFIED opinions only,
- an appendix with every verified quote and a link to the second in the video,
- what was not counted: UNVERIFIED quotes and UNRESOLVED names (to add to the alias table).

Research only; nothing here is a signal.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine

from quant.config.settings import Settings
from quant.sentiment import tables
from quant.sentiment.summarize import inputs_hash, load_opinions, load_summaries

MARKS = {"BULLISH": "▲", "BEARISH": "▼", "CONDITIONAL": "◆", "NEUTRAL": "●"}
HORIZON = {"SHORT": "S", "MEDIUM": "M", "LONG": "L", "UNSPECIFIED": ""}
ACTION = {"BOUGHT": "bought", "SOLD": "sold", "HOLDING": "holds", "NONE": ""}


def _clock(seconds) -> str:
    if seconds is None or pd.isna(seconds):
        return "link"
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes // 60}:{minutes % 60:02d}:{secs:02d}" if minutes >= 60 else f"{minutes:02d}:{secs:02d}"


def _link(o) -> str:
    at = "" if o.start_seconds is None or pd.isna(o.start_seconds) else f"?t={int(o.start_seconds)}"
    return f"https://youtu.be/{o.item_id}{at}"


def _condition(text: str) -> str:
    """The condition as "if ...", without doubling a leading "if" the extractor already wrote."""
    text = text.strip()
    return "" if not text else ("if " + text[3:] if text.lower().startswith("if ") else f"if {text}")


def _creator_marks(channel: str, mine: pd.DataFrame) -> str:
    """One entry per creator: all their marks for this stock, e.g. "Chip Stock Investor ▲L ▼S (holds)"."""
    marks = " ".join(f"{MARKS[o.stance]}{HORIZON[o.horizon]}" for o in mine.itertuples(index=False))
    extras = list(dict.fromkeys(e for o in mine.itertuples(index=False)
                                for e in (_condition(o.condition), ACTION[o.action]) if e))
    return f"{channel} {marks}" + (f" ({'; '.join(extras)})" if extras else "")


def stock_table(group: pd.DataFrame, pool: set[str]) -> pd.DataFrame:
    rows = []
    for ticker, mine in group.groupby("ticker"):
        creators = mine.groupby("channel")["stance"].apply(set)
        count = {stance: sum(1 for stances in creators if stance in stances) for stance in MARKS}
        rows.append({
            "Stock": ticker, "Pool": "★" if ticker in pool else "",
            "Bull": count["BULLISH"], "Bear": count["BEARISH"], "Cond": count["CONDITIONAL"],
            "Neutral": count["NEUTRAL"], "Net": count["BULLISH"] - count["BEARISH"],
            "Creators": ", ".join(_creator_marks(channel, theirs) for channel, theirs in mine.groupby("channel")),
            "_creators": len(creators),
        })
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["_abs"] = frame["Net"].abs()
    frame = frame.sort_values(["_creators", "_abs", "Stock"], ascending=[False, False, True])
    return frame.drop(columns=["_creators", "_abs"])


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in frame.itertuples(index=False):
        cells = [f"{v:+d}" if c == "Net" else str(v) for c, v in zip(columns, row)]
        lines.append("| " + " | ".join(cell.replace("|", "/") for cell in cells) + " |")
    return lines


def models_used(day: pd.DataFrame) -> str:
    """Which extraction model the day's videos come from, e.g. "sonnet (3 videos), haiku (2 videos)"."""
    videos = day.drop_duplicates("item_id")["model"].value_counts()
    return ", ".join(f"{model} ({n} video{'s' if n > 1 else ''})" for model, n in videos.items())


def _other_model(o, settings: Settings) -> str:
    return "" if o.model == settings.sentiment.extraction.model else f"by {o.model}"


def render_day(day_date: date, day: pd.DataFrame, summaries: dict[str, dict], settings: Settings) -> str:
    verified = day[day["verification"] == "VERIFIED"]
    pool = set(settings.market.stocks)
    extraction, summary = settings.sentiment.extraction, settings.sentiment.summary
    first, last = day["published_at"].min(), day["published_at"].max()
    lines = [
        f"# Creator opinions, effective {day_date:%Y-%m-%d (%a)}",
        "",
        f"{len(verified)} verified opinion(s) from {verified['channel'].nunique()} creator(s) on "
        f"{verified['ticker'].nunique()} ticker(s), published {first:%Y-%m-%d %H:%M} .. {last:%Y-%m-%d %H:%M} UTC; "
        "they count from this session's open. Research only, not advice: a creator's opinion is not a signal.",
        "",
        f"Extraction {extraction.backend} {models_used(day)}, prompt v{extraction.prompt_version} ({extraction.model} "
        f"accepted provisionally; quotes from another model are marked); summaries {summary.backend}/{summary.model}, "
        f"prompt v{summary.prompt_version}, written from verified opinions only. ▲ bullish ▼ bearish ◆ conditional "
        "● neutral; S/M/L = short / medium / long term; ★ = in your stock pool.",
        "",
        "## Market mood",
        "",
        summaries.get(tables.MARKET, {}).get("summary", "_No summary yet: run `quant sentiment summarize`._"),
        "",
    ]
    for title, kind in (("Stocks", "STOCK"), ("ETFs and indices", "ETF")):
        group = verified[verified["asset_type"] == kind]
        if group.empty:
            continue
        table = stock_table(group, pool)
        lines += [f"## {title}", ""] + _markdown_table(table) + [""]
        for ticker in table["Stock"]:
            entry = summaries.get(ticker)
            text = entry["summary"] if entry else "_no summary_"
            flag = " (creators disagree)" if entry and entry["disagreement"] else ""
            lines.append(f"- **{ticker}**{flag}: {text}")
        lines.append("")
    lines += ["## Appendix: quotes", ""]
    for ticker, mine in verified.groupby("ticker"):
        lines.append(f"### {ticker}")
        for o in mine.itertuples(index=False):
            extras = [e for e in (_condition(o.condition), ACTION[o.action],
                                  "auto captions" if o.is_generated == 1 else "", _other_model(o, settings)) if e]
            lines.append(f"- **{o.channel}** {MARKS[o.stance]} {o.stance} {o.horizon}, conviction {o.conviction}"
                         + (f" ({'; '.join(extras)})" if extras else "")
                         + f" · [{_clock(o.start_seconds)}]({_link(o)}) · _{str(o.title)[:80]}_")
            lines.append(f"  > {o.quote}")
            if o.reasoning:
                lines.append(f"  Reason: {o.reasoning}")
        lines.append("")
    unverified = day[day["verification"] == "UNVERIFIED"]
    unresolved = day[day["verification"] == "UNRESOLVED"]
    if len(unverified) or len(unresolved):
        lines += ["## Not counted", ""]
        if len(unverified):
            lines.append(f"- {len(unverified)} opinion(s) whose quote did not match the transcript: "
                         + ", ".join(f"{o.ticker} ({o.channel})" for o in unverified.itertuples(index=False)))
        if len(unresolved):
            names = unresolved["name_as_said"].value_counts()
            lines.append("- Unresolved names (add real stocks to the alias table): "
                         + ", ".join(f"{name} (x{count})" for name, count in names.items()))
        lines.append("")
    return "\n".join(lines)


def resolve_ticker(text: str, aliases: dict[str, tuple[str, ...]]) -> str:
    """A ticker as typed (nvda, BRK.B) or a name from the alias table (英伟达, Nvidia)."""
    from quant.sentiment.aliases import normalize_ticker

    wanted = text.strip()
    for ticker, names in aliases.items():
        if any(wanted.casefold() == name.casefold() for name in names):
            return ticker
    return normalize_ticker(wanted)


def stock_history(engine: Engine, settings: Settings, ticker: str, start: date | None = None,
                  end: date | None = None, include_all: bool = False) -> pd.DataFrame:
    """Every opinion on one ticker, oldest first; VERIFIED only unless include_all."""
    opinions = load_opinions(engine, settings, start, end, verified_only=not include_all)
    return opinions[opinions["ticker"] == ticker].sort_values(["effective_date", "published_at", "channel"])


def format_history(ticker: str, history: pd.DataFrame, engine: Engine, settings: Settings,
                   pool: set[str]) -> str:
    if history.empty:
        return f"No opinions on {ticker} yet."
    verified = history[history["verification"] == "VERIFIED"]
    counts = verified["stance"].value_counts()
    first, last = history["effective_date"].min(), history["effective_date"].max()
    lines = [
        f"{ticker}{' (in your stock pool)' if ticker in pool else ''}: {len(history)} opinion(s) from "
        f"{history['channel'].nunique()} creator(s), effective {first:%Y-%m-%d} .. {last:%Y-%m-%d}",
        "  " + ", ".join(f"{MARKS[s]} {s.lower()} {counts.get(s, 0)}" for s in MARKS),
        "Research only, not advice: a creator's opinion is not a signal.",
        "",
    ]
    for day_date, day in history.groupby("effective_date", sort=True):
        lines.append(f"== {day_date:%Y-%m-%d (%a)}")
        everyone = load_opinions(engine, settings, day_date, day_date, verified_only=True)
        summary = load_summaries(engine, settings, day_date, inputs_hash(everyone, settings)).get(ticker)
        if summary:
            lines.append(f"   Summary: {summary['summary']}")
        for o in day.itertuples(index=False):
            extras = [e for e in (_condition(o.condition), ACTION[o.action],
                                  "auto captions" if o.is_generated == 1 else "", _other_model(o, settings),
                                  "" if o.verification == "VERIFIED" else o.verification.lower()) if e]
            lines.append(f"   {MARKS[o.stance]} {o.channel}: {o.stance} {o.horizon}, conviction {o.conviction}"
                         + (f" ({'; '.join(extras)})" if extras else ""))
            lines.append(f"     \"{o.quote}\"")
            lines.append(f"     {_link(o)}  ({_clock(o.start_seconds)}, {str(o.title)[:60]})")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_reports(engine: Engine, settings: Settings, start: date | None = None, end: date | None = None) -> list[Path]:
    opinions = load_opinions(engine, settings, start, end)
    directory = Path(settings.sentiment.summary.reports_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for day_date, day in opinions.groupby("effective_date"):
        verified = day[day["verification"] == "VERIFIED"]
        summaries = load_summaries(engine, settings, day_date, inputs_hash(verified, settings)) if len(verified) else {}
        path = directory / f"{day_date:%Y-%m-%d}.md"
        path.write_text(render_day(day_date, day, summaries, settings), encoding="utf-8")
        paths.append(path)
    return paths
