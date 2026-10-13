"""The extraction prompt and its output schema (sentiment_proposal.md sections 4.1-4.2).

`PROMPTS` holds every prompt version ever used, so opinions stored with an old
`prompt_version` can always be traced to the exact text. Change the prompt by adding a new
version and bumping sentiment.extraction.prompt_version; never edit a version in use.
"""

from __future__ import annotations

STANCES = ("BULLISH", "BEARISH", "NEUTRAL", "CONDITIONAL")
HORIZONS = ("SHORT", "MEDIUM", "LONG", "UNSPECIFIED")
ACTIONS = ("BOUGHT", "SOLD", "HOLDING", "NONE")
ASSET_TYPES = ("STOCK", "ETF")

OPINION_SCHEMA = {
    "type": "object",
    "properties": {
        "opinions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "US ticker, e.g. NVDA, BRK-B; UNKNOWN if unsure"},
                    "name_as_said": {"type": "string", "description": "how the creator refers to it"},
                    "asset_type": {"type": "string", "enum": list(ASSET_TYPES)},
                    "stance": {"type": "string", "enum": list(STANCES)},
                    "horizon": {"type": "string", "enum": list(HORIZONS)},
                    "condition": {"type": "string", "description": "only for CONDITIONAL, else empty"},
                    "conviction": {"type": "integer", "minimum": 1, "maximum": 3},
                    "action": {"type": "string", "enum": list(ACTIONS)},
                    "quote": {"type": "string", "description": "verbatim from the excerpt, original language"},
                    "start_seconds": {"type": "integer", "description": "the [seconds] mark of the quote's line"},
                    "reasoning": {"type": "string", "description": "one English sentence: the creator's reason"},
                },
                "required": ["ticker", "name_as_said", "asset_type", "stance", "horizon", "condition", "conviction",
                             "action", "quote", "start_seconds", "reasoning"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["opinions"],
    "additionalProperties": False,
}

PROMPT_V1 = """\
You extract stock opinions from an excerpt of a finance YouTube video's transcript. The
video may be in English or Chinese. Return JSON that matches the schema; return
{"opinions": []} when the excerpt has no opinions.

WHAT COUNTS
- Record only the creator's OWN view on a specific US-listed stock or ETF: "I think",
  "I'm buying", "I'd avoid it", "我看好", "我会加仓", "我不会碰".
- Do NOT record: views the creator only relays (analyst ratings, news, what "the market"
  or "people" think, viewer questions) unless the creator clearly takes that side
  themselves; stocks mentioned in passing without a view; stocks read from a list without
  comment.
- A view on the whole market or an index is recorded on its ETF: S&P 500 / "the market" ->
  SPY, Nasdaq 100 -> QQQ, Dow -> DIA, Russell 2000 -> IWM.

FIELDS
- ticker: the US ticker. Use the alias table below; caption errors are common (e.g. "英伟大"
  for 英伟达, a company name heard as a similar word). Class shares with a dash: BRK-B. If you cannot tell which company is
  meant, use UNKNOWN and put the words used in name_as_said.
- stance: BULLISH or BEARISH; NEUTRAL for an explicit "no strong view / fairly valued /
  hold, neither add nor sell"; CONDITIONAL when the view depends on a condition such as a
  price ("I'd buy below 120", "if it holds 370"), with the condition in `condition`.
- horizon: SHORT (days to weeks), MEDIUM (months), LONG (a year or more), UNSPECIFIED if
  not said. "Bearish short term, bullish long term" is TWO opinions with different horizons.
- conviction: how strongly the creator says it: 1 mild or hedged, 2 clear, 3 strong.
  It is not your confidence.
- action: what the creator says THEY did or do with their own money: BOUGHT (bought,
  added, opened), SOLD (sold, trimmed, closed), HOLDING (owns it, keeps it), else NONE.
  A recommendation to others ("you should buy") is a stance, not an action.
- quote: copy one or two consecutive sentences VERBATIM from the excerpt that show the
  view, in the original language and script. Do not translate, paraphrase, shorten inside
  the sentence, or join distant passages; leave out the [seconds] marks.
- start_seconds: the [seconds] mark at the start of the line where the quote begins.
- reasoning: one short English sentence with the creator's reason.

One opinion per stock and horizon in this excerpt; if the creator changes their mind
within it, keep the final view.

ALIAS TABLE (ticker: names creators use)
{aliases}
"""

PROMPTS = {1: PROMPT_V1}


def system_prompt(version: int, alias_table: str) -> str:
    if version not in PROMPTS:
        raise ValueError(f"no extraction prompt version {version}; known: {', '.join(map(str, PROMPTS))}")
    return PROMPTS[version].replace("{aliases}", alias_table)


SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "market": {
            "type": "object",
            "properties": {"summary": {"type": "string"},
                           "opinion_ids": {"type": "array", "items": {"type": "integer"}}},
            "required": ["summary", "opinion_ids"],
            "additionalProperties": False,
        },
        "stocks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "summary": {"type": "string"},
                    "disagreement": {"type": "boolean"},
                    "opinion_ids": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["ticker", "summary", "disagreement", "opinion_ids"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["market", "stocks"],
    "additionalProperties": False,
}

SUMMARY_PROMPT_V1 = """\
You write the daily digest of what finance YouTube creators said about US stocks. The input
lists one trading day's opinions, grouped by ticker; each opinion has an id, the creator, the
stance, horizon, conviction, the creator's own action, a condition if any, the creator's
reason and a verbatim quote (English or Chinese).

Return JSON matching the schema, in English:

- stocks: one entry per ticker in the input, in the same order. `summary` is one to three
  sentences: what the bulls and the bears each argue, and which views are conditional (with
  the condition). Name the creators. `disagreement` is true when creators take opposite sides.
  `opinion_ids` lists the ids you used; use only ids listed under that ticker.
- market: two to four sentences on the overall mood of the day across all opinions (which
  way most creators lean, recurring themes, views on SPY / QQQ if any), citing ids in
  `opinion_ids`.

Rules: use ONLY the opinions given; add no facts, prices, targets or news of your own. Do not
recommend anything and do not give your own view. Keep the creators' uncertainty: a mild or
conditional view must not read as a firm call.
"""

SUMMARY_PROMPTS = {1: SUMMARY_PROMPT_V1}


def summary_prompt(version: int) -> str:
    if version not in SUMMARY_PROMPTS:
        raise ValueError(f"no summary prompt version {version}; known: {', '.join(map(str, SUMMARY_PROMPTS))}")
    return SUMMARY_PROMPTS[version]


def user_message(channel: str, title: str, published: str, part: int, parts: int, lines: list[str]) -> str:
    return (f"Channel: {channel}\nVideo title: {title}\nPublished: {published}\n"
            f"Transcript excerpt {part} of {parts}; each line starts with its [seconds] mark:\n\n" + "\n".join(lines))
