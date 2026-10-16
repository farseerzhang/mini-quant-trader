# Creator & Crowd Opinion Factor (YouTube, Reddit) — Proposal

Status: approved 2026-10-12; revised 2026-10-13 after review (extraction backends, Claude-written summaries, scorecard statistics, forward-only promotion; decisions in section 14). S0 done (section 12.1); S1 built 2026-10-13 (section 12.2); S2 steps 1-5 built (section 12.3); S2 review done quickly and kept as given, so S2 is accepted provisionally with Sonnet (section 12.3); the Haiku comparison and Gemini backend are postponed. S3 built (section 12.4); it is done once you accept a week of reports.
Date: 2026-10-12

---

# 1. Goal

Collect commentary on individual US stocks from a few finance YouTube creators (English and Chinese), extract structured opinions from their transcripts, and summarize them by **date × stock** to show who is bullish and who is bearish. Once enough data has been collected, measure each creator's historical hit rate to decide whether their opinions can feed the existing system as a sentiment factor.

As in `proposal.md`, this is a research tool. **A creator's opinion is not a trading signal**, and it does not enter `generate-signals` until it passes the pre-registered forward test of section 9. Until then its realistic use is as reading material: before acting on a BUY or EXIT, you can see what the creators said about that stock.

## Out of scope (first version)

- No transcription of videos without captions (Whisper is deferred to S5).
- No scheduler. Like the main system, the CLI is run on demand.
- **Not part of `run-daily`.** Sentiment commands depend on YouTube and an LLM; a failure there must never block the evening forward-test freeze. They run as separate commands.
- No changes to existing modules or tables in `quant/` beyond the one line in `quant/cli.py` that registers the subcommands. All new code goes in `quant/sentiment/` and new tables.
- No Streamlit page yet. Output is the command line and Markdown reports.
- No paid API. Extraction runs on the Claude Code Pro subscription or the free Gemini API (section 4.1).

---

# 2. Pipeline

```
config/settings.yaml (sentiment.channels)
        │
        ▼
 ① List videos      yt-dlp (no key needed)          → sentiment_videos
        │
        ▼
 ② Fetch captions   youtube-transcript-api           → sentiment_transcripts
        │           (no captions → status = NO_TRANSCRIPT, skipped)
        ▼
 ③ Extract opinions Extractor backend (claude_code or gemini), JSON output
        │           → sentiment_opinions
        │           (each opinion carries a quote and timestamp, and the quote
        │            is checked against the transcript)
        ▼
 ④ Summarize        the day's VERIFIED opinions → one paragraph per stock
        │           + one market-mood paragraph  → sentiment_summaries
        ▼
 ⑤ Daily report     date × stock × bullish/bearish + summaries
        │           → reports/sentiment/<date>.md
        ▼
 ⑥ Hit rate         excess return vs the pool 5/20/60 sessions after the opinion
                    → reports/sentiment/scorecard.md
```

Every step is idempotent and keyed by `video_id` (or date × ticker for summaries). A re-run only processes new or failed items, matching the main system's convention (proposal section 30).

---

# 3. Data sources

## 3.1 Video list: `yt-dlp`

- `yt-dlp --flat-playlist` reads a channel's `/videos` page for `video_id`, title and duration. The metadata of each new video is then read once for the exact publish time.
- Advantage: no Google Cloud key needed. Disadvantage: like `yfinance`, it is an unofficial tool and can break.
- It therefore sits behind an interface (`VideoSource`), just like market data, so it can later be replaced with the YouTube Data API v3 without affecting anything downstream.
- Filters: skip Shorts (duration < 3 minutes) and live-stream replays (configurable), and keep only videos inside the lookback window (default 14 days).

## 3.2 Captions: `youtube-transcript-api`

- Language priority: manual captions before auto-generated ones. English channels use `en`; Chinese channels try `zh-Hans`, then `zh-Hant`, then `zh`.
- Whether captions are manual or auto-generated is recorded (`is_generated`). Auto-generated captions contain many errors, so reports flag them, and hit rates can be computed separately for them.
- Traditional Chinese is converted to Simplified (`opencc`) before extraction so one alias table covers both. The converted text is stored next to the raw text, and both extraction and the quote check (section 4.3) use the converted text.
- The raw transcript, with the start second of every segment, is stored in full in the database. This follows "raw data is the source of truth" (proposal 4.3): a new prompt, model or backend can re-extract from the stored transcripts without fetching them again.
- Rate limiting: a random wait of a few seconds between requests. After one failed retry a video is marked `FAILED` and retried on the next run. Runs only on this machine or a home network. The S4 backfill is spread over several days (a per-run cap) so the IP is not blocked.

---

# 4. Opinion extraction

## 4.1 Extractor backends

All extraction goes through one `Extractor` interface: transcript chunk + alias table in, a list of opinions (section 4.2) out. `sentiment.extraction.backend` picks the backend, and every opinion stores the `backend` and `model` that produced it.

| Backend | What it is | Cost | Limits |
|---|---|---|---|
| `claude_code` (default until S2 decides) | Runs Claude Code non-interactively (`claude -p`) with the prompt and the chunk on stdin, using your Pro subscription | Included in Pro | Pro usage limits (they reset every few hours, plus a weekly cap) |
| `gemini` | Google Gemini API with a free API key from Google AI Studio | Free | Requests-per-minute and requests-per-day caps on the free tier |

**`claude_code`**

- Called from Python as a subprocess, with the chunk on stdin, from a neutral working directory (so no project `CLAUDE.md` or memory is loaded):

  ```
  claude -p --model <model> --tools "" --strict-mcp-config --no-session-persistence
         --output-format json --system-prompt <extraction prompt> --json-schema <opinion schema>
  ```

  `--system-prompt` replaces Claude Code's own system prompt, which cuts the overhead per call from about 36k tokens to about 1.3k. `--json-schema` enforces the reply format; the result is in `structured_output`. Our code still validates the values; one retry on an invalid reply, then the video is marked `FAILED`. Verified on 2026-10-13 with Claude Code 2.1.296 (Haiku, a Chinese sentence → `{"ticker": "NVDA", "stance": "BULLISH"}`).
- Not `--bare`: that mode accepts only an API key and ignores the subscription login.
- `ANTHROPIC_API_KEY` must **not** be set in the environment; if it is, Claude Code bills the API instead of the subscription. The backend refuses to run when it is set.
- Needs only the `claude` command on PATH, logged in with the subscription. No API key.

**`gemini`**

- Uses the API's response schema, so the reply is valid JSON by construction; our code still validates the values.
- The key is read only from `GEMINI_API_KEY` in `.env`.
- On the free tier Google may use prompts and responses to improve its products. The input is public YouTube transcripts and Reddit posts, so this is accepted. **Nothing private is ever sent**: no `positions.csv`, `trades.csv`, signals or account values. The summary step (section 8.3) sends only opinions.
- The free tier's limits, available models and regional availability change over time. They are checked at the start of S2; the model name lives only in config.

**Both backends**

- Long videos are split into chunks of about 10 minutes with a small overlap. Each chunk is extracted on its own, then the results are merged (section 4.5).
- When a backend reports a usage or rate limit, the run stops cleanly; the remaining videos stay `PENDING` for the next run.
- `max_videos_per_run` caps each run, so the 6-month backfill (about 500 videos) is spread over one to two weeks.

**Choosing in S2**: the gold set (section 4.6) compares `claude_code` with Sonnet, `claude_code` with Haiku and `gemini` with a Flash-class model. The default becomes the backend that reaches the accuracy target; if more than one does, `gemini`, so Pro usage stays free for coding.

## 4.2 Fields of each opinion

| Field | Meaning |
|---|---|
| `ticker` | US ticker. It must be found in the alias table or the ticker list, otherwise the opinion is marked `UNRESOLVED` |
| `stance` | `BULLISH` / `BEARISH` / `NEUTRAL` / `CONDITIONAL` |
| `horizon` | `SHORT` (weeks) / `MEDIUM` (months) / `LONG` (a year or more) / `UNSPECIFIED` |
| `condition` | The condition of a conditional opinion, e.g. "buy below 120"; empty otherwise |
| `conviction` | 1–3, how strongly the creator says it (not the model's confidence) |
| `action` | Whether the creator says they bought, added, sold or hold it themselves: `BOUGHT` / `SOLD` / `HOLDING` / `NONE` |
| `quote` | The original sentence from the transcript, not paraphrased |
| `start_seconds` | Where the quote occurs in the video; forms a `youtu.be/<id>?t=<seconds>` link for checking it |
| `reasoning` | One sentence summarizing the creator's reason |

Rules (written into the prompt):

- Only record the creator's **own** opinion. Relaying analyst ratings, news or viewer questions does not count unless the creator explicitly takes a side.
- "Bearish short term, bullish long term" becomes two opinions with different `horizon` values.
- Stocks mentioned in passing without a view are not recorded.
- ETFs and indices (SPY, QQQ) are recorded as usual with `asset_type = ETF/INDEX` and summarized separately from stocks.

## 4.3 Guarding against fabrication

- **Quote check**: after extraction, the `quote` must be found in its transcript chunk (fuzzy match after normalizing whitespace, punctuation, and full-width vs half-width characters; Chinese is compared on the Simplified text). Opinions that fail are marked `UNVERIFIED` and excluded from summaries and the scorecard, but kept for inspection. The threshold (`quote_match_threshold`, 0.85 as a starting value) is set per language from the gold set in S2, because character-level similarity in Chinese behaves differently from word-level similarity in English.
- **Ticker check**: the ticker returned by the model must be in the alias table or the SEC company ticker list (`company_tickers.json`, cached locally).
- Every opinion records its `backend`, `model` and `prompt_version`. Re-extracting after a prompt or backend change creates new rows and keeps the old ones, so the two can be compared.

## 4.4 Stock alias table

`config/sentiment_aliases.yaml`, maintained by hand, with the model allowed to suggest additions:

```yaml
NVDA: [英伟达, 辉达, 老黄, Nvidia]
AAPL: [苹果, Apple]
GOOGL: [谷歌, 谷歌A, Alphabet, Google]
TSLA: [特斯拉, Tesla]
BRK-B: [伯克希尔, 巴菲特的公司, Berkshire]
```

Common misrecognitions in Chinese auto-captions (e.g. "英伟达" transcribed as "英伟大") are added to the alias table too. References the model cannot resolve are output as `UNRESOLVED` and listed at the end of the report, so you can decide whether to add them.

## 4.5 Merging within a video

Chunks overlap, and a creator can return to a stock later in the same video. After all chunks are extracted, opinions on the same ticker and horizon within one video are merged into one:

- the **last** stated stance wins (creators usually conclude at the end), and the earlier quotes are kept in the appendix;
- `conviction` is the highest stated; `action` is kept if any chunk states one;
- duplicates from the chunk overlap (same quote) are dropped.

## 4.6 Gold set and accuracy target (S2)

10 videos (decision 11; both languages, manual and auto captions) are labeled by hand: every opinion with ticker, stance and horizon. Each backend/model is scored on:

- **opinion precision**: extracted opinions that are real;
- **opinion recall**: real opinions that were found (a missed bearish call skews hit rates, so recall matters most);
- **stance accuracy** on the matched opinions.

Target: precision, recall and stance accuracy each ≥ 85%. With 10 videos the counts are reported next to the percentages, since one video can move them several points (roughly ±10 points); if two backends end up close, 10 more videos are labeled to separate them.

The labels come from you, not from a model: the `claude_code` extractor is the same model family as the assistant, so labels from either would share its blind spots and the score would only measure agreement with itself.

As built (`quant sentiment gold-export`): videos are picked with channels taking turns in config order and, within a channel, in `video_id` order (arbitrary but fixed). `research/sentiment_gold/` holds `videos.csv` (one row per video, `status` done / done-none), `opinions.csv` (ticker, stance, horizon, condition, conviction, action, time, note) and `LABELING.md`; it holds no transcript text and is committed. Readable transcripts go to `data/sentiment_gold/transcripts/` (git-ignored, deletable). Later runs never touch the label files. **No extraction output of the gold videos is shown before they are labeled**; S2's `extract` must hold them back until their status is done / done-none.

**What was actually done (decision 12).** The set grew to 20 videos (`gold-export --add 10`), and all 20 were labeled with an AI tool (GPT or MAI), not by a person. They are therefore an **AI reference** from a third model family, kept in `research/sentiment_ai_reference/` with its provenance in `SOURCE.md`:

- S2 reports precision, recall and stance **agreement** with this reference, never "accuracy". It is not from Claude or Gemini, so it does not simply echo either backend, but where they disagree either side can be wrong.
- The 85% figure is no longer a pass/fail gate, because agreement with another model is not correctness. Backends are compared on agreement as one input; the quality gate for S2 becomes **your review of the disagreements**: for each opinion where the extractor and the reference differ, you read the quoted passage and say which is right (or neither). That review, not the agreement number, decides whether extraction is good enough.
- A human-labeled set can still be added later (option A, about 5 videos) if the review leaves doubt.

---

# 5. Timing convention (no look-ahead)

- The video's publish time `published_at` (UTC) is stored.
- An opinion's **effective time** is the open of the first NYSE session after the publish time, following the `next_open` convention of proposal 4.5. A video published during market hours also only counts from the next session's open.
- Hit rates are always measured from the effective time, so they never use prices the creator could not yet see.
- Captions can appear a few hours after a video is published. We record the publish time, not the fetch time, so backtests are unaffected. Live signals, if ever built, must use the fetch time instead.

---

# 6. Database

Six new tables in the same SQLite file, defined in `quant/sentiment/tables.py` on the shared `metadata`, so `quant/persistence/tables.py` does not change. Every run still writes a `pipeline_runs` row (command = `sentiment-*`).

The dev database is disposable and can be rebuilt, so there are no migrations before S4.

### sentiment_channels
`channel_id` (PK), `handle`, `name`, `language` (en/zh), `style_note`, `added_at`. Synchronized from config, which is the source of truth (as with the `symbols` table).

### sentiment_videos
`video_id` (PK), `channel_id`, `title`, `published_at`, `duration_seconds`, `transcript_status` (`PENDING` / `OK` / `NO_TRANSCRIPT` / `FAILED` / `SKIPPED`), `fetched_at`.

### sentiment_transcripts
`video_id` (PK), `language`, `is_generated`, `segments_json` (`[{start, duration, text}]`), `segments_simplified_json` (Chinese only; null otherwise), `char_count`, `fetched_at`.

### sentiment_opinions
`opinion_id` (PK), `source` (`youtube` / `reddit`), `item_id` (YouTube `video_id` or Reddit post id), `channel_id` (YouTube channel or `r/<subreddit>`), `author` (Reddit username; null for YouTube), `published_at`, `effective_date`, `ticker`, `asset_type`, `stance`, `horizon`, `condition`, `conviction`, `action`, `quote`, `start_seconds` (null for Reddit), `reasoning`, `verification` (`VERIFIED` / `UNVERIFIED` / `UNRESOLVED`), `backend`, `model`, `prompt_version`, `extracted_at`.

Unique constraint: `(source, item_id, ticker, horizon, backend, model, prompt_version)` — one merged opinion per video, stock and horizon (section 4.5) for each extractor, so the S2 comparison and later re-extractions never overwrite each other.

`source`, `item_id` and `author` exist from S1 so that Reddit (section 15) can share this table and the downstream report and scorecard code without a schema change.

Only opinions from the configured backend, model and prompt version (`sentiment.extraction.*`) feed the report, summaries and scorecard.

### sentiment_summaries
`date` (effective date), `ticker` (or `_MARKET` for the market-mood paragraph), `summary`, `disagreement` (bool), `opinion_ids_json`, `backend`, `model`, `prompt_version`, `created_at`. Primary key: `(date, ticker, backend, model, prompt_version)`. Reports are rebuilt from this table without calling a model again.

### sentiment_prices
Daily prices of mentioned tickers outside the stock pool, fetched through the same `YFinanceProvider` and stored in the same layout as `daily_prices`, so the main pool is not polluted.

---

# 7. CLI

A group of subcommands under the existing `quant` command:

```bash
quant sentiment fetch      [--days 14] [--channel <handle>]   # ①②: video list + captions
quant sentiment extract    [--limit N] [--reextract] [--yes]  # ③: only videos without opinions for the current backend/model/prompt_version
quant sentiment summarize  [--from 2026-10-01] [--to ...]     # ④: one paragraph per stock and day
quant sentiment report     [--from 2026-10-01] [--to ...]     # ⑤: date × stock summary
quant sentiment scorecard                                     # ⑥: hit rates
quant sentiment run                                           # ①–⑤ in order
```

Before running, `extract` prints the backend, the number of videos and chunks to process and the estimated token count. Above `sentiment.extraction.max_videos_per_run` it stops and needs `--yes` to continue (and still respects the backend's limits, section 4.1).

---

# 8. Output

## 8.1 Daily summary (`reports/sentiment/<date>.md`)

```
2026-10-09 (effective 2026-10-10)

Market mood: <one paragraph from sentiment_summaries, ticker _MARKET>

| Stock | Pool | Bull | Bear | Cond | Net | Creators |
|-------|------|------|------|------|-----|----------|
| NVDA  | yes  | 3    | 0    | 1    | +3  | Chip Stock Investor ▲, NaNa说美股 ▲, Financial Education ▲, 投资TALK君 ◆ (buy below 160) |
| TSLA  | yes  | 0    | 2    | 0    | -2  | ...      |

NVDA: <summary paragraph> [opinions 812, 815, 820, 823]
TSLA: <summary paragraph> [opinions ...]

Quotes and video links for every opinion are in the appendix.
Unresolved references: 英伟大 (×2), "the data-center one" (×1)
```

Net = bullish count − bearish count; `CONDITIONAL` and `NEUTRAL` are not counted. A creator counts at most once per stock per day. `Pool` marks stocks in the `settings.yaml` pool.

## 8.2 Creator scorecard (`reports/sentiment/scorecard.md`)

**Excess return.** For an opinion on stock X with effective date d and window h (5 / 20 / 60 sessions): X's return from the open of d over h sessions, minus the equal-weight return of the stock pool over the same sessions (the expected return of a random pool stock on that day). Returns use split- and dividend-adjusted prices, the same as `daily_prices`. A market-beta-adjusted excess return is reported alongside as a secondary column.

- Bullish opinions: hit if the excess return is positive.
- Bearish opinions: hit if the excess return is negative.

**Counting.** Within one creator, stock and stance, only the first opinion in each h-session window counts; repeats inside the window are dropped. A daily-wrap channel would otherwise contribute the same NVDA call twenty times to one 20-session window.

**Significance.** Permutation test: each counted opinion keeps its creator, date and stance, and its stock is replaced by a random stock from the pool on that date; repeated 10,000 times. This asks "do the creator's picks beat a random pick on the same days", keeps the dates (so days when many opinions cluster stay clustered) and works even for creators who are almost always bullish, where shuffling bullish/bearish labels would have little power.

**Scope.**

- Stocks in the pool are scored as above. Out-of-pool tickers are reported separately and are descriptive only (the pool is not their natural benchmark).
- Tickers whose prices cannot be fetched (e.g. delisted) are listed, not dropped silently.
- Each cell reports the sample size, hit rate, mean excess return and the permutation p-value. Cells with fewer than `min_samples` (30) show the numbers only, with no conclusion.
- Many cells are shown (creators × horizon labels × windows, optionally split by `is_generated`), so the report states that the p-values are not corrected and that only the pre-registered test of section 9 can lead to a decision.

**Expectations.** Telling a 55% hit rate from 50% takes roughly 600 independent samples. Thirty samples are enough to display numbers, not to conclude anything.

## 8.3 Summaries (step ④)

The extractor backend also writes a short readable summary per stock and day, built **from that day's VERIFIED opinions only**, never from the raw transcripts:

- **Input**: per stock, each verified opinion's creator, stance, horizon, condition, quote and link; for the market paragraph, the day's opinion counts and the ETF/index opinions.
- **Output** (JSON): `{ticker, summary, disagreement, opinion_ids}`. The code rejects a summary that cites an opinion id not in its input.
- **Content**: what the bulls and the bears each argue, where creators disagree and which views are conditional. No recommendation, no price target and no view of its own.
- **Display only**: summaries never enter the scorecard or the features.
- The input is small (a few thousand tokens a day), so a cheap model is enough; S2 checks a handful of summaries by hand.

---

# 9. Integration with the quant system (S6, later)

**Forward-only evidence.** Everything collected before the S6 pre-registration, including the S4 backfill, is exploration: it may suggest which creators, windows and features to test, but it cannot be the test. The S6 test uses only opinions published **after** the pre-registration date, the same discipline as the holdout in the main system (proposal 13.1 / 18.1).

It becomes a feature only when all of these hold:

1. At least **6 months** of opinions have been collected after the pre-registration date (with a 20-session window, 3 months would give only about three non-overlapping periods). This matches the first look of the main forward test.
2. The pre-registration fixes, before any forward data: one **primary test** (planned: the 20-session window, VERIFIED stock opinions in the pool, all creators pooled, the permutation test of section 8.2), the significance level, and a Holm correction for any secondary tests (single creators, other windows). The primary test must pass.
3. The opinions cover enough stocks in the pool (threshold set in the pre-registration).

Candidate features then: `sentiment_net_5d` (net bullish count over the last 5 sessions), `sentiment_consensus` (agreement), `sentiment_contrarian` (inverted for creators whose hit rate is significantly negative). They are added as new columns of `features`, which requires bumping `features.version` and re-registering models under the post-Phase-5 rules.

**Forward test.** A sentiment-based model never replaces the live strategy during an open forward-test run (proposal Phase 10). At most it joins the run as a new shadow at the next restart, and earns its place the same way the other shadows do.

**Limitations of the backfill.**

- We picked these creators today, and they may still be around precisely because they were right in the past (survivorship of creators).
- Creators delete or make private videos whose calls went wrong, and the backfill only sees what is still public (survivorship of videos).
- Backfilled data is therefore for exploration only. The real evidence comes from forward logging, in the same spirit as proposal section 15.5.

---

# 10. Configuration (added to `config/settings.yaml`)

```yaml
sentiment:
  lookback_days: 14
  min_duration_seconds: 180          # skip Shorts
  include_live: false
  request_delay_seconds: [8, 15]     # random wait range (as built; [2, 5] got the IP blocked)
  max_captions_per_run: 10           # captions per run, oldest first; the rest waits for the next run
  no_captions_retry_days: 3          # NO_TRANSCRIPT videos are asked again while younger than this
  channels:                          # chosen after S0 (section 12.1); as built, each also has channel_id and name
    - {handle: "@FinancialEducation",   language: en, style: "aggressive growth"}
    - {handle: "@chipstockinvestor",    language: en, style: "semiconductors / AI"}
    - {handle: "@NaNaShuoMeiGu",        language: zh, style: "daily market wrap"}
    - {handle: "@yttalkjun",            language: zh, style: "macro + sectors"}
    - {handle: "@Zac-Stock-Journal",    language: zh, style: "technical buy points, own holdings"}
    - {handle: "@nick_stockcafe",       language: zh, style: "own buys, Traditional Chinese"}
  extraction:
    backend: claude_code             # claude_code | gemini; decided in S2
    model: sonnet                    # placeholder; decided in S2 (a Gemini model name for gemini)
    prompt_version: 1
    chunk_minutes: 10
    quote_match_threshold: 0.85      # starting value; set per language in S2
    max_videos_per_run: 40
  # gold: as built, sentiment.gold {videos: 10, labels_dir, transcripts_dir} (section 4.6)
  summary:
    enabled: true
    backend: claude_code             # same choices as extraction
    model: haiku                     # placeholder; decided in S2
    prompt_version: 1
  evaluation:
    horizons: [5, 20, 60]
    benchmark: pool_equal_weight     # excess return vs a random pool stock on the same day
    min_samples: 30
    permutations: 10000
```

The handles were verified in S0; their `channel_id`s are listed in section 12.1. Members-only videos (Chip Stock Investor posts about half of its videos that way) are recorded as `SKIPPED`.

Secrets are read only from environment variables loaded from `.env`: `GEMINI_API_KEY` (gemini backend) and, later, the Reddit credentials (section 15). **`.env` must be added to `.gitignore` before any key is written**; it is not ignored today. The `claude_code` backend needs no key, only the `claude` command logged in with your subscription. No YouTube key is needed.

---

# 11. Code layout

```
quant/sentiment/
  __init__.py
  tables.py         # the 6 new tables
  sources.py        # VideoSource interface + YtDlpSource
  transcripts.py    # caption fetching, Traditional→Simplified, chunking
  aliases.py        # alias table + SEC ticker list check
  backends.py       # Extractor interface + ClaudeCodeBackend (claude -p) + GeminiBackend
  extract.py        # chunk loop, JSON validation, quote check, merging (section 4.5)
  prompts.py        # extraction and summary prompts (with prompt_version)
  summarize.py      # step ④ (section 8.3)
  aggregate.py      # date × stock summary
  scorecard.py      # hit rates, permutation test
  reddit.py         # S3b: ApeWisdom + praw fetch, mention counting (section 15)
  report.py         # Markdown reports
  cli.py            # subcommands, registered in quant/cli.py (one line)
config/sentiment_aliases.yaml
tests/sentiment/
  fixtures/         # saved real caption excerpts (a few in each language)
  test_*.py         # offline: VideoSource, the claude subprocess and the Gemini client are mocked
```

New dependencies are regular dependencies in `pyproject.toml`, so one `pip install -e .` installs everything (decision 10): `yt-dlp`, `youtube-transcript-api`, `opencc-python-reimplemented`, `python-dotenv`, and `google-genai` from S2. The core commands still never import them: the sentiment modules import them only when a `quant sentiment` command runs, so a broken scraper cannot affect `run-daily`.

The `claude_code` backend calls the installed `claude` command, so it adds no Python package.

---

# 12. Phases

| Phase | Content | Done when |
|---|---|---|
| **S0 Spike** | Verify the `channel_id` of the 4 candidates; count videos in the last 14 days, caption coverage (manual / auto / none) and average length | A table is handed to you and the final channel list is chosen from it (done, section 12.1) |
| **S1 Fetch** | `.env` in `.gitignore`; tables, `fetch` command, idempotency, rate limiting, tests | Re-runs write no duplicates; a run after a network failure fills the gaps |
| **S2 Extract** | Prompts, both backends, JSON validation, quote check, merging, alias table, AI reference of 20 videos (decision 12); check the Gemini free tier's current limits, models and availability | Agreement with the AI reference reported per backend; you have reviewed the disagreements and accept the extraction quality (section 4.6); backend, models and quote thresholds chosen |
| **S3 Report** | Summaries (section 8.3) + daily report + appendix with quotes and links | You have reviewed a week of reports and accept the quality |
| **S4 Scorecard** | Historical backfill (6 months, spread over runs) + hit rates + permutation test | Report produced; labeled exploratory; conclusions not overstated |
| **S6 pre-registration** | Write the S6 test (section 9) **before looking at forward hit rates**; forward logging continues | Pre-registration recorded with its date |
| **S3b Reddit** | Mention counts (ApeWisdom, then Reddit API) + extraction of DD / high-score posts (section 15); deferred until S4 has been read | Daily mention table and Reddit opinions appear in the daily summary; Reddit API access approved |
| ~~**S5 Whisper**~~ | Dropped (decision 13): no transcription of videos without captions | - |
| **S6 Factor** | Run the pre-registered test after ≥ 6 months of forward data; if it passes, add to `features` | Separate pre-registration (section 9) |

## 12.1 S0 findings (2026-10-12, window 2026-09-28 .. 2026-10-12)

| Channel | Handle / channel_id | Videos in 14 days | Captions | Notes |
|---|---|---|---|---|
| Joseph Carlson | `@JosephCarlsonShow` | 0 | — | Now posts every 1–2 months (last: 2026-08-12, before that 2026-06-08) |
| Financial Education | `@FinancialEducation` / `UCnMn36GT_H0X-w5_ckLtlgQ` | 4 | 4/4 auto (`en`) | ~35–40 min, ~38k characters per transcript; auto captions read cleanly |
| 美投讲美股 | `@MeiTouJun` / `UCBUH38E0ngqvmTqdchWunwQ` | 2 public (+ members-only) | 0 — captions disabled | ~1 video a week publicly; more macro/theme than single-stock calls |
| 视野环球财经 | `@RhinoFinance` / `UCFQsi7WaF5X41tcuOryDk8w` | 9 public + 2 members-only | 0 — captions disabled | Near-daily, ~20–25 min, titles packed with tickers (NVDA, META, ORCL, …) |

Consequences:

- **Both Chinese channels disable captions**, including auto captions, so they can only be covered by transcribing audio (S5 Whisper must move forward). The machine has an RTX 5070 (12 GB), enough for a local Whisper large model; `ffmpeg` is not installed, but `faster-whisper` decodes audio without it.
- **Joseph Carlson posts too rarely** to contribute to a daily summary.
- **Members-only videos** (both Chinese channels) cannot be read and are recorded as `SKIPPED`.
- The handles in section 10 are corrected to `@MeiTouJun` and `@RhinoFinance`.

Decision: replace Joseph Carlson and both Chinese channels with channels that publish captions, keeping the first version captions-only (Whisper stays in S5).

### Replacement candidates (same window; at most 10 videos checked per channel)

| Channel | Handle / channel_id | Long videos in 14 days | Captions | Content |
|---|---|---|---|---|
| Chip Stock Investor | `@chipstockinvestor` / `UC3aD-gfmHV_MhMmcwyIu1wA` | 10 (5 public, 5 members-only) | manual `en` on all public | Semiconductors/AI single stocks (ON, MU, neoclouds); clean text |
| Ticker Symbol: YOU | `@TickerSymbolYOU` / `UC7kCeZ53sli_9XwuQeFxLqw` | 1 | auto `en` | Good single-stock calls but about every 1–2 weeks |
| NaNa说美股 | `@NaNaShuoMeiGu` / `UCFhJ8ZFg9W4kLwFTBBNIjOw` | 10+ (daily) | manual `zh-Hans` + `zh-Hant` | Daily market wrap: indices, sectors, single stocks, options; very clean text |
| 投资TALK君 | `@yttalkjun` / `UCRBrH2qS7yKGMNSmjnj8gcw` | 9 | manual `zh` on 7/9 | Macro-heavy (CPI, Fed, rates) with sector/single-stock views; clean text |
| Zac美股笔记 | `@Zac-Stock-Journal` / `UCawshLsWXrFnYhAlnmn0h2A` | 3 | manual `zh-Hans` + `zh-Hant` | Explicit buy-point calls and real holdings, but half the videos are technical-analysis lessons |
| Nick 美股咖啡館 | `@nick_stockcafe` / `UCjrP2TtSTifuRJ76hW2IW1A` | 4 | manual `zh-TW` | Explicit buys and own positions, but the captions are noisy (misrecognized words) |
| 老李玩钱 | `@老李玩钱` | 4 | none | Captions disabled |
| 北美韭菜日记, Money Manta | | 0 | — | Inactive since 2023 / 2022 |
| 猴哥财经 | `@猴哥财经` | 0 | — | Last video 2026-09-23 |

### Second spike (2026-10-13, window 2026-09-29 .. 2026-10-13)

Caption availability read from each video page with yt-dlp (`subtitles` / `automatic_captions`); the method was checked on a NaNa video with known manual captions (it shows `zh-Hans`, `zh-Hant`).

| Channel | Handle / channel_id | Long videos in 14 days | Captions | Content |
|---|---|---|---|---|
| 贝拉聊财金 (often written 贝拉聊财经) | `@bellafinance` / `UCVomjkM_t0EcctTWSE1Jvxg` | 10+ (daily, ~13-20 min) | none (no manual, no auto) on 10/10 | Daily market and macro commentary with sector and single-stock picks (MU, ASML, TSM) |
| Jefftalkstocks | `@Jefftalkstocks` / `UCsmkTPubhBtCCk1w373ty8w` | 0 public (2 members-only among the 25 newest) | none on the latest public video | Last public video 2026-07-30 (MSFT earnings); inactive publicly |

Consequences: neither can join. 贝拉聊财金 would need audio transcription, like 美投讲美股 and 视野环球财经; with S5 dropped (decision 13), all three stay out.

The selected six publish about 20 long videos a week in total (about 3 a day), which fits the daily limits of either backend.

### Third spike and five more channels (2026-10-16)

You asked for creators who also post on Xiaohongshu; scraping Xiaohongshu itself was ruled out (no content API, anti-bot protection, terms of use). You then asked for Chinese and English YouTube channels in general.

- **Method:** candidates came from YouTube searches (Chinese: 美股 分析 / 个股 / 财报 / 科技股 / 复盘; English: stock analysis / best stocks / deep dive / earnings reaction / portfolio buys). For each, the 12 newest uploads were listed, and the captions of the two newest long videos were read with `youtube-transcript-api` (`list`).
- **Left out:**
  - captions disabled: 老李玩钱, Kelly Tsai, 美股小头狼 Stock Alpha Research, XTrader猫姐美股, 小左第一视角, Victor美股投资;
  - videos unreadable: JohnLu谈股;
  - promotional titles (profit targets, "will explode") and news networks (CNBC, Bloomberg), which have many speakers.
- **Usable but not added:** 幂笈投资, Sam谈美股, Amy说美股@致富学院 (zh); Ticker Symbol: YOU, Couch Investor, BWB - Business With Brian, Tom Nash, Daniel Pronk (en).

**Added (2026-10-16):**

| Channel | Handle / channel_id | Followers | Typical length | Captions | Why |
|---|---|---|---|---|---|
| 老汤美股财经 | `@laotangmeigu` / `UCeTEWFsNC3eeUsn9hvsDALQ` | 154k | 8-14 min | manual `zh-Hans` | Single-stock analysis |
| Ethan的美股笔记 | `@EthanFinanceNote` / `UCvf_c9B7iL0x7aqdhxI4zRg` | 22k | 13-26 min | manual `zh-Hans` | Talks about his own trades (the `action` field) |
| Parkev Tatevosian, CFA | `@parkevtatevosiancfa9544` / `UCwKB_00dPL3x5XmHF9IJCrg` | 878k | 7-9 min | auto `en` | One stock per video with a buy or sell verdict |
| Everything Money | `@EverythingMoney` / `UChBVf9YnourrEDTsbbwJPRA` | 417k | 21-39 min | auto `en` | Valuation-based calls, a value view |
| Dividend Talks | `@DividendTalks` / `UClLyc3C2K3C-RAh7eEJAwyg` | 128k | 23-34 min | auto `en` | Dividend stocks: the pool's utilities and REITs |

**Consequences:**
- There are now 11 channels.
- **Parkev Tatevosian posts about 4 videos a day** (31 long ones in the first fetch's 7 days), more than estimated in the spike. They are short (about 4,000-7,000 caption characters), so each is about one extraction call. With him, the 11 channels give roughly 60-70 long videos a week.
- Extraction needs about 60-90 more Sonnet calls a week. Watch the Pro usage for the first week. If it is too much, Parkev is the first channel to drop or to limit.
- `max_captions_per_run: 10` makes the first fetch of the new channels' last 14 days take a few runs; the backlog clears oldest first.
- The scorecard (S4) and the forward-only test (S6) judge each creator from their own first opinion on, so creators added later are not mixed with earlier ones.

## 12.4 S3 as built (2026-10-13)

- `quant sentiment summarize [--from D] [--to D] [--force]` and `quant sentiment report [--from D] [--to D]`; both join `quant sentiment run` (fetch, extract, summarize, report; each step runs even after an earlier one had problems). Code: `quant/sentiment/summarize.py`, `quant/sentiment/report.py`; summary prompt v1 and schema in `prompts.py`; table `sentiment_summaries` (section 6).
- Settings `sentiment.summary`: backend `claude_code`, model `haiku`, prompt version 1, `reports_dir: reports/sentiment`.
- One call per trading day for all of its stocks plus the market paragraph. A summary may cite only its own stock's opinion ids; one that cites others is dropped (shown as "no summary"). Each summary stores a hash of its opinion ids and the extractor, so a day whose opinions change is summarized again, and the report shows only summaries written from exactly the current opinions.
- Report per effective date: market mood; tables for stocks and ETFs (Bull / Bear / Cond / Neutral / Net per creator, each creator once per column, all their marks in one entry; ★ = in the stock pool); the summaries; an appendix with every verified quote, a link to the second and the video title (auto captions flagged); "Not counted": unverified quotes and unresolved names.
- First run: 9 trading days (2026-09-30 .. 2026-10-12), 9 Haiku calls, 105 stock summaries, none dropped.
- Seen in the first reports, for prompt version 2 after your week of reading: the summaries repeat "No bearish view was given" for single-creator stocks; the extractor sometimes turns one quote into two opinions with different horizons and stances (MRVL on 2026-10-12: BULLISH LONG and NEUTRAL from the same sentence).
- `quant sentiment stock TICKER [--from D] [--to D] [--all]` (read-only): every opinion on one stock over time, grouped by effective date, with that day's summary, quotes and links; the ticker may be typed as a name from the alias table (美光 -> MU). Added because one daily report often shows a single creator per stock.
- Also for prompt version 2: the summaries sometimes write opinion ids into the text ("Financial Education (37)").
- **Sonnet and Haiku in one report (2026-10-15).** `sentiment.extraction.report_models: [sonnet, haiku]`: reports, summaries, `sentiment stock` and the dashboard take, per video, the opinions of the first listed model that extracted it, so a video extracted by both counts once (Sonnet's). The report header says which model each day's videos come from; quotes not from `extraction.model` are marked "by haiku". Haiku has not been reviewed like Sonnet (S2 step 6-8 postponed), so Haiku days are weaker evidence. First run: 2026-10-08 .. 10-12 mixed, 10-13 .. 10-15 Haiku only; 6 days summarized again.
- Done when (section 12): you have read a week of reports and accept their quality.

## 12.3 S2 steps 1-5 as built (2026-10-13)

- `quant sentiment extract [--limit N] [--video ID] [--model M] [--reextract] [--yes]`; code in `quant/sentiment/` (`prompts.py`, `aliases.py`, `backends.py`, `extract.py`); tables `sentiment_extractions` (one row per video and extractor, even with no opinions; OK or FAILED) and `sentiment_opinions` (section 6, plus `name_as_said`, `quote_score` and `other_quotes_json`; the unique key also has `name_as_said`, so different UNKNOWN names stay apart).
- Prompt version 1 in `prompts.py` (`PROMPTS` keeps every version); the alias table is part of the system prompt. The JSON schema allows asset types STOCK and ETF; a view on the market or an index is recorded on its ETF (SPY, QQQ, DIA, IWM).
- **Ticker list: Nasdaq Trader symbol directory instead of the SEC list.** The SEC refuses requests without a contact email in the request header. The directory covers every Nasdaq, NYSE and other US-listed stock and ETF, needs no key, is cached in `data/sentiment/us_tickers.json` and refreshed monthly. OTC ADRs (e.g. HESAY, LVMUY) are not in it; they are known only through the alias table.
- `claude_code`: the system prompt is passed with `--system-prompt-file` (on Windows `claude` is a .CMD wrapper, and cmd.exe cannot pass multi-line text with `&` or quotes, nor more than about 8,000 characters); `--json-schema` with the compact schema. Refuses to run while `ANTHROPIC_API_KEY` is set.
- Quote check: compared on letters and digits after NFKC and case folding (so punctuation and full-width differences do not count), exact substring first, else the best sliding-window ratio; `start_seconds` is taken from the line where the quote matched, not from the model.
- Merging (section 4.5): the last stated stance wins among VERIFIED opinions if there are any.
- Effective date: the first NYSE session whose open is after `published_at` (`quant/data/calendar.py: first_session_opening_after`).
- Videos of the gold set / AI reference whose `status` is not done are held back.
- "Win Resorts" (the auto-caption spelling of Wynn) was left out of the alias table and the prompt on purpose: it was found only by checking the AI reference, so it would inflate agreement on that row.
- Live check: one NaNa video outside the AI reference (`khnPevsQ_TE`, 13 min) with Haiku: 2 calls, about 12k input and 4k output tokens, 5 opinions, all quotes matched exactly.
- First full run (2026-10-13, Sonnet): all 25 stored videos, 68 calls, about 462k input and 36k output tokens, 129 opinions, all VERIFIED, none UNRESOLVED. Rough agreement with the AI reference on its 20 videos, matched on video and stock: 81 pairs in both (89% of the reference's 91, 85% of Sonnet's 95), same direction on 72 of 81; the largest differences are in NaNa's news wraps (own view or news?).
- `quant sentiment review-export [--model M]` (S2 step 7, built 2026-10-13): matches the extractor's opinions with the labels on video and stock, prints the agreement and writes the disagreements (kinds `only_reference`, `only_extractor`, `direction`) with quote, link and the transcript around it to `data/sentiment_gold/reviews/` (git-ignored: it holds transcript text). Verdict per case: `reference`, `extractor`, `both` or `neither`. Sonnet / prompt v1: 33 cases (10, 14, 9).
- `quant sentiment run` (section 7) is built for steps 1-3 so far: `fetch`, then `extract`, which runs even when the fetch had problems; the summary and report steps join it with S3.
- **Review of Sonnet / prompt v1 (2026-10-13), kept as given (option A).** 33 disagreements: 31 judged `extractor`, 2 `both`, 0 `reference`, 0 `neither`, no comments, filled in about 7.5 minutes (about 14 s per case). Verdicts without transcript text: `research/sentiment_ai_reference/verdicts-claude_code-sonnet-v1.csv`. Taken literally, Sonnet made no error on any disputed case, which is not plausible for an extractor; some cases were debatable (e.g. TLT mapped from a bond-yield view the creator never tied to TLT; JBL's "割韭菜" read as bullish). The review is therefore **weak evidence**: S2 is accepted **provisionally** with Sonnet as the extractor, and the quote and ticker checks remain the only hard guards. A careful review (or human labels) should come before any S6 use of the opinions.
- **Postponed (2026-10-13):** S2 steps 6-8: the review file of the about 33 disagreements (10 only in the reference, 14 only from Sonnet, 9 with a different direction) with your verdicts, the Haiku comparison and the Gemini backend. Resume with the review; no extraction result counts as accepted before it.

## 12.2 S1 as built (2026-10-13)

- `quant sentiment fetch [--days N] [--channel HANDLE]`; code in `quant/sentiment/` (`tables.py`, `sources.py`, `transcripts.py`, `store.py`, `fetch.py`, `cli.py`), registered in `quant/cli.py` (an import, the parser and the dispatch entry). Tests: `tests/test_sentiment_fetch.py`, offline with fake sources (flat in `tests/` like the other test files).
- Tables so far: `sentiment_channels`, `sentiment_videos` (plus `live_status`, `availability`, `note` for the skip or failure reason, and `attempts`), `sentiment_transcripts` (plus `segments_simplified_json`). The other three tables come with S2–S4.
- `sentiment.channels` in `config/settings.yaml` also holds each `channel_id` and `name`. `.env` is in `.gitignore`.
- Listing: members-only videos and Shorts are SKIPPED from the channel listing alone (no extra request); every other new video's page is read once; the listing stops at the first video older than the lookback. Captions are fetched oldest first; one retry, then FAILED; a block from YouTube stops the run.
- First full run (all six channels, 14 days): YouTube blocked caption requests (`IpBlocked`) after 19, with waits of 2-5 s. Changed: waits of 8-15 s, and at most 10 captions per run (`sentiment.max_captions_per_run`), oldest first; the run prints its progress line by line. The 6-month backfill in S4 therefore takes many runs over several days.
- Captions often appear hours after a video is published, so a `NO_TRANSCRIPT` video is asked again on later runs while younger than `sentiment.no_captions_retry_days` (3); these retries come after every video not tried yet.
- Live check on 2026-10-13: Chip Stock Investor (7 days: 4 captioned, 4 members-only) and Nick 美股咖啡館 (10 days: 3 captioned, `zh-TW`, Simplified copy stored); a second run wrote nothing. Nick's captions are noisy as found in S0 (e.g. "香蕉臺灣", probably "相較台灣").

---

# 13. Risks

| Risk | Mitigation |
|---|---|
| `yt-dlp` or the caption endpoint breaks, or the IP gets rate-limited | Behind an interface; rate limiting; backfill spread over days; run locally; failed videos marked `FAILED` and retried next run |
| A sentiment failure breaks the daily pipeline | Not part of `run-daily`; separate commands only |
| Gray area under YouTube's terms | Only a few channels, personal research only, transcripts not redistributed |
| Many errors in Chinese auto-captions | Alias table includes common misrecognitions; `is_generated` recorded; hit rates can be split by it |
| The model fabricates opinions or tickers | Quote check + ticker check; failures are excluded from summaries and the scorecard; summaries built only from verified opinions and must cite them |
| Pro usage limits reached (`claude_code`) | Per-run cap; the run stops cleanly and resumes next run; `gemini` as the alternative backend |
| Gemini free-tier limits, models or terms change | Checked in S2; model name in config; backend switchable without schema changes |
| Gemini free tier may use inputs to improve Google's products | Only public transcripts, posts and opinions are sent; never positions, trades, signals or account values |
| `claude -p` returns an invalid reply | `--json-schema` enforces the format; values validated; one retry; then `FAILED` |
| Claude Code bills the API instead of Pro | The backend refuses to run while `ANTHROPIC_API_KEY` is set |
| Vague opinions that are hard to classify | `CONDITIONAL` / `NEUTRAL` categories; quotes kept for manual review |
| High-beta bias: bullish calls on tech look right in rising markets | Excess return vs the pool on the same days; permutation against random pool picks; beta-adjusted column |
| Repeated calls inflate the sample | One opinion per creator × stock × stance per window |
| Many cells produce false discoveries | One pre-registered primary test; Holm correction for secondary tests; descriptive cells marked as uncorrected |
| Small samples make hit rates unreliable | Minimum sample size, permutation test, power stated in the report; backfilled data for exploration only; S6 on forward data only |
| Survivorship of creators and of their videos | Backfill is exploration only (section 9) |
| Reddit API approval delayed or refused | Start with ApeWisdom mention counts (no key); apply early so approval overlaps S1–S3 |
| Reddit history cannot be backfilled | Treat Reddit as forward-only from the day logging starts; no Reddit backtest before enough history exists |
| WSB sarcasm, memes and pump posts | Extraction only on DD / high-score posts; `stance` from the post body, not the title; mention counts kept separate from opinions |

---

# 14. Decisions

## 2026-10-12

1. **Channels**: initially Joseph Carlson, Financial Education, 美投讲美股 and 视野环球财经. After S0 (section 12.1): Financial Education, Chip Stock Investor, NaNa说美股, 投资TALK君, Zac美股笔记 and Nick 美股咖啡館.
2. **Backfill**: 6 months in S4.
3. **Stock scope**: every US stock mentioned; stocks in the `settings.yaml` pool are marked separately in the summary.
4. **Report language**: English.

## 2026-10-13 (after review)

5. **No paid API.** Two extractor backends: `claude_code` (Claude Code `claude -p` on the Pro subscription) and `gemini` (free Gemini API). The S2 gold set picks one.
6. **Summaries**: written by the extractor backend from verified opinions only, display only (section 8.3).
7. **Benchmark**: excess return against the pool on the same days, tested against random pool picks on the same dates (section 8.2).
8. **Promotion**: S6 uses only forward data after its pre-registration, at least 6 months, one primary test (section 9).
9. **Reddit (S3b)** deferred until S4 has been read; the Reddit API application can still be submitted early.
10. **One install**: the sentiment packages are regular dependencies, not an optional group (`pip install -e .` installs everything).
11. **Gold set of 10 videos**, labeled by you (option B); labels committed under `research/sentiment_gold/`, transcripts deletable.
12. **No human gold set; an AI reference instead** (2026-10-13). The 20 labeled videos were labeled with an AI tool (GPT or MAI; model not recorded), not by hand. They are kept as an *AI reference* in `research/sentiment_ai_reference/` (provenance in its `SOURCE.md`). Consequences in section 4.6: S2 reports agreement, not accuracy, and the quality gate moves to your review.
13. **No audio transcription (2026-10-13).** S5 (local Whisper) is dropped: you judged it too costly. The system stays captions-only, so channels without captions (美投讲美股, 视野环球财经, 贝拉聊财金) are not covered; new channels must publish captions.

---

# 15. Phase S3b: Reddit (added 2026-10-12)

Deferred (decision 9): the pool is large caps (S&P 100 and similar), where Reddit attention mostly echoes the news; retail-attention effects in the research literature are strongest in small caps. S3b starts only if S4 gives a reason to. Submitting the Reddit API application early costs nothing.

## 15.1 Why, and what is different

YouTube gives a few named creators making explicit calls, so each can be scored. Reddit (especially r/wallstreetbets) gives a crowd: thousands of short, mostly anonymous posts. Its main signal is **attention and crowd mood** (how often a ticker is mentioned, how fast that rises, the bull/bear balance), not a per-person track record. It is therefore a complementary input, not more of the same.

Reddit is handled in two layers:

| Layer | What | Cost | Used for |
|---|---|---|---|
| **A. Mentions** | Daily mention count per ticker per subreddit | Cheap, no LLM | Attention factor |
| **B. Opinions** | Extraction (the configured backend, section 4.1) from DD and high-score posts only | LLM, capped per run | Same opinion table, report and scorecard as YouTube |

## 15.2 Sources

- **Subreddits**: r/wallstreetbets, r/stocks, r/investing, r/ValueInvesting (configurable).
- **ApeWisdom** (S3b-1): its public feed of ticker mentions and rank changes across subreddits. Needs no key and no Reddit approval, so it can start immediately. Its endpoints, terms and rate limits are verified directly before use; the history it serves is not relied on.
- **Reddit Data API** via `praw` (S3b-2): a "script" app with OAuth, read-only, at most about 100 requests/min (free non-commercial tier). New apps reportedly need manual approval that can take weeks, so **you can submit the application from your Reddit account early**. Credentials are read only from `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET` and `REDDIT_USER_AGENT` in `.env`.
- Once the Reddit API works, layer A is computed from our own fetched posts as well, so it no longer depends on ApeWisdom alone; both are stored with their `provider` so they can be compared.

## 15.3 What is fetched

- Each run reads the `new` and `top` (day) listings of every configured subreddit and stores **posts only**, not comments. Comments multiply the volume many times over and are mostly noise; they can be reconsidered after S3b.
- Each post is stored once with its score, comment count and upvote ratio **at fetch time**, and `fetched_at`. Runs happen at a fixed time of day so these snapshots are comparable.
- Posts deleted or removed later are deleted locally on the next run that notices it, as Reddit's Data API terms require for stored content (to be confirmed against the current terms in S3b-2). Text stays local and is never redistributed.

## 15.4 Ticker detection (layer A)

WSB text is full of words that collide with tickers (`A`, `IT`, `ON`, `ALL`, `DD`, `YOLO`, `CEO`, `NOW`). A mention counts only if it is:

- a cashtag (`$NVDA`), or
- an uppercase word in the SEC ticker list **and** not on a stop list (`config/sentiment_reddit_stoplist.yaml`), or
- a company name or nickname from the alias table (section 4.4), which gets Reddit nicknames added (e.g. `TSLA: [..., Elon's car company]`, `GME: [..., GameStop]`).

The same post counts at most once per ticker. Title and body are both scanned.

## 15.5 Opinion extraction (layer B)

A post is sent to the extractor only if one of these holds at fetch time:

- its flair is DD / Due Diligence / Fundamentals / Technical Analysis, or
- its score is ≥ `reddit.min_score` and its body is ≥ `reddit.min_chars` characters.

Then the same prompt rules as YouTube apply (section 4.2), with Reddit-specific additions:

- Stance comes from the body, not the title (titles are often jokes or bait).
- Sarcasm, memes and "to the moon" without reasoning are recorded as `NEUTRAL` or skipped.
- `action` uses position screenshots or statements in the text ("I bought 50 calls").
- The quote check (section 4.3) runs against title + body. `start_seconds` is null.

A per-run cap (`reddit.max_posts_per_run`) and a token estimate before each run apply, as for YouTube.

## 15.6 Timing

`published_at` is the post's `created_utc`; `effective_date` follows section 5 (next NYSE open).

Selecting posts by score adds a subtle look-ahead: a post's score at fetch time is partly earned after it was published. For research this is accepted and documented. For any live feature, selection must use only the score known at the decision time, which the fixed-time daily fetch provides.

## 15.7 Database

Two new tables (prefixed `sentiment_` like every table of this module); opinions go into the shared `sentiment_opinions` (section 6) with `source = reddit`.

### sentiment_reddit_posts
`post_id` (PK), `subreddit`, `author`, `created_utc`, `title`, `body`, `flair`, `score`, `num_comments`, `upvote_ratio`, `url`, `fetched_at`, `extraction_status` (`PENDING` / `SELECTED` / `DONE` / `NOT_SELECTED` / `DELETED`).

### sentiment_reddit_mentions
`date`, `ticker`, `subreddit`, `provider` (`apewisdom` / `reddit`), `mentions`, `upvotes` (sum of scores of mentioning posts; ApeWisdom's own figure for that provider), `rank`, `fetched_at`. Primary key: `(date, ticker, subreddit, provider)`.

## 15.8 Output

The daily summary (section 8.1) gets a **Reddit attention** section: the top tickers by mentions, each with its change against its own 20-day average (`mention_z`), and Reddit opinions listed alongside YouTube ones with a source marker.

The scorecard (section 8.2) scores Reddit **per subreddit**, not per user, because most users post too rarely. Individual authors are scored only once they have at least `evaluation.min_samples` opinions.

## 15.9 Factors (with S6)

Candidates for the S6 pre-registration: `reddit_mention_z` (attention spike), `reddit_mention_trend_5d`, `reddit_net_stance_5d`. The research literature on retail attention suggests both short-term momentum and later reversal, so the pre-registration tests both signs rather than assuming one.

**Reddit cannot be backfilled** in practice: listings only reach about the last 1,000 posts, and Pushshift is no longer open to the public. Reddit evidence therefore comes only from forward logging, starting the day S3b-1 runs. It is not eligible for S6 until the same 6-month minimum (section 9) has passed.

## 15.10 CLI and configuration

```bash
quant sentiment reddit-fetch   [--subreddit wallstreetbets]   # posts + mentions
quant sentiment extract        [--source youtube|reddit]       # shared extraction
quant sentiment run                                            # now includes reddit-fetch
```

```yaml
sentiment:
  reddit:
    enabled: false                   # true once S3b-1 is built
    subreddits: [wallstreetbets, stocks, investing, ValueInvesting]
    providers: [apewisdom]           # add reddit after API approval
    listings: [new, top_day]
    min_score: 100
    min_chars: 800
    extract_flairs: [DD, Due Diligence, Fundamentals, Technical Analysis]
    max_posts_per_run: 30
    request_delay_seconds: [1, 2]
```

## 15.11 Steps

| Step | Content | Done when |
|---|---|---|
| **Early (you, optional)** | Apply for Reddit API access (script app, non-commercial research) | Application submitted |
| **S3b-1** | ApeWisdom mentions → `sentiment_reddit_mentions`; Reddit attention section in the daily report | A week of daily mention tables reviewed |
| **S3b-2** | `praw` fetch → `sentiment_reddit_posts`; own mention counts; DD/high-score extraction; deletion handling | Extraction precision, recall and stance accuracy ≥ 85% on a hand-labeled set of ~20 posts; own counts agree broadly with ApeWisdom |

Not in S3b: comments, Stocktwits, X/Twitter, Chinese forums (雪球, 富途). These can be proposed separately later.
