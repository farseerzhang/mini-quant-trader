# Creator & Crowd Opinion Factor (YouTube, Reddit) — Proposal

Status: approved 2026-10-12 (decisions in section 14). S0 done (section 12.1); S1 next.
Date: 2026-10-12

---

# 1. Goal

Collect commentary on individual US stocks from a few finance YouTube creators (English and Chinese), extract structured opinions from their transcripts, and summarize them by **date × stock** to show who is bullish and who is bearish. Once enough data has been collected, measure each creator's historical hit rate to decide whether their opinions can feed the existing system as a sentiment factor.

As in `proposal.md`, this is a research tool. **A creator's opinion is not a trading signal**, and it does not enter `generate-signals` until it passes the hit-rate test.

## Out of scope (first version)

- No transcription of videos without captions (Whisper is deferred to S5).
- No scheduler. Like the main system, the CLI is run on demand.
- No changes to existing modules or tables in `quant/`, and nothing touches the current uncommitted changes. All new code goes in `quant/sentiment/` and new tables.
- No Streamlit page yet. Output is the command line and Markdown reports.

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
 ③ Extract opinions Claude API, structured output    → sentiment_opinions
        │           (each opinion carries a quote and timestamp, and the quote
        │            is checked against the transcript)
        ▼
 ④ Daily report     date × stock × bullish/bearish   → reports/sentiment/<date>.md
        │
        ▼
 ⑤ Hit rate         excess return vs SPY 5/20/60 sessions after the opinion
                    → reports/sentiment/scorecard.md
```

Every step is idempotent and keyed by `video_id`. A re-run only processes new or failed videos, matching the main system's convention (proposal section 30).

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
- Traditional Chinese is converted to Simplified (`opencc`) before extraction so one alias table covers both.
- The raw transcript, with the start second of every segment, is stored in full in the database. This follows "raw data is the source of truth" (proposal 4.3): a new prompt or model can re-extract from the stored transcripts without fetching them again.
- Rate limiting: a random wait of a few seconds between requests. After one failed retry a video is marked `FAILED` and retried on the next run. Runs only on this machine or a home network.

---

# 4. Opinion extraction

## 4.1 Model

- The Anthropic Python SDK is used, with the model name set in config (`sentiment.extraction.model`).
- In S2, about 20 videos are labeled by hand as a small gold set, and Haiku 5.5 (`claude-haiku-5-5`) and Sonnet 5.5 (`claude-sonnet-5-5`) are compared on it for accuracy and cost before one is chosen. No model is assumed before then.
- Tool use / structured output enforces a JSON result. The fixed system prompt and the stock alias table come first in the request, with prompt caching enabled.
- Long videos are split into chunks of about 10 minutes with a small overlap. Each chunk is extracted on its own, then the results are merged and deduplicated by stock.

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

- **Quote check**: after extraction, the `quote` must be found in its transcript chunk (fuzzy match after normalizing whitespace and punctuation, configurable threshold). Opinions that fail are marked `UNVERIFIED` and excluded from summaries, but kept for inspection.
- **Ticker check**: the ticker returned by the model must be in the alias table or the SEC company ticker list (`company_tickers.json`, cached locally).
- Every opinion records its `prompt_version` and `model`. Re-extracting after a prompt change creates a new version and keeps the old one, so the two can be compared.

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

---

# 5. Timing convention (no look-ahead)

- The video's publish time `published_at` (UTC) is stored.
- An opinion's **effective time** is the open of the first NYSE session after the publish time, following the `next_open` convention of proposal 4.5. A video published during market hours also only counts from the next session's open.
- Hit rates are always measured from the effective time, so they never use prices the creator could not yet see.
- Captions can appear a few hours after a video is published. We record the publish time, not the fetch time, so backtests are unaffected. Live signals, if ever built, must use the fetch time instead.

---

# 6. Database

Four new tables in the same SQLite file, defined in `quant/sentiment/tables.py` on the shared `metadata`, so `quant/persistence/tables.py` does not change. Every run still writes a `pipeline_runs` row (command = `sentiment-*`).

The dev database is disposable and can be rebuilt, so there are no migrations before S4.

### sentiment_channels
`channel_id` (PK), `handle`, `name`, `language` (en/zh), `style_note`, `added_at`. Synchronized from config, which is the source of truth (as with the `symbols` table).

### sentiment_videos
`video_id` (PK), `channel_id`, `title`, `published_at`, `duration_seconds`, `transcript_status` (`PENDING` / `OK` / `NO_TRANSCRIPT` / `FAILED` / `SKIPPED`), `fetched_at`.

### sentiment_transcripts
`video_id` (PK), `language`, `is_generated`, `segments_json` (`[{start, duration, text}]`), `char_count`, `fetched_at`.

### sentiment_opinions
`opinion_id` (PK), `source` (`youtube` / `reddit`), `item_id` (YouTube `video_id` or Reddit post id), `channel_id` (YouTube channel or `r/<subreddit>`), `author` (Reddit username; null for YouTube), `published_at`, `effective_date`, `ticker`, `asset_type`, `stance`, `horizon`, `condition`, `conviction`, `action`, `quote`, `start_seconds` (null for Reddit), `reasoning`, `verification` (`VERIFIED` / `UNVERIFIED` / `UNRESOLVED`), `model`, `prompt_version`, `extracted_at`.

Unique constraint: `(source, item_id, ticker, horizon, prompt_version)`.

`source`, `item_id` and `author` exist from S1 so that Reddit (section 15) can share this table and the downstream report and scorecard code without a schema change.

---

# 7. CLI

A group of subcommands under the existing `quant` command:

```bash
quant sentiment fetch      [--days 14] [--channel <handle>]   # ①②: video list + captions
quant sentiment extract    [--limit N] [--reextract]          # ③: only videos without opinions for the current prompt_version
quant sentiment report     [--from 2026-10-01] [--to ...]     # ④: date × stock summary
quant sentiment scorecard                                     # ⑤: hit rates
quant sentiment run                                           # ①–④ in order
```

Before running, `extract` prints the number of videos to process and the estimated token count. Above `sentiment.extraction.max_videos_per_run` it stops and needs `--yes` to continue.

---

# 8. Output

## 8.1 Daily summary (`reports/sentiment/<date>.md`)

```
2026-10-09 (effective 2026-10-10)

| Stock | Bull | Bear | Cond | Net | Creators |
|-------|------|------|------|-----|----------|
| NVDA  | 3    | 0    | 1    | +3  | Joseph Carlson ▲, 美投讲美股 ▲, Chip Stock Investor ▲, 视野环球财经 ◆ (buy below 160) |
| TSLA  | 0    | 2    | 0    | -2  | ...      |

Quotes and video links for every opinion are in the appendix.
Unresolved references: 英伟大 (×2), "the data-center one" (×1)
```

Net = bullish count − bearish count; `CONDITIONAL` and `NEUTRAL` are not counted. A creator counts at most once per stock per day.

## 8.2 Creator scorecard (`reports/sentiment/scorecard.md`)

For each creator × horizon, returns 5 / 20 / 60 sessions after the effective time:

- Bullish opinions: hit if the stock's excess return over SPY is positive.
- Bearish opinions: hit if the excess return is negative.
- Reports the sample size, hit rate, mean excess return, and a permutation-test p-value. The baseline shuffles the bullish/bearish labels within the same creator and the same set of stocks.
- Cells with fewer than 30 samples show the numbers only, with no conclusion drawn.

Prices come from the existing `daily_prices`. Mentioned tickers outside the stock pool are fetched through the same `YFinanceProvider` into a separate `sentiment_prices` table, so the main pool is not polluted.

---

# 9. Integration with the quant system (S6, later)

It becomes a feature only when all of these hold:

1. At least 3 months of opinions have been collected.
2. In a **pre-registered** test, at least one creator (or group of creators) has a 20-session excess hit rate significantly above chance, following the same pre-registration practice as proposal 13.1 / 18.1.
3. The opinions cover enough stocks in the pool.

Candidate features then: `sentiment_net_5d` (net bullish count over the last 5 sessions), `sentiment_consensus` (agreement), `sentiment_contrarian` (inverted for creators whose hit rate is significantly negative). They are added as new columns of `features`, which requires bumping `features.version` and re-registering models under the post-Phase-5 rules.

**Limitation**: older YouTube videos can be backfilled, so backtests are possible, but the backfill has survivorship bias: we picked these creators today, and they may still be around precisely because they were right in the past. Backfilled data is therefore for exploration only. The real evidence comes from forward logging starting today, in the same spirit as proposal section 15.5.

---

# 10. Configuration (added to `config/settings.yaml`)

```yaml
sentiment:
  lookback_days: 14
  min_duration_seconds: 180          # skip Shorts
  include_live: false
  request_delay_seconds: [2, 5]      # random wait range
  channels:                          # chosen after S0 (section 12.1)
    - {handle: "@FinancialEducation",   language: en, style: aggressive growth}
    - {handle: "@chipstockinvestor",    language: en, style: semiconductors / AI}
    - {handle: "@NaNaShuoMeiGu",        language: zh, style: daily market wrap}
    - {handle: "@yttalkjun",            language: zh, style: macro + sectors}
    - {handle: "@Zac-Stock-Journal",    language: zh, style: technical buy points, own holdings}
    - {handle: "@nick_stockcafe",       language: zh, style: own buys, Traditional Chinese}
  extraction:
    model: claude-sonnet-5-5         # decided after the S2 comparison
    prompt_version: 1
    chunk_minutes: 10
    quote_match_threshold: 0.85
    max_videos_per_run: 40
  evaluation:
    horizons: [5, 20, 60]
    benchmark: SPY
    min_samples: 30
```

The handles were verified in S0; their `channel_id`s are listed in section 12.1. Members-only videos (Chip Stock Investor posts about half of its videos that way) are recorded as `SKIPPED`.

The API key is read only from the `ANTHROPIC_API_KEY` environment variable (you put it in `.env`; `.env` must be added to `.gitignore` first, as it is not ignored today). The first version needs no YouTube key.

---

# 11. Code layout

```
quant/sentiment/
  __init__.py
  tables.py         # the 4 new tables
  sources.py        # VideoSource interface + YtDlpSource
  transcripts.py    # caption fetching, Traditional→Simplified, chunking
  aliases.py        # alias table + SEC ticker list check
  extract.py        # Claude calls, structured output, quote check
  prompts.py        # system prompt (with prompt_version)
  aggregate.py      # date × stock summary
  scorecard.py      # hit rates
  reddit.py         # S3b: ApeWisdom + praw fetch, mention counting (section 15)
  report.py         # Markdown reports
  cli.py            # subcommands, registered in quant/cli.py (one line)
config/sentiment_aliases.yaml
tests/sentiment/
  fixtures/         # saved real caption excerpts (a few in each language)
  test_*.py         # offline: VideoSource and the Anthropic client are mocked
```

New dependencies go in an optional group, so the main install is unaffected:

```toml
[project.optional-dependencies]
sentiment = ["yt-dlp", "youtube-transcript-api", "anthropic", "opencc-python-reimplemented", "python-dotenv"]
```

`quant/cli.py` currently has uncommitted changes. Registering the subcommands needs one line there, which I will add after you commit.

---

# 12. Phases

| Phase | Content | Done when |
|---|---|---|
| **S0 Spike** | Verify the `channel_id` of the 4 candidates; count videos in the last 14 days, caption coverage (manual / auto / none) and average length | A table is handed to you and the final channel list is chosen from it |
| **S1 Fetch** | Tables, `fetch` command, idempotency, rate limiting, tests | Re-runs write no duplicates; a run after a network failure fills the gaps |
| **S2 Extract** | Prompt, structured output, quote check, alias table, hand-labeled gold set of ~20 videos | Ticker and stance accuracy both ≥ 85% on the gold set; model chosen |
| **S3 Report** | Daily summary + appendix with quotes and links | You have reviewed a week of reports and accept the quality |
| **S4 Scorecard** | Historical backfill (e.g. 6 months) + hit rates + permutation test | Report produced; conclusions follow section 9 and are not overstated |
| **S3b Reddit** | Mention counts (ApeWisdom, then Reddit API) + extraction of DD / high-score posts (section 15) | Daily mention table and Reddit opinions appear in the daily summary; Reddit API access approved |
| **S5 Whisper** | Local transcription of videos without captions (optional) | Decided by the caption coverage found in S0 |
| **S6 Factor** | Pre-register per section 9, then add to `features` | Separate pre-registration |

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

---

# 13. Risks

| Risk | Mitigation |
|---|---|
| `yt-dlp` or the caption endpoint breaks, or the IP gets rate-limited | Behind an interface; rate limiting; run locally; failed videos marked `FAILED` and retried next run |
| Gray area under YouTube's terms | Only a few channels, personal research only, transcripts not redistributed |
| Many errors in Chinese auto-captions | Alias table includes common misrecognitions; `is_generated` recorded; hit rates can be split by it |
| The model fabricates opinions or tickers | Quote check + ticker check; failures are excluded from summaries |
| Vague opinions that are hard to classify | `CONDITIONAL` / `NEUTRAL` categories; quotes kept for manual review |
| Small samples make hit rates unreliable | Minimum sample size and permutation test; backfilled data for exploration only |
| API cost runs away | Token estimate before each run; per-run cap; transcripts fetched once and re-extractable |
| Reddit API approval delayed or refused | Start with ApeWisdom mention counts (no key); apply early so approval overlaps S1–S3 |
| Reddit history cannot be backfilled | Treat Reddit as forward-only from the day logging starts; no Reddit backtest before enough history exists |
| WSB sarcasm, memes and pump posts | Extraction only on DD / high-score posts; `stance` from the post body, not the title; mention counts kept separate from opinions |

---

# 14. Decisions (2026-10-12)

1. **Channels**: initially Joseph Carlson, Financial Education, 美投讲美股 and 视野环球财经. After S0 (section 12.1): Financial Education, Chip Stock Investor, NaNa说美股, 投资TALK君, Zac美股笔记 and Nick 美股咖啡館.
2. **Backfill**: 6 months in S4.
3. **Stock scope**: every US stock mentioned; stocks in the `settings.yaml` pool are marked separately in the summary.
4. **Report language**: English.

---

# 15. Phase S3b: Reddit (added 2026-10-12)

## 15.1 Why, and what is different

YouTube gives a few named creators making explicit calls, so each can be scored. Reddit (especially r/wallstreetbets) gives a crowd: thousands of short, mostly anonymous posts. Its main signal is **attention and crowd mood** (how often a ticker is mentioned, how fast that rises, the bull/bear balance), not a per-person track record. It is therefore a complementary input, not more of the same.

Reddit is handled in two layers:

| Layer | What | Cost | Used for |
|---|---|---|---|
| **A. Mentions** | Daily mention count per ticker per subreddit | Cheap, no LLM | Attention factor |
| **B. Opinions** | Claude extraction from DD and high-score posts only | LLM, capped per run | Same opinion table, report and scorecard as YouTube |

## 15.2 Sources

- **Subreddits**: r/wallstreetbets, r/stocks, r/investing, r/ValueInvesting (configurable).
- **ApeWisdom** (S3b-1): its public feed of ticker mentions and rank changes across subreddits. Needs no key and no Reddit approval, so it can start immediately. Its endpoints, terms and rate limits are verified directly before use; the history it serves is not relied on.
- **Reddit Data API** via `praw` (S3b-2): a "script" app with OAuth, read-only, at most about 100 requests/min (free non-commercial tier). New apps reportedly need manual approval that can take weeks, so **you submit the application from your Reddit account now**, and it overlaps S1–S3. Credentials are read only from `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET` and `REDDIT_USER_AGENT` in `.env`.
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

A post is sent to Claude only if one of these holds at fetch time:

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

**Reddit cannot be backfilled** in practice: listings only reach about the last 1,000 posts, and Pushshift is no longer open to the public. Reddit evidence therefore comes only from forward logging, starting the day S3b-1 runs. It is not eligible for S6 until the same 3-month minimum (section 9) has passed.

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
| **Now (you)** | Apply for Reddit API access (script app, non-commercial research) | Application submitted |
| **S3b-1** | ApeWisdom mentions → `sentiment_reddit_mentions`; Reddit attention section in the daily report | A week of daily mention tables reviewed |
| **S3b-2** | `praw` fetch → `sentiment_reddit_posts`; own mention counts; DD/high-score extraction; deletion handling | Extraction accuracy ≥ 85% on a hand-labeled set of ~20 posts; own counts agree broadly with ApeWisdom |

Not in S3b: comments, Stocktwits, X/Twitter, Chinese forums (雪球, 富途). These can be proposed separately later.
