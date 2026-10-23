# Creator & Crowd Opinion Factor (YouTube, Reddit) — Proposal

Status (2026-10-22):
- **Built:** S0 (channels, 12.1), S1 fetch (12.2), S2 extraction (12.3; accepted **provisionally** with Sonnet after a weak review), S3 summaries and daily reports (12.4), plus `sentiment stock` and the dashboard's Sentiment page. 11 channels.
- **Accepted:** S3, by the owner on 2026-10-22 after a week of daily reports (12.4).
- **Next:** S4, the 6-month backfill and the exploratory scorecard (8.2): built (12.6); the backfill is not run yet.
- **Registered:** the S6 test (9.1, 2026-10-22): forward videos from 2026-10-23 count; it runs once in about July 2027.
- **Not built:** S4 scorecard, the S6 test code and factor, the Gemini backend, S2 steps 6-8 (Haiku comparison), S3b Reddit (deferred, decision 9). S5 Whisper is dropped (decision 13).

Approved 2026-10-12, revised 2026-10-13 after review (decisions in section 14). This file was compacted on 2026-10-19: plans now describe what was built, the spike tables keep their findings, and the deferred Reddit plan is shortened. The full earlier text is in git.

---

# 1. Goal

Collect commentary on single US stocks from a few finance YouTube creators (English and Chinese), extract structured opinions from their captions, and summarize them by **date × stock**: who is bullish, who is bearish. Once enough data exists, measure each creator's hit rate, to decide whether the opinions can become a sentiment factor.

A research tool. **A creator's opinion is not a trading signal**; it enters `generate-signals` only after the pre-registered forward test of section 9. Until then it is reading material: before acting on a BUY or EXIT, you can see what the creators said.

**Out of scope:**
- Videos without captions (no audio transcription, decision 13).
- **`run-daily`:** sentiment depends on YouTube and an LLM, and a failure there must never block the evening freeze. The commands run on their own.
- Paid APIs: extraction runs on the Claude Code Pro subscription (or, not built yet, the free Gemini API).
- Changes to the core modules: everything is in `quant/sentiment/` and new tables; `quant/cli.py` registers the commands, and the dashboard only reads.

---

# 2. Pipeline

```
sentiment.channels (config/settings.yaml)
 ① list videos      yt-dlp, no key                   -> sentiment_videos
 ② fetch captions   youtube-transcript-api           -> sentiment_transcripts (none: NO_TRANSCRIPT, skipped)
 ③ extract          the extractor backend, JSON      -> sentiment_extractions, sentiment_opinions
                    (each opinion has a quote, checked against the transcript)
 ④ summarize        the day's VERIFIED opinions      -> sentiment_summaries (one paragraph per stock + market mood)
 ⑤ daily report                                      -> reports/sentiment/<date>.md, dashboard Sentiment page
 ⑥ scorecard        excess return 5/20/60 sessions   -> reports/sentiment/scorecard.md (S4, not built)
```

Every step is idempotent, keyed by `video_id` (or date × ticker for summaries): a rerun only processes new or failed items.

---

# 3. Data sources

## 3.1 Video list: `yt-dlp`

- `yt-dlp --flat-playlist` lists a channel's videos (id, title, duration) without a Google key; each new video's page is read once for the exact publish time.
- An unofficial tool that can break, so it sits behind an interface (`VideoSource`) and could be replaced by the YouTube Data API.
- Skipped: Shorts (< 3 minutes), live replays (configurable), members-only videos (recorded `SKIPPED`), videos outside the lookback window.

## 3.2 Captions: `youtube-transcript-api`

- Manual captions before auto-generated; English `en`, Chinese `zh-Hans`, then `zh-Hant`, then `zh`. `is_generated` is recorded (auto captions have many errors; reports flag them).
- Traditional Chinese is converted to Simplified (`opencc`) and stored next to the raw text; extraction and the quote check use the Simplified text.
- The full transcript with each segment's start second is stored, so a new prompt or model can re-extract without fetching again.
- **Rate limiting:** a random wait of 8-15 s between requests and at most 10 captions per run, oldest first (waits of 2-5 s got the IP blocked after 19 requests, 12.2). One retry, then `FAILED`, retried next run. Run on this machine only.
- **Caption pacing (12.5):** captions only 6 h after the last caption fetch and 8 h after a blocked one; the listing always runs; `--force-captions` ignores both.

---

# 4. Opinion extraction

## 4.1 Extractor backends

One `Extractor` interface: a transcript chunk and the alias table in, a list of opinions out. Every opinion stores the `backend` and `model` that produced it.

**`claude_code` (built, in use):** runs `claude -p` on the Pro subscription, from a neutral working directory (no project `CLAUDE.md` or memory):

```
claude -p --model <model> --tools "" --strict-mcp-config --no-session-persistence
       --output-format json --system-prompt-file <file> --json-schema <opinion schema>
```

- `--system-prompt-file` replaces Claude Code's own system prompt (about 1.3k instead of 36k tokens per call) and, on Windows, avoids cmd.exe's limits on multi-line arguments. `--json-schema` enforces the reply format; the values are still validated; one retry, then `FAILED`.
- Not `--bare` (it accepts only an API key). **It refuses to run while `ANTHROPIC_API_KEY` is set**, because Claude Code would then bill the API instead of the subscription.

**`gemini` (planned, not built):** the free Gemini API with `GEMINI_API_KEY` from `.env`. Its free tier may use inputs to improve Google's products; only public transcripts and opinions would be sent, **never `positions.csv`, trades, signals or account values**. When it is compared (S2 step 8): the backend that agrees well enough wins; if both do, `gemini`, so the Pro usage stays free for coding.

**Both:** chunks of about 10 minutes with a 30 s overlap, merged afterwards (4.5); a usage or rate limit stops the run cleanly, the rest stays pending; `max_videos_per_run` (40) caps a run.

## 4.2 Fields of each opinion

| Field | Meaning |
|---|---|
| `ticker` | US ticker, found in the alias table or the ticker list; otherwise `UNRESOLVED` |
| `stance` | `BULLISH` / `BEARISH` / `NEUTRAL` / `CONDITIONAL` |
| `horizon` | `SHORT` (weeks) / `MEDIUM` (months) / `LONG` (a year or more) / `UNSPECIFIED` |
| `condition` | e.g. "buy below 120"; empty otherwise |
| `conviction` | 1-3, how strongly the creator says it (not the model's confidence) |
| `action` | what the creator says they did: `BOUGHT` / `SOLD` / `HOLDING` / `NONE` |
| `quote` | the original sentence, not paraphrased |
| `start_seconds` | where the quote is, for a `youtu.be/<id>?t=<seconds>` link |
| `reasoning` | one sentence: the creator's reason |

Prompt rules: only the creator's **own** view (relayed analyst ratings, news or viewer questions do not count unless the creator takes a side); "bearish short term, bullish long term" is two opinions; passing mentions are not recorded; ETFs and index views are recorded on their ETF (SPY, QQQ, DIA, IWM) and summarized apart from stocks.

## 4.3 Guarding against fabrication

- **Quote check:** the quote must be found in its chunk (compared on letters and digits after NFKC and case folding; exact substring first, else the best sliding-window ratio, threshold 0.85). Failures are `UNVERIFIED`: kept for inspection, excluded from summaries and the scorecard. `start_seconds` comes from where the quote matched, not from the model.
- **Ticker check:** in the alias table or the US ticker list (Nasdaq Trader's symbol directory, 12.3).
- Every opinion records `backend`, `model` and `prompt_version`; a re-extraction adds rows and keeps the old ones.

## 4.4 Stock alias table

`config/sentiment_aliases.yaml`, kept by hand; it is part of the system prompt.

```yaml
NVDA: [英伟达, 辉达, 老黄, Nvidia]
BRK-B: [伯克希尔, 巴菲特的公司, Berkshire]
```

Common auto-caption misrecognitions are added too ("英伟大"). Names the model cannot resolve are listed at the end of the report, so you can add real stocks.

## 4.5 Merging within a video

Opinions on the same ticker and horizon within one video become one: the **last** stated stance wins (among VERIFIED opinions if there are any), the earlier quotes are kept; the highest `conviction`; an `action` if any chunk states one; overlap duplicates dropped.

## 4.6 Gold set and quality gate (S2)

- **Planned:** 10 videos labeled by you; precision, recall (missed bearish calls skew hit rates) and stance accuracy ≥ 85% each. Labels from a person, not a model, because the extractor and the assistant share blind spots.
- **Done instead (decision 12):** 20 videos labeled with another AI tool (GPT or MAI), kept as an **AI reference** in `research/sentiment_ai_reference/` (provenance in `SOURCE.md`). S2 therefore reports **agreement**, never accuracy, and 85% is no gate. **The gate is your review of the disagreements** (`quant sentiment review-export`): for each case you read the passage and judge `reference`, `extractor`, `both` or `neither`.
- `gold-export` picks the videos (channels taking turns, then by `video_id`); labels are committed, readable transcripts are in `data/sentiment_gold/transcripts/` (git-ignored). **Extraction output of a gold video is not shown before it is labeled**: `extract` holds such videos back.
- A human-labeled set (about 5 videos) can still be added if the review leaves doubt.

---

# 5. Timing convention (no look-ahead)

- `published_at` (UTC) is stored. An opinion's **effective date** is the first NYSE session whose open is after it (`first_session_opening_after`); a video published during market hours counts from the next open.
- Hit rates are measured from the effective date, never from prices the creator could not see.
- Captions may appear hours after publication; the publish time is what counts for research. A live signal, if ever built, must use the fetch time instead.

---

# 6. Database (as built)

Tables in the same SQLite file (`quant/sentiment/tables.py`, every name `sentiment_*`); no core table changes. Every run writes a `pipeline_runs` row. Since the forward test (2026-10-12) the database is not disposable: changes are new tables only.

| Table | Content |
|---|---|
| `sentiment_channels` | channel id, handle, name, language, style; synchronized from config |
| `sentiment_videos` | title, `published_at`, duration, `transcript_status` (`PENDING` / `OK` / `NO_TRANSCRIPT` / `FAILED` / `SKIPPED`), live status, availability, attempts, note |
| `sentiment_transcripts` | language, `is_generated`, segments with start seconds, the Simplified copy |
| `sentiment_extractions` | one row per video and extractor (OK or FAILED), even with no opinions |
| `sentiment_opinions` | the fields of 4.2 plus `source` (youtube / reddit), `item_id`, `channel_id`, `author`, `published_at`, `effective_date`, `asset_type`, `name_as_said`, `quote_score`, `other_quotes_json`, `verification` (`VERIFIED` / `UNVERIFIED` / `UNRESOLVED`), backend, model, prompt version |
| `sentiment_summaries` | per effective date and ticker (`_MARKET` for the mood paragraph): summary, disagreement, opinion ids, backend, model, prompt version |

- `sentiment_opinions` is unique on (source, item, ticker, horizon, name as said, backend, model, prompt version): one merged opinion per video, stock and horizon per extractor, so comparisons and re-extractions never overwrite each other. `source`, `item_id` and `author` let Reddit share the table later.
- Reports use the opinions of `sentiment.extraction.report_models` (12.4).
- Planned, not built: `sentiment_prices` (prices of mentioned stocks outside the pool, for S4).

---

# 7. CLI (as built)

```bash
quant sentiment fetch [--days N] [--channel HANDLE]              # ① ②
quant sentiment extract [--limit N] [--video ID] [--model M] [--reextract] [--yes]   # ③
quant sentiment summarize [--from D] [--to D] [--force]          # ④
quant sentiment report [--from D] [--to D]                        # ⑤
quant sentiment run                                               # ① .. ⑤; each step runs even after a problem
quant sentiment stock TICKER [--from D] [--to D] [--all]          # one stock over time (read-only)
quant sentiment gold-export [--add N]                             # 4.6
quant sentiment review-export [--model M]                         # 4.6
```

`extract` prints the backend, the videos and chunks to process and a token estimate; above `max_videos_per_run` it needs `--yes`.

---

# 8. Output

## 8.1 Daily report (`reports/sentiment/<date>.md`, and the dashboard's Sentiment page)

Per effective date: the market mood; tables for stocks and for ETFs (Bull / Bear / Cond / Neutral / Net, each creator once per column with all their marks in one entry; ★ = in the stock pool); the summaries; an appendix with every verified quote, a link to the second and the video title (auto captions flagged); "Not counted": unverified quotes and unresolved names. Net = bullish minus bearish creators.

## 8.2 Creator scorecard (S4, planned)

- **Excess return:** for an opinion on stock X, effective date d, window h (5 / 20 / 60 sessions): X's return from the open of d over h sessions minus the equal-weight return of the pool over the same sessions (a random pool stock on that day); adjusted prices. A beta-adjusted excess return alongside.
- **Hit:** bullish with positive excess return, bearish with negative.
- **Counting:** per creator, stock and stance, only the first opinion in each h-session window (a daily-wrap channel would otherwise count one call twenty times).
- **Significance:** a permutation test (10,000 times): each opinion keeps its creator, date and stance, and its stock is replaced by a random pool stock of that date. It asks whether the picks beat random picks on the same days, and works for creators who are almost always bullish.
- **Scope:** pool stocks scored; out-of-pool tickers descriptive only; unpriceable tickers listed. Each cell: sample size, hit rate, mean excess return, p-value; below `min_samples` (30) numbers only. P-values are uncorrected; only section 9 can lead to a decision.
- **Power:** telling 55% from 50% takes about 600 independent samples.

## 8.3 Summaries (step ④)

- Written by the configured backend (Haiku) **from that day's VERIFIED opinions only**, never from transcripts; one call per trading day for all stocks plus the market paragraph.
- Content: what the bulls and the bears argue, disagreements, conditional views; no recommendation, no price target, no view of its own.
- A summary may cite only its own stock's opinion ids, otherwise it is dropped. Each stores a hash of its opinion ids and the extractor, so a day whose opinions change is summarized again, and the report shows only summaries of exactly the current opinions.
- **Display only:** never in the scorecard or features.

---

# 9. Integration with the quant system (S6, later)

**Forward-only evidence.** Everything before the S6 pre-registration, the S4 backfill included, is exploration: it may suggest what to test, but cannot be the test. S6 uses only opinions published **after** its pre-registration date.

It becomes a feature only when all hold:
1. at least **6 months** of opinions after the pre-registration date (as the forward test's first look);
2. the pre-registration fixes, before any forward data, one **primary test** (planned: 20-session window, VERIFIED pool-stock opinions, all creators pooled, the permutation test of 8.2), the significance level and a Holm correction for secondary tests; the primary test must pass;
3. enough pool stocks covered (threshold in the pre-registration).

Candidate features: `sentiment_net_5d`, `sentiment_consensus`, `sentiment_contrarian` (inverted for creators with a significantly negative hit rate); new `features` columns with a `features.version` bump. **Never replaces the live strategy during an open forward-test run**; at most a new shadow at the next restart.

**Backfill limits:** the creators were picked today and may still be around because they were right (survivorship of creators); losing calls get deleted (survivorship of videos). Backfilled data is exploration only.

## 9.1 S6 pre-registration (registered 2026-10-22, R = 2026-10-22; nothing built yet)

Written before any hit rate or excess return of any creator has been computed: S4 (the scorecard) is not built, and no opinion has been scored against prices. Confirmed by the owner on 2026-10-22, with the choices marked *(confirm)* as written and haiku as the extraction model. **R = 2026-10-22:** the test counts videos published from 2026-10-23 00:00 UTC; 6 months of them run to 2027-04-22, so the test runs once in about July 2027 (after the 60-session windows close).

**Question.** Do the stocks these YouTube creators call bullish (or bearish) do better (or worse) over the next 20 sessions than random pool stocks on the same days?

**Forward data only:**
- Opinions from videos **published after the end of day R (UTC)**. Everything before, the S4 backfill included, is exploration and never enters the test.
- **Creators:** the 11 channels configured on day R (12.1). A channel added later is excluded from the test; a channel that stops publishing stays in with what it published.
- A video deleted after its captions were fetched stays in (its transcript is stored), so deletions after the fact cannot hide wrong calls.

**Which opinions count:**
- `verification = VERIFIED` (quote found in the transcript), `stance` BULLISH or BEARISH (NEUTRAL and CONDITIONAL left out), any horizon.
- `ticker` in the 81-stock pool as configured on day R (ETFs and out-of-pool stocks are descriptive only).
- **Extraction fixed** *(confirm)*: backend `claude_code`, model **haiku**, `prompt_version` 1 — what the daily runs use. A forward video without a haiku v1 extraction is extracted with exactly that before the test. A later prompt version or model is kept apart and does not enter the test. (Alternative: sonnet, the model reviewed in S2, at a higher Pro usage cost.)
- **Counting:** per creator, stock and stance, only the first opinion in each 20-session window (as 8.2), so a daily-wrap channel repeating a call does not count it many times.

**Outcome of one opinion:**
- From the **open of its effective date** d (section 5: the first session opening after publication) over 20 sessions: the stock's total return minus the equal-weight total return of the pool stocks over the same sessions.
- **Signed** by stance: +excess for BULLISH, -excess for BEARISH.

**Primary test** (one, all creators pooled):
- **Statistic:** the mean signed 20-session excess return over all counted opinions.
- **Null:** a permutation test with 10,000 draws (seed fixed in code): each opinion keeps its creator, date and stance, and its stock is replaced by a random pool stock that has a price on that date. p = the share of draws whose mean is at least the observed mean (one-sided: the creators' picks beat random picks).
- **Pass:** p < **0.05**.

**Secondary tests** (reported with a Holm correction over the three, at 0.05; they cannot rescue a failed primary):
1. the same statistic over 5 sessions;
2. the same statistic over 60 sessions;
3. the pooled hit rate (share of counted opinions with a positive signed excess return) against the same permutation null.

**When it runs** *(confirm)*:
- **Once**, after **6 months** of forward opinions (videos published R+1 .. R + 6 months) plus the 60 sessions their longest window needs, so about **9 months after R**.
- **Minimum data:** at least **200 counted opinions on at least 30 different pool stocks**. If either is not met at 6 months, the collection period is extended to **12 months** (and the run moves accordingly); if still not met, the test is reported as **not run (too little data)**, not as failed.
- With about 200 opinions, only a fairly large effect can be told from noise (8.2: about 600 samples are needed to tell a 55% from a 50% hit rate). A failure is therefore weak evidence that there is no effect.

**Data-quality check before the run** *(confirm)*:
- S2 was accepted only provisionally, and haiku was not reviewed (12.3). So before the test, **50 counted opinions drawn at random (fixed seed)** are shown to the owner with their quote and video link; the owner marks each ticker and stance as right or wrong.
- At least **90% right** (45 of 50): the test runs. Otherwise it is reported as **not run (extraction not reliable enough)**; the extraction is improved, and the test needs a new registration with a new R. The check is done before any test statistic is computed.

**Descriptive only** (decides nothing): per creator (opinions, hit rate, mean signed excess return, permutation p-value, only where at least 30 counted opinions), per horizon, per language, bullish against bearish, ETFs and out-of-pool tickers, and the share of opinions that are bullish.

**If the primary test passes:** no live change during an open forward-test run. The candidate features of section 9 (`sentiment_net_5d`, `sentiment_consensus`, `sentiment_contrarian`) would be built and tested as a **separate, pre-registered trial**; at most a new shadow at the next forward-test restart.

**If it fails or does not run:** sentiment stays display only. The same forward data is not re-tested with another window, subset of creators or counting rule; a new question needs a new registration and new forward data.

**Confirmed (2026-10-22, owner):**
- (a) the question, the primary test (20 sessions, all creators pooled, permutation, one-sided, p < 0.05) and the three Holm-corrected secondary tests;
- (b) the opinions counted: VERIFIED, BULLISH or BEARISH, pool stocks, any horizon, the 11 channels of day R, the counting rule;
- (c) the extraction fixed to claude_code / haiku / prompt version 1 (or sonnet instead);
- (d) the timing: once after 6 months of forward opinions (about 9 months after R), at least 200 counted opinions on 30 stocks, else extended to 12 months, else not run;
- (e) the data-quality check: 50 random opinions checked by the owner, at least 90% right, before any statistic is computed;
- (f) R = the day of confirmation.

All of (a) to (f) were confirmed as written on 2026-10-22, with haiku for (c) and R = 2026-10-22. From here on, a change to any of them is a design change recorded below this section before any forward statistic is computed, never a silent edit.

---

# 10. Configuration (as built)

`sentiment` in `config/settings.yaml`:
- fetching: `lookback_days` (7), `min_duration_seconds` (180), `include_live` (false), `request_delay_seconds` ([8, 15]), `max_captions_per_run` (10), `no_captions_retry_days` (3), `min_hours_between_caption_fetches` (6), `cooldown_hours_after_block` (8);
- `gold`: `videos`, `labels_dir` (`research/sentiment_ai_reference`), `transcripts_dir`;
- `extraction`: `backend` (`claude_code`), `model` (`haiku` since 2026-10-22, the model S4 and S6 score), `report_models` ([sonnet, haiku]), `prompt_version` (1), `chunk_minutes` (10), `chunk_overlap_seconds` (30), `quote_match_threshold` (0.85), `max_videos_per_run` (40), `call_timeout_seconds`, `aliases_path`, `tickers_cache`;
- `summary`: `backend`, `model` (`haiku`), `prompt_version`, `reports_dir`;
- `channels`: 11 channels with handle, channel id, name, language and style (12.1).

- `evaluation` (S4, 12.6): `horizons` ([5, 20, 60]), `min_samples` (30), `permutations` (10,000), `seed`, `model` (haiku), `scored_until` (2026-10-22, never later), `backfill_from` (2026-04-23).

Secrets only from `.env` (git-ignored): `GEMINI_API_KEY` and, later, the Reddit credentials. No YouTube key.

---

# 11. Code layout (as built)

```
quant/sentiment/
  tables.py        section 6
  sources.py       VideoSource + yt-dlp (3.1)
  transcripts.py   captions, Traditional -> Simplified, chunks (3.2)
  store.py, fetch.py   steps ① ②
  aliases.py       alias table, US ticker list (4.3, 4.4)
  backends.py      Extractor + claude_code (4.1)
  prompts.py       extraction and summary prompts, every version kept
  extract.py       chunks, validation, quote check, merging (4)
  summarize.py     step ④ (8.3)
  report.py        daily report, `sentiment stock` (8.1)
  gold.py, review.py   gold export and review export (4.6)
  cli.py           `quant sentiment ...`
config/sentiment_aliases.yaml
tests/test_sentiment_{fetch,extract,gold,review,report}.py, tests/test_dashboard_sentiment.py   (offline, fakes)
```

The packages (`yt-dlp`, `youtube-transcript-api`, `opencc-python-reimplemented`, `python-dotenv`) are regular dependencies (decision 10) but imported only when a `quant sentiment` command runs, so a broken scraper cannot affect `run-daily`.

---

# 12. Phases

| Phase | Content | Status |
|---|---|---|
| S0 | channel spikes | done (12.1) |
| S1 | fetch | done (12.2) |
| S2 | extraction; AI reference; your review of disagreements | steps 1-5 done, accepted provisionally with Sonnet (12.3); steps 6-8 (Haiku comparison, Gemini) postponed |
| S3 | summaries and daily reports | done: accepted 2026-10-22 (12.4) |
| S4 | 6-month backfill, scorecard, permutation test; exploratory | not built |
| S6 pre-registration | the S6 test, **before looking at forward hit rates** | registered 2026-10-22 (9.1) |
| S3b | Reddit (section 15) | deferred until S4 has been read |
| ~~S5~~ | Whisper | dropped (decision 13) |
| S6 | the pre-registered test after ≥ 6 months of forward data | later |

## 12.1 S0 findings and channel choice (2026-10-12 .. 10-16)

**First candidates (2026-10-12):** Financial Education was kept (4 videos in 14 days, auto `en` captions that read cleanly). Joseph Carlson was dropped (a video every 1-2 months). 美投讲美股 (`@MeiTouJun`) and 视野环球财经 (`@RhinoFinance`) were dropped because both disable captions, auto captions included. Members-only videos cannot be read.

**Replacements (same window):**

| Channel | Handle / channel_id | Captions | Content |
|---|---|---|---|
| Chip Stock Investor | `@chipstockinvestor` / `UC3aD-gfmHV_MhMmcwyIu1wA` | manual `en` (about half the videos are members-only) | semiconductors / AI |
| NaNa说美股 | `@NaNaShuoMeiGu` / `UCFhJ8ZFg9W4kLwFTBBNIjOw` | manual `zh-Hans` + `zh-Hant` | daily market wrap, very clean |
| 投资TALK君 | `@yttalkjun` / `UCRBrH2qS7yKGMNSmjnj8gcw` | manual `zh` on 7/9 | macro with sector and stock views |
| Zac美股笔记 | `@Zac-Stock-Journal` / `UCawshLsWXrFnYhAlnmn0h2A` | manual `zh-Hans` + `zh-Hant` | buy points and own holdings |
| Nick 美股咖啡館 | `@nick_stockcafe` / `UCjrP2TtSTifuRJ76hW2IW1A` | manual `zh-TW`, noisy | own buys |

Not taken: Ticker Symbol: YOU (too rare then), 老李玩钱 (no captions), 北美韭菜日记 and Money Manta (inactive), 猴哥财经 (stopped). **Second spike (2026-10-13):** 贝拉聊财金 has no captions; Jefftalkstocks is inactive publicly.

**Third spike (2026-10-16), five added** (you asked for creators beyond the first six; scraping Xiaohongshu was ruled out: no API, anti-bot protection, terms):

| Channel | Handle / channel_id | Captions | Why |
|---|---|---|---|
| 老汤美股财经 | `@laotangmeigu` / `UCeTEWFsNC3eeUsn9hvsDALQ` | manual `zh-Hans` | single-stock analysis |
| Ethan的美股笔记 | `@EthanFinanceNote` / `UCvf_c9B7iL0x7aqdhxI4zRg` | manual `zh-Hans` | his own trades |
| Parkev Tatevosian, CFA | `@parkevtatevosiancfa9544` / `UCwKB_00dPL3x5XmHF9IJCrg` | auto `en` | one stock per video, buy or sell |
| Everything Money | `@EverythingMoney` / `UChBVf9YnourrEDTsbbwJPRA` | auto `en` | valuation-based calls |
| Dividend Talks | `@DividendTalks` / `UClLyc3C2K3C-RAh7eEJAwyg` | auto `en` | dividend stocks (the pool's utilities, REITs) |

- Left out in that spike: captions disabled (Kelly Tsai, 美股小头狼, XTrader猫姐美股, 小左第一视角, Victor美股投资), unreadable (JohnLu谈股), promotional titles, news networks. Usable but not added: 幂笈投资, Sam谈美股, Amy说美股, Ticker Symbol: YOU, Couch Investor, BWB, Tom Nash, Daniel Pronk.
- **11 channels, about 60-70 long videos a week.** Parkev posts about 4 a day (short, about one call each): watch the Pro usage; he is the first to drop if it is too much. The scorecard judges each creator from their own first opinion on.
- Checked again later on your request: 视野环球财经, 美投君 and 阳光财经 all disable captions, so not added.

## 12.2 S1 as built (2026-10-13)

- Listing: members-only videos and Shorts are skipped from the listing alone; every other new video's page is read once; the listing stops at the first video older than the lookback. Captions oldest first; a block from YouTube stops the run.
- **First full run:** YouTube blocked caption requests (`IpBlocked`) after 19 with waits of 2-5 s; hence 8-15 s waits and 10 captions per run (3.2). A backfill takes many runs over several days.
- Captions often appear hours after publication, so a `NO_TRANSCRIPT` video is asked again while younger than 3 days, after every untried video.
- Nick's captions are noisy, as found in S0 ("香蕉臺灣", probably "相較台灣").

## 12.3 S2 steps 1-5 as built (2026-10-13)

- Prompt version 1 (`PROMPTS` keeps every version), the alias table in the system prompt; the schema allows STOCK and ETF.
- **Ticker list: Nasdaq Trader's symbol directory**, not the SEC list (the SEC refuses requests without a contact email): every US-listed stock and ETF, cached in `data/sentiment/us_tickers.json`, refreshed monthly. OTC ADRs (HESAY, LVMUY) are known only through the alias table.
- Gold / AI-reference videos whose status is not done are held back. "Win Resorts" (the auto-caption spelling of Wynn) was kept out of the alias table on purpose: it was found only by reading the AI reference, so adding it would inflate agreement.
- **First full run (Sonnet):** 25 videos, 68 calls, about 462k input and 36k output tokens, 129 opinions, all VERIFIED. Agreement with the AI reference, matched on video and stock: 81 pairs (89% of the reference's 91, 85% of Sonnet's 95), the same direction on 72; most differences in NaNa's news wraps (own view or news?).
- **Review (option A, kept as given):** 33 disagreements (10 only in the reference, 14 only from Sonnet, 9 different direction): 31 judged `extractor`, 2 `both`, 0 `reference`, filled in about 7.5 minutes. Verdicts in `research/sentiment_ai_reference/verdicts-claude_code-sonnet-v1.csv`. "Sonnet never wrong" is not plausible, and some cases were debatable (TLT from a bond-yield view; "割韭菜" read as bullish). So the review is **weak evidence**: S2 is accepted **provisionally**, the quote and ticker checks are the only hard guards, and **a careful review or human labels must come before any S6 use**.
- Postponed: S2 steps 6-8 (the Haiku comparison and the Gemini backend).

## 12.4 S3 as built (2026-10-13)

- Summaries with Haiku (8.3) and the daily report (8.1), both in `sentiment run`. First run: 9 trading days, 9 Haiku calls, 105 summaries, none dropped.
- `quant sentiment stock TICKER`: every opinion on one stock over time, with that day's summary, quotes and links; a name from the alias table works too (美光 -> MU). Added because a daily report often shows one creator per stock.
- **For prompt version 2, after your week of reading:** summaries repeat "No bearish view was given" for single-creator stocks and sometimes write opinion ids into the text ("Financial Education (37)"); the extractor sometimes turns one quote into two opinions (MRVL 2026-10-12: BULLISH LONG and NEUTRAL from one sentence).
- **Sonnet and Haiku together (2026-10-15):** `report_models: [sonnet, haiku]`: reports, summaries, `sentiment stock` and the dashboard take, per video, the opinions of the first listed model that extracted it (so a video counts once). The header says which model each day comes from, and Haiku quotes are marked "by haiku". **Haiku was not reviewed** like Sonnet, so Haiku days are weaker evidence.
- **Dashboard:** a Sentiment page (the daily report: mood, tables, summaries, quotes) and opinion marks on the Stock detail chart.
- **Accepted (2026-10-22, owner):** after a week of daily reports (2026-10-19 .. 10-23), with no new problems reported. The known summary issues above stay open for a later summary prompt version 2 (display only; it does not touch S6). The extraction prompt stays at version 1, which S6 (9.1) uses.

## 12.5 Caption-fetch pacing (planned and built 2026-10-19)

**Problem.** YouTube still blocks caption requests (`IpBlocked`) with 8-15 s waits and 10 captions per run. The blocks depend on the **time since the last full run**, not on the pace within a run:

```text
fetch started (UTC)   hours since the previous fetch   result
2026-10-18 07:50      -                                10 captions
2026-10-18 10:14      2.4                              blocked at the first caption
2026-10-18 14:36      4.4                              10 captions
2026-10-19 14:45      24.1                             10 captions
2026-10-19 17:01      2.3                              blocked at the first caption
```

Each caption costs about three requests (the video page, the player API and the caption file), so a full run sends about 30 caption requests plus the listing. YouTube seems to allow about that many per window of a few hours from one home IP. Five runs are too few to measure the window; 2.3-2.4 hours was too short twice, and 4.4 hours was enough once. The listing (`yt-dlp`) was not blocked in either failed run.

**Plan** (code in `quant/sentiment/fetch.py` and `cli.py`; no new table: the times come from the `sentiment-fetch` rows of `pipeline_runs`):
1. **Minimum gap.** The caption step runs only if the last fetch that requested captions started at least `sentiment.min_hours_between_caption_fetches` (new, default **6**) hours ago. A run that only listed, or found no captions waiting, does not count.
2. **Cooldown after a block.** After a blocked run, the caption step waits `sentiment.cooldown_hours_after_block` (new, default **8**) hours from that run, so retries do not extend the block. A blocked run is recognised by a fixed marker in its stored message (today's "stopped: YouTube refused caption requests"), kept as a constant shared by the writer and the check.
3. **What still runs.** The listing runs as before, so new videos are stored and the lookback window keeps moving. Only the caption step is skipped, with one line: `captions: next fetch after 2026-10-19 23:01 UTC (cooldown after a block)` or `(6 h since the last caption fetch)`. That run is stored as `OK` with this line in its message; `sentiment run` continues with extraction and summaries of the captions already stored, as today.
4. **Override.** `--force-captions` on `sentiment fetch` and `sentiment run` ignores both waits, for a manual run when you know the IP has changed. The run's message records that it was forced.
5. **Unchanged:** `request_delay_seconds` [8, 15], `max_captions_per_run` 10, oldest first.
6. **Tests** (fake clock, fake sources, in-memory database): a caption fetch 5 hours after the last is skipped and the listing still runs; 6 hours after, it runs; after a blocked run, 7 hours is skipped and 8 runs; a listing-only run does not move the gap; `--force-captions` runs at once and is recorded.
7. **Documents:** sections 3.2 and 10, `config/settings.yaml` and the README's sentiment section.

**Backlog.** On 2026-10-19, 13 videos inside the 7-day window are PENDING (the oldest from 2026-10-15, so it leaves the window on 2026-10-22). At 10 captions per allowed run and up to four runs a day, the backlog clears within a day.

**If blocks continue with 6 hours between runs:** the next step is fewer requests per video (captions read from the `yt-dlp` page data already loaded for the listing), which needs a check that its text matches today's captions. It would be planned separately.

**As built (2026-10-19), as planned:**
- `next_caption_fetch` in `fetch.py` reads the `sentiment-fetch` rows of `pipeline_runs` from the last max(6, 8) hours. A run requested captions if its message has the marker `captions requested` or the block marker; runs stored before 12.5 have no marker, so one with "N captions" (N > 0) also counts. The wait ends at the later of the last caption fetch + 6 h and the last block + 8 h. Times are taken from the stored run, which is written when the run ends.
- `run_message` writes the stored message for the CLI, with the markers `captions requested`, `captions forced` and `captions skipped until <time> UTC (<why>)`, so the writer and the check share the constants.
- A skipped caption step does not make the run fail: it is stored as `OK`.
- Tests: `CaptionPacingTest` and `test_a_second_fetch_waits_unless_forced` in `test_sentiment_fetch.py`, and two invalid-settings cases.

## 12.6 S4 plan: 6-month backfill and the exploratory scorecard (planned and confirmed 2026-10-22, not built)

**Purpose.** Learn how the creators behave (how often they call stocks, how bullish they are, which stocks) and get a first, exploratory look at whether their calls are followed by excess returns. **S4 decides nothing** (section 9): the creators were picked today and losing videos may be deleted, so backfilled results are biased and only suggest what later questions to ask. The choices marked *(confirm)* were confirmed by the owner on 2026-10-22 as written (see the end of this section).

**The rule that protects S6** *(confirm)*: the scorecard scores **only opinions from videos published up to the end of 2026-10-22 (R)**. Forward opinions (from 2026-10-23) are fetched and extracted every day as now, but **never scored before the S6 test runs** (9.1), so no forward hit rate is seen early.

### Backfill

- **Window** *(confirm)*: videos published **2026-04-23 .. 2026-10-22** (six months before R), for the 11 channels of R; the videos already stored are kept.
- **Size:** at the current pace (about 150 long videos in the 23 days since 2026-09-29, about 6.5 a day) about **1,000 more videos** need captions and extraction. Shorts, members-only videos and live replays are skipped as now.
- **The forward run comes first** *(confirm)*:
  - Today the caption queue is oldest first within the lookback window (3.2), and a backfill would put about 1,000 old videos at its front. With 10 captions per run and at least 6 hours between caption fetches (12.5), new videos could then wait longer than `lookback_days` (7) and be lost to S6.
  - So the queue changes to: **videos inside the lookback window first** (oldest first, as now), **then backfill videos** (oldest first) with the rest of the run's 10 captions. A forward video never waits behind the backfill.
- **How it runs:** a backfill option on `sentiment fetch` and `sentiment run` (for example `--backfill-from 2026-04-23`) lists each channel back to that date once and stores the videos as PENDING; every later run then fetches backfill captions after the forward ones. With about 6-7 forward videos a day and up to 40 captions a day, the backfill takes **about 4-6 weeks** of daily runs. The first listing reads about 1,000 video pages (8-15 s apart, about 3-4 hours once).
- **Extraction** *(confirm)*: the same `claude_code` / **haiku** / prompt version 1 as the forward data (9.1), so backfill and forward opinions are comparable. About 1,000 videos at up to 40 per run uses Pro usage over several weeks; the forward videos are extracted first in each run.
- **Missing captions stay missing:** a backfill video whose captions cannot be fetched (none, disabled, failed twice) stays as it is; nothing is filled in from other sources.

### Scorecard (`quant sentiment scorecard`, section 8.2 as written)

- **Opinions scored:** VERIFIED, BULLISH or BEARISH, ticker in the 81-stock pool, published up to R; the counting rule of 8.2 (first opinion per creator, stock and stance in each window).
- **Excess return** for windows of **5, 20 and 60 sessions** from the open of the effective date: the stock's total return minus the pool's equal-weight total return over the same sessions; a beta-adjusted excess return alongside (beta of the stock against SPY over the 252 sessions before the effective date).
- **Hit:** a bullish call with a positive excess return, a bearish call with a negative one.
- **Permutation test** (10,000 draws, fixed seed): each opinion keeps its creator, date and stance; its stock is replaced by a random pool stock with a price that day. Reported per creator and pooled.
- **Each cell:** counted opinions, hit rate, mean excess return, p-value; below `min_samples` (30) only the counts. P-values are **uncorrected and exploratory** (11 creators x 3 windows x 2 measures, so several "significant" cells are expected by chance).
- **Descriptive:** the share of bullish calls per creator, the stocks named most often, horizons, conviction, calls on ETFs and on out-of-pool stocks (counted, not scored), and unpriceable tickers.
- **Out-of-pool stocks** *(confirm)*: not priced in this version; `sentiment_prices` (6) stays unbuilt. They are listed with their counts only.
- **Output:** `reports/sentiment/scorecard-<date>.md`; no dashboard page in this version. Computed from the stored opinions and prices on each run; **no new table**.
- **Settings:** a new `sentiment.evaluation` section (horizons [5, 20, 60], `min_samples` 30, `permutations` 10,000, `scored_until` 2026-10-22, `backfill_from` 2026-04-23).

### Tests and documents

- Tests (fakes and in-memory databases only, never YouTube, Yahoo or `claude -p`): forward videos are fetched before backfill videos and never wait behind them; the backfill listing stops at its date; the scorecard ignores opinions published after R; excess returns, the counting rule and hits on hand-made prices; the permutation test on a case with a known answer; cells below 30 show counts only.
- Documents: this section (as built), sections 3.2, 8.2, 10 and 12, the README's sentiment section.

**Confirmed (2026-10-22, owner):**
- (a) the S6 protection: only opinions published up to 2026-10-22 are ever scored by S4;
- (b) the backfill window 2026-04-23 .. 2026-10-22;
- (c) the caption queue: forward videos first, backfill after;
- (d) backfill extraction with haiku, prompt version 1;
- (e) out-of-pool stocks counted but not priced in this version;
- (f) the scorecard as in 8.2 (5 / 20 / 60 sessions, pool equal-weight benchmark, beta-adjusted alongside, permutation test, 30 minimum), exploratory only.

All of (a) to (f) were confirmed as written on 2026-10-22. A later change is recorded below this section before it is built.

**As built (2026-10-22; the backfill not run yet):**
- **Settings:** `sentiment.evaluation` (horizons [5, 20, 60], `min_samples` 30, `permutations` 10,000, `seed` 42, `model` haiku, `scored_until` 2026-10-22, `backfill_from` 2026-04-23). Loading the settings **refuses a `scored_until` after 2026-10-22** (the S6 registration date), and the scorecard caps it there again, so forward opinions cannot be scored by mistake.
- **Backfill:** `--backfill` on `sentiment fetch` and `sentiment run` lists each channel back to `backfill_from` (one long listing; afterwards ordinary runs are enough). The caption queue is now: forward videos (inside `lookback_days`) oldest first, then the recent no-caption retries, then **backfill videos (from `backfill_from` up to the lookback) oldest first**. This applies on every run, with or without `--backfill`, so backfill captions keep coming after the forward ones until none is left.
- **Extraction order:** forward videos (published within `lookback_days`) first, then older ones, oldest first. **`sentiment run` now extracts at most one batch (`max_videos_per_run`, 40) per run** instead of refusing a larger backlog, so the backfill never stops the daily run; `sentiment extract` on its own still asks for `--yes` above 40.
- **The model:** the backfill must be extracted with haiku (d). `extraction.model` was set to **haiku** on 2026-10-22 (owner), so a plain `sentiment run` extracts with the model S4 and S6 score. Side effects: `review-export` now defaults to haiku, and in the daily reports the quotes taken from sonnet extractions are now the ones marked "by sonnet" (`report_models` still prefers sonnet where a video has both).
- **Scorecard:** `quant sentiment scorecard` (`quant/sentiment/scorecard.py`) writes `reports/sentiment/scorecard-<date>.md`. As 8.2 and (f): windows of 5, 20 and 60 sessions from the open of the effective date to the open h sessions later, against the pool's equal weight; beta-adjusted alongside (252 daily returns before the effective date, at least 200); the counting rule; the one-sided permutation test per creator and pooled; counts only below 30. Descriptive tables: per creator (videos, opinions, bullish share, pool / other stocks / ETFs), the stocks named most often, stances, horizons, conviction. Opinions whose window has not ended are counted as pending and left out. No new table.
- **Change before the first backfill run (2026-10-23, owner):** a backfill listing reads up to **10 entries per day** of the window (`backfill_scan_limit`, about 1,840 for 183 days) instead of 3 (`scan_limit`, about 560). Parkev Tatevosian posts about 2.7 long videos a day, so the normal cap could have stopped his listing before 2026-04-23. The flat listing costs only a request per page of entries, and the walk still stops at the first video older than the window, so the higher cap adds no video-page requests. Any listing (backfill or not) that runs out of entries before reaching the window's start now prints "the listing reached its cap ... older videos may be missing".
- **First backfill run (2026-10-23, owner):** the listing took about 6 hours (06:54 .. 12:57 UTC); all 11 channels reached 2026-04-23, with no listing cut off. About **1,550 videos** wait for captions (Parkev Tatevosian alone 868, about 4.7 a day), more than the 1,000 estimated, so the backfill needs about 7 weeks at four runs a day. Two of Parkev's videos (`PpKdPGeCmEk`, `O7kSEcUtvi8`) could not be read during a short DNS outage and are **left out by the owner's choice**. The switch to haiku made every existing day count as changed once, so 20 days were summarized again.
- **Tests:** `tests/test_sentiment_scorecard.py` (the settings guard, the backfill listing and the forward-first caption queue over two runs, the backfill listing cap and the cut-off warning, the extraction order, the excess return, the counting rule, a known good and bad picker in the permutation test, counts-only cells, and the command scoring only haiku opinions up to `scored_until`).

---

# 13. Risks

| Risk | Mitigation |
|---|---|
| yt-dlp or the caption endpoint breaks; the IP gets blocked | interface; 8-15 s waits, 10 captions per run; local runs; `FAILED` retried |
| A sentiment failure breaks the daily pipeline | not in `run-daily`; packages imported only by sentiment commands |
| YouTube's terms | a few channels, personal research, transcripts not redistributed |
| Auto-caption errors (Chinese) | misrecognitions in the alias table; `is_generated` recorded |
| Fabricated opinions or tickers | quote and ticker checks; summaries only from verified opinions and must cite them |
| Pro usage limits; billing the API by mistake | per-run caps, clean stop; refuses to run with `ANTHROPIC_API_KEY` set |
| Private data sent to a free API | only public transcripts and opinions, never positions, trades, signals or values |
| Tech-heavy bullish calls look right in rising markets | excess return vs the pool on the same days; permutation against random pool picks; beta-adjusted column |
| Repeated calls; many cells; small samples | one opinion per creator × stock × stance per window; one pre-registered primary test with Holm; minimum samples and stated power; S6 forward data only |
| Survivorship of creators and videos | backfill is exploration only |
| Reddit: approval delayed, no backfill, sarcasm and pumps | ApeWisdom first (no key); forward-only; extraction only from DD / high-score posts, stance from the body |

---

# 14. Decisions

2026-10-12:
1. **Channels:** first Joseph Carlson, Financial Education, 美投讲美股, 视野环球财经; after S0 the six of 12.1 (11 since 2026-10-16).
2. **Backfill:** 6 months, in S4.
3. **Stock scope:** every US stock mentioned; pool stocks marked.
4. **Report language:** English.

2026-10-13 (after review):

5. **No paid API:** backends `claude_code` (Pro subscription) and `gemini` (free).
6. **Summaries:** by the extractor backend, from verified opinions only, display only.
7. **Benchmark:** excess return against the pool on the same days, tested against random pool picks.
8. **Promotion:** S6 on forward data after its pre-registration only, at least 6 months, one primary test.
9. **Reddit (S3b)** deferred until S4 has been read; the API application can be submitted early.
10. **One install:** the sentiment packages are regular dependencies.
11. **Gold set of 10 videos labeled by you.** *(Replaced by 12.)*
12. **An AI reference instead of human labels:** 20 videos labeled with GPT or MAI (model not recorded), in `research/sentiment_ai_reference/`; S2 reports agreement, and the gate is your review (4.6).
13. **No audio transcription:** S5 (Whisper) dropped as too costly; captions only, so channels without captions are not covered.

---

# 15. Phase S3b: Reddit (deferred, decision 9)

The pool is large caps, where Reddit attention mostly echoes the news; retail-attention effects are strongest in small caps. S3b starts only if S4 gives a reason.

- **Why it differs:** YouTube gives a few named creators who can each be scored; Reddit (r/wallstreetbets, r/stocks, r/investing, r/ValueInvesting) gives a crowd, whose signal is **attention and mood**, not a personal track record.
- **Two layers:** (A) daily mention counts per ticker and subreddit, no LLM; (B) opinions extracted from DD and high-score posts only, into the shared `sentiment_opinions` (`source = reddit`), report and scorecard.
- **Sources:** ApeWisdom's public mention feed first (no key; its terms and limits checked before use), then the Reddit Data API via `praw` (script app, read-only, about 100 requests a minute; approval can take weeks, so the application can go in early; credentials only from `.env`).
- **Fetched:** posts only, not comments, from the `new` and `top` (day) listings, with score, comments and upvote ratio at fetch time, at a fixed time of day. Posts deleted on Reddit are deleted locally, as the API terms require; text stays local.
- **Ticker detection:** a cashtag (`$NVDA`), or an uppercase word in the ticker list and not on a stop list (`A`, `IT`, `ON`, `ALL`, `DD`, `YOLO`, `CEO`), or a name or nickname from the alias table; once per post and ticker.
- **Extraction** only for flairs DD / Due Diligence / Fundamentals / Technical Analysis, or a score ≥ 100 with a body ≥ 800 characters; stance from the body, not the title; sarcasm and "to the moon" without reasoning as `NEUTRAL` or skipped; a per-run cap.
- **Timing:** `effective_date` as in section 5. Selecting by score at fetch time is a slight look-ahead (part of the score comes later): accepted for research; a live feature must use the score known at decision time.
- **Tables:** `sentiment_reddit_posts` (post, subreddit, author, time, title, body, flair, score, comments, upvote ratio, extraction status) and `sentiment_reddit_mentions` (date, ticker, subreddit, provider, mentions, upvotes, rank).
- **Output:** a Reddit attention section in the daily report (top tickers by mentions with `mention_z` against their own 20-day average); scored per subreddit, and per author only with ≥ 30 opinions.
- **Factors (with S6):** `reddit_mention_z`, `reddit_mention_trend_5d`, `reddit_net_stance_5d`, both signs tested (momentum or reversal). **No backfill is possible** (listings reach about 1,000 posts; Pushshift is closed), so forward logging only, with the same 6-month minimum.
- **Steps:** S3b-1 ApeWisdom mentions and the attention section (done when a week of tables is reviewed); S3b-2 `praw` fetch, own counts, extraction, deletion handling (done when extraction reaches 85% precision, recall and stance accuracy on about 20 hand-labeled posts and own counts broadly agree with ApeWisdom). Not in S3b: comments, Stocktwits, X, Chinese forums (雪球, 富途).
