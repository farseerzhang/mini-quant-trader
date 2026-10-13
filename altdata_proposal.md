# Free Structured Alternative Data — Proposal (Draft)

Status: approved 2026-10-13 (decisions in section 15). Starts after sentiment S1, with EDGAR only.
Date: 2026-10-13

Companion to `sentiment_proposal.md`. That proposal turns **text** (YouTube, Reddit) into opinions with an LLM and mostly has to be logged going forward. This one covers **already-structured records** from free, official sources that have **years of history**, so they can be researched on the existing development folds now.

---

# 1. Goal

Add four free, official data sources as candidate stock features, and test them with the same discipline as the existing features (proposal sections 11 and 15):

| Source | Publisher | Signal |
|---|---|---|
| **Form 4 insider transactions** | SEC EDGAR | Executives and directors buying or selling their own company's stock with their own money |
| **Schedule 13D filings** | SEC EDGAR | An investor taking a > 5% stake with intent to influence the company (activists) |
| **Short-sale volume and short interest** | FINRA | How much selling is short, and how many shares are held short |
| **Wikipedia pageviews** | Wikimedia | Retail attention to a company |

As everywhere in this project, a new input is a hypothesis. It enters a model only after a pre-registered test on development data (section 8), never because it "should" work.

## Out of scope

- **13F fund holdings**: quarterly, published up to 45 days after quarter end. Too slow for a strategy that rebalances every 5 sessions.
- **Congressional trades**: House filings are PDFs, disclosures lag up to 45 days, and the evidence is mixed.
- **RSS feeds** (Substack, blogs, podcasts): text, so they belong in the sentiment pipeline (`sentiment_proposal.md`) as another source type.
- **Analyst ratings and EPS estimate revisions**: probably the strongest signal of all, but no free source covers them reliably (section 11).
- Paid data of any kind.

---

# 2. Ground rule: available-at time

Every alternative data row has an **event date** (when something happened) and an **available-at time** (when the public could first see it). They can be far apart:

| Source | Event date | Available at |
|---|---|---|
| Form 4 | Transaction date | EDGAR acceptance time (due within 2 business days, sometimes late) |
| 13D | Date the 5% threshold was crossed | EDGAR acceptance time (due within 5 business days since February 2024; 10 calendar days before) |
| FINRA short-sale volume | Trade date | That evening (FINRA publishes daily files after the close) |
| FINRA short interest | Settlement date (mid-month and end of month) | FINRA publication date, about a week or more later |
| Wikipedia pageviews | UTC day | Next UTC day |

**Features use only rows available before the close of session T** (proposal 4.5: decision at the close of T, fill at the open of T+1). Every table stores both times, and every feature query filters on `available_at`. This is the main way alternative data creates look-ahead bias, so it gets its own tests (section 9).

Specific cases:

- **Form 4 / 13D** filed after 16:00 ET on day T count from session T+1.
- **Short interest** is used from its publication date, never from its settlement date.
- **Wikipedia**: a UTC day ends at 20:00 ET (19:00 in winter), after the 16:00 close, and is published only the next day. Session T therefore uses UTC days up to **T−1**.
- **FINRA daily short volume** for day T is published after the close, so session T uses data up to **T−1**.

---

# 3. Sources

## 3.1 SEC EDGAR — Form 4

- **Backfill**: the SEC's quarterly *Insider Transactions Data Sets* (structured Forms 3/4/5 as tab-separated files, 2006 onward as far as I know). This avoids parsing hundreds of thousands of filings one at a time.
- **Incremental**: the EDGAR daily index (`form.idx`) lists new filings; each new Form 4 XML for a pool company is fetched and parsed.
- **Rules**: a `User-Agent` header with your name and email is required, and at most 10 requests per second. The downloader stays well below that (configurable, default 5/s).
- **Ticker → CIK**: SEC `company_tickers.json`, cached; checked against the pool on every update. `BRK-B` and other share classes map to one CIK.
- **What is kept**: one row per transaction line, with the insider's role (officer / director / 10% owner, title), transaction code, shares, price, shares owned afterwards, whether it is direct or indirect ownership, and the Rule 10b5-1 flag (on Form 4 since 2023).
- **What counts as a signal**: only transaction code **P** (open-market purchase) and **S** (open-market sale). Grants (A), option exercises (M), tax withholding (F) and gifts (G) are stored but are not decisions about the stock. Sales under a 10b5-1 plan are pre-scheduled and flagged so they can be excluded.
- **Amendments** (Form 4/A) replace the original by accession number; the original row is kept with `superseded_by`.

## 3.2 SEC EDGAR — Schedule 13D

- Same index and rules as Form 4. Filings since December 2024 are structured XML; earlier ones are HTML/text, from which only the filer, the subject company, the filing date and the percentage are extracted. Amendments (13D/A) are kept as separate events.
- 13G (passive holders) is stored but not a candidate signal: it says "large holder", not "activist".
- Events are rare in a pool of large caps; this source is expected to fire a few times a year at most (section 10).

## 3.3 FINRA — short-sale volume (daily)

- FINRA publishes a daily file of short-sale volume and total volume per symbol (Reg SHO daily files).
- **Important caveat**: these files cover only trades reported to FINRA's facilities (mostly **off-exchange** trades), not exchange volume. And "short volume" includes market makers selling short to fill buy orders, so it is typically 40–50% of the reported volume for any stock. The level is therefore not the short position; only **changes in the ratio** relative to the stock's own history can carry information.
- Symbols are mapped with the same rules as `daily_prices` (e.g. `BRK-B` vs `BRK.B` / `BRK/B`).

## 3.4 FINRA — short interest (twice a month)

- Shares held short per stock at each mid-month and end-of-month settlement date, from FINRA's free data / query API.
- Derived: short interest as a percentage of shares outstanding (shares outstanding from the SEC XBRL "company facts" API, using the value available at that time), and days to cover (short interest ÷ average daily volume from `daily_prices`).
- How far back the free FINRA series goes is checked in the A0 spike; older history may only be available elsewhere.

## 3.5 Wikimedia — pageviews

- Wikimedia REST API, per article per day, `agent=user` (excludes known crawlers and automated traffic), English Wikipedia, all access types. No key; a `User-Agent` with contact details is required. History starts in July 2015.
- **Ticker → article** is a hand-maintained map in config (`config/altdata_wikipedia.yaml`), because names are ambiguous ("Apple", "Oracle", "Target"). Redirect and rename history is checked in A0: if an article was renamed, the views of the old title are added for the period before the rename.
- Company articles also get traffic for reasons unrelated to the stock (product launches, scandals, a famous CEO). That is part of "attention" and is accepted as noise.

---

# 4. Database

New tables in the same SQLite file, defined in `quant/altdata/tables.py` on the shared `metadata`. Every table name starts with `altdata_`. The existing tables do not change. Every run writes a `pipeline_runs` row (command = `altdata-*`).

### altdata_entities
`symbol` (PK), `cik`, `wikipedia_article`, `finra_symbol`, `updated_at`. Synchronized from config and `company_tickers.json`.

### altdata_insider_transactions
`accession_no`, `line_no`, `symbol`, `cik`, `insider_cik`, `insider_name`, `is_officer`, `is_director`, `is_ten_percent_owner`, `officer_title`, `transaction_date`, `transaction_code`, `acquired_disposed` (A/D), `shares`, `price`, `shares_owned_after`, `direct_indirect`, `is_10b5_1`, `filed_at`, `available_at`, `superseded_by`. Primary key: `(accession_no, line_no)`.

### altdata_activist_filings
`accession_no` (PK), `form_type` (`SC 13D` / `SC 13D/A` / `SC 13G` / `SC 13G/A`), `symbol`, `filer_name`, `filer_cik`, `event_date`, `percent_of_class`, `filed_at`, `available_at`.

### altdata_short_volume
`symbol`, `trade_date`, `altdata_short_volume`, `short_exempt_volume`, `total_volume`, `available_at`. Primary key: `(symbol, trade_date)`.

### altdata_short_interest
`symbol`, `settlement_date`, `altdata_short_interest`, `avg_daily_volume`, `days_to_cover`, `shares_outstanding`, `published_date`. Primary key: `(symbol, settlement_date)`.

### altdata_wikipedia_pageviews
`symbol`, `date` (UTC), `article`, `views`, `available_at`. Primary key: `(symbol, date)`.

### altdata_features
Date-keyed per-symbol candidate features (section 7), versioned like `features` (`altdata_version`), **kept separate from `features`**. Only a feature that passes section 8 is moved into `features`, with a `features.version` bump and re-registration of models as the main proposal requires once models exist.

Raw tables are the source of truth (proposal 4.3): features can always be rebuilt from them.

---

# 5. CLI

```bash
quant altdata update   [--source form4|13d|short_volume|short_interest|wikipedia] [--since 2006-01-01]
quant altdata features                                  # rebuild altdata_features from the raw tables
quant altdata research [--pre-registration <id>]        # section 8: IC and backtest report
```

`update` is incremental and idempotent, like the market data downloader (proposal section 9): each source remembers the last date it holds, re-fetches a short overlap window to catch late filings and revisions, and upserts. The first run backfills; later runs take seconds to minutes.

`run-daily` is not changed in this proposal. A source joins it only once a feature built from it is promoted into `features`.

---

# 6. Data quality

Following proposal section 10:

- **Form 4**: price ≤ 0 on P/S lines, shares ≤ 0, transaction dates after the filing date, or a price more than 50% away from that day's close → `data_quarantine`.
- **Short volume**: short volume > total volume, or a missing day for a symbol that traded → quarantine / gap report.
- **Short interest**: a change of more than 10× between consecutive reports → flagged for review (often a split or a symbol change).
- **Pageviews**: zero views for a pool company article (usually a wrong title or a rename) → flagged.
- **Coverage report** after each update: rows per source per symbol per year, so gaps are visible before any research is run.

---

# 7. Candidate features

Computed per symbol at the close of each session T from rows with `available_at` before that close. The list is fixed **before** any result is seen (section 8). Windows are in trading sessions.

| Feature | Definition | Idea |
|---|---|---|
| `insider_net_buy_value_60d` | Σ value of P minus Σ value of non-10b5-1 S, by officers and directors, over 60 sessions, ÷ market cap | Insiders trading with discretion |
| `insider_buyers_60d` | Number of distinct officers/directors with a P transaction in 60 sessions | "Cluster buying", a well-documented pattern |
| `activist_13d_120d` | 1 if a new 13D (not an amendment) became available in the last 120 sessions, else 0 | Activist involvement |
| `short_volume_ratio_z` | (5-session mean of short ÷ total volume − its 120-session mean) ÷ its 120-session std | Unusual short selling |
| `short_interest_pct` | Latest published short interest ÷ shares outstanding | Crowded shorts |
| `short_interest_change` | Change in `short_interest_pct` between the last two publications | Shorts building or covering |
| `days_to_cover` | Latest short interest ÷ 20-session average daily volume | Squeeze potential / bearish conviction |
| `pageviews_z` | (log of 5-day mean views − log of 60-day mean) ÷ std of the daily log changes | Attention spike |

Missing data is missing, not zero. A stock with no Form 4 in the window gets 0 insider buying (no transactions happened), but a stock before its source's history starts gets NULL, and the model's existing missing-value handling applies.

---

# 8. Research protocol

Same principles as the existing feature research (proposal section 11, 15.4):

1. **Pre-registration** (written into this file with a date, before any result): the 8 features above, their definitions, the target (the existing 5-day excess return label), the folds (the existing 11 development folds, 2014-09-26 .. 2025-10-01), and the pass criteria below.
2. **Stage 1, feature check** (cheap, per feature): rank information coefficient (Spearman correlation between the feature and the next 5-day excess return across stocks on each date), averaged over dates, with a Newey-West t-statistic; reported per fold. **Pass**: mean IC t-stat ≥ 2 over the development period **and** the same sign in at least 7 of 11 folds. With 8 features tested, about one could pass at |t| ≥ 2 by chance alone, so a pass is a candidate, not a finding.
3. **Stage 2, model trial** (only for features that pass stage 1): one retrain of the current best predictor configuration with the passing features added, everything else unchanged, judged by the existing gates in `retrain.gates`. Each trial is added to the trial log.
4. **No holdout.** The current holdout has already been evaluated twice (proposal 15.4.8 and the HMM holdout check that follows it) and must not be used to choose features. Evidence beyond the development folds comes from forward testing.

Coverage limits that apply to the protocol:

- **Wikipedia** starts in July 2015, so `pageviews_z` exists for 10 of the 11 folds (the first fold only from mid-2015). The report states the coverage per fold.
- **Short interest** coverage depends on what A0 finds; a feature covering fewer than 8 folds is reported but cannot pass.

---

# 9. Tests

Offline, with fixtures saved from real files:

- Parsers: a few real Form 4 XMLs (including an amendment, an indirect holding, a 10b5-1 sale), a 13D in XML and in HTML, a FINRA daily file, a short-interest extract, a pageviews response.
- **Look-ahead tests** (the important ones): a Form 4 accepted at 16:05 ET is not visible to session T's features; short interest is invisible between its settlement and publication dates; session T's pageviews feature uses UTC day T−1 at most; a backfilled row with a late `available_at` does not change features for earlier sessions.
- Idempotency: running `update` twice changes nothing; an amendment replaces its original.
- Symbol mapping: `BRK-B` resolves for all sources.

---

# 10. Expectations

Honest priors, written down before any data is seen:

- **Insider buying** is the best-documented effect of the four, but most of the evidence comes from small and mid caps. In a pool of 81 large caps, open-market purchases by insiders are rare (executives are mostly paid in stock and sell), so `insider_buyers_60d` will be zero most of the time. A weak or absent result would not be surprising.
- **13D** events are too rare in this pool to test statistically. The feature is kept for completeness; stage 1 will most likely be inconclusive.
- **Short interest** has a documented relationship with lower future returns, again stronger in smaller stocks; large caps rarely have high short interest.
- **Short-volume ratio** is noisy because of market-maker activity; any signal is expected to be small.
- **Pageviews** measure attention, which the literature links to short-term moves and later reversals. The sign is not assumed.

So the most likely outcome is that one or none of the eight features passes. That is still useful: it closes the question cheaply with free data, before paying for anything (section 11).

---

# 11. Configuration (added to `config/settings.yaml`)

```yaml
altdata:
  version: 1
  user_agent: "mini-quant-trader <your name> <your email>"   # required by SEC and Wikimedia; you fill this in
  sec:
    max_requests_per_second: 5
    insider_backfill_from: 2006-01-01
    signal_codes: [P, S]
    exclude_10b5_1_sales: true
  finra:
    short_volume_from: 2009-08-01        # adjusted to what A0 finds
  wikipedia:
    project: en.wikipedia
    agent: user
    articles_file: config/altdata_wikipedia.yaml
  features:
    insider_window: 60
    activist_window: 120
    short_volume_short_window: 5
    short_volume_long_window: 120
    pageviews_short_window: 5
    pageviews_long_window: 60
```

No API keys are needed for any of the four sources.

---

# 12. Code layout

```
quant/altdata/
  __init__.py
  tables.py           # section 4
  entities.py         # ticker → CIK / Wikipedia article / FINRA symbol
  edgar.py            # index, Form 4 + 13D fetch and parse, insider data sets backfill
  finra.py            # short volume + short interest
  wikipedia.py        # pageviews
  availability.py     # available-at rules (section 2), shared by all sources
  features.py         # section 7
  research.py         # section 8: IC report, trial wiring
  quality.py          # section 6
  cli.py              # subcommands, registered in quant/cli.py (one line)
config/altdata_wikipedia.yaml
tests/altdata/
  fixtures/
  test_*.py
```

No new dependencies: HTTP via `requests` (already installed with yfinance), XML via `xml.etree`, pandas for the rest.

---

# 13. Phases

| Phase | Content | Done when |
|---|---|---|
| **A0 Spike** | Confirm endpoints, formats, rate limits and **history start dates** for all four sources on 3 pool stocks; confirm the insider data sets cover 2006 onward; resolve Wikipedia titles for the whole pool | A short findings table appended here; config dates set from it |
| **A1 EDGAR** | Form 4 + 13D: backfill, incremental update, amendments, quality checks, tests | Coverage report for 81 stocks, 2006 → today |
| **A2 FINRA** | Short volume + short interest | Coverage report; known gaps documented |
| **A3 Wikipedia** | Pageviews + title map | Coverage report from July 2015 |
| **A4 Research** | Pre-registration (section 8) → features → stage 1 → stage 2 for passing features | Report in `reports/altdata/`; outcome recorded here, whatever it is |
| **A5 Integration** | Only for features that passed: move into `features` (version bump), add the source to `run-daily`, forward-log | Under the main proposal's rules |

A1–A3 are independent and can be built in any order. A4 needs all three.

---

# 14. Risks

| Risk | Mitigation |
|---|---|
| Look-ahead through publication lags | `available_at` on every row; dedicated tests (section 9) |
| Too many features tried → false discoveries | Fixed list of 8 before any result; t ≥ 2 plus fold consistency; stage 2 through the existing gates and trial log |
| Large-cap pool dilutes effects documented in small caps | Stated as a prior (section 10); a null result is an acceptable outcome |
| FINRA short-volume misread as short position | Only ratio changes against the stock's own history are used |
| Wikipedia title ambiguity and renames | Hand-made map; rename history checked in A0; zero-view check |
| SEC blocks the client for exceeding limits | User-Agent with contact, 5 req/s cap, backoff on 403/429 |
| A free source changes format | Each source behind its own parser with fixture tests; a failed update is a `FAILED` stage, not a crash of other sources |

---

# 15. Decisions (2026-10-13)

1. **Scope**: EDGAR first (A0 for EDGAR, then A1: Form 4 + 13D). Wikipedia (A3) and FINRA (A2) stay in this proposal but are deferred; each is started only on your go-ahead.
2. **User-Agent**: asked during implementation, before the first request to the SEC.
3. **Module name**: `quant/altdata/`.
4. **Timing**: after sentiment S1.
5. **Table names**: every table of this module starts with `altdata_` (section 4), so alt data, sentiment (`sentiment_*`) and the core quant tables never mix.
