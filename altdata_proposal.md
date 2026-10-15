# Free Structured Alternative Data — Proposal (Draft)

Status: approved 2026-10-13 (decisions in section 15). A0 for EDGAR done 2026-10-13 (section 13.1); A1 (Form 4 only) done 2026-10-13 (section 13.2); decision 8 confirmed; A2 and A3 deferred (decision 10); A4 round 1 done 2026-10-14: no feature promoted (section 8.1.2).
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
| ~~`activist_13d_120d`~~ | Dropped with 13D (decision 7) | - |
| `opportunistic_sellers_60d` | Number of distinct officers and directors with an **opportunistic** sale in the last 60 sessions (definition below) | Sales that break an insider's own pattern; routine and pre-planned sales carry little information |
| `insider_sell_fraction_60d` | Largest fraction of their direct holdings that any officer or director sold in the last 60 sessions (definition below); 0 if none sold | A CEO selling 40% stands out against routine 2% trims |
| `short_volume_ratio_z` | (5-session mean of short ÷ total volume − its 120-session mean) ÷ its 120-session std | Unusual short selling |
| `short_interest_pct` | Latest published short interest ÷ shares outstanding | Crowded shorts |
| `short_interest_change` | Change in `short_interest_pct` between the last two publications | Shorts building or covering |
| `days_to_cover` | Latest short interest ÷ 20-session average daily volume | Squeeze potential / bearish conviction |
| `pageviews_z` | (log of 5-day mean views − log of 60-day mean) ÷ std of the daily log changes | Attention spike |

**Sale-side definitions** (added 2026-10-13, confirmed 2026-10-14 as decision 8; written before any return was looked at, only filing counts):

- *Qualifying sale*: an `S` line (open-market sale) whose filing is available by the close of session T (`available_on` <= T), reported by an owner who is an officer or a director (10% owners alone do not count), and whose filing is not flagged 10b5-1 (`is_10b5_1 = 1`; the flag exists only from 2023, so before that no sale is excluded by it).
- *Routine / opportunistic* (after Cohen, Malloy and Pomorski 2012): for a sale in calendar month m of year y, the insider is **classifiable** if they had at least one P or S line in each of the years y-1, y-2 and y-3 (counting only filings available by session T). A classifiable insider is **routine** if they traded in month m in each of those three years, otherwise **opportunistic**. Insiders who are not classifiable count as neither. The classification uses only past filings, so it is free of look-ahead, and it covers the years before the 10b5-1 flag.
- `opportunistic_sellers_60d`: distinct insiders (owner CIK) with at least one qualifying, opportunistic sale among the filings that became available in the 60 sessions up to T. For a jointly filed Form 4, every officer or director listed on it counts.
- `insider_sell_fraction_60d`: for each insider with qualifying sales of directly held shares (`direct_indirect = D`) in the 60 sessions up to T, (shares sold) / (shares sold + shares owned after the last of those sales); the feature is the largest value across insiders, 0 if there was no qualifying sale. Lines without a share count are left out.
- Both are NULL before a stock has three years of Form 4 history in the tables (the routine test needs it), for `insider_sell_fraction_60d` too, so the two start on the same date.

Missing data is missing, not zero. A stock with no Form 4 in the window gets 0 insider buying (no transactions happened), but a stock before its source's history starts gets NULL, and the model's existing missing-value handling applies.

---

# 8. Research protocol

Same principles as the existing feature research (proposal section 11, 15.4):

1. **Pre-registration** (written into this file with a date, before any result): the 8 features above, their definitions, the target (the existing 5-day excess return label), the folds (the existing 11 development folds, 2014-09-26 .. 2025-10-01), and the pass criteria below.
2. **Stage 1, feature check** (cheap, per feature): rank information coefficient (Spearman correlation between the feature and the next 5-day excess return across stocks on each date), averaged over dates, with a Newey-West t-statistic; reported per fold. **Pass**: mean IC t-stat ≥ 2 over the development period **and** the same sign in at least 7 of 11 folds. With 9 features tested (the 8 original ones without the dropped 13D feature, plus the two sale-side features of section 7), about one could pass at |t| ≥ 2 by chance alone, so a pass is a candidate, not a finding.
3. **Stage 2, model trial** (only for features that pass stage 1): one retrain of the current best predictor configuration with the passing features added, everything else unchanged, judged by the existing gates in `retrain.gates`. Each trial is added to the trial log.
4. **No holdout.** The current holdout has already been evaluated twice (proposal 15.4.8 and the HMM holdout check that follows it) and must not be used to choose features. Evidence beyond the development folds comes from forward testing.

## 8.1 Pre-registration, A4 round 1: the Form 4 features (registered 2026-10-14, before any feature value or result is computed)

Written before `altdata_features` exists and before any return is joined to Form 4 data; only filing counts (section 13.2) have been looked at.

**Features (4, decision 10)**, computed for every pool stock on every NYSE session from 2006 on, from filings with `available_on` <= T only:

1. `insider_net_buy_value_60d` = (sum of shares x price of `P` lines) minus (sum of shares x price of `S` lines in filings not flagged 10b5-1), over lines in filings that became available in the 60 sessions up to and including T and that list at least one officer or director as a reporting owner, divided by the market capitalization at the close of T. Market capitalization = the close of T (as traded) x the latest shares outstanding available at T, where shares outstanding is the SEC XBRL company-facts value `dei:EntityCommonStockSharesOutstanding`, available from the session after the date it was filed; if a filing reports several share classes for the same date, they are summed (for BRK-B this understates the economic value of the A shares; accepted as a known approximation). A transaction line counts once even when several owners filed it jointly. NULL when no shares outstanding is available yet.
2. `insider_buyers_60d` = the number of distinct officers or directors (owner CIK) listed on filings with at least one `P` line that became available in the 60 sessions up to T.
3. `opportunistic_sellers_60d` and 4. `insider_sell_fraction_60d`: as defined in section 7 (decision 8).

"Officer or director" = an owner with `is_officer = 1` or `is_director = 1` (10% owners alone do not count). Quarantined lines (section 13.2) are not used. A 4/A is used like any filing (its lines as reported); duplicates between an amendment and its original are not removed in this round. Before a stock's first Form 4, features 1-2 are 0 once its first filing exists in the tables and NULL before; features 3-4 follow section 7 (NULL until three years of history).

**Target**: the existing label's `excess` column (open T+1 -> open T+1+5 return of the stock minus SPY's, as in `quant/models/labels.py`), the same target as every earlier feature study.

**Folds and period**: the existing 11 development folds (`walk_forward_folds`, expanding, 2014-09-26 .. 2025-10-01), built with the holdout start **fixed at 2025-10-10**, the boundary of both earlier holdout checks (`proposal.md` 15.4.8): `quant retrain` derives the holdout from the last 12 months of stored sessions, so its boundary moves forward every day and would otherwise move already-viewed holdout days into development. **No holdout** (section 8, point 4).

**Stage 1 (per feature)**: the daily cross-sectional rank IC (`rank_ic`, sessions with at least 10 stocks with both values) over the sessions of the 11 validation windows; the t-statistic of `ic_summary` (mean over every 5th session, i.e. non-overlapping 5-day labels; not Newey-West as section 8 first said, for consistency with the earlier feature studies of `proposal.md` 15.4.5 and 15.4.7). **Pass** = |t| >= 2 **and** the mean IC of at least 7 of the 11 folds has the sign of the overall mean IC. 4 features are tested, so with |t| >= 2 about 0.2 passes are expected by chance; a pass is a candidate, not a finding. Reported per feature: mean IC, t, folds with the same sign, coverage (sessions and stocks with a value) per fold.

**Stage 2 (only if at least one feature passes)**: one trial with all passing features added to the 35 configured inputs, otherwise the 15.4.6 / 15.4.7 configuration unchanged (logistic regression C = 1.0, sigmoid calibration, 5-day label, `ml_select` top 10 / keep 10 / every 5 sessions, 81 stocks, the same folds, embargo and gates in `retrain.gates`); rows without a value for a passing feature are dropped (as `with_candidates` does). Judged by the gates against `regime_switch` with the rule-based regime. One trial, added to the trial log; nothing is registered or promoted by it. A passing trial would lead to A5 (into `features`, a `quant retrain`, a shadow in the next forward-test run).

**If no feature passes**: the result is recorded here and round 1 ends; no redefinition or re-test of these four features on the same data.

**Report**: `reports/altdata/a4-round1-<timestamp>.md`, and the outcome written below, whatever it is.

**Amendment (2026-10-14, data fixes found while computing the features, before any return was joined to them).** A first computation of `altdata_features` showed values of `insider_net_buy_value_60d` up to 224 times the market cap and late insider histories for several stocks. Causes and fixes:

1. *Earlier CIKs.* Six more pool stocks moved to a new CIK at a reorganization, so their earlier filings were missing: GOOGL (Google Inc. 1288776), AVGO (Avago 1441634, Broadcom Ltd 1649338), DIS (old Walt Disney Co 1001039), LIN (Praxair 884905; LIN's price history also goes back through Praxair), BLK (old BlackRock 1364742), PSA (Public Storage Inc 318380). Each was verified through its submissions list and added to `altdata.sec.extra_ciks`; the quarters are reloaded. SE is not a CIK problem: as a foreign issuer its insiders filed no Form 4 before 2026.
2. *Wrong share counts.* Some XBRL values are off by orders of magnitude (PSA: 173 and 171,858 shares instead of about 175 million). A share count is not used if it is below 50% or above 200% of the median of the stock's counts with an as-of date within two years of it, itself included (excluding itself let two wrong values pull the median far enough to drop a correct count).
3. *The same trade in several filings.* Related entities file separate Form 4s for one transaction (META 2012: one sale in three filings). For `insider_net_buy_value_60d`, `P` / `S` lines with the same stock, transaction date, code, shares and price in filings that became available on the same session count once. The insider-counting features are unaffected (they count distinct owners).

The features, target, folds, tests and pass rules above are unchanged.

### 8.1.1 Stage 1 result (2026-10-14, after the pre-registration and its amendment were committed, a2d880c)

`quant altdata research`, report `reports/altdata/a4-round1-20261014T200853Z.md`; 11 folds 2014-09-26 .. 2025-10-01, 2,770 sessions, coverage 84-97% of the pool per fold.

```text
feature                     mean IC   t      folds same sign   result
insider_net_buy_value_60d   -0.0178   -2.32  7/11              PASS
insider_buyers_60d          -0.0041   -0.96  7/11              fail
opportunistic_sellers_60d   +0.0140   +1.90  7/11              fail
insider_sell_fraction_60d   +0.0077   +1.04  7/11              fail
```

- **The sign is the opposite of the informed-insider story**: more net insider *selling* went with *higher* 5-day excess returns, and the two sale features lean the same way. The likely reading is that insiders sell after rises (vesting after good results, sales into rallies), so the feature partly stands for recent price strength, which the 35 inputs already carry. Stage 2 tests whether anything is left.
- **The effect faded**: every fold of 2014-2020 is negative (-0.016 .. -0.061), folds 7-10 (2020-2024) are positive and fold 11 is about zero. The pass rests on exactly 7 of 11 folds, carried by the early years.
- One pass among 4 features (about 0.2 expected by chance) at t = -2.32: a candidate, not a finding.
- Expectation written before stage 2: no improvement over the 35 inputs, for both reasons above.

### 8.1.2 Stage 2 result and outcome of round 1 (2026-10-14)

`quant altdata research --stage 2`, report `reports/altdata/a4-round1-stage2-20261014T202350Z.md`. Logistic regression C = 1.0 with the 35 inputs plus `insider_net_buy_value_60d`; the 11 folds' validation windows unchanged, their training windows cut to the trial rows (299,107 of 369,391 rows, from 2009-04-30, when the XBRL share counts start); alongside, the same model without the feature on the same rows (descriptive).

```text
portfolio              CAGR   Sharpe  max DD   turnover  years > regime_switch  AUC     gates
with the feature       13.8%  0.94    -22.1%   17.3x     7/11                   0.5330  FAILED (Sharpe)
same rows, no feature  13.8%  0.94    -22.0%   17.4x     7/11                   0.5329  (descriptive)
regime_switch          13.2%  0.97    -25.0%    4.1x     -                      -       baseline
```

- **The feature adds nothing**: with and without it the model is the same to the fourth decimal of AUC (0.5330 vs 0.5329) and in every trading figure. What stage 1 found is already in the 35 inputs, as expected (insiders sell after rises, which the price features carry).
- The trial fails `beats_baseline_after_costs` (Sharpe 0.94 vs 0.97); the other gates pass, but the failure is the model's, not the feature's: it fails the same way without the feature.
- (The lower Sharpe than the 81-stock retrain of `proposal.md` 15.4.6, 1.04, comes from the training windows starting in 2009 instead of 2006, not from the feature.)

**Outcome of A4 round 1: no Form 4 feature is promoted; no A5.** Per the pre-registration, these four features are not redefined or re-tested on the same data. The data (`altdata_insider_*`, `altdata_features`) stays for reading and for any later, separately pre-registered question.

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
- **Insider selling** is frequent (about 139,000 sale lines 2006-2026 against 2,700 purchase lines) but mostly routine: diversification, taxes, 10b5-1 programs, founders' steady sales. The literature finds the information in the opportunistic part only, so a small effect at most is expected; a few large holders dominate the raw counts, which is why both features count insiders or fractions, not lines or dollars.
- **13D** events are too rare in this pool to test statistically (dropped, decision 7). The feature is kept for completeness; stage 1 will most likely be inconclusive.
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

## 13.1 A0 findings, EDGAR (2026-10-13)

User-Agent `mini-quant-trader Chenghao Zhang chenghao.zhang@outlook.de` (`altdata.user_agent`), at most 4 requests per second in the spike.

| Check | Finding | Consequence |
|---|---|---|
| Ticker -> CIK (`company_tickers.json`, 10,445 entries) | All 81 pool stocks map, BRK-B included (via `BRK-B`); no two pool tickers share a CIK | `altdata_entities` can be built from it; recheck on every update |
| Insider Transactions Data Sets | 83 quarterly zips, **2006 Q1 .. 2026 Q3**, 12-17 MB each (about 1.2 GB for the backfill; only pool rows are kept). The newest quarters moved to `/files/datastandardsinnovation/...`, older ones are under `/files/structureddata/...`: the downloader reads the links from the index page instead of building URLs | `insider_backfill_from: 2006-01-01` confirmed |
| Files and fields (2026 Q2) | `SUBMISSION`, `REPORTINGOWNER`, `NONDERIV_TRANS` (+ derivative tables, footnotes, signatures). All fields of section 3.1 exist: transaction date, code, shares, price, acquired/disposed, shares owned after, direct/indirect, owner relationship and title | Parser as planned |
| Filing time | The data sets have only `FILING_DATE` (a date, `30-JUN-2026` format), **no acceptance time**. The submissions API (`data.sec.gov/submissions/CIK##########.json`) has `acceptanceDateTime` (e.g. AAPL Form 4 filed 2026-10-08, accepted 2026-10-09T02:30:32Z, i.e. 22:30 ET, after the close) | **Backfill rows are available from the session after their filing date** (conservative: a filing on day D may be accepted until late evening). Incremental rows use the exact acceptance time (after 16:00 ET -> next session) |
| 10b5-1 flag | `AFF10B5ONE` on the **filing**, not per transaction line; values mixed (`true`/`false`/`1`/`0`), empty before the 2023 rule | Stored per filing, normalized to 1/0/NULL; before 2023 unknown (NULL), so `exclude_10b5_1_sales` can only apply from 2023 |
| Volume for the pool (2026 Q2) | 1,695 filings (1,635 Form 4); 4,517 non-derivative lines; codes S 3,039, A 499, M 392, F 365, G 85, **P 12** (NKE 5, LULU 2, BRK-B 2, ADBE, AVGO, CAT 1 each); sales concentrated (DELL 1,170 lines, SE 599). XOM had no filing that quarter | As expected in section 10: **insider purchases are very rare in this pool**, so `insider_buyers_60d` will be 0 almost always; the sale side is dominated by a few holders' programs |
| Form 4 XML (live) | `ownershipDocument` with `aff10b5One`, owner name, role and title, transaction lines; the raw XML is the primary document without its `xslF345X06/` prefix | Incremental parser works on the raw XML |
| 13D | A company's submissions list holds both 13Ds **about** it and 13Ds it **filed about others** (e.g. BRK-B, LLY as investors). The recent lists show 1-4 per company for 20 pool stocks since 2014, but which side they are on needs each filing's header (`SUBJECT COMPANY` vs `FILED BY`) | 13D needs the header check in A1; events about pool companies are rare (section 10 holds) |

Decided (section 15, decisions 6 and 7): the conservative availability rule above; A1 builds Form 4 only.

## 13.2 A1 as built (2026-10-13, Form 4 only)

- `quant altdata update [--max-quarters N] [--no-live] [--reload]` and `quant altdata coverage` (writes `reports/altdata/coverage-form4.md`); code in `quant/altdata/` (`tables.py`, `edgar.py`, `form4.py`, `cli.py`), registered in `quant/cli.py`. Tests: `tests/test_altdata_form4.py`, offline with a synthetic quarterly zip, a Form 4 XML and a fake SEC client.
- **Tables differ from section 4**: a Form 4 can be filed jointly by several owners for the same transactions (e.g. a fund and its manager), so the owner fields moved out of the transaction rows: `altdata_insider_filings` (one row per filing: document type, filing date, acceptance time when known, `available_on`, the 10b5-1 flag, source), `altdata_insider_owners` (per filing and owner: name, director / officer / 10% owner / other, title) and `altdata_insider_transactions` (per non-derivative line: date, code, shares, price, shares owned after, direct / indirect). `altdata_entities` (ticker -> CIK) and `altdata_sync` (one row per loaded quarter, plus the live updates) as planned.
- `available_on` is the session at whose close a filing may first be used: data-set filings the session after their filing date (decision 6), live filings the first session closing after their acceptance time.
- The 10b5-1 flag is per filing (`AFF10B5ONE` / `aff10b5One`), normalized to 1 / 0 / NULL.
- Amendments (4/A) are stored with `date_of_original`; linking them to the filing they amend (the data sets give only the original's date) is left to the feature step (A4).
- Live filings are fetched only once every published quarter is loaded, after the newest quarter's end. A first live run with only one quarter loaded reached back to 2006 through the submissions lists and was stopped before it stored anything; the rule above prevents that.
- A live filing keeps its exact acceptance time when its quarter's data set is loaded later. A filing whose issuer is not the pool company (e.g. a pool company reporting as an owner of another company) is skipped.
- **Earlier CIKs** (`altdata.sec.extra_ciks`): `company_tickers.json` maps XOM to a new holding company (ExxonMobil Holdings Corp, CIK 2115436), while all of Exxon's insider filings are under Exxon Mobil Corp (CIK 34088). The setting maps earlier CIKs to a pool stock; `XOM: [34088]`. The first backfill missed XOM; a `--reload` brought in 1,603 XOM filings (2006-01-04 .. 2026-07-01).
- **Result (2026-10-13)**: all 81 stocks, 83 quarters (2006 Q1 .. 2026 Q3) plus the live filings since 2026-09-30: 118,487 filings, 2,699 purchase lines (P), 139,671 sale lines (S), 771 lines quarantined (mostly sales without a price). Short histories because of recent listings: SPCX from 2026-06, SE (as a Form 4 filer) from 2026-03, FIG from 2025-08, SNDK from 2025-02, TEM from 2024-06, GEV from 2024-04. The 10b5-1 flag is present from 2023 (about 1 in 5 filings flagged since 2024). Coverage report: `reports/altdata/coverage-form4.md`. A1 is done.
- Quality (section 6): P / S lines without a positive price or shares, dated after their filing, or priced more than 50% away from that day's as-traded close go to `data_quarantine` (provider `sec-form4`) and are not stored.
- **Settings removed (2026-10-15, cleanup):** `altdata.sec.signal_codes` and `altdata.sec.exclude_10b5_1_sales`, which section 5 planned, were never read. The features always use P and S lines and always leave out 10b5-1 sales, as fixed by the A4 pre-registration (sections 7 and 8.1), so changing the settings had no effect. The rule is now stated only in the code (`quant/altdata/features.py`) and in sections 7 and 8.1.

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
6. **Availability of backfilled Form 4 rows** (2026-10-13, after A0): the data sets have no acceptance time, so a backfilled filing counts from the **session after its filing date**; incrementally fetched filings use their exact acceptance time (after 16:00 ET -> next session).
7. **A1 builds Form 4 only** (2026-10-13). Schedule 13D is dropped from A1: too rare in the pool, and it needs a header check to tell filings about a company from filings by it. `activist_13d_120d` leaves the candidate list of section 7.
8. **Sale-side features (confirmed 2026-10-14):** `opportunistic_sellers_60d` and `insider_sell_fraction_60d` are added to the candidate list of section 7, with the definitions there, before A4 and before any return was looked at. The list of section 7 is now fixed for A4: 9 features (8 original without `activist_13d_120d`, plus these two).
9. **Order (2026-10-14):** the phases run one after another, A2 (FINRA) and A3 (Wikipedia) before A4, so A4 tests all 9 features at once. This revises decision 1, which had deferred A2 and A3. A2 starts with its own A0 spike (FINRA history start dates), as A0 so far covered EDGAR only.
10. **A2 and A3 deferred again (2026-10-14, replaces decision 9).** FINRA and Wikipedia stay deferred, each started only on your go-ahead. **A4 (round 1) tests the four Form 4 features only**: `insider_net_buy_value_60d`, `insider_buyers_60d`, `opportunistic_sellers_60d`, `insider_sell_fraction_60d`, with the definitions of section 7, so the multiple-testing count of section 8 is 4 for this round. The five FINRA / Wikipedia features stay on the list; if those sources are built later, they get their own pre-registered round.
