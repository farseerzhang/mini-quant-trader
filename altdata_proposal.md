# Free Structured Alternative Data — Proposal

Status (2026-10-19):
- **Built:** Form 4 insider trades (A1, section 13.2), analyst changes (13.4), option chains (13.7, the only source in `run-daily`).
- **Researched, nothing promoted:** A4 round 1, Form 4 (section 8.1.2); A4 round 2, analysts (8.2.1).
- **Deferred:** Schedule 13D (dropped, decision 7), FINRA and Wikipedia (decision 10).
- **Open:** a pre-registration after about 6 months of option data. The options timing check (13.7) is done: the evening read has the session's volume and the previous session's open interest.

First written 2026-10-13, approved the same day (section 15). This file was compacted on 2026-10-19: the plans for sources that were never built were shortened, and the spike notes trimmed to their findings. Section 8 (the pre-registrations and their results) and the sale-side definitions in section 7 are unchanged, word for word. The full earlier text is in git (commit a7480b8).

Companion to `sentiment_proposal.md`, which turns **text** into opinions and must be logged going forward. This one covers **already-structured records** with **years of history**, so they can be researched on the development folds now.

---

# 1. Goal

Free data sources as candidate stock features, tested with the discipline of the main proposal (sections 11 and 15). A new input is a hypothesis; it enters a model only after a pre-registered test on development data (section 8).

Planned on 2026-10-13: Form 4 insider trades and Schedule 13D activist stakes (SEC EDGAR), short-sale volume and short interest (FINRA), Wikipedia pageviews (Wikimedia). Added later from Yahoo: analyst rating and price-target changes (13.3), option chains (13.5) and earnings dates (13.6, spike only).

**Out of scope:** 13F holdings (quarterly, 45 days late); congressional trades (PDFs, late, mixed evidence); RSS feeds (text: the sentiment pipeline's job); paid data. (Analyst ratings were first listed here as having no free source; Yahoo's turned out usable, 13.3.)

---

# 2. Ground rule: available-at time

Every row has an **event date** and an **available-at time** (when the public could first see it). **Features use only rows available before the close of session T** (decision at the close of T, fill at the open of T+1). Every table stores both, and every feature filters on availability: this is how alternative data most often creates look-ahead.

| Source | Event | Counts from |
|---|---|---|
| Form 4, quarterly data sets | transaction date | the session after its filing date (no acceptance time in the data sets; decision 6) |
| Form 4, fetched live | transaction date | the first session closing after its EDGAR acceptance time |
| Analyst changes | Yahoo's UTC timestamp | the first session closing after it (13.3) |
| Option chains | the session | read after that session's close, before the next open (13.7) |
| 13D (deferred) | 5% threshold crossed | acceptance time (due within 5 business days since 2024-02, 10 days before) |
| FINRA short volume (deferred) | trade date | published that evening: session T uses up to T-1 |
| FINRA short interest (deferred) | settlement date | its publication date, about a week later, never the settlement date |
| Wikipedia pageviews (deferred) | UTC day | the next day: session T uses UTC days up to T-1 |

---

# 3. Sources

## 3.1 SEC EDGAR — Form 4 (built)

- **Backfill** from the SEC's quarterly Insider Transactions Data Sets (2006 Q1 on); **live** filings from each company's EDGAR submissions list after the newest quarter.
- A `User-Agent` with name and email is required; at most 10 requests per second (the loader keeps to 5).
- Ticker -> CIK from `company_tickers.json`, checked on every update, plus earlier CIKs of reorganized companies (`altdata.sec.extra_ciks`, 13.2 and 8.1).
- One row per transaction line. Only codes **P** (open-market purchase) and **S** (open-market sale) are decisions about the stock; grants, exercises, tax withholding and gifts are stored but not used. Sales under a 10b5-1 plan (flagged per filing since 2023) are left out of the sale features.

## 3.2 SEC EDGAR — Schedule 13D (dropped, decision 7)

Too rare in a pool of large caps, and a company's filing list mixes 13Ds about it with 13Ds it filed about others (13.1).

## 3.3 FINRA — short-sale volume (deferred)

Daily Reg SHO files. They cover only trades reported to FINRA (mostly off-exchange), and "short volume" includes market makers' short sales, typically 40-50% of the reported volume. Only changes of the ratio against the stock's own history could carry information.

## 3.4 FINRA — short interest (deferred)

Shares held short at each mid-month and month-end settlement date. Derived: % of shares outstanding (the XBRL count available at the time) and days to cover. How far back the free series goes was never checked.

## 3.5 Wikimedia — pageviews (deferred)

Daily views per article (`agent=user`, English Wikipedia), from July 2015. Ticker -> article needs a hand-made map (names are ambiguous: "Apple", "Oracle"), with renames checked.

---

# 4. Database (as built)

Tables in the same SQLite file (`quant/altdata/tables.py`), every name starting with `altdata_`; no existing table changes.

| Table | Content |
|---|---|
| `altdata_entities` | ticker -> CIK |
| `altdata_insider_filings`, `altdata_insider_owners`, `altdata_insider_transactions` | Form 4: one row per filing, per filing and owner, per transaction line (13.2) |
| `altdata_shares_outstanding` | XBRL `dei:EntityCommonStockSharesOutstanding` per stock and date (for market caps, 8.1) |
| `altdata_features` | the four Form 4 features of round 1 (8.1), kept apart from `features` |
| `altdata_analyst_changes` | analyst changes, one row per version of a change (13.4) |
| `altdata_option_summaries`, `altdata_option_contracts` | option chains, per session (13.7) |
| `altdata_sync` | one row per loaded quarter, live update, analyst download or options read |

Raw tables are the source of truth: features can always be rebuilt from them. A feature that passes section 8 would move into `features`, with a `features.version` bump (A5). The tables planned for the deferred sources (`altdata_activist_filings`, `altdata_short_volume`, `altdata_short_interest`, `altdata_wikipedia_pageviews`) were never created.

---

# 5. CLI (as built)

```bash
quant altdata update [--max-quarters N] [--no-live] [--reload]   # Form 4 (13.2)
quant altdata coverage                                           # Form 4 filings and P / S lines per stock and year
quant altdata features                                           # rebuild altdata_features (round 1)
quant altdata research [--round 1|2] [--stage 1|2]                # A4 (section 8)
quant altdata analysts [--symbol T ...]                          # analyst changes (13.4), weekly
quant altdata options [--compare] [--symbol T ...]               # option chains (13.7), stage 7 of run-daily
```

Every run writes a `pipeline_runs` row. Only the option chains are in `run-daily`; any other source joins it only when a feature built from it is promoted.

---

# 6. Data quality

- **Form 4:** P / S lines without a positive price or shares, dated after their filing, or priced more than 50% away from that day's as-traded close go to `data_quarantine` (provider `sec-form4`); 771 lines in the first backfill. Share counts off by orders of magnitude are dropped (8.1, amendment point 2).
- **Analyst changes:** zero targets and empty text stored as NULL; repeated rows counted once.
- **Option chains:** volume 0 stored as NULL (no trade); IVs outside 1%..500% are not used in the summaries.
- **Coverage reports** before any research: `quant altdata coverage` for Form 4; the readiness checks of 8.2 for analysts.

---

# 7. Candidate features

Computed per stock at the close of each session T from rows available by that close. The list was fixed **before** any result (section 8); windows are in trading sessions.

| Feature | Definition | Status |
|---|---|---|
| `insider_net_buy_value_60d` | value of P minus value of non-10b5-1 S by officers and directors, 60 sessions, / market cap | round 1: stage 1 pass, stage 2 nothing added |
| `insider_buyers_60d` | distinct officers / directors with a P in 60 sessions ("cluster buying") | round 1: fail |
| `opportunistic_sellers_60d` | distinct officers / directors with an opportunistic sale in 60 sessions (below) | round 1: fail |
| `insider_sell_fraction_60d` | the largest fraction of direct holdings any officer or director sold in 60 sessions (below) | round 1: fail |
| ~~`activist_13d_120d`~~ | dropped with 13D (decision 7) | - |
| `short_volume_ratio_z` | (5-session mean of short / total volume - its 120-session mean) / its 120-session std | deferred |
| `short_interest_pct`, `short_interest_change`, `days_to_cover` | short interest / shares outstanding; its change between the last two reports; short interest / 20-session average volume | deferred |
| `pageviews_z` | (log 5-day mean views - log 60-day mean) / std of the daily log changes | deferred |

The analyst features of round 2 are defined in section 8.2.

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

## 8.2 Pre-registration, A4 round 2: analyst price targets and ratings (registered 2026-10-16, before any feature value or result is computed)

Written after the A0b spike (section 13.3) and the loader (13.4). Only counts, coverage and timestamps have been looked at; no feature value has been computed and no return has been joined to analyst data.

**Data.**
- `altdata_analyst_changes` **as of the download of 2026-10-16**: the rows with `first_seen` <= 2026-10-16 <= `last_seen`, i.e. that download's view. Later downloads therefore cannot change the research input.
- That history is Yahoo's view today, not point in time (13.3, problem 4); this is accepted for this round and stated in the report.
- Each change counts from its `available_on` session: the first session closing after its UTC timestamp.
- If a firm has several changes for the same stock that become available on the same session, only the latest by timestamp counts.

**Features (3)**, per pool stock and NYSE session T, from changes with `available_on` <= T only. Windows are in sessions.

1. `target_revision_20d`: the mean of ln(current target / prior target) over the changes that became available in the 20 sessions up to and including T and that have both targets, each value clipped to [-ln 2, +ln 2] (a halving or doubling at most).
   - **Idea:** analysts raising or cutting their targets (the strongest documented analyst signal for returns).
   - 0 when the stock had no such change in the window.
2. `net_rating_changes_20d`: the number of `up` minus the number of `down` actions among the changes that became available in the 20 sessions up to T.
   - **Idea:** rating changes.
   - 0 when there were none.
3. `target_upside`: for each firm, its latest change with a current target that became available in the 250 sessions up to T. The feature is the median of those targets divided by the as-traded close of T, minus 1.
   - **Idea:** the analysts' implied upside (with no clear sign in the literature).
   - NULL when no firm has a target in the window.

- **Stock splits.** Targets are in as-traded dollars.
  - For `target_upside`, a target set before a split is divided by the split ratios between its session and T, from the stored `corporate_actions`.
  - For `target_revision_20d`, a change that becomes available within 5 sessions after a split of the stock is left out, because its prior target may still be the unadjusted one.
- **Before a stock's first change in the table, all three are NULL.** This includes stocks with a known gap in history (GE before 2021-10, LIN before 2024-11).
- **Counts grow fourfold over the years** (13.3, problem 2):
  - Stage 1 is a cross-sectional rank IC per session, so a level shift that is common to all stocks does not matter.
  - In stage 2, each feature enters the model as its **cross-sectional percentile rank on that session** (0..1, ties averaged), not as the raw value.

**Target, folds, stage 1** exactly as in round 1 (section 8.1):
- the label's `excess` column;
- the same 11 development folds, with the holdout start fixed at 2025-10-10 (validation windows 2014-09-26 .. 2025-10-01);
- daily rank IC on sessions with at least 10 stocks that have a value, with the `ic_summary` t-statistic on every 5th session;
- **Pass** = |t| >= 2 **and** at least 7 of 11 folds with the overall sign.
- The direction is not fixed in advance: either sign can pass, as in round 1.
- With 3 features tested, about 0.15 passes are expected by chance.
- Reported per fold: the coverage, as sessions and stocks with a value.

**Stage 2 (only for features that pass):**
- One trial, as round 1's stage 2: logistic regression C = 1.0 with the 35 inputs plus the passing features as percentile ranks; everything else is unchanged (label, `ml_select`, 81 stocks, folds, embargo, `retrain.gates`).
- The training windows are cut to the trial rows, which start in 2012 at the earliest. The validation windows are kept.
- Alongside, the same model without the features on the same rows, as a descriptive reference.
- One trial, added to the trial log; nothing is registered or promoted. A pass would lead to A5.
- **No holdout** (section 8, point 4).

**If no feature passes:** the result is recorded here and round 2 ends, with no redefinition or re-test of these features on this data. Other windows, other normalizations or the rating-level features would each be a new pre-registration.

**Expectation, stated before any result:** most likely no improvement. Large caps absorb analyst news within hours, analysts tend to raise targets after rises (so the price features already carry part of it), and the data is not point in time.

**Report:** `reports/altdata/a4-round2-<timestamp>.md`, and the outcome written here, whatever it is.

**Confirmed as drafted (2026-10-16):**
- (a) the three features and their definitions, including the 20- and 250-session windows, the ±ln 2 clip and the split rules;
- (b) the data as of the 2026-10-16 download;
- (c) percentile ranks as the stage-2 inputs;
- (d) the protocol and pass rule of round 1, unchanged.

**As built (2026-10-16, before any stage is run):**
- **Code:** the features are in `quant/altdata/analyst_features.py`, computed in memory (`altdata_features` keeps round 1's columns only).
  - `AS_OF` = 2026-10-16 fixes the download.
  - The split rules use `corporate_actions` `SPLIT` rows. Their ratio is the share multiple, e.g. 10 for NVDA on 2024-06-10. GE's spin-off adjustments are stored as splits too (1.281, 1.253) and are applied the same way.
- **Shared with round 1:** stage 1 and stage 2 are the round-1 code in `quant/altdata/research.py`, now taking the round number. The reports are `a4-round2-*.md`, and the trial is named `ml_lr_analysts`.
- **Command:** `quant altdata research --round 2 [--stage 1|2]`.
- **Readiness check** on the real data, coverage only: share of stock-sessions with a value over 2014-09-26 .. 2025-10-01: 87% (revision and rating features), 84% (`target_upside`). The features compute in about 2 s.
- **Tests:** `tests/test_altdata_analyst_features.py` covers:
  - the windows, NULL before the first change, and the clip;
  - the rating count, the median of each firm's latest target and the 250-session expiry;
  - split adjustment and exclusion, percentile ranks, and the as-of view with one change per firm and session.

### 8.2.1 Stage 1 result and outcome of round 2 (2026-10-16)

`quant altdata research --round 2`, report `reports/altdata/a4-round2-20261016T113155Z.md`. 11 folds 2014-09-26 .. 2025-10-01, 2,770 sessions; coverage 81-98% of the pool per fold (76-98% for `target_upside`).

```text
feature                  mean IC   t      folds same sign   result
target_revision_20d      +0.0079   +1.07  7/11              fail
net_rating_changes_20d   +0.0006   +0.38  4/11              fail
target_upside            -0.0014   -0.07  5/11              fail
```

- **No feature passes**, so there is no stage 2.
- `target_revision_20d` has the expected sign (raised targets, slightly higher 5-day excess returns) and 7 of 11 folds, but t = 1.07: far from the pre-registered 2. Its folds swing both ways (-0.021 .. +0.034).
- The rating count and the implied upside are indistinguishable from zero.
- This fits the expectation written before the run: in large caps, analyst news is priced before the next session's open, which is when the label starts.

**Outcome of A4 round 2: no analyst feature is promoted; no stage 2, no A5.**
- Per the pre-registration, these three features are not redefined or re-tested on this data (other windows, normalizations, or a shorter target).
- The loader and its table stay. Weekly downloads keep building a point-in-time record, which a later, separately pre-registered question could use, for example on forward data only.

---

# 9. Tests

All offline (fixtures, fake clients, in-memory databases); no test calls the SEC or Yahoo.

- `tests/test_altdata_form4.py`: the quarterly data sets and live Form 4 XML, the availability rules, backfill then live (idempotent), earlier CIKs, quarantine, the SEC rate limit and retries.
- `tests/test_altdata_features.py`: the round-1 features (the 60-session window, the three-year wait, routine / opportunistic and the sell fraction, one trade in several filings, 10b5-1 sales, share classes and wrong counts) and the A4 wiring (stage 2 rows and folds, the pass rule).
- `tests/test_altdata_analysts.py`, `tests/test_altdata_analyst_features.py`: the analyst loader (13.4) and the round-2 features (8.2).
- `tests/test_altdata_options.py`: the options loader (13.7); the run-daily tests replace its stage with a stand-in.

The key look-ahead tests: a filing accepted after 16:00 ET is not visible at that session's close; a change counts from the first session closing after it; an options read during the session is refused.

---

# 10. Expectations

Priors written on 2026-10-13, before any data was seen:

- **Insider buying** is the best-documented effect, but mostly in small and mid caps. In 81 large caps, purchases are rare (executives are paid in stock and sell), so `insider_buyers_60d` would be zero most of the time.
- **Insider selling** is frequent but mostly routine (diversification, taxes, 10b5-1 plans, founders' steady sales); the information is in the opportunistic part only, so a small effect at most.
- **13D** events are too rare in this pool to test.
- **Short interest** predicts lower returns, mostly in smaller stocks; large caps rarely have high short interest. The **short-volume ratio** is noisy (market makers).
- **Pageviews** measure attention, linked to short-term moves and later reversals; no sign assumed.

The most likely outcome was one or no passing feature, which still closes the question cheaply with free data. (Round 1: one stage-1 pass that added nothing; round 2: none.)

---

# 11. Configuration (as built)

```yaml
altdata:
  version: 1
  user_agent: "mini-quant-trader <name> <email>"   # sent to the SEC with every request
  sec:
    max_requests_per_second: 5                      # the SEC allows 10
    insider_backfill_from: 2006-01-01
    extra_ciks: {XOM: [34088], GOOGL: [1288776], ...}   # earlier CIKs of a pool stock (13.2, 8.1)
```

No API keys. The analyst and options loaders have no settings: their constants are in their modules.

---

# 12. Code layout (as built)

```
quant/altdata/
  tables.py              section 4
  edgar.py, form4.py     SEC client; Form 4 data sets, live filings, coverage (13.2)
  shares.py              XBRL shares outstanding
  features.py            round-1 features (8.1)
  analysts.py            analyst loader (13.4)
  analyst_features.py    round-2 features (8.2)
  options.py             option chains loader (13.7)
  research.py            A4 stage 1 and 2, both rounds
  cli.py                 `quant altdata ...`, registered in quant/cli.py
```

No new dependencies beyond the project's (`requests`, `yfinance`, pandas, `xml.etree`).

---

# 13. Phases

| Phase | Content | Status |
|---|---|---|
| A0 | Spikes: endpoints, formats, history | EDGAR done (13.1); analysts (13.3), options (13.5), earnings dates (13.6) |
| A1 | EDGAR loader | Form 4 done (13.2); 13D dropped |
| A2, A3 | FINRA, Wikipedia | deferred (decision 10) |
| A4 | Pre-registered research (section 8) | round 1 (Form 4) and round 2 (analysts): nothing promoted |
| A5 | Integration of a passing feature | none |

## 13.1 A0 findings, EDGAR (2026-10-13)

- **Ticker -> CIK:** all 81 pool stocks map in `company_tickers.json` (10,445 entries), BRK-B included; no shared CIKs.
- **Insider Transactions Data Sets:** 83 quarterly zips, 2006 Q1 .. 2026 Q3, 12-17 MB each (about 1.2 GB; only pool rows are kept). The newest quarters moved to another path, so the loader reads the links from the index page. Every field of section 3.1 is there.
- **Filing time:** the data sets have only `FILING_DATE`, no acceptance time; the submissions API has `acceptanceDateTime` (e.g. an AAPL Form 4 filed 2026-10-08, accepted 22:30 ET). Hence decision 6.
- **10b5-1 flag:** per filing (`AFF10B5ONE`), mixed values, empty before 2023: normalized to 1 / 0 / NULL, so the exclusion applies from 2023 only.
- **Volume (2026 Q2):** 1,635 Form 4s, 4,517 lines: S 3,039, A 499, M 392, F 365, G 85, **P 12**. Purchases are very rare; sales are concentrated (DELL 1,170 lines, SE 599).
- **13D:** a company's list holds 13Ds about it and 13Ds it filed about others; telling them apart needs each filing's header (decision 7).

## 13.2 A1 as built (2026-10-13, Form 4 only)

- **Tables differ from the first plan:** a Form 4 can be filed jointly by several owners (e.g. a fund and its manager), so owners are a separate table (section 4).
- `available_on` is the session at whose close a filing may first be used (section 2). Amendments (4/A) are stored with `date_of_original`, used like any filing.
- Live filings are fetched only once every published quarter is loaded (a first live run reached back to 2006 and was stopped before storing anything). A live filing keeps its exact acceptance time when its quarter is loaded later. Filings whose issuer is not the pool company are skipped.
- **Earlier CIKs:** `company_tickers.json` maps XOM to a new holding company (CIK 2115436), while Exxon's filings are under 34088; `--reload` brought in 1,603 XOM filings. Six more were found in A4 (8.1, amendment point 1).
- **Result:** 81 stocks, 2006 Q1 .. 2026 Q3 plus live filings: 118,487 filings, 2,699 P lines, 139,671 S lines, 771 lines quarantined. Short histories from recent listings: SPCX (2026-06), SE as a Form 4 filer (2026-03), FIG, SNDK, TEM, GEV. About 1 in 5 filings flagged 10b5-1 since 2024.
- **Removed 2026-10-15:** the settings `altdata.sec.signal_codes` and `exclude_10b5_1_sales` were never read; the rule (P and S, 10b5-1 sales out) is fixed by sections 7 and 8.1 and the code.

## 13.3 A0b spike: analyst rating and price-target changes (2026-10-16; data only, no returns looked at)

- **Source:** `yfinance` 1.7.0 `Ticker.upgrades_downgrades` (Yahoo's web data, personal use). Fields: `GradeDate` (to the second), `Firm`, `FromGrade`, `ToGrade`, `Action` (main / up / down / reit / init), `priceTargetAction`, `currentPriceTarget`, `priorPriceTarget`. (`analyst_price_targets` is today's consensus only; `recommendations` covers months only.)
- **Coverage:** 38,836 rows, 213 firms, 2011-12-08 .. 2026-10-15, all 81 stocks. Three in four rows are "maintain"; targets raised 19,912, lowered 9,687. Rows per year grow from about 1,200 (2012, 62 stocks) to 5,000 (2024, 78 stocks).
- **Timestamps are UTC**, checked against dated news (a Benzinga roundup at 09:15 ET lists changes Yahoo stamps 11:53-12:41, i.e. pre-market if UTC; TheFly published an NFLX downgrade two hours before Yahoo's time). Yahoo's time can be late, never early in these checks, which is safe against look-ahead.
- **Problems:** history from 2012 only; counts not comparable over time (features must be relative); missing histories (GE before 2021-10, LIN before 2024-11, BRK-B 22 rows); **not point in time** (today's view of the past); 2-32% of rows without a target, depending on the year.
- **Assessment:** usable for one pre-registered round from 2012, mainly the price-target revision; low prior. Led to the loader (13.4) and round 2 (8.2).

## 13.4 Analyst changes loader (designed and built 2026-10-16)

- **Changes, not copies:** a full download is 2.8 MB and 39,000 rows; each change (stock, timestamp, firm) is stored once per version of its content, with `first_seen` and `last_seen` (download dates). New growth: about 5,000 changes a year.
- A new change is inserted; the same content only moves `last_seen`; different content is a new version; a change no longer there keeps its row and is counted as "gone"; a failed stock changes nothing.
- **Why:** from the first download (2026-10-16) on, `first_seen` makes the data point in time; a change appearing later with an old timestamp is a Yahoo backfill; versions show how much Yahoo revises. The first download's history stays today's view.
- **As built:** `quant/altdata/analysts.py`, table `altdata_analyst_changes` (`content_hash` per version, `available_on` = the first session closing after `grade_time`). A full download logs one `altdata_sync` row (`analysts-yahoo`); a `--symbol` run does not. About 2 minutes (1 s between stocks); **weekly**, not in `run-daily`.

## 13.5 A0c spike: option chains, "unusual options activity" (2026-10-18; data only, no returns looked at)

The question (owner): can sudden large option bets on one stock be detected, as a sign that someone knows something?

- **Source:** `yfinance` 1.7.0 `Ticker.options` and `option_chain(expiry)`. Per contract: strike, last price, bid, ask, that day's volume, open interest, Yahoo's IV, last trade time.
- **Not there:** no history (only today's chains), and **no trades or side**: one volume per contract and day, which can be buying or selling, opening or closing, or a spread's leg. Trade-level "large order" signals need paid data (OPRA trades, flow services).
- **Coverage** (expiries within 60 days, read Sunday for Friday 2026-10-16): all 81, 19 s, 46,316 contracts (PSA 12 .. NVDA 2,973), 7.3 MB a day as CSV, too much to keep whole. Thin option markets: PSA traded 99 contracts that day, AEP 426, SHW 477.
- **Quality:** 43% of contracts did not trade that day (stale last price and IV: use traded contracts or the bid/ask mid); 6.7% open interest zero (median per stock).
- **Measures, once:** put/call volume median 0.76; volume / OI 0.24; ATM call IV 40% (AAPL's 29.5% matches a hand Black-Scholes check); skew (OTM put IV - ATM call IV) +1.6%. **Yahoo's call-minus-put IV spread is +2.3% for most stocks**, where the literature finds it slightly negative: Yahoo's IVs break put-call parity, so this measure needs our own IVs.
- **Problems:** (1) **earnings** raise option volume and IV for every stock; (2) **timing:** when is Yahoo's volume final, and is the evening's open interest the previous session's (OCC publishes the next morning)?; (3) contract counts change over time: ratios or ranks only; (4) **mega caps:** informed option trading is documented mostly in small stocks and takeover targets; (5) collection starts now, so a weak first test after 6-12 months.
- **Decided (owner, 2026-10-18):** the timing check and an earnings-dates spike (13.6), and a loader (13.7); a pre-registration after about 6 months of data.

## 13.6 A0d spike: earnings dates (2026-10-18; data only, no returns looked at)

- **Source:** `yfinance` `get_earnings_dates(limit=100)` (Yahoo's maximum); all 81 in 15 s. Date and time (New York), EPS estimate, reported EPS, surprise; the next report included.
- **Coverage:** 100 quarters for most stocks (median first date 2002-04-17; 59 of 81 from 2006 or earlier); recent listings from their listing. About 63% of reports before the open, 35% after the close (the time decides the first session that can react). Gaps in PEP and COST; duplicate days for 18 stocks; no next date for HD and LOW.
- **55 of the 81 report within the next 30 days**, so the 30-day IVs of 13.5 are mostly earnings IVs now.
- **Assessment:** good enough to mark the sessions around each report. Not point in time (future dates move). A loader that keeps each download's next date, as in 13.4, only when an options feature is designed.

## 13.7 Option chains loader (designed and built 2026-10-18)

**Stored per session** (the session the chains belong to):
- `altdata_option_summaries`, one row per stock: spot (the stored as-traded close; Yahoo's price only if none), counts of expiries (within 60 days), contracts and contracts traded, call and put volume and open interest, and Yahoo's IVs on the expiry nearest 30 calendar days, from traded contracts only: ATM call and put (strike / spot 0.95..1.05), OTM put (0.80..0.95), OTM call (1.05..1.20).
- `altdata_option_contracts`, at most 24 per stock, all of Yahoo's fields, each flagged with why it was kept:
  - **the 30-day grid:** calls and puts at the strikes nearest 85, 90, .., 115% of spot (14), so our own IVs, skew and call-put spread can be computed later;
  - **the unusual:** the 5 largest volumes, and the 5 largest volume / open interest with a volume of at least 100 (any expiry within 60 days).

**When a read counts:** only between a session's close and the next open (a weekend read is the last session's); a read during the session is refused. **The first stored read of a session is kept**, so the stored values are what was known that evening; a stock missing from it is filled in later. `--compare` reads again, stores nothing and prints what moved.

**In `run-daily`:** stage 7, after the forward test; about 20 s; never stops the run. One `altdata_sync` row per read (`options-yahoo`, key = the session). Research only: nothing live reads these tables. Tests: section 9.

**First read** (Sunday 2026-10-18, for 2026-10-16): 81 stocks, 1,663 contracts; 312 KB of database growth, about 79 MB a year. Found, for the feature design:
- Fewer grid contracts than 14 x 81: on stocks with few strikes, the nearest strike repeats.
- **The largest volume / open interest are mostly new contracts** (NVDA's five: weeklies with open interest 0-12 against volumes of 1,300-3,600). A feature on this ratio needs a minimum open interest or contract age.
- **Volume alone misleads:** AAPL's 19,000 puts at $0.01 are $19,000 in premium; a premium (volume x price x 100) is the better size measure.

**Timing check, to run:** `quant altdata options --compare` on Monday 2026-10-19 about an hour after `run-daily`, and on Tuesday 2026-10-20 before 13:30 UTC. Unchanged volume with changed open interest on Tuesday would confirm that the evening read has the session's final volume and the previous session's open interest.

**Timing check, result (2026-10-19 / 10-20):** the stored read of 2026-10-19 (20:40 UTC, 40 minutes after the close) was compared three times with `--compare`:

```text
compare read              total volume   total open interest   contracts found   volume changed   OI changed
2026-10-19 evening        unchanged      unchanged             1,737 of 1,737    0                0
2026-10-20 07:34 UTC      8.5M (-46%)    0.28M (-99%)          1,676 of 1,737    0                1,554
2026-10-20 12:33 UTC      7.6M (-52%)    39.5M (+2.2%)         1,676 of 1,737    11               1,227
```

- **The evening read has the session's final volume:** unchanged later that evening, and unchanged on every contract found again at 07:34.
- **Its open interest is the previous session's:** by 12:33 open interest had changed on 1,227 of 1,676 contracts (total +2.2%). So the stored open interest of session T is the open interest at the end of T-1; the open interest of T is only known the next morning. A volume / open interest ratio from one read is therefore volume(T) over open interest(T-1).
- **The 61 contracts not found again** were all in high-volume names (AAPL, AMD, AMZN, AVGO, GOOGL, META, MSFT, MU, NVDA, SPCX, TSLA): same-day contracts that expired on 2026-10-19. They explain the drop in total volume.
- **Overnight reads are not usable.** At 07:34 Yahoo's open interest was broken or half-loaded (total -99%, zeros for AMZN, FIG, INTC, TEM). At 12:33 volume had changed on 11 contracts and fallen for some stocks without a kept contract changing (e.g. PLTR), so Yahoo seems to start resetting volume before the open.
- **Consequence for the loader (not changed yet):** it accepts any read between the close and the next open and keeps the first one stored. A `run-daily` run the next morning, after a missed evening, would store broken open interest or reduced volume for that session. A fix (e.g. store option reads only within a few hours after the close) would be planned here before it is built.

**Read-window guard (planned and built 2026-10-20).** Store a session's option read only within `altdata.options_max_hours_after_close` (new, default **4**) hours after that session's close.
- **Why 4 hours:** the reads known to be good were 40 minutes after the close and once more later that evening; 07:34 UTC the next morning (11.5 hours) was broken. 4 hours (until 00:00 UTC after a normal close, 21:00 UTC after an early close) covers a normal evening `run-daily` with room for a late one, and stays well before the overnight reset. The close comes from the NYSE calendar (`session_close`), so early closes count.
- **Where:** `options_session` in `quant/altdata/options.py` gets a second refusal next to "the market is open": `too late for the session of <date> (its read window closed at <time> UTC); no option data is stored for it`. `update_options` raises it before any request, so nothing is read from Yahoo.
- **`--compare` is not limited by it:** it stores nothing, so it stays allowed at any time outside market hours (needed for checks like the one above). Only storing is refused.
- **In `run-daily` (stage 7):** a late run prints that line and the stage counts as skipped, not failed; as today it never stops the run.
- **No backfill:** a session whose window was missed stays without option rows, as a missed forward-test session does.
- **The read already stored for 2026-10-16** was made on Sunday 2026-10-18 at 15:15 UTC, 43 hours after its close, before this was known. It is not changed or deleted. Research uses only reads whose `read_at` is inside the window (computed from the stored `read_at` and the calendar), so it is left out there; this is stated in the options pre-registration.
- **Tests:** a read 3 hours after the close is stored; 5 hours after is refused before any fetch; an early close (e.g. the day after Thanksgiving) uses its own close; a weekend read of Friday's session is refused; `--compare` still works outside the window; run-daily's stage 7 reports a late read as skipped (with the option stage patched, as in `tests/test_cli_daily.py`).
- **Documents:** this section, `config/settings.yaml` and the README's option section.
- **As built (2026-10-20), as planned:** `session_close` in `quant/data/calendar.py`; `options_session(now, calendar, max_hours_after_close)` refuses a late read only when the limit is given, which `update_options` does and `compare_options` does not. The refusal is an `OutsideWindowError`, so the existing handling applies: the command prints `Option chains not read: ...` and returns 0, and `run-daily`'s stage 7 goes on. A missing stock is filled in only inside the window too. Tests in `test_altdata_options.py`; the earlier gap-fill test now fills the gap 3.5 hours after the close instead of on Monday morning.

## 13.8 Quarantine clean-up: Form 4 duplicates and the status warning (planned and built 2026-10-19)

**Found (2026-10-19, read-only look at `data_quarantine`):** 2,511 rows, all from the last 7 days, so `quant status` warns every day. 2,373 are Form 4 lines (`sec-form4`, research only), 138 are prices (`yfinance`).
- **Duplicates:** `store_filings` replaces a source key's filings on every load, but `queries.quarantine_rows` only appends. The loads of 2026-10-13 and 10-14 quarantined the same lines again: about 1,540 rows repeat an earlier one exactly (provider, symbol, date, reason, raw). The price loader does the same on a re-read (SPY 2026-10-16 twice).
- **The quarantined Form 4 lines are correctly kept out:** preferred shares, depositary shares and the BRK class A price (price more than 50% from the close, 1,321), and lines with no price (932, mostly COST 2006). Nothing to change in `quality_problem`.
- **The price rows are expected:** 126 `EURUSD=X` days (known since 2026-10-17, run-2 design in `proposal.md`), 9 `^IRX` days of zero yield in 2020, and three session-day bars read right after the close that the next morning's read replaced with valid ones.

**Plan:**
1. **No exact duplicates.** `queries.quarantine_rows` skips a row whose (provider, symbol, date, reason, raw) is already stored, for every caller (Form 4 and prices). It checks in code, one query per call; no unique index, so `data_quarantine` is not altered. It returns the number of rows actually added; the loaders' messages ("N quarantined") keep counting the lines found, so a re-load still reports them.
2. **The status warning counts prices only.** `queries.count_quarantined` gets an optional `provider`; `quant status` passes `market.provider` (`yfinance`) for the 7-day warning. The status line shows both parts: `quarantined rows   138 prices, 2373 altdata in total; 0 prices this week`.
3. **Existing duplicates are left alone.** Removing the about 1,540 repeated rows is deleting rows, so it is a separate step only with your explicit yes (after `backup-database`); the plan works without it.
4. **Tests** (in-memory database): a second identical quarantine call adds nothing; a row differing only in `raw` or `reason` is added; status warns on a new price row and not on a new `sec-form4` row; the status line's two counts.

**As built (2026-10-19):** steps 1, 2 and 4 as planned; step 3 not done (the existing duplicates stay). `quarantine_rows` compares dates as the Date column stores them; the dashboard caption shows the same two counts. Tests in `test_store.py` and `test_status.py`; full suite 486 OK. On the real database `status` now reads "142 prices, 2369 altdata in total; 142 prices this week": the warning is the price rows of 2026-10-12 .. 10-17 (126 of them `EURUSD=X`) and ends once they are older than 7 days (after 2026-10-24).

---

# 14. Risks

| Risk | Mitigation |
|---|---|
| Look-ahead through publication lags | availability on every row; look-ahead tests (section 9) |
| Too many features tried | fixed lists before any result; |t| >= 2 plus fold consistency; stage 2 through the existing gates and trial log |
| Large caps dilute effects documented in small caps | stated as a prior (section 10); a null result is acceptable |
| Data not point in time (Yahoo) | stated in each pre-registration; the loaders' first-seen dates build a point-in-time record from now on |
| SEC blocks the client | User-Agent with contact, 5 requests/s, backoff |
| A free source changes format | each source behind its own parser and tests; a failed source does not stop the others |

---

# 15. Decisions

1. (2026-10-13) Scope: EDGAR first; FINRA and Wikipedia deferred. *(Revised by 9, then 10.)*
2. (2026-10-13) The SEC User-Agent was asked before the first request.
3. (2026-10-13) Module `quant/altdata/`.
4. (2026-10-13) Timing: after sentiment S1.
5. (2026-10-13) Every table name starts with `altdata_`.
6. (2026-10-13, after A0) **Backfilled Form 4 filings count from the session after their filing date**; live filings from their exact acceptance time.
7. (2026-10-13) **A1 builds Form 4 only**; 13D is dropped, and `activist_13d_120d` with it.
8. (2026-10-14) **The sale-side features** `opportunistic_sellers_60d` and `insider_sell_fraction_60d` are added, with the definitions of section 7, before any return was looked at.
9. (2026-10-14) FINRA and Wikipedia before A4, so A4 tests all 9 features. *(Replaced by 10.)*
10. (2026-10-14) **FINRA and Wikipedia deferred again; A4 round 1 tests the four Form 4 features only** (multiple-testing count 4). The five FINRA / Wikipedia features would get their own pre-registered round.

Later decisions are recorded where they were made: round 2 (8.2, confirmed 2026-10-16) and the options work (13.5, 2026-10-18).
