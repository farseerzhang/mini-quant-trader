"""Tables of the alternative data module (altdata_proposal.md section 4), on the shared metadata.

Registered when this module is imported, like quant/sentiment/tables.py; every name starts
with `altdata_`. A1 has Form 4 only (decision 7). A Form 4 can be filed jointly by several
reporting owners (e.g. a fund and its manager) for the same transactions, so owners and
transaction lines are separate tables, both keyed by the filing's accession number.

Times are stored as naive UTC; `available_on` is the session at whose close a filing may
first be used (altdata_proposal.md section 2, decision 6).
"""

from sqlalchemy import Column, Date, DateTime, Float, Integer, PrimaryKeyConstraint, String, Table, Text, UniqueConstraint

from quant.persistence.tables import metadata

# Ticker -> SEC CIK, from company_tickers.json, checked on every update.
altdata_entities = Table(
    "altdata_entities",
    metadata,
    Column("symbol", String, primary_key=True),
    Column("cik", Integer, nullable=False),
    Column("name", String),
    Column("updated_at", DateTime, nullable=False),
)

altdata_insider_filings = Table(
    "altdata_insider_filings",
    metadata,
    Column("accession_no", String, primary_key=True),
    Column("symbol", String, nullable=False),
    Column("cik", Integer, nullable=False),
    Column("document_type", String, nullable=False),  # 4 | 4/A
    Column("period_of_report", Date),
    Column("date_of_original", Date),  # amendments: the filing date of the original
    Column("filing_date", Date, nullable=False),
    Column("accepted_at", DateTime),  # live filings only; the quarterly data sets have no time
    Column("available_on", Date, nullable=False),
    Column("is_10b5_1", Integer),  # 1 / 0; NULL before the 2023 rule or when not reported
    Column("source", String, nullable=False),  # dataset (quarterly zip) | live (EDGAR, per filing)
    Column("source_key", String, nullable=False),  # e.g. 2026q2, or live
    Column("ingested_at", DateTime, nullable=False),
)

altdata_insider_owners = Table(
    "altdata_insider_owners",
    metadata,
    Column("accession_no", String, nullable=False),
    Column("owner_cik", Integer, nullable=False),
    Column("owner_name", String),
    Column("is_director", Integer, nullable=False),
    Column("is_officer", Integer, nullable=False),
    Column("is_ten_percent_owner", Integer, nullable=False),
    Column("is_other", Integer, nullable=False),
    Column("officer_title", String),
    PrimaryKeyConstraint("accession_no", "owner_cik", name="pk_altdata_insider_owners"),
)

altdata_insider_transactions = Table(
    "altdata_insider_transactions",
    metadata,
    Column("accession_no", String, nullable=False),
    Column("line_no", Integer, nullable=False),  # order of the non-derivative lines in the filing
    Column("security_title", String),
    Column("transaction_date", Date),
    Column("transaction_code", String),  # P purchase, S sale, A grant, M exercise, F tax, G gift, ...
    Column("acquired_disposed", String),  # A | D
    Column("shares", Float),
    Column("price", Float),
    Column("shares_owned_after", Float),
    Column("direct_indirect", String),  # D | I
    PrimaryKeyConstraint("accession_no", "line_no", name="pk_altdata_insider_transactions"),
)

# Shares outstanding from SEC XBRL company facts (dei:EntityCommonStockSharesOutstanding), for
# market capitalization (section 8.1). One row per reported value; several share classes of the
# same filing and date are summed when used.
altdata_shares_outstanding = Table(
    "altdata_shares_outstanding",
    metadata,
    Column("symbol", String, nullable=False),
    Column("cik", Integer, nullable=False),
    Column("accession_no", String, nullable=False),
    Column("as_of", Date, nullable=False),  # the date the count refers to ("end")
    Column("filed", Date, nullable=False),
    Column("seq", Integer, nullable=False),  # several values for one filing and date: one per class
    Column("shares", Float, nullable=False),
    Column("form", String),
    PrimaryKeyConstraint("symbol", "accession_no", "as_of", "seq", name="pk_altdata_shares_outstanding"),
)

# Candidate features (section 7, pre-registration 8.1), per stock and session, kept apart from
# `features`; only a feature that passes section 8 moves there (A5).
altdata_features = Table(
    "altdata_features",
    metadata,
    Column("symbol", String, nullable=False),
    Column("date", Date, nullable=False),
    Column("altdata_version", Integer, nullable=False),
    Column("insider_net_buy_value_60d", Float),
    Column("insider_buyers_60d", Float),
    Column("opportunistic_sellers_60d", Float),
    Column("insider_sell_fraction_60d", Float),
    PrimaryKeyConstraint("symbol", "date", "altdata_version", name="pk_altdata_features"),
)

# Analyst rating and price-target changes from Yahoo (section 13.4): one row per version of a
# change, identified by stock, timestamp and firm. A row is never updated except `last_seen`;
# a change whose content differs in a later download gets a new row, a change that disappears
# keeps its row (its `last_seen` stops). `first_seen` is the first download that had it.
altdata_analyst_changes = Table(
    "altdata_analyst_changes",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("symbol", String, nullable=False),
    Column("grade_time", DateTime, nullable=False),  # Yahoo's GradeDate, naive UTC (section 13.3)
    Column("firm", String, nullable=False),
    Column("from_grade", String),
    Column("to_grade", String),
    Column("action", String),  # main | up | down | reit | init
    Column("target_action", String),  # Raises | Lowers | Announces | Maintains | Removes | Adjusts
    Column("current_target", Float),  # NULL when Yahoo gives none (0)
    Column("prior_target", Float),
    Column("content_hash", String, nullable=False),
    Column("available_on", Date, nullable=False),  # the first session closing after grade_time
    Column("first_seen", Date, nullable=False),
    Column("last_seen", Date, nullable=False),
    UniqueConstraint("symbol", "grade_time", "firm", "content_hash", name="uq_altdata_analyst_changes"),
)

# Option chains from Yahoo (section 13.7), one read per session: the first stored read of a
# session is kept, so a row is what was known that evening. IVs are Yahoo's, from contracts
# traded that session, on the expiry nearest 30 calendar days.
altdata_option_summaries = Table(
    "altdata_option_summaries",
    metadata,
    Column("session", Date, nullable=False),
    Column("symbol", String, nullable=False),
    Column("read_at", DateTime, nullable=False),  # naive UTC
    Column("spot", Float, nullable=False),
    Column("expiries", Integer, nullable=False),  # within 60 days
    Column("contracts", Integer, nullable=False),
    Column("traded_contracts", Integer, nullable=False),  # volume > 0 that session
    Column("call_volume", Float, nullable=False),
    Column("put_volume", Float, nullable=False),
    Column("call_open_interest", Float, nullable=False),
    Column("put_open_interest", Float, nullable=False),
    Column("iv_expiry", Date),  # the expiry nearest 30 days; NULL when nothing traded
    Column("atm_call_iv", Float),  # strike / spot 0.95 .. 1.05
    Column("atm_put_iv", Float),
    Column("otm_put_iv", Float),  # 0.80 .. 0.95
    Column("otm_call_iv", Float),  # 1.05 .. 1.20
    PrimaryKeyConstraint("session", "symbol", name="pk_altdata_option_summaries"),
)

# Per stock and session, at most 24 contracts with all of Yahoo's fields: the 30-day grid (the
# strike nearest 85 .. 115% of spot, calls and puts) and the day's unusual contracts.
altdata_option_contracts = Table(
    "altdata_option_contracts",
    metadata,
    Column("session", Date, nullable=False),
    Column("contract_symbol", String, nullable=False),  # e.g. NVDA261019C00220000
    Column("symbol", String, nullable=False),
    Column("expiry", Date, nullable=False),
    Column("side", String, nullable=False),  # call | put
    Column("strike", Float, nullable=False),
    Column("bid", Float),
    Column("ask", Float),
    Column("last_price", Float),
    Column("last_trade_at", DateTime),  # naive UTC
    Column("volume", Float),  # NULL: no trade that session
    Column("open_interest", Float),
    Column("yahoo_iv", Float),
    Column("in_grid", Integer, nullable=False),  # 1: one of the 14 grid contracts
    Column("top_volume", Integer, nullable=False),  # 1: among the 5 largest volumes
    Column("top_volume_over_oi", Integer, nullable=False),  # 1: among the 5 largest volume / OI (volume >= 100)
    PrimaryKeyConstraint("session", "contract_symbol", name="pk_altdata_option_contracts"),
)

# What has been loaded: one row per quarterly data set, and one for the live updates.
altdata_sync = Table(
    "altdata_sync",
    metadata,
    Column("source", String, nullable=False),  # form4-dataset | form4-live | analysts-yahoo | options-yahoo
    Column("key", String, nullable=False),  # 2026q2 | live | the download date
    Column("status", String, nullable=False),  # OK | FAILED
    Column("filings", Integer, nullable=False),
    Column("lines", Integer, nullable=False),
    Column("message", Text),
    Column("fetched_at", DateTime, nullable=False),
    PrimaryKeyConstraint("source", "key", name="pk_altdata_sync"),
)
