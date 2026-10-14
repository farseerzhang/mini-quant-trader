"""Tables of the alternative data module (altdata_proposal.md section 4), on the shared metadata.

Registered when this module is imported, like quant/sentiment/tables.py; every name starts
with `altdata_`. A1 has Form 4 only (decision 7). A Form 4 can be filed jointly by several
reporting owners (e.g. a fund and its manager) for the same transactions, so owners and
transaction lines are separate tables, both keyed by the filing's accession number.

Times are stored as naive UTC; `available_on` is the session at whose close a filing may
first be used (altdata_proposal.md section 2, decision 6).
"""

from sqlalchemy import Column, Date, DateTime, Float, Integer, PrimaryKeyConstraint, String, Table, Text

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

# What has been loaded: one row per quarterly data set, and one for the live updates.
altdata_sync = Table(
    "altdata_sync",
    metadata,
    Column("source", String, nullable=False),  # form4-dataset | form4-live
    Column("key", String, nullable=False),  # 2026q2 | live
    Column("status", String, nullable=False),  # OK | FAILED
    Column("filings", Integer, nullable=False),
    Column("lines", Integer, nullable=False),
    Column("message", Text),
    Column("fetched_at", DateTime, nullable=False),
    PrimaryKeyConstraint("source", "key", name="pk_altdata_sync"),
)
