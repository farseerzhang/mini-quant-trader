"""Tables of the broker snapshots (forward-test run 2 design, proposal Phase 10), on the shared metadata.

New tables only; nothing existing changes. One snapshot per broker and NYSE session: the account,
its positions, and the exchange rate used to express it in USD. Deposits and withdrawals are kept
separately so that they never count as gains or losses.
"""

from sqlalchemy import Column, Date, DateTime, Float, Integer, PrimaryKeyConstraint, String, Table

from quant.persistence.tables import metadata

broker_account_snapshots = Table(
    "broker_account_snapshots",
    metadata,
    Column("session", Date, nullable=False),  # the NYSE session the values belong to
    Column("broker", String, nullable=False),  # trading212 | ibkr
    Column("account_id", String),
    Column("account_currency", String, nullable=False),
    Column("fx_to_usd", Float, nullable=False),  # USD per unit of the account currency on the session
    Column("value_account_currency", Float, nullable=False),
    Column("cash_account_currency", Float),
    Column("value_usd", Float, nullable=False),
    Column("cash_usd", Float),
    Column("statement_date", String),  # IBKR's statement date; for Trading 212 the date read
    Column("fetched_at", DateTime, nullable=False),  # UTC
    PrimaryKeyConstraint("session", "broker", name="pk_broker_account_snapshots"),
)

broker_position_snapshots = Table(
    "broker_position_snapshots",
    metadata,
    Column("session", Date, nullable=False),
    Column("broker", String, nullable=False),
    Column("broker_ticker", String, nullable=False),  # e.g. FB_US_EQ, BMWd_EQ, BRK B
    Column("symbol", String, nullable=False),  # the project's symbol (META, BRK-B) or the broker's short name
    Column("name", String),
    Column("asset_type", String),  # STOCK, ETF, STK, ...
    Column("in_pool", Integer, nullable=False),  # 1 if the symbol is in market.symbols
    Column("quantity", Float, nullable=False),
    Column("price", Float),
    Column("price_currency", String),
    Column("value_usd", Float, nullable=False),
    PrimaryKeyConstraint("session", "broker", "broker_ticker", name="pk_broker_position_snapshots"),
)

broker_cash_flows = Table(
    "broker_cash_flows",
    metadata,
    Column("broker", String, nullable=False),
    Column("reference", String, nullable=False),  # the broker's own id of the transaction
    Column("happened_at", DateTime, nullable=False),  # UTC (IBKR: the date at 00:00)
    Column("session", Date, nullable=False),  # the first NYSE session on or after its date
    Column("kind", String, nullable=False),  # DEPOSIT | WITHDRAWAL
    Column("amount", Float, nullable=False),  # signed: + into the account, - out of it
    Column("currency", String, nullable=False),
    Column("amount_usd", Float, nullable=False),
    PrimaryKeyConstraint("broker", "reference", name="pk_broker_cash_flows"),
)
