"""`quant broker ...` subcommands; registered in quant/cli.py. Read-only; nothing is stored yet."""

from __future__ import annotations

import argparse
import sys

from quant.config.settings import Settings


def add_parser(commands, db_help: str) -> None:
    broker = commands.add_parser("broker", help="read-only access to your Trading 212 and IBKR accounts "
                                 "(credentials from .env; never places orders)")
    sub = broker.add_subparsers(dest="broker_command", required=True)
    check = sub.add_parser("check-connection", help="read each configured account once and print a summary "
                           "(positions, cash, currency); stores nothing")
    check.add_argument("--only", choices=["trading212", "ibkr"], help="check one broker only")
    sync = sub.add_parser(
        "sync-accounts", help="store today's snapshot of each account (positions, cash, value in USD) and any new "
        "deposits or withdrawals. Trading 212: after the US close, before 07:00 UTC; IBKR: its latest statement "
        "(the previous session). `quant run-daily` runs it every evening")
    sync.add_argument("--only", choices=["trading212", "ibkr"], help="sync one broker only")
    sync.add_argument("--db", help=db_help)


def run(args: argparse.Namespace, settings: Settings) -> int:
    if args.broker_command == "check-connection":
        return _check_connection(args, settings)
    if args.broker_command == "sync-accounts":
        return _sync_accounts(args, settings)
    raise ValueError(f"unknown broker command {args.broker_command!r}")


def _sync_accounts(args: argparse.Namespace, settings: Settings) -> int:
    import quant.broker.tables  # noqa: F401  registers the broker tables before init_db
    from quant.broker.sync import sync_accounts
    from quant.persistence import queries
    from quant.persistence.database import create_db_engine, init_db

    engine = create_db_engine(args.db or settings.database.path, settings.database.busy_timeout_ms)
    init_db(engine)
    try:
        print("Reading the broker accounts (read-only)...", flush=True)
        result = sync_accounts(engine, settings, args.only)
        for line in result.stored:
            print(f"  stored {line}")
        for note in result.notes:
            print(f"  {note}")
        print(f"  {result.flows_added} new deposit(s) / withdrawal(s) stored")
        for broker, error in result.failed.items():
            print(f"  {broker}: FAILED: {error}", file=sys.stderr)
        queries.record_pipeline_run(engine, "broker-sync-accounts", "sync", "FAILED" if result.failed else "OK",
                                    message="; ".join(result.stored + [f"{b} failed" for b in result.failed]))
        return 1 if result.failed else 0
    finally:
        engine.dispose()


def _check_connection(args: argparse.Namespace, settings: Settings) -> int:
    from quant.broker.ibkr_flex import IbkrFlexClient, IbkrFlexCredentials, IbkrFlexError
    from quant.broker.trading212 import Trading212Client, Trading212Credentials, Trading212Error, trading212_symbol

    pool = set(settings.market.stocks)
    failed = False
    if args.only in (None, "trading212"):
        print("Trading 212 (read-only; requests spaced about 5 s apart):", flush=True)
        try:
            account = Trading212Client(Trading212Credentials.from_environment()).account()
        except (Trading212Error, OSError) as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
            failed = True
        else:
            symbols = [trading212_symbol(p.ticker) for p in account.positions]
            in_pool = sum(s in pool for s in symbols)
            print(f"  account currency {account.currency}; value {_money(account.cash_total)}, free cash "
                  f"{_money(account.cash_free)}; {len(account.positions)} position(s), {in_pool} in the stock pool")
            for position, symbol in sorted(zip(account.positions, symbols), key=lambda pair: pair[1])[:40]:
                print(f"    {position.ticker:16s} -> {symbol:8s} {position.quantity:>12,.4f}  price "
                      f"{_money(position.current_price)}{'' if symbol in pool else '  (not in the pool)'}")
    if args.only in (None, "ibkr"):
        print("IBKR Flex Web Service (read-only report; generation can take up to a minute):", flush=True)
        try:
            account = IbkrFlexClient(IbkrFlexCredentials.from_environment()).account()
        except (IbkrFlexError, OSError) as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
            failed = True
        else:
            stocks = [p for p in account.positions if p.asset_category in (None, "STK")]
            in_pool = sum(p.symbol.replace(" ", "-") in pool for p in stocks)
            print(f"  base currency {account.base_currency}; report date {account.report_date}; cash "
                  f"{_money(account.ending_cash_base)}; {len(account.positions)} position(s) "
                  f"({len(stocks)} stocks, {in_pool} in the stock pool)")
            for p in sorted(account.positions, key=lambda p: p.symbol)[:40]:
                print(f"    {p.symbol:10s} {p.asset_category or '':4s} {p.quantity:>12,.4f}  value "
                      f"{_money(p.position_value)} {p.currency or ''}")
    return 1 if failed else 0


def _money(value) -> str:
    return "n/a" if value is None else f"{value:,.2f}"
