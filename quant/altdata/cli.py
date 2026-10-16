"""`quant altdata ...` subcommands (altdata_proposal.md section 5); registered in quant/cli.py."""

from __future__ import annotations

import argparse
import logging
import sys

from quant.config.settings import Settings
from quant.persistence import queries
from quant.persistence.database import create_db_engine, init_db

log = logging.getLogger("quant.altdata")


def add_parser(commands, db_help: str) -> None:
    altdata = commands.add_parser(
        "altdata", help="free structured alternative data (research only; not part of run-daily): SEC Form 4")
    sub = altdata.add_subparsers(dest="altdata_command", required=True)
    update = sub.add_parser(
        "update", help="Form 4 insider transactions for the pool: quarterly data sets not loaded yet, then live "
        "filings after the newest quarter (first run: a backfill of about 1.2 GB of downloads)")
    update.add_argument("--max-quarters", type=int, metavar="N", help="load at most N quarterly data sets this run")
    update.add_argument("--no-live", action="store_true", help="only the quarterly data sets")
    update.add_argument("--reload", action="store_true", help="load every quarter again (replaces them)")
    update.add_argument("--db", help=db_help)
    report = sub.add_parser("coverage", help="Form 4 filings and P / S lines per stock and year (read-only)")
    report.add_argument("--db", help=db_help)
    features = sub.add_parser("features", help="rebuild altdata_features (the four Form 4 candidate features) "
                              "from the stored filings and shares outstanding")
    features.add_argument("--db", help=db_help)
    analysts = sub.add_parser(
        "analysts", help="analyst rating and price-target changes from Yahoo for the pool: stores new and changed "
        "rows only, with the date first and last seen (altdata_proposal.md 13.4; about 2 minutes)")
    analysts.add_argument("--symbol", action="append", metavar="TICKER", help="only this stock (repeatable)")
    analysts.add_argument("--db", help=db_help)
    research = sub.add_parser("research", help="A4: rank IC of the candidate features on the development folds, "
                              "then the model trial (pre-registered, altdata_proposal.md 8.1 and 8.2; no holdout)")
    research.add_argument("--round", type=int, choices=[1, 2], default=1, dest="round_no",
                          help="1: the Form 4 features (8.1); 2: the analyst features (8.2)")
    research.add_argument("--stage", type=int, choices=[1, 2], default=1,
                          help="1: rank IC per feature; 2: the pre-registered model trial with the features "
                          "that passed stage 1")
    research.add_argument("--db", help=db_help)


def _error(message: str) -> None:
    print(message, file=sys.stderr)
    log.error(message)


def _engine(settings: Settings, db_override: str | None):
    import quant.altdata.tables  # noqa: F401  registers the altdata tables before init_db

    engine = create_db_engine(db_override or settings.database.path, settings.database.busy_timeout_ms)
    init_db(engine)
    return engine


def run(args: argparse.Namespace, settings: Settings) -> int:
    if args.altdata_command == "update":
        return _run_update(args, settings)
    if args.altdata_command == "coverage":
        return _run_coverage(args, settings)
    if args.altdata_command == "features":
        return _run_features(args, settings)
    if args.altdata_command == "research":
        return _run_research(args, settings)
    if args.altdata_command == "analysts":
        return _run_analysts(args, settings)
    raise ValueError(f"unknown altdata command {args.altdata_command!r}")


def _run_update(args: argparse.Namespace, settings: Settings) -> int:
    from quant.altdata.edgar import SecClient, SecError
    from quant.altdata.form4 import coverage, update_form4, write_coverage

    if args.max_quarters is not None and args.max_quarters < 0:
        _error("--max-quarters must not be negative")
        return 1

    def progress(line: str) -> None:
        print(line, flush=True)
        log.info("altdata: %s", line.strip())

    client = SecClient(settings.altdata.user_agent, settings.altdata.sec.max_requests_per_second)
    engine = _engine(settings, args.db)
    try:
        try:
            result = update_form4(engine, settings, client, progress, live=not args.no_live,
                                  max_quarters=args.max_quarters, reload=args.reload)
        except SecError as exc:
            _error(f"SEC: {exc}")
            queries.record_pipeline_run(engine, "altdata-update", "form4", "FAILED", message=str(exc)[:500])
            return 1
        path = write_coverage(coverage(engine), settings.retrain.reports_dir)
        print(f"\nSummary: {len(result.quarters_loaded)} quarter(s) loaded, {len(result.quarters_failed)} failed; "
              f"{result.live_filings} live filing(s); {result.quarantined} line(s) quarantined; "
              f"{client.requests} request(s) to the SEC")
        print(f"Coverage: {path}")
        for key, error in result.quarters_failed:
            _error(f"  {key}: {error}")
        for error in result.errors[:10]:
            _error(f"  {error}")
        queries.record_pipeline_run(engine, "altdata-update", "form4", "OK" if result.ok else "FAILED",
                                    message=f"{len(result.quarters_loaded)} quarters, {result.live_filings} live, "
                                    f"{len(result.quarters_failed)} failed quarters, {len(result.errors)} errors")
        return 0 if result.ok else 1
    finally:
        engine.dispose()


def _run_features(args: argparse.Namespace, settings: Settings) -> int:
    from quant.altdata.features import FEATURES, compute_features, store_features

    engine = _engine(settings, args.db)
    try:
        frame = compute_features(engine, settings)
        rows = store_features(engine, frame, settings.altdata.version)
        filled = frame[list(FEATURES)].notna().mean()
        print(f"altdata_features v{settings.altdata.version}: {rows:,} rows "
              f"({frame['symbol'].nunique()} stocks, {frame['date'].nunique():,} sessions)")
        for name in FEATURES:
            print(f"  {name:28s} value on {filled[name]:.0%} of stock-sessions")
        queries.record_pipeline_run(engine, "altdata-features", "features", "OK", message=f"{rows} rows")
        return 0
    finally:
        engine.dispose()


def _run_research(args: argparse.Namespace, settings: Settings) -> int:
    from quant.altdata.features import load_features
    from quant.altdata.research import development_folds, stage1, write_report
    from quant.models.dataset import load_dataset

    engine = _engine(settings, args.db)
    try:
        dataset = load_dataset(engine, settings)
        labelled = dataset.labelled()
        target = labelled.frame["excess"].unstack("symbol")
        if args.round_no == 2:
            from quant.altdata.analyst_features import compute_analyst_features, percentile_ranks

            features = compute_analyst_features(engine, settings, target.index)
            if all(f.isna().all().all() for f in features.values()):
                _error("no analyst changes stored; run `quant altdata analysts` first")
                return 1
        else:
            features = load_features(engine, settings.altdata.version)
            if all(f.empty for f in features.values()):
                _error("altdata_features is empty; run `quant altdata features` first")
                return 1
        folds = development_folds(labelled.sessions, settings)
        results = stage1(features, target, folds, settings.model.target.horizon_days)
        if args.stage == 2:
            from quant.altdata.research import stage2, write_stage2_report

            passing = [r.name for r in results if r.passed]
            if not passing:
                print(f"No feature passes stage 1, so there is no stage 2 (pre-registration "
                      f"{'8.2' if args.round_no == 2 else '8.1'}).")
                return 0
            print(f"Stage 2 with {', '.join(passing)}: two walk-forward runs and their backtest (several minutes)",
                  flush=True)
            if args.round_no == 2:  # stage-2 inputs are each session's percentile ranks (8.2 (c))
                features = {name: percentile_ranks(features[name]) for name in passing}
            outcome = stage2(engine, settings, features, passing, round_no=args.round_no)
            report = write_stage2_report(outcome, settings.retrain.reports_dir)
            e = outcome.evidence
            print(f"  trial: Sharpe {e.sharpe:.2f} vs regime_switch {e.baseline_sharpe:.2f}, max DD "
                  f"{e.max_drawdown:.1%} vs {e.baseline_max_drawdown:.1%}, years higher {e.folds_higher_sharpe}/"
                  f"{e.folds}; gates {'PASSED' if outcome.gates.passed else 'FAILED'}")
            r = outcome.reference_evidence
            print(f"  same rows without the feature (descriptive): Sharpe {r.sharpe:.2f}, max DD "
                  f"{r.max_drawdown:.1%}, years higher {r.folds_higher_sharpe}/{r.folds}")
            print(f"  AUC {outcome.trial_auc:.4f} with the feature, {outcome.reference_auc:.4f} without")
            print(f"\n{report}")
            queries.record_pipeline_run(engine, "altdata-research", f"round{args.round_no}-stage2",
                                        "OK", message=f"{','.join(passing)}: gates "
                                        f"{'PASSED' if outcome.gates.passed else 'FAILED'}")
            return 0
        path = write_report(results, folds, settings.retrain.reports_dir, round_no=args.round_no)
    finally:
        engine.dispose()
    print(f"A4 round {args.round_no}, stage 1 ({len(folds)} folds, {folds[0].validation_start.date()} .. "
          f"{folds[-1].validation_end.date()}; pass = |t| >= 2 and >= 7 folds same sign):")
    for r in results:
        print(f"  {r.name:28s} mean IC {r.mean_ic:+.4f}  t {r.t_stat:+.2f}  folds {r.folds_same_sign}/{r.folds}  "
              f"{'PASS' if r.passed else 'fail'}")
    print(f"\n{path}")
    return 0


def _run_analysts(args: argparse.Namespace, settings: Settings) -> int:
    from quant.altdata.analysts import update_analysts

    symbols = [s.upper() for s in args.symbol] if args.symbol else None
    unknown = [s for s in symbols or [] if s not in settings.market.stocks]
    if unknown:
        _error(f"not in the stock pool: {', '.join(unknown)}")
        return 1

    def progress(line: str) -> None:
        print(line, flush=True)
        log.info("altdata analysts: %s", line.strip())

    engine = _engine(settings, args.db)
    try:
        count = len(symbols or settings.market.stocks)
        print(f"Analyst changes from Yahoo for {count} stock(s), one request each:", flush=True)
        result = update_analysts(engine, settings, symbols, progress=progress)
        print(f"\nSummary {result.day}: {len(result.symbols_ok)} stock(s) read, {result.seen:,} changes seen; "
              f"{result.new:,} new, {result.changed:,} changed, {result.gone:,} gone since the previous download")
        for symbol, error in result.failed.items():
            _error(f"  {symbol}: {error}")
        queries.record_pipeline_run(engine, "altdata-analysts", "analysts", "OK" if result.ok else "FAILED",
                                    message=f"{result.new} new, {result.changed} changed, {result.gone} gone, "
                                    f"{len(result.failed)} failed")
        return 0 if result.ok else 1
    finally:
        engine.dispose()


def _run_coverage(args: argparse.Namespace, settings: Settings) -> int:
    from quant.altdata.form4 import coverage, write_coverage

    engine = _engine(settings, args.db)
    try:
        frame = coverage(engine)
    finally:
        engine.dispose()
    path = write_coverage(frame, settings.retrain.reports_dir)
    if frame.empty:
        print("No Form 4 filings stored yet; run `quant altdata update`.")
    else:
        years = frame.groupby("year")[["filings", "P", "S"]].sum()
        print(years.to_string())
    print(f"\n{path}")
    return 0
