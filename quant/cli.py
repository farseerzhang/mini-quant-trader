import argparse
import logging
import sys
import time
from datetime import date

import pandas as pd

from quant.backtest.engine import run_backtests
from quant.backtest.persist import save_backtest
from quant.backtest.report import development_report, holdout_report, metrics_table
from quant.config.settings import DEFAULT_CONFIG_PATH, ConfigError, Settings, load_settings
from quant.data.store import sync_market_data, sync_symbol_table
from quant.data.yfinance_provider import MarketDataError, YFinanceProvider
from quant.features.pipeline import calculate_and_store_features
from quant.forward.real import TradesError
from quant.forward.report import build_report, format_report
from quant.forward.freeze import freeze_session
from quant.forward.paper import update_paper_equity
from quant.logs import close_logging, configure_logging
from quant.models.ml_backtest import BASELINE
from quant.models.compare import comparison_table, format_comparison, write_comparison
from quant.models.retrain import IMPLEMENTED_MODELS, run_retrain
from quant.ops.promote import FALLBACK, PromotionError, promote_model, retire_model
from quant.ops.status import collect_status, format_status
from quant.persistence import queries
from quant.persistence.database import create_db_engine, init_db
from quant.regime.active import calculate_and_store_active_regime
from quant.regime.pipeline import calculate_and_store_regimes
from quant.regime.retrain import run_regime_retrain
from quant.altdata import cli as altdata_cli
from quant.broker import cli as broker_cli
from quant.sentiment import cli as sentiment_cli
from quant.signals.pipeline import generate_and_store_signals
from quant.signals.positions import PositionsError
from quant.strategies import build_strategies


log = logging.getLogger("quant.cli")


def _error(message: str) -> None:
    """Print to stderr and log: every refusal and failure the user sees is also in the log."""
    print(message, file=sys.stderr)
    log.error(message)


def _iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a date in YYYY-MM-DD format") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quant")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"path to the settings file (default: {DEFAULT_CONFIG_PATH})",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    db_help = "SQLite database path (overrides database.url in the settings file)"

    update_command = commands.add_parser(
        "update-market-data",
        help="incrementally update market.symbols and market.market_symbols through the last completed session",
    )
    update_command.add_argument(
        "--initial-date",
        type=_iso_date,
        help="start date for symbols not yet stored (overrides market.initial_history)",
    )
    update_command.add_argument("--db", help=db_help)

    feature_command = commands.add_parser(
        "calculate-features",
        help="calculate and persist versioned technical and market features",
    )
    feature_command.add_argument("--db", help=db_help)

    regime_command = commands.add_parser(
        "calculate-regime",
        help="calculate and persist the rule-based BULL/SIDEWAYS/BEAR regime from stored features",
    )
    regime_command.add_argument("--db", help=db_help)

    trial_command = commands.add_parser(
        "regime-bond-trial",
        help="the pre-registered one-time trial of section 12.5: bond-market fear inputs in the rule score "
        "(research only; writes reports/regime/bond-trial-*.md, stores nothing)",
    )
    trial_command.add_argument("--db", help=db_help)

    combo_command = commands.add_parser(
        "regime-combo-trial",
        help="the pre-registered one-time trial of section 13.3: the rule-based regime with an HMM BEAR trigger "
        "(research only; writes reports/regime/combo-trial-*.md, stores nothing)",
    )
    combo_command.add_argument("--db", help=db_help)

    holdout_command = commands.add_parser(
        "regime-rule-v2-holdout",
        help="the pre-registered one-time holdout check of rule_based-v2 (section 12.5.3; holdout evaluation #3; "
        "writes reports/regime/rule-v2-holdout-*.md, stores nothing)",
    )
    holdout_command.add_argument("--db", help=db_help)

    rate_shocks_command = commands.add_parser(
        "regime-rate-shocks-trial",
        help="the pre-registered one-time trial of section 12.7: interest-rate shock inputs (3-month yield drop, "
        "10-year yield jump) in the rule score (research only; writes reports/regime/rate-shocks-trial-*.md, "
        "stores nothing)",
    )
    rate_shocks_command.add_argument("--db", help=db_help)

    strategy_trial_command = commands.add_parser(
        "strategy-trial",
        help="a pre-registered one-time strategy trial of section 17.1.5 against regime_switch "
        "(research only; writes reports/strategies/<trial>-trial-*.md, stores nothing)",
    )
    strategy_trial_command.add_argument("trial", choices=list(STRATEGY_TRIALS),
                                        help="dual-momentum: trial A (SPY / IEF / cash, monthly); "
                                        "momentum-regime: trial B (12-1 momentum top 20 inside the regime timing); "
                                        "volatility-target: trial C (regime_switch scaled to a 15%% volatility target); "
                                        "sector-rotation: trial D (top 3 SPDR sector ETFs inside the regime timing); "
                                        "top3-momentum: trial E (the 3 strongest pool stocks of the last 12 months, "
                                        "quarterly; section 17.1.6)")
    strategy_trial_command.add_argument("--db", help=db_help)

    backtest_command = commands.add_parser(
        "backtest",
        help="backtest the rule-based strategies against SPY, equal weight and a random baseline, and save the results",
    )
    backtest_command.add_argument(
        "--strategy",
        action="append",
        metavar="NAME",
        help="run only this strategy: trend, mean_reversion, regime_switch, or regime_switch_bear50 (one "
        "regime_switch_bear<N> per extra strategies.regime_switch.bear_exposures value). Repeatable; default: "
        "all. Benchmarks (SPY, equal weight, random baseline) always run",
    )
    backtest_command.add_argument(
        "--evaluate-holdout",
        action="store_true",
        help="also show and save the holdout period; every evaluation is logged (section 15.2)",
    )
    backtest_command.add_argument("--db", help=db_help)

    retrain_command = commands.add_parser(
        "retrain",
        help="MANUAL: search, validate walk-forward, evaluate the gates and register a CANDIDATE "
        "(never changes an active model)",
    )
    retrain_command.add_argument("--role", choices=["predictor", "regime"], required=True)
    retrain_command.add_argument(
        "--model",
        help="predictor type to train: logistic_regression, random_forest, xgboost, or all (each in turn, then a "
        "comparison table); default model.type in the settings file. The regime role trains the HMM"
    )
    retrain_command.add_argument(
        "--evaluate-holdout",
        action="store_true",
        help="also trade the holdout and check its gate; every evaluation is logged (section 15.2)",
    )
    retrain_command.add_argument("--db", help=db_help)

    signals_command = commands.add_parser(
        "generate-signals",
        help="BUY / HOLD / REDUCE / EXIT for the latest session from the stored regime and features and "
        "the positions file, with reasons; saved in the signals table",
    )
    signals_command.add_argument("--reasons", action="store_true", help="also print the reason of every signal")
    signals_command.add_argument("--db", help=db_help)

    promote_command = commands.add_parser(
        "promote",
        help="MANUAL: make a candidate the ACTIVE model of its role (the previous one is RETIRED); "
        "refuses a candidate whose gates failed unless --override is given",
    )
    promote_command.add_argument("model_id", help="model_id from model_versions (also the report's file name)")
    promote_command.add_argument(
        "--override", metavar="REASON", help="promote although the gates failed; the reason is logged"
    )
    promote_command.add_argument("--db", help=db_help)

    retire_command = commands.add_parser(
        "retire",
        help="MANUAL: set an ACTIVE model to RETIRED; its role falls back to the rule-based version",
    )
    retire_command.add_argument("model_id", help="the ACTIVE model to retire (see `quant status`)")
    retire_command.add_argument("--reason", help="why; logged with the change")
    retire_command.add_argument("--db", help=db_help)

    freeze_command = commands.add_parser(
        "forward-test",
        help="forward test: freeze tonight's targets of live and the shadows, update the paper and real values "
        "(also the last stage of run-daily)",
    )
    freeze_command.add_argument(
        "--start", action="store_true",
        help="ONCE, before the next open: start a forward-test run with your Trading 212 and IBKR accounts, "
        "combined in USD, as real (both brokers' snapshots of the latest session are needed: run "
        "`quant run-daily` after the close, and once more the next day for IBKR's statement)",
    )
    freeze_command.add_argument(
        "--end", action="store_true",
        help="end the running forward test (kept, never deleted; needs --reason); then --start begins a new run",
    )
    freeze_command.add_argument("--reason", help="why the run is ended (logged with it)")
    freeze_command.add_argument("--db", help=db_help)

    forward_command = commands.add_parser(
        "forward-report",
        help="forward test: every portfolio (live, shadows, your account, SPY) since the start (read-only)",
    )
    forward_command.add_argument("--run", type=int, help="show this run (default: the running one)")
    forward_command.add_argument("--db", help=db_help)

    status_command = commands.add_parser(
        "status",
        help="last runs, data freshness, the active model per role and warnings (read-only)",
    )
    status_command.add_argument("--db", help=db_help)

    backup_command = commands.add_parser(
        "backup-database",
        help="copy the database to database.backup_directory (checked, dated; keeps the newest "
        "database.backups_to_keep copies). Safe while the dashboard is open",
    )
    backup_command.add_argument("--db", help=db_help)

    daily_command = commands.add_parser(
        "run-daily",
        help="update market data, calculate features and the regime, then generate signals "
        "(stops at the first failed stage)",
    )
    daily_command.add_argument(
        "--initial-date",
        type=_iso_date,
        help="start date for symbols not yet stored (overrides market.initial_history)",
    )
    daily_command.add_argument("--reasons", action="store_true", help="also print the reason of every signal")
    daily_command.add_argument("--db", help=db_help)

    sentiment_cli.add_parser(commands, db_help)
    altdata_cli.add_parser(commands, db_help)
    broker_cli.add_parser(commands, db_help)
    return parser


def _provider(settings: Settings) -> YFinanceProvider:
    if settings.market.provider != "yfinance":
        raise ConfigError(f"unsupported market.provider: {settings.market.provider!r}")
    return YFinanceProvider(request_timeout_seconds=settings.market.request_timeout_seconds)


def _engine(settings: Settings, db_override: str | None):
    engine = create_db_engine(
        db_override or settings.database.path, settings.database.busy_timeout_ms
    )
    init_db(engine)
    return engine


# Outcome of the market-data stage: some symbols failing still leaves usable data.
_OK, _PARTIAL, _FAILED = "ok", "partial", "failed"


def _run_update(args: argparse.Namespace, settings: Settings) -> int:
    return 0 if _update_market_data(args, settings) == _OK else 1


def _update_market_data(args: argparse.Namespace, settings: Settings) -> str:
    engine = _engine(settings, args.db)
    try:
        sync_symbol_table(engine, settings.market.sectors, settings.market.market_symbols)
        report = sync_market_data(
            engine,
            _provider(settings),
            settings.market.universe,
            initial_date=args.initial_date or settings.market.initial_history,
            provider_name=settings.market.provider,
            calendar_name=settings.market.calendar,
        )
    except (MarketDataError, ValueError) as exc:
        _error(f"Market data update failed: {exc}")
        return _FAILED
    finally:
        engine.dispose()
    print(f"Market data updated through {report.as_of}")
    for summary in report.summaries:
        print(
            f"{summary.symbol}: fetched {summary.fetched_prices}, stored {summary.stored_prices}, "
            f"quarantined {summary.quarantined}; {summary.stored_actions} new corporate actions"
        )
    for symbol in report.skipped:
        print(f"{symbol}: already current")
    for failure in report.failures:
        _error(f"{failure.symbol}: FAILED: {failure.error}")
    return _PARTIAL if report.failures else _OK


def _run_features(args: argparse.Namespace, settings: Settings) -> int:
    engine = _engine(settings, args.db)
    try:
        summary = calculate_and_store_features(engine, settings.features)
    except ValueError as exc:
        _error(f"Feature calculation failed: {exc}")
        return 1
    finally:
        engine.dispose()
    print(
        f"Calculated features through {summary.data_as_of} for {summary.symbols} symbols "
        f"({summary.symbol_rows} symbol rows, {summary.market_rows} market rows; "
        f"versions {summary.feature_version}/{summary.market_feature_version})."
    )
    return 0


def _run_regime(args: argparse.Namespace, settings: Settings) -> int:
    engine = _engine(settings, args.db)
    try:
        summary = calculate_and_store_regimes(engine, settings.regime, settings.features)
        active = calculate_and_store_active_regime(engine, settings)
    except ValueError as exc:
        _error(f"Regime calculation failed: {exc}")
        return 1
    finally:
        engine.dispose()
    counts = ", ".join(f"{name} {count}" for name, count in summary.sessions_by_regime.items())
    print(
        f"Calculated {summary.rows} {summary.model_version} regimes "
        f"from {summary.first_date} through {summary.last_date}."
    )
    print(f"Latest: {summary.latest_regime} (score {summary.latest_score:+d})")
    print(f"Sessions: {counts}; average dwell {summary.average_dwell_days:.1f} sessions")
    if active is not None:
        print(f"ACTIVE regime model {active.model_id}: {active.rows} regimes through {active.last_date.date()}, "
              f"latest {active.latest_regime} (probability {active.latest_probability:.0%}); the signals use it")
    return 0


def _run_bond_trial(args: argparse.Namespace, settings: Settings) -> int:
    from datetime import datetime, timezone

    from quant.regime.bond_trial import V1, V2, run_bond_trial, write_trial_report
    from quant.regime.comparison import portfolio_name

    engine = _engine(settings, args.db)
    try:
        print("Section 12.5 trial: v1 and v2 regimes, regime_switch with each (a few minutes)", flush=True)
        outcome = run_bond_trial(engine, settings)
        path = write_trial_report(outcome, settings.retrain.reports_dir, datetime.now(timezone.utc))
        verdict = "PASSED" if outcome.gates.passed else "FAILED"
        queries.record_pipeline_run(engine, command="regime-bond-trial", stage="trial", status="OK",
                                    message=f"section 12.5: gates {verdict}")
    except ValueError as exc:
        _error(f"Trial failed to run: {exc}")
        return 1
    finally:
        engine.dispose()
    v = outcome.comparison.versus(portfolio_name("regime_switch", V2), portfolio_name("regime_switch", V1))
    print(f"regime_switch Sharpe {v['sharpe']:.4f} with {V2} vs {v['baseline_sharpe']:.4f} with {V1}; "
          f"higher in {v['folds_higher_sharpe']}/{v['folds']} years; gates {verdict}")
    for result in outcome.gates.results:
        print(f"  {result.name:38s} {result.status:5s} {result.observed}")
    print(f"\n{path}")
    return 0


def _run_combo_trial(args: argparse.Namespace, settings: Settings) -> int:
    from datetime import datetime, timezone

    from quant.regime.combo_trial import COMBO, V1, run_combo_trial, write_combo_report
    from quant.regime.comparison import portfolio_name

    engine = _engine(settings, args.db)
    try:
        print("Section 13.3 trial: walk-forward HMM (11 yearly fits, then two point-in-time re-runs), the "
              "rule-based regime, regime_switch with each (several minutes)", flush=True)
        outcome = run_combo_trial(engine, settings)
        path = write_combo_report(outcome, settings.retrain.reports_dir, datetime.now(timezone.utc))
        verdict = "PASSED" if outcome.gates.passed else "FAILED"
        queries.record_pipeline_run(engine, command="regime-combo-trial", stage="trial", status="OK",
                                    message=f"section 13.3: gates {verdict}")
    except ValueError as exc:
        _error(f"Trial failed to run: {exc}")
        return 1
    finally:
        engine.dispose()
    v = outcome.comparison.versus(portfolio_name("regime_switch", COMBO), portfolio_name("regime_switch", V1))
    print(f"regime_switch Sharpe {v['sharpe']:.4f} combined vs {v['baseline_sharpe']:.4f} with {V1}; "
          f"gates {verdict}")
    for result in outcome.gates.results:
        print(f"  {result.name:28s} {result.status:5s} {result.observed}")
    print(f"\n{path}")
    return 0


def _run_rule_v2_holdout(args: argparse.Namespace, settings: Settings) -> int:
    from datetime import datetime, timezone

    from quant.regime.bond_trial import run_holdout_check, write_holdout_report

    engine = _engine(settings, args.db)
    try:
        print("Section 12.5.3: holdout check of rule_based-v2 (holdout evaluation #3)", flush=True)
        check = run_holdout_check(engine, settings)
        path = write_holdout_report(check, settings.retrain.reports_dir, datetime.now(timezone.utc))
        queries.record_pipeline_run(engine, command="regime-rule-v2-holdout", stage="holdout", status="OK",
                                    message=f"section 12.5.3 (holdout evaluation #3): {check.decision}")
    except ValueError as exc:
        _error(f"Holdout check failed to run: {exc}")
        return 1
    finally:
        engine.dispose()
    print(f"{check.start.date()} .. {check.end.date()}: Sharpe {'at least as high' if check.sharpe_ok else 'lower'}, "
          f"max drawdown {'no deeper' if check.drawdown_ok else 'deeper'} -> {check.decision}")
    print(f"\n{path}")
    return 0


def _run_rate_shocks_trial(args: argparse.Namespace, settings: Settings) -> int:
    from datetime import datetime, timezone

    from quant.regime.rate_shocks_trial import run_rate_shocks_trial, write_rate_shocks_report

    engine = _engine(settings, args.db)
    try:
        print("Section 12.7 trial: rule_based-v1 and rule_rate_shocks-v1, regime_switch with each (a few minutes)",
              flush=True)
        outcome = run_rate_shocks_trial(engine, settings)
        path = write_rate_shocks_report(outcome, settings.retrain.reports_dir, datetime.now(timezone.utc))
        verdict = "PASSED" if outcome.gates.passed else "FAILED"
        queries.record_pipeline_run(engine, command="regime-rate-shocks-trial", stage="trial", status="OK",
                                    message=f"section 12.7: gates {verdict}")
    except ValueError as exc:
        _error(f"Trial failed to run: {exc}")
        return 1
    finally:
        engine.dispose()
    print(f"gates {verdict}")
    for result in outcome.gates.results:
        print(f"  {result.name:28s} {result.status:5s} {result.observed}")
    print(f"\n{path}")
    return 0


# strategy-trial <name> -> (trial letter, strategy name) of proposal section 17.1.5
STRATEGY_TRIALS = {"dual-momentum": ("A", "dual_momentum"), "momentum-regime": ("B", "momentum_regime"),
                   "volatility-target": ("C", "regime_switch_voltarget"), "sector-rotation": ("D", "sector_rotation"),
                   "top3-momentum": ("E", "top3_momentum_12m_quarterly")}


def _strategy_trial_functions(trial: str):
    """(run, write report) of one strategy trial."""
    if trial == "dual-momentum":
        from quant.strategies.dual_momentum_trial import run_dual_momentum_trial, write_dual_momentum_report

        return run_dual_momentum_trial, write_dual_momentum_report
    if trial == "momentum-regime":
        from quant.strategies.momentum_regime_trial import run_momentum_regime_trial, write_momentum_regime_report

        return run_momentum_regime_trial, write_momentum_regime_report
    if trial == "volatility-target":
        from quant.strategies.volatility_target_trial import (
            run_volatility_target_trial,
            write_volatility_target_report,
        )

        return run_volatility_target_trial, write_volatility_target_report
    if trial == "sector-rotation":
        from quant.strategies.sector_rotation_trial import run_sector_rotation_trial, write_sector_rotation_report

        return run_sector_rotation_trial, write_sector_rotation_report
    from quant.strategies.top3_momentum_trial import run_top3_momentum_trial, write_top3_momentum_report

    return run_top3_momentum_trial, write_top3_momentum_report


def _run_strategy_trial(args: argparse.Namespace, settings: Settings) -> int:
    from datetime import datetime, timezone

    letter, name = STRATEGY_TRIALS[args.trial]
    run_trial, write_report = _strategy_trial_functions(args.trial)
    engine = _engine(settings, args.db)
    try:
        section = "17.1.6" if letter == "E" else "17.1.5"
        print(f"Section {section} trial {letter}: {name} against regime_switch with rule_based-v1 (a few minutes)",
              flush=True)
        outcome = run_trial(engine, settings)
        path = write_report(outcome, settings.retrain.reports_dir, datetime.now(timezone.utc))
        verdict = "PASSED" if outcome.gates.passed else "FAILED"
        queries.record_pipeline_run(engine, command="strategy-trial", stage=args.trial, status="OK",
                                    message=f"section {section} trial {letter}: gates {verdict}")
    except ValueError as exc:
        _error(f"Trial failed to run: {exc}")
        return 1
    finally:
        engine.dispose()
    print(f"gates {verdict}")
    for result in outcome.gates.results:
        print(f"  {result.name:28s} {result.status:5s} {result.observed}")
    print(f"\n{path}")
    return 0


def _run_backup_database(args: argparse.Namespace, settings: Settings) -> int:
    import sqlite3

    from quant.ops.backup import BackupError, backup_database

    database = settings.database
    try:
        result = backup_database(args.db or database.path, database.backup_directory, database.backups_to_keep)
    except (BackupError, sqlite3.Error, OSError) as exc:
        _error(f"Backup failed: {exc}")
        return 1
    counts = ", ".join(f"{table} {count:,}" for table, count in result.row_counts.items())
    print(f"Backed up to {result.path} ({result.size_bytes / 1e6:,.0f} MB); integrity check OK; rows: {counts}")
    print(f"{result.kept} backup(s) kept in {database.backup_directory}"
          + (f"; deleted {len(result.deleted)} older: {', '.join(p.name for p in result.deleted)}"
             if result.deleted else ""))
    return 0


def _run_backtest(args: argparse.Namespace, settings: Settings) -> int:
    strategies = build_strategies(settings)
    if args.strategy:
        known = {strategy.name for strategy in strategies}
        unknown = [name for name in args.strategy if name not in known]
        if unknown:
            _error(f"Unknown strategy: {', '.join(unknown)}; choose from {', '.join(sorted(known))}")
            return 1
        strategies = [strategy for strategy in strategies if strategy.name in args.strategy]
    show_holdout = args.evaluate_holdout or not settings.holdout_requires_flag

    engine = _engine(settings, args.db)
    try:
        report = run_backtests(engine, settings, strategies)
        previous_evaluations = queries.count_holdout_evaluations(engine)
        batch_id = save_backtest(engine, settings, report, include_holdout=show_holdout)
        periods = report.periods
        names = ", ".join(result.name for result in report.results)
        if show_holdout:
            queries.record_pipeline_run(
                engine,
                command="backtest",
                stage="holdout",
                status="OK",
                data_as_of_date=periods.end.date(),
                message=f"batch {batch_id}: holdout {periods.holdout_start.date()}..{periods.end.date()} "
                f"evaluated for {names}",
            )
        queries.record_pipeline_run(
            engine,
            command="backtest",
            stage="backtest",
            status="OK",
            data_as_of_date=periods.end.date(),
            message=f"batch {batch_id}: {names}",
        )
    except ValueError as exc:
        _error(f"Backtest failed: {exc}")
        return 1
    finally:
        engine.dispose()

    print(development_report(report))
    print()
    if show_holdout:
        print(holdout_report(report, previous_evaluations + 1))
    else:
        print(
            f"HOLDOUT {periods.holdout_start.date()} .. {periods.end.date()}: not shown "
            f"(evaluated {previous_evaluations} time(s) so far; use --evaluate-holdout to view it)"
        )
    print()
    print(f"Saved as batch {batch_id} in backtest_runs and backtest_equity.")
    return 0


def _print_retrain(outcome) -> None:
    folds, best = outcome.folds, outcome.search
    name = f"ml_{outcome.model_type}"
    print(
        f"{outcome.model_type}: {len(folds)} walk-forward folds validating "
        f"{folds[0].validation_start.date()} .. {folds[-1].validation_end.date()}; "
        f"holdout from {outcome.holdout_start.date()}"
    )
    print(f"\nSearch ({best.trial_count} trials; best {best.best_parameters}):")
    trials = best.trials
    params = [c for c in trials.columns if c.startswith("param_")]
    view = trials[params].rename(columns=lambda c: c[len("param_"):])
    view["score"] = trials["score"].map(lambda v: f"{v:+.3%}")
    view["log-loss skill"] = trials["log_loss_skill_mean"].map(lambda v: f"{v:+.2%}")
    view["AUC"] = trials["auc_mean"].map(lambda v: f"{v:.3f}")
    print(view.to_string())

    print(f"\nWalk-forward backtest {outcome.backtest.start.date()} .. {outcome.backtest.end.date()} (after costs):")
    print(metrics_table(outcome.backtest.overall))
    if outcome.holdout is not None:
        print(
            f"\nHOLDOUT {outcome.holdout.start.date()} .. {outcome.holdout.end.date()} "
            f"(evaluation #{outcome.holdout_evaluation_number}; logged in pipeline_runs):"
        )
        print(metrics_table(outcome.holdout.overall))
        print("If this result changes a rule or setting, the holdout is consumed (section 15.2).")

    print(f"\nGates: {'PASSED' if outcome.gates.passed else 'FAILED'}")
    print(outcome.gates.table().to_string())
    registration = outcome.registration
    print(
        f"\nRegistered {registration.model_id} (version {registration.version}) as CANDIDATE."
        f"\n  model:  {registration.model_path}\n  report: {registration.report_path}"
    )
    active = outcome.active
    if active is None:
        print("No ACTIVE predictor: run-daily uses rule-based signals only. Nothing was changed.")
    else:
        print(
            f"ACTIVE predictor stays {active['model_id']} ({active['model_type']} v{active['version']}, "
            f"gates {'passed' if active['gates_passed'] else 'failed'}). Nothing was changed; "
            "promoting a candidate is `quant promote` (Phase 9)."
        )
    if not outcome.gates.passed:
        print(f"The candidate failed its gates; under section 32 the rule-based {BASELINE} remains what ships.")


def _print_regime_retrain(outcome) -> None:
    folds, regime = outcome.folds, outcome.walk_forward.regime
    print(
        f"hmm: {len(folds)} yearly fits; filtered regime {regime.index[0].date()} .. {regime.index[-1].date()} "
        f"(out of sample; holdout untouched)"
    )
    shares = ", ".join(f"{label} {(regime == label).mean():.0%}" for label in ("BULL", "SIDEWAYS", "BEAR"))
    print(f"  sessions: {shares}; average length {outcome.evidence.average_dwell:.0f} sessions; "
          f"lowest agreement between consecutive fits {outcome.evidence.agreement.min():.1%}")
    print(f"\nTrading with each regime {outcome.comparison.start.date()} .. {outcome.comparison.end.date()} (after costs):")
    print(metrics_table(outcome.comparison.overall))
    print(f"\nGates: {'PASSED' if outcome.gates.passed else 'FAILED'}")
    print(outcome.gates.table().to_string())
    registration = outcome.registration
    print(
        f"\nRegistered {registration.model_id} (version {registration.version}) as REGIME CANDIDATE."
        f"\n  model:  {registration.model_path}\n  report: {registration.report_path}"
    )
    active = outcome.active
    if active is None:
        print("No ACTIVE regime model: run-daily uses the rule-based regime. Nothing was changed.")
    else:
        print(f"ACTIVE regime stays {active['model_id']}. Nothing was changed; promoting is `quant promote` (Phase 9).")
    if not outcome.gates.passed:
        print("The candidate failed its gates; the rule-based regime remains the one in use.")


def _run_retrain(args: argparse.Namespace, settings: Settings) -> int:
    if args.role == "regime":
        if args.evaluate_holdout:
            _error("--evaluate-holdout does not apply to --role regime: no regime gate uses the holdout")
            return 1
        if args.model and args.model != "hmm":
            _error(f"--role regime trains the HMM; --model {args.model} is not a regime model")
            return 1
        engine = _engine(settings, args.db)
        try:
            outcome = run_regime_retrain(engine, settings)
        except ValueError as exc:
            _error(f"Retrain failed: {exc}")
            return 1
        finally:
            engine.dispose()
        _print_regime_retrain(outcome)
        return 0
    model_type = args.model or settings.model.type
    evaluate_holdout = args.evaluate_holdout or not settings.holdout_requires_flag
    if model_type == "all":
        return _run_retrain_all(args, settings, evaluate_holdout)
    engine = _engine(settings, args.db)
    try:
        outcome = run_retrain(engine, settings, model_type, evaluate_holdout)
    except (NotImplementedError, ValueError) as exc:
        _error(f"Retrain failed: {exc}")
        return 1
    finally:
        engine.dispose()
    _print_retrain(outcome)
    return 0


def _run_retrain_all(args: argparse.Namespace, settings: Settings, evaluate_holdout: bool) -> int:
    if evaluate_holdout:
        _error("--model all does not evaluate the holdout: viewing it once per model would use it up. Compare "
               "with --model all, then check one chosen model with --model <type> --evaluate-holdout")
        return 1
    outcomes = []
    for number, model_type in enumerate(IMPLEMENTED_MODELS, start=1):
        print(f"\n===== [{number}/{len(IMPLEMENTED_MODELS)}] {model_type} =====")
        engine = _engine(settings, args.db)
        try:
            outcome = run_retrain(engine, settings, model_type, evaluate_holdout=False)
        except (NotImplementedError, ValueError) as exc:
            _error(f"Retrain of {model_type} failed: {exc}")
            return 1
        finally:
            engine.dispose()
        _print_retrain(outcome)
        outcomes.append(outcome)
    table = comparison_table(outcomes)
    folds = outcomes[0].folds
    print(f"\n===== Comparison: walk-forward {folds[0].validation_start.date()} .. {folds[-1].validation_end.date()}, "
          "after costs =====")
    print(format_comparison(table).drop(columns="candidate").to_string())
    path = write_comparison(outcomes, table, settings.retrain.reports_dir)
    print(f"\nSaved {path}. Descriptive only: the best of several looks better than it is; the gates are unchanged, "
          "and nothing was promoted.")
    return 0


def _print_signals(run, show_reasons: bool) -> None:
    signals = run.signals
    targets = signals.targets
    if run.predictions is not None:
        p = run.predictions
        print(f"ACTIVE predictor {p.model_id} ({p.model_type}): {p.rows} predictions for "
              f"{p.first_date.date()}..{p.last_date.date()}; ml_select drives the signals")
    print(
        f"Signals for {signals.session.date()} (fill at the next open): {targets.strategy} ({run.model_version}), "
        f"{targets.regime} regime ({targets.regime_source}), exposure {targets.regime_exposure:.0%}"
    )
    print(
        f"Portfolio {signals.portfolio_value:,.2f}: target {targets.invested:.1%} invested, {targets.cash:.1%} cash; "
        f"cash after the suggested trades about {signals.estimated_cash_after:,.2f} (before costs)"
    )
    rows = signals.rows
    if rows.empty:
        print("No signals: nothing held and nothing wanted.")
    else:
        table = pd.DataFrame({
            "signal": rows["signal"],
            **({"prob": targets.stocks.loc[rows.index, "probability"].map(lambda v: f"{v:.2f}" if v == v else "-")}
               if "probability" in targets.stocks.columns else {}),
            "target": rows["target_weight"].map(lambda v: f"{v:.1%}"),
            "current": rows["current_weight"].map(lambda v: f"{v:.1%}"),
            "shares": rows["suggested_shares"].map(lambda v: f"{v:+g}" if v else "0"),
            "close": rows["close"].map(lambda v: f"{v:,.2f}"),
            "value": rows["trade_value"].map(lambda v: f"{v:+,.0f}" if v else "0"),
        })
        print(table.to_string())
    print("  " + ", ".join(f"{name} {count}" for name, count in signals.counts.items()))
    if show_reasons:
        print()
        for symbol, row in rows.iterrows():
            print(f"{symbol}: {row['reason']}")
    for warning in signals.warnings:
        print(f"warning: {warning}")
    stored = f"{run.positions_stored} position rows" if run.positions_stored else "no positions snapshot"
    print(f"Saved {run.signals_stored} signals and {stored} for {signals.session.date()}. "
          "Research output, not advice.")


def _run_signals(args: argparse.Namespace, settings: Settings) -> int:
    engine = _engine(settings, args.db)
    try:
        run = generate_and_store_signals(engine, settings)
    except (PositionsError, ValueError, NotImplementedError) as exc:
        _error(f"Signal generation failed: {exc}")
        return 1
    finally:
        engine.dispose()
    _print_signals(run, args.reasons)
    return 0


def _run_promote(args: argparse.Namespace, settings: Settings) -> int:
    engine = _engine(settings, args.db)
    try:
        promotion = promote_model(engine, settings, args.model_id, args.override)
    except PromotionError as exc:
        _error(f"Promotion refused: {exc}")
        return 1
    finally:
        engine.dispose()
    print(f"{promotion.model_id} is now the ACTIVE {promotion.role} model.")
    if promotion.retired:
        print(f"  {promotion.retired} is RETIRED.")
    if promotion.override_reason:
        print(f"  OVERRIDE logged: the gates failed ({', '.join(promotion.failed_gates) or 'not passed'}); "
              f"reason: {promotion.override_reason}")
    if promotion.stale_predictor:
        print(f"warning: the ACTIVE predictor {promotion.stale_predictor} was trained on another regime; "
              "retrain it (`quant status` warns until then)")
    undo = (f"`quant promote {promotion.retired}`" if promotion.retired
            else f"`quant retire {promotion.model_id}` (back to {FALLBACK[promotion.role]})")
    print(f"The other role is unchanged. Undo with {undo}.")
    return 0


def _run_retire(args: argparse.Namespace, settings: Settings) -> int:
    engine = _engine(settings, args.db)
    try:
        role = retire_model(engine, settings, args.model_id, args.reason)
    except PromotionError as exc:
        _error(f"Retire refused: {exc}")
        return 1
    finally:
        engine.dispose()
    print(f"{args.model_id} is RETIRED. {role} now uses {FALLBACK[role]}; run `quant run-daily` (or generate-signals) "
          f"to refresh the signals. Undo with `quant promote {args.model_id}`.")
    return 0


def _end_forward(args: argparse.Namespace, settings: Settings) -> int:
    if getattr(args, "start", False):
        _error("--end and --start are separate steps: end the run, then start a new one after a close")
        return 1
    reason = (args.reason or "").strip()
    if not reason:
        _error('--end needs --reason, e.g. --reason "started with the wrong positions" (it is logged)')
        return 1
    engine = _engine(settings, args.db)
    try:
        run = queries.end_forward_run(engine, reason)
        if run is None:
            _error("No forward test is running; nothing to end.")
            return 1
        frozen = len(queries.frozen_sessions(engine, "live", run["run_id"]))
        queries.record_pipeline_run(engine, command="forward-test", stage="end", status="OK",
                                    message=f"run {run['run_id']} (since {run['started_on']}, {frozen} frozen "
                                            f"sessions) ended: {reason}")
    finally:
        engine.dispose()
    print(f"Forward test run {run['run_id']} (since {run['started_on']}, {frozen} frozen sessions) ENDED: {reason}. "
          f"It is kept: `quant forward-report --run {run['run_id']}`.")
    print("To start a new run: both brokers' snapshots of the latest session are needed (`quant run-daily` "
          "after its close, and once more the next day for IBKR's statement); then `quant forward-test --start` "
          "before the next open.")
    return 0


def _run_forward(args: argparse.Namespace, settings: Settings) -> int:
    if getattr(args, "end", False):
        return _end_forward(args, settings)
    start = bool(getattr(args, "start", False))
    engine = _engine(settings, args.db)
    try:
        before = queries.forward_start(engine)
        # Every new run reads real from the brokers (run 2 design); run 1 (positions file + trade log)
        # stays readable in the reports but is no longer started.
        result = freeze_session(engine, settings, start=start, real_from_brokers=True)
        if not result.started:
            print("Forward test not started: run `python -m quant run-daily` after a close, and once more the next "
                  "day for IBKR's statement; then `python -m quant forward-test --start` (before the next open).")
            return 0
        if start and before is not None:
            print(f"Forward test run {before['run_id']} is already running (since {before['started_on']}); "
                  'nothing to start. To restart: `quant forward-test --end --reason "..."` first.')
        elif before is None:
            begun = queries.forward_start(engine)
            print(f"Forward test run {begun['run_id']} STARTED on {begun['started_on']} with "
                  f"{begun['start_value']:,.2f} "
                  f"({begun['value_source']}). From now on run `quant run-daily` every evening after the "
                  "close: it stores the broker snapshots and freezes the session.")
        paper = update_paper_equity(engine, settings)
    except (TradesError, ValueError) as exc:
        _error(f"Forward test failed: {exc}")
        return 1
    finally:
        engine.dispose()
    session = result.session.date()
    if result.frozen:
        print(f"Froze {session} for {', '.join(result.frozen)} (deadline {result.deadline:%Y-%m-%d %H:%M} UTC).")
    elif result.missed:
        missing = [n for n in ("live", *settings.forward_test.shadows) if n not in result.already]
        if result.already:  # e.g. a shadow added after this session's deadline
            print(f"{session} was already frozen for {', '.join(result.already)}; left as it was.")
            _error(f"{session} was NOT frozen for {', '.join(missing)}: the next open "
                   f"({result.deadline:%Y-%m-%d %H:%M} UTC) has passed. Counted as missed for "
                   f"{'it' if len(missing) == 1 else 'them'} (no backfill).")
        else:
            _error(f"{session} was NOT frozen: the next open ({result.deadline:%Y-%m-%d %H:%M} UTC) has passed. "
                   "It is counted as missed (no backfill); run run-daily each evening after the close.")
    else:
        print(f"{session} was already frozen ({', '.join(result.already)}); left as it was.")
    if paper is not None:
        print(f"Paper and real values updated through {paper.sessions[-1].date()} "
              f"({len(paper.sessions) - 1} forward sessions). See `quant forward-report`.")
    return 0


def _run_forward_report(args: argparse.Namespace, settings: Settings) -> int:
    engine = _engine(settings, args.db)
    try:
        report = build_report(engine, settings, getattr(args, "run", None))
    except (TradesError, ValueError) as exc:
        _error(f"Forward report failed: {exc}")
        return 1
    finally:
        engine.dispose()
    print(format_report(report))
    return 0


def _run_status(args: argparse.Namespace, settings: Settings) -> int:
    engine = _engine(settings, args.db)
    try:
        report = collect_status(engine, settings)
    finally:
        engine.dispose()
    print(format_status(report))
    return 0


def _record_daily(args: argparse.Namespace, settings: Settings, status: str, message: str) -> None:
    engine = _engine(settings, args.db)
    try:
        queries.record_pipeline_run(engine, command="run-daily", stage="run-daily", status=status, message=message)
    finally:
        engine.dispose()


def _run_daily(args: argparse.Namespace, settings: Settings) -> int:
    code = _run_daily_stages(args, settings)
    if code == 0:
        _record_daily(args, settings, "OK", "every stage finished")
    else:
        _record_daily(args, settings, "FAILED", "a stage failed or some symbols did not update (see the output)")
    return code


def _run_daily_broker_sync(args: argparse.Namespace, settings: Settings) -> int:
    """The broker snapshots inside run-daily: never stops the run. Without credentials in .env it
    is skipped; a broker that fails (or Trading 212 outside its window) is reported and the other
    stages go on, because a missing snapshot is only carried forward in `real`."""
    import os

    from dotenv import load_dotenv

    load_dotenv()
    if not (os.environ.get("TRADING212_API_KEY") or os.environ.get("IBKR_FLEX_TOKEN")):
        print("  skipped: no broker credentials in .env (see .env.example)")
        return 0
    sync_args = argparse.Namespace(only=None, db=getattr(args, "db", None))
    if broker_cli.run(argparse.Namespace(broker_command="sync-accounts", **vars(sync_args)), settings) != 0:
        _error("  a broker snapshot was not stored (see above); run-daily continues")
    return 0


def _run_daily_options(args: argparse.Namespace, settings: Settings) -> int:
    """The option chains inside run-daily (altdata_proposal.md 13.7): research only, last, never
    stops the run. Outside the window (the market open) it is skipped with a note."""
    options_args = argparse.Namespace(altdata_command="options", compare=False, symbol=None,
                                      db=getattr(args, "db", None))
    if altdata_cli.run(options_args, settings) != 0:
        _error("  some option chains were not stored (see above); research data only, run-daily is not affected")
    return 0


def _run_daily_stages(args: argparse.Namespace, settings: Settings) -> int:
    """Run the pipeline stages in order (proposal section 23, steps 1-13).

    Every stage persists its own results and pipeline_runs row, so a failure
    leaves earlier stages intact and a rerun resumes safely. A stage that fails
    outright stops the run, because later stages read its output. Symbols that
    fail to download do not stop it: features and the regime are calculated
    from the data that is stored, and the exit code still reports the failure.
    """
    print("[1/7] update-market-data")
    market_data = _update_market_data(args, settings)
    if market_data == _FAILED:
        _error("run-daily stopped: market data update failed")
        return 1
    for step, name, run in (
        (2, "calculate-features", _run_features),
        (3, "calculate-regime", _run_regime),
        (4, "broker sync-accounts", _run_daily_broker_sync),
        (5, "generate-signals", _run_signals),
        (6, "forward-test", _run_forward),
        (7, "option chains (research)", _run_daily_options),
    ):
        print(f"[{step}/7] {name}")
        if run(args, settings) != 0:
            _error(f"run-daily stopped: {name} failed")
            return 1
    if market_data == _PARTIAL:
        _error("run-daily finished, but some symbols failed to update (see above)")
        return 1
    print("run-daily finished")
    return 0


COMMANDS = {
    "update-market-data": lambda args, settings: _run_update(args, settings),
    "calculate-features": lambda args, settings: _run_features(args, settings),
    "calculate-regime": lambda args, settings: _run_regime(args, settings),
    "regime-bond-trial": lambda args, settings: _run_bond_trial(args, settings),
    "regime-combo-trial": lambda args, settings: _run_combo_trial(args, settings),
    "regime-rule-v2-holdout": lambda args, settings: _run_rule_v2_holdout(args, settings),
    "regime-rate-shocks-trial": lambda args, settings: _run_rate_shocks_trial(args, settings),
    "strategy-trial": lambda args, settings: _run_strategy_trial(args, settings),
    "promote": lambda args, settings: _run_promote(args, settings),
    "retire": lambda args, settings: _run_retire(args, settings),
    "status": lambda args, settings: _run_status(args, settings),
    "backup-database": lambda args, settings: _run_backup_database(args, settings),
    "forward-report": lambda args, settings: _run_forward_report(args, settings),
    "forward-test": lambda args, settings: _run_forward(args, settings),
    "generate-signals": lambda args, settings: _run_signals(args, settings),
    "run-daily": lambda args, settings: _run_daily(args, settings),
    "backtest": lambda args, settings: _run_backtest(args, settings),
    "retrain": lambda args, settings: _run_retrain(args, settings),
    "sentiment": sentiment_cli.run,
    "altdata": altdata_cli.run,
    "broker": broker_cli.run,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.config)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)  # no log yet: its settings are the problem
        return 1
    if args.command not in COMMANDS:
        parser.error(f"unknown command: {args.command}")
        return 2
    handler = configure_logging(settings.logging, getattr(args, "db", None) or settings.database.path)
    words = " ".join(argv if argv is not None else sys.argv[1:])
    log.info("command started: %s", words)
    started = time.monotonic()
    try:
        code = COMMANDS[args.command](args, settings)
    except ConfigError as exc:
        _error(f"Configuration error: {exc}")
        code = 1
    except Exception:
        log.exception("command crashed: %s", words)
        close_logging(handler)
        raise
    log.log(logging.INFO if code == 0 else logging.WARNING, "command finished: %s -> exit %d in %.1fs",
            args.command, code, time.monotonic() - started)
    close_logging(handler)
    return code
