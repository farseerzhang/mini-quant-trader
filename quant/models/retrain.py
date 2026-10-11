"""`quant retrain --role predictor`: what the manual retrain automates (proposal section 16.2).

1. Build the dataset from the stored data up to the last session.
2. Generate the purged walk-forward folds and the holdout window from the config.
3. Search the configured hyperparameters (every candidate on every fold).
4. Run the leak checks (part of the gates).
5. Backtest the best candidate walk-forward as `ml_select` against `regime_switch`,
   SPY, equal weight and the random baseline, after costs.
6. Only with ``evaluate_holdout``: train on all development data, trade the holdout,
   and log the evaluation.
7. Evaluate the pre-defined gates.
8. Refit the chosen configuration on all development data (section 15.5).
9. Register it as a CANDIDATE with its report.
10. Never change an ACTIVE model; that is `quant promote`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine

from quant.backtest.data import load_prices
from quant.backtest.engine import split_periods
from quant.config.settings import Settings
from quant.models.dataset import load_dataset
from quant.models.final import refit_final
from quant.models.folds import Fold, holdout_fold, walk_forward_folds
from quant.models.gates import GateReport, PredictorEvidence, collect_evidence, evaluate_predictor_gates
from quant.models.ml_backtest import BASELINE, MLBacktest, run_ml_backtest
from quant.models.predictors import FIXED_PARAMETERS
from quant.models.registry import PREDICTOR, Registration, register_candidate
from quant.models.search import SearchResult, search
from quant.models.walk_forward import run_walk_forward
from quant.persistence import queries
from quant.regime.pipeline import RULE_METHOD
from quant.strategies import PANEL_COLUMNS

IMPLEMENTED_MODELS = tuple(FIXED_PARAMETERS)


@dataclass(frozen=True)
class RetrainOutcome:
    model_type: str
    folds: list[Fold]
    holdout_start: pd.Timestamp
    search: SearchResult
    backtest: MLBacktest
    holdout: MLBacktest | None
    holdout_evaluation_number: int | None
    evidence: PredictorEvidence
    gates: GateReport
    registration: Registration
    active: dict | None


def portfolio_name(model_type: str) -> str:
    return f"ml_{model_type}"


def run_retrain(engine: Engine, settings: Settings, model_type: str, evaluate_holdout: bool) -> RetrainOutcome:
    if model_type not in IMPLEMENTED_MODELS:
        raise NotImplementedError(
            f"model type {model_type!r} is not implemented yet (XGBoost arrives in Phase 6); "
            f"use one of: {', '.join(IMPLEMENTED_MODELS)}"
        )
    stocks = list(settings.market.stocks)
    embargo = settings.model.embargo_days
    dataset = load_dataset(engine, settings)
    holdout_start = split_periods(dataset.sessions, settings.holdout_months).holdout_start
    labelled_sessions = dataset.labelled().sessions
    folds = walk_forward_folds(labelled_sessions, holdout_start, settings.retrain.validation, embargo)

    result = search(dataset, folds, model_type, settings)
    prices = load_prices(engine, [*stocks, "SPY"])
    panel = queries.load_feature_panel(engine, settings.features.feature_version, stocks, PANEL_COLUMNS)
    regimes = queries.load_regimes(engine, RULE_METHOD, settings.regime.rule_version)
    name = portfolio_name(model_type)
    backtest = run_ml_backtest({name: result.best}, folds, prices, panel, regimes, settings)

    holdout = None
    evaluation_number = None
    holdout_metrics = None
    if evaluate_holdout:
        fold = holdout_fold(labelled_sessions, holdout_start, embargo)
        trained = run_walk_forward(dataset, [fold], model_type, result.best_parameters, settings)
        holdout = run_ml_backtest({name: trained}, [fold], prices, panel, regimes, settings)
        holdout_metrics = {
            "sharpe": float(holdout.overall.loc[name, "sharpe"]),
            "baseline_sharpe": float(holdout.overall.loc[BASELINE, "sharpe"]),
        }
        evaluation_number = queries.count_holdout_evaluations(engine) + 1

    evidence = collect_evidence(
        name, result.best, backtest, folds, prices.close.index, holdout_start, embargo, holdout_metrics
    )
    gates = evaluate_predictor_gates(settings.retrain.gates["predictor"], evidence)

    final = refit_final(dataset, holdout_start, result.best, settings)
    registration = register_candidate(
        engine, settings, final, result, folds, backtest, name, evidence, gates,
        models_dir=Path(settings.retrain.models_dir), reports_dir=Path(settings.retrain.reports_dir),
    )
    last = dataset.sessions[-1].date()
    if evaluate_holdout:
        queries.record_pipeline_run(
            engine, command="retrain", stage="holdout", status="OK", data_as_of_date=last,
            message=f"{registration.model_id}: holdout {holdout.start.date()}..{holdout.end.date()} evaluated "
            f"(Sharpe {holdout_metrics['sharpe']:.2f} vs {BASELINE} {holdout_metrics['baseline_sharpe']:.2f})",
        )
    queries.record_pipeline_run(
        engine, command="retrain", stage="retrain", status="OK", data_as_of_date=last,
        message=f"registered {registration.model_id} as CANDIDATE; gates {'PASSED' if gates.passed else 'FAILED'}; "
        f"{result.trial_count} trials",
    )
    active = queries.active_model(engine, PREDICTOR)
    return RetrainOutcome(
        model_type=model_type,
        folds=folds,
        holdout_start=holdout_start,
        search=result,
        backtest=backtest,
        holdout=holdout,
        holdout_evaluation_number=evaluation_number,
        evidence=evidence,
        gates=gates,
        registration=registration,
        active=dict(active) if active is not None else None,
    )
