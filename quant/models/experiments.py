"""Pre-registered PREDICTOR trials: one variant of the label horizon and trading rule (section 15.4).

A trial changes only what its spec names, then runs the full chain on the
development folds: labels -> dataset -> purged walk-forward -> ``ml_select``
backtest against ``regime_switch`` -> the configured gates (unchanged). For a
horizon of h sessions the label spans h sessions, the embargo is h + 1 and the
selection is refreshed every h sessions. Every trial run counts in the trial log.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sqlalchemy import Engine

from quant.backtest.data import Prices
from quant.backtest.engine import split_periods
from quant.config.settings import Settings
from quant.models.dataset import load_dataset
from quant.models.folds import walk_forward_folds
from quant.models.gates import GateReport, PredictorEvidence, collect_evidence, evaluate_predictor_gates
from quant.models.ml_backtest import MLBacktest, run_ml_backtest
from quant.models.walk_forward import WalkForwardResult, run_walk_forward
from quant.strategies.rules import Panel


@dataclass(frozen=True)
class TrialSpec:
    model_type: str
    parameters: dict[str, Any]
    horizon_days: int
    keep_rank: int

    @property
    def name(self) -> str:
        return f"{self.model_type} h{self.horizon_days} keep{self.keep_rank}"


@dataclass(frozen=True)
class TrialResult:
    spec: TrialSpec
    settings: Settings
    walk_forward: WalkForwardResult
    backtest: MLBacktest
    evidence: PredictorEvidence
    gates: GateReport


def trial_settings(settings: Settings, spec: TrialSpec) -> Settings:
    """The settings of a trial: horizon, matching embargo and rebalance, and keep band."""
    horizon = spec.horizon_days
    model = dataclasses.replace(
        settings.model,
        target=dataclasses.replace(settings.model.target, horizon_days=horizon),
        embargo_days=horizon + 1,
    )
    select = dataclasses.replace(
        settings.strategies.ml_select, keep_rank=spec.keep_rank, rebalance_every_sessions=horizon
    )
    return dataclasses.replace(
        settings, model=model, strategies=dataclasses.replace(settings.strategies, ml_select=select)
    )


def run_trial(
    engine: Engine,
    settings: Settings,
    spec: TrialSpec,
    prices: Prices,
    panel: Panel,
    regimes: pd.Series,
) -> TrialResult:
    trial = trial_settings(settings, spec)
    dataset = load_dataset(engine, trial)
    periods = split_periods(dataset.sessions, trial.holdout_months)
    folds = walk_forward_folds(
        dataset.labelled().sessions, periods.holdout_start, trial.retrain.validation, trial.model.embargo_days
    )
    name = f"ml_{spec.model_type}"
    walk_forward = run_walk_forward(dataset, folds, spec.model_type, spec.parameters, trial)
    backtest = run_ml_backtest({name: walk_forward}, folds, prices, panel, regimes, trial)
    evidence = collect_evidence(
        name, walk_forward, backtest, folds, prices.close.index, periods.holdout_start, trial.model.embargo_days
    )
    gates = evaluate_predictor_gates(trial.retrain.gates["predictor"], evidence)
    return TrialResult(spec, trial, walk_forward, backtest, evidence, gates)
