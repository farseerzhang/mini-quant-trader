"""`quant retrain --role regime`: what the manual retrain automates for the HMM (sections 13.1, 16.2).

1. Build the same purged walk-forward folds as the predictors (holdout excluded).
2. Fit the HMM yearly, filter each validation window forward, apply the dwell rule.
3. Trade the regime-dependent strategies with the HMM and with the rule-based regime.
4. Evaluate the pre-defined regime gates (including a point-in-time re-run).
5. Fit the final HMM on all purged development sessions.
6. Register it as a REGIME CANDIDATE with a validation report.
7. Never change the active regime; that is `quant promote`.
"""

from __future__ import annotations

import dataclasses
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sqlalchemy import Engine

from quant.backtest.data import load_prices
from quant.backtest.engine import split_periods
from quant.backtest.persist import config_hash, git_commit
from quant.config.settings import Settings
from quant.models.dataset import load_dataset
from quant.models.folds import Fold, development_sessions, walk_forward_folds
from quant.models.gates import GateReport
from quant.models.registry import CANDIDATE, Registration, library_versions, next_version
from quant.persistence import queries, tables
from quant.regime.comparison import RegimeComparison, compare_regimes
from quant.regime.gates import RegimeEvidence, collect_regime_evidence, evaluate_regime_gates
from quant.regime.hmm import FittedHMM, fit_hmm, load_observations
from quant.regime.chart import Ribbon, regime_chart_html, regime_chart_svg
from quant.regime.hmm_walk_forward import HMMWalkForward, apply_min_dwell, run_hmm_walk_forward
from quant.regime.pipeline import RULE_METHOD
from quant.regime.report import regime_report
from quant.strategies import PANEL_COLUMNS

REGIME = "REGIME"
HMM_TYPE = "hmm"


@dataclass(frozen=True)
class RegimeRetrainOutcome:
    folds: list[Fold]
    walk_forward: HMMWalkForward
    comparison: RegimeComparison
    evidence: RegimeEvidence
    gates: GateReport
    final: FittedHMM
    registration: Registration
    active: dict | None


def load_final_hmm(path: Path) -> FittedHMM:
    model = joblib.load(path)
    if not isinstance(model, FittedHMM):
        raise ValueError(f"{path} does not hold a FittedHMM")
    return model


# Labelled on the timeline chart (when inside the chart window).
CHART_EVENTS = [
    ("2008-10-10", "2008 crisis"), ("2011-08-08", "2011"), ("2016-02-11", "2015-16"),
    ("2018-12-24", "2018 Q4"), ("2020-03-23", "2020 COVID"), ("2022-10-12", "2022"), ("2025-04-08", "2025 tariffs"),
]


def timeline_regime(
    final: FittedHMM, walk_forward: HMMWalkForward, observations: pd.DataFrame, settings: Settings
) -> pd.Series:
    """The HMM regime for the whole history: out of sample from the first walk-forward session,
    and before it the final fit's filtered regime (in sample: that fit has seen those years)."""
    start = walk_forward.regime.index[0]
    earlier = observations.loc[observations.index < start]
    in_sample = apply_min_dwell(final.filtered(earlier).idxmax(axis=1), settings.regime.min_dwell_days)
    timeline = pd.concat([in_sample, walk_forward.regime])
    timeline.attrs["out_of_sample_start"] = start
    return timeline


def write_regime_charts(
    directory: Path, model_id: str, spy: pd.Series, rule: pd.Series, hmm: pd.Series
) -> dict[str, str]:
    """``<model_id>_regimes.svg`` (embedded in the report) and ``.html`` (interactive), from the
    first session with both regimes to the end of the walk-forward window (holdout excluded)."""
    hmm_out_of_sample_start = hmm.attrs.get("out_of_sample_start")
    start = max(rule.dropna().index[0], hmm.dropna().index[0])
    end = hmm.index[-1]
    price = spy.loc[start:end]
    ribbons = [
        Ribbon("HMM regime", hmm, in_sample_until=hmm_out_of_sample_start),
        Ribbon("Rule-based", rule),
    ]
    svg_path = directory / f"{model_id}_regimes.svg"
    html_path = directory / f"{model_id}_regimes.html"
    svg_path.write_text(regime_chart_svg(price, ribbons, CHART_EVENTS), encoding="utf-8")
    html_path.write_text(
        regime_chart_html(price, ribbons, CHART_EVENTS, title=f"Market regimes: {model_id}"), encoding="utf-8"
    )
    return {"svg": svg_path.name, "html": html_path.name}


def run_regime_retrain(engine: Engine, settings: Settings) -> RegimeRetrainOutcome:
    stocks = list(settings.market.stocks)
    embargo = settings.model.embargo_days
    dataset = load_dataset(engine, settings)
    holdout_start = split_periods(dataset.sessions, settings.holdout_months).holdout_start
    folds = walk_forward_folds(dataset.labelled().sessions, holdout_start, settings.retrain.validation, embargo)

    observations = load_observations(engine, settings)
    walk_forward = run_hmm_walk_forward(observations, folds, settings)
    rule = queries.load_regimes(engine, RULE_METHOD, settings.regime.rule_version)
    prices = load_prices(engine, [*stocks, "SPY"])
    panel = queries.load_feature_panel(engine, settings.features.feature_version, stocks, PANEL_COLUMNS)
    comparison = compare_regimes({"rule": rule, "hmm": walk_forward.regime}, folds, prices, panel, settings)
    evidence = collect_regime_evidence(observations, folds, settings, walk_forward, comparison)
    gates = evaluate_regime_gates(settings.retrain.gates["regime"], evidence)

    development = development_sessions(dataset.labelled().sessions, holdout_start, embargo)
    final = fit_hmm(observations.loc[observations.index.isin(development)], settings.regime.hmm, settings.retrain.search.seed)

    now = datetime.now(timezone.utc)
    model_id = f"{HMM_TYPE}-{now:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
    version = next_version(engine, HMM_TYPE)
    model_path = Path(settings.retrain.models_dir) / HMM_TYPE / f"{model_id}.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, model_path)
    report_path = Path(settings.retrain.reports_dir) / HMM_TYPE / f"{model_id}.md"  # mirrors models/hmm/
    report_path.parent.mkdir(parents=True, exist_ok=True)
    spy = prices.close["SPY"]
    chart_files = write_regime_charts(
        report_path.parent, model_id, spy, rule, timeline_regime(final, walk_forward, observations, settings)
    )
    report_path.write_text(
        regime_report(model_id, final, walk_forward, rule, comparison, evidence, gates, spy, now, chart_files),
        encoding="utf-8",
    )

    regime = walk_forward.regime
    metrics = {
        "window": [str(comparison.start.date()), str(comparison.end.date())],
        "share": {label: float((regime == label).mean()) for label in ("BULL", "SIDEWAYS", "BEAR")},
        "average_dwell": evidence.average_dwell,
        "lowest_agreement": float(evidence.agreement.min()),
        "sharpe": evidence.sharpe,
        "baseline_sharpe": evidence.baseline_sharpe,
    }
    row = {
        "model_id": model_id,
        "role": REGIME,
        "model_type": HMM_TYPE,
        "version": version,
        "status": CANDIDATE,
        "depends_on": None,
        "training_start": final.train_start.date(),
        "training_end": final.train_end.date(),
        "feature_version": settings.features.market_feature_version,
        "target_definition": json.dumps({
            "regime": "3-state Gaussian HMM, filtered; states ordered by mean return (BULL, SIDEWAYS, BEAR)",
            "observations": list(settings.regime.hmm.observations),
            "min_dwell_days": settings.regime.min_dwell_days,
        }),
        "hyperparameters": json.dumps(dataclasses.asdict(settings.regime.hmm), sort_keys=True),
        "metrics": json.dumps(metrics, sort_keys=True),
        "gates_passed": int(gates.passed),
        "gate_results": json.dumps({r.name: {"status": r.status, "observed": r.observed, "required": r.required}
                                    for r in gates.results}),
        "trial_count": 1,  # one pre-registered configuration (random starts are not trials)
        "holdout_evaluated": 0,
        "validation_report_path": report_path.as_posix(),
        "model_path": model_path.as_posix(),
        "library_versions": json.dumps(library_versions(), sort_keys=True),
        "config_hash": config_hash(settings),
        "git_commit": git_commit(),
        "created_at": now,
    }
    with engine.begin() as conn:
        conn.execute(tables.model_versions.insert().values(**row))
    registration = Registration(model_id, version, model_path, report_path, gates.passed)
    queries.record_pipeline_run(
        engine, command="retrain", stage="retrain-regime", status="OK", data_as_of_date=dataset.sessions[-1].date(),
        message=f"registered {model_id} as REGIME CANDIDATE; gates {'PASSED' if gates.passed else 'FAILED'}",
    )
    active = queries.active_model(engine, REGIME)
    return RegimeRetrainOutcome(
        folds=folds, walk_forward=walk_forward, comparison=comparison, evidence=evidence, gates=gates,
        final=final, registration=registration, active=dict(active) if active is not None else None,
    )
