"""Register a refit predictor as a CANDIDATE (proposal sections 6.10, 16 and 22).

Registration saves the model file and the validation report and adds one
``model_versions`` row. It never changes another row: no model becomes ACTIVE
here, and nothing that `run-daily` uses changes. Models are never overwritten;
each registration is a new id and the next version number of its model type.
"""

from __future__ import annotations

import dataclasses
import json
import platform
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sqlalchemy import Engine, func, select

from quant.backtest.persist import config_hash, git_commit
from quant.config.settings import Settings
from quant.models.final import FinalModel
from quant.models.folds import Fold
from quant.models.gates import GateReport, PredictorEvidence
from quant.models.ml_backtest import MLBacktest
from quant.models.report import validation_report
from quant.models.search import SearchResult
from quant.persistence import tables

PREDICTOR = "PREDICTOR"
CANDIDATE, ACTIVE, RETIRED = "CANDIDATE", "ACTIVE", "RETIRED"


@dataclass(frozen=True)
class Registration:
    model_id: str
    version: int
    model_path: Path
    report_path: Path
    gates_passed: bool


def library_versions() -> dict[str, str]:
    import vectorbt
    import xgboost

    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "vectorbt": vectorbt.__version__,
    }


def target_description(settings: Settings) -> str:
    t = settings.model.target
    return (
        f"{t.kind}: stock minus {t.benchmark} total return from the open of T+1 to the open of "
        f"T+1+{t.horizon_days}, above max({t.k:g} x {t.volatility_window_days}-session volatility x "
        f"sqrt({t.horizon_days}), {t.min_threshold:.2%})"
    )


def _next_version(engine: Engine, model_type: str) -> int:
    stmt = select(func.max(tables.model_versions.c.version)).where(tables.model_versions.c.model_type == model_type)
    with engine.connect() as conn:
        return int(conn.execute(stmt).scalar_one() or 0) + 1


def register_candidate(
    engine: Engine,
    settings: Settings,
    final: FinalModel,
    search: SearchResult,
    folds: list[Fold],
    backtest: MLBacktest,
    portfolio: str,
    evidence: PredictorEvidence,
    gates: GateReport,
    models_dir: Path = Path("models"),
    reports_dir: Path = Path("reports"),
    now: datetime | None = None,
) -> Registration:
    now = now or datetime.now(timezone.utc)
    model_id = f"{final.model_type}-{now:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
    version = _next_version(engine, final.model_type)
    model_path = final.save(models_dir / final.model_type / f"{model_id}.joblib")
    report_path = reports_dir / f"{model_id}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        validation_report(
            model_id, final, search, folds, backtest, portfolio, evidence, gates,
            target_description(settings), now,
        ),
        encoding="utf-8",
    )

    walk = search.best.summary()
    comparison = backtest.versus_baseline(portfolio)
    metrics = {
        "walk_forward": {key: float(value) for key, value in walk.items()},
        "backtest": {key: float(value) for key, value in comparison.items()},
        "window": [str(backtest.start.date()), str(backtest.end.date())],
    }
    row = {
        "model_id": model_id,
        "role": PREDICTOR,
        "model_type": final.model_type,
        "version": version,
        "status": CANDIDATE,
        "depends_on": settings.regime.rule_version,  # the regime indicator columns are inputs
        "training_start": final.training_start.date(),
        "training_end": final.training_end.date(),
        "feature_version": settings.features.feature_version,
        "target_definition": json.dumps(dataclasses.asdict(settings.model.target), sort_keys=True),
        "hyperparameters": json.dumps(final.parameters, sort_keys=True),
        "metrics": json.dumps(metrics, sort_keys=True),
        "gates_passed": int(gates.passed),
        "gate_results": json.dumps({r.name: {"status": r.status, "observed": r.observed, "required": r.required}
                                    for r in gates.results}),
        "trial_count": search.trial_count,
        "holdout_evaluated": int(evidence.holdout_sharpe is not None),
        "validation_report_path": report_path.as_posix(),
        "model_path": model_path.as_posix(),
        "library_versions": json.dumps(library_versions(), sort_keys=True),
        "config_hash": config_hash(settings),
        "git_commit": git_commit(),
        "created_at": now,
    }
    with engine.begin() as conn:
        conn.execute(tables.model_versions.insert().values(**row))
    return Registration(model_id, version, model_path, report_path, gates.passed)
