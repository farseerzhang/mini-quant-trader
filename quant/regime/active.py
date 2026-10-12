"""The ACTIVE REGIME model in daily use (proposal sections 6.6, 16.3 and 23 step 6).

`calculate-regime` always stores the rule-based regime. When an HMM is ACTIVE it also runs
that model's final fit forward over the stored observations: filtered probabilities only
(row t uses sessions up to t), the same minimum dwell as everywhere else, stored in
``market_regimes`` as method HMM with ``model_version`` = the model id. The signal engine
then reads the regime of the model in use; with none ACTIVE it reads the rule-based one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine

from quant.config.settings import Settings
from quant.persistence import queries
from quant.regime.hmm import load_observations
from quant.regime.hmm_walk_forward import apply_min_dwell
from quant.regime.pipeline import RULE_METHOD

REGIME = "REGIME"
HMM_METHOD = "HMM"
COMMAND = "calculate-regime"


class ActiveModelError(ValueError):
    """The ACTIVE model cannot be used as configured; the message says why."""


@dataclass(frozen=True)
class RegimeInUse:
    method: str  # RULE or HMM
    model_version: str  # rule_based-v<N> or the HMM model id
    regimes: pd.Series


@dataclass(frozen=True)
class ActiveRegimeSummary:
    model_id: str
    rows: int
    last_date: pd.Timestamp
    latest_regime: str
    latest_probability: float


def load_active_hmm(row, settings: Settings):
    """The ACTIVE model's final fit, checked against the configured observations."""
    from quant.regime.retrain import load_final_hmm  # retrain pulls in the backtest stack

    if row["model_type"] != "hmm":
        raise ActiveModelError(f"ACTIVE regime model {row['model_id']} is a {row['model_type']}; only hmm is supported")
    path = Path(row["model_path"])
    if not path.exists():
        raise ActiveModelError(f"ACTIVE regime model file {path} is missing; retrain or promote another model")
    fit = load_final_hmm(path)
    configured = tuple(settings.regime.hmm.observations)
    if tuple(fit.columns) != configured:
        raise ActiveModelError(
            f"ACTIVE regime model {row['model_id']} uses {', '.join(fit.columns)}, but regime.hmm.observations is "
            f"{', '.join(configured)}; retrain it or change the setting back"
        )
    return fit


def active_hmm_regime(fit, observations: pd.DataFrame, min_dwell_days: int) -> tuple[pd.Series, pd.DataFrame]:
    """The filtered regime per session and the probabilities behind it (pure; point in time)."""
    probabilities = fit.filtered(observations)
    return apply_min_dwell(probabilities.idxmax(axis=1), min_dwell_days), probabilities


def calculate_and_store_active_regime(engine: Engine, settings: Settings) -> ActiveRegimeSummary | None:
    """With an ACTIVE HMM, recompute its regime history and replace the stored one; else None."""
    row = queries.active_model(engine, REGIME)
    if row is None:
        return None
    fit = load_active_hmm(row, settings)
    observations = load_observations(engine, settings)
    if observations.empty:
        raise ActiveModelError("no HMM observations stored; calculate features first")
    regime, probabilities = active_hmm_regime(fit, observations, settings.regime.min_dwell_days)
    rows = [
        {"date": stamp.date(), "regime": label, "regime_score": None,
         "confidence": float(probabilities.loc[stamp, label])}
        for stamp, label in regime.items()
    ]
    stored = queries.replace_regimes(engine, HMM_METHOD, row["model_id"], rows)
    last = regime.index[-1]
    summary = ActiveRegimeSummary(row["model_id"], stored, last, regime.iloc[-1],
                                  float(probabilities.loc[last, regime.iloc[-1]]))
    queries.record_pipeline_run(
        engine, command=COMMAND, stage="active-regime", status="OK", data_as_of_date=last.date(),
        message=f"Stored {stored} regimes of ACTIVE {row['model_id']}; latest {summary.latest_regime} "
                f"(probability {summary.latest_probability:.0%})",
    )
    return summary


def regime_in_use(engine: Engine, settings: Settings) -> RegimeInUse:
    """The regime the signals use: the ACTIVE model's stored regime, else the rule-based one."""
    row = queries.active_model(engine, REGIME)
    if row is None:
        version = settings.regime.rule_version
        return RegimeInUse(RULE_METHOD, version, queries.load_regimes(engine, RULE_METHOD, version))
    regimes = queries.load_regimes(engine, HMM_METHOD, row["model_id"])
    if regimes.empty:
        raise ActiveModelError(f"no stored regime for the ACTIVE model {row['model_id']}; run `quant calculate-regime`")
    rule = queries.load_regimes(engine, RULE_METHOD, settings.regime.rule_version)
    if not rule.empty and regimes.index[-1] < rule.index[-1]:
        raise ActiveModelError(
            f"the ACTIVE model {row['model_id']}'s regime ends {regimes.index[-1].date()}, before the latest "
            f"session {rule.index[-1].date()}; run `quant calculate-regime`"
        )
    return RegimeInUse(HMM_METHOD, row["model_id"], regimes)
