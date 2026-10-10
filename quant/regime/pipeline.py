"""Calculate and persist the rule-based market regime from stored features."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from sqlalchemy import Engine

from quant.config.settings import FeatureSettings, RegimeSettings
from quant.persistence import queries
from quant.regime.rule_based import BEAR, BULL, SIDEWAYS, average_dwell, calculate_rule_regimes

RULE_METHOD = "RULE"


@dataclass(frozen=True)
class RegimeSummary:
    model_version: str
    rows: int
    first_date: date
    last_date: date
    latest_regime: str
    latest_score: int
    sessions_by_regime: dict[str, int]
    average_dwell_days: float


def calculate_and_store_regimes(
    engine: Engine, regime_settings: RegimeSettings, feature_settings: FeatureSettings
) -> RegimeSummary:
    """Recompute the full rule-based regime history and replace the stored one."""
    inputs = queries.load_regime_inputs(
        engine, feature_settings.feature_version, feature_settings.market_feature_version
    )
    if inputs.empty:
        raise ValueError(
            f"No {feature_settings.market_feature_version} market features stored; "
            "calculate features before the regime"
        )
    regimes = calculate_rule_regimes(inputs, regime_settings)
    if regimes.empty:
        raise ValueError("No session has all five regime inputs; more history is needed")

    rows = [
        {
            "date": stamp.date(),
            "regime": row.regime,
            "regime_score": float(row.regime_score),
            "confidence": float(row.confidence),
        }
        for stamp, row in regimes.iterrows()
    ]
    model_version = regime_settings.rule_version
    stored = queries.replace_regimes(engine, RULE_METHOD, model_version, rows)

    latest = regimes.iloc[-1]
    counts = regimes["regime"].value_counts()
    summary = RegimeSummary(
        model_version=model_version,
        rows=stored,
        first_date=regimes.index[0].date(),
        last_date=regimes.index[-1].date(),
        latest_regime=str(latest["regime"]),
        latest_score=int(latest["regime_score"]),
        sessions_by_regime={name: int(counts.get(name, 0)) for name in (BULL, SIDEWAYS, BEAR)},
        average_dwell_days=average_dwell(regimes["regime"]),
    )
    queries.record_pipeline_run(
        engine,
        command="calculate-regime",
        stage="regime",
        status="OK",
        data_as_of_date=summary.last_date,
        message=(
            f"Stored {stored} {model_version} regimes; latest {summary.latest_regime} "
            f"(score {summary.latest_score:+d})"
        ),
    )
    return summary
