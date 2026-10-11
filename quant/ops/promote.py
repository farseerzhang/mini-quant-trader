"""`quant promote <model_id> [--override "<reason>"]` (proposal sections 16.1 and 16.3).

Makes a candidate the ACTIVE model of its role, in one transaction:

- the previous ACTIVE model of that role becomes RETIRED; other roles are untouched;
- a candidate whose gates failed is refused, unless an override reason is given;
- the promotion, and any override with its reason, is logged in pipeline_runs.

Promoting a REGIME model leaves an ACTIVE predictor trained on another regime in place
but stale: `quant status` warns until it is retrained (its ``depends_on`` no longer
matches the regime in use).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import Engine, select

from quant.config.settings import Settings
from quant.models.registry import ACTIVE, CANDIDATE, RETIRED
from quant.persistence import queries, tables

COMMAND = "promote"


class PromotionError(ValueError):
    """The candidate cannot be promoted as asked; the message says why."""


@dataclass(frozen=True)
class Promotion:
    model_id: str
    role: str
    model_type: str
    retired: str | None  # the previous ACTIVE model of the role
    override_reason: str | None
    failed_gates: tuple[str, ...]
    stale_predictor: str | None  # ACTIVE predictor that was trained on another regime


def _failed_gates(row) -> tuple[str, ...]:
    try:
        results = json.loads(row["gate_results"] or "{}")
    except json.JSONDecodeError:
        return ()
    return tuple(f"{name} ({result.get('observed', '?')})" for name, result in results.items()
                 if result.get("status") == "FAIL")


def promote_model(engine: Engine, settings: Settings, model_id: str, override_reason: str | None = None) -> Promotion:
    models = tables.model_versions
    with engine.connect() as conn:
        row = conn.execute(select(models).where(models.c.model_id == model_id)).mappings().first()
    if row is None:
        raise PromotionError(f"no model {model_id!r} in model_versions (see `quant status` and the reports folder)")
    if row["status"] == ACTIVE:
        raise PromotionError(f"{model_id} is already the ACTIVE {row['role']} model")
    if override_reason is not None and not override_reason.strip():
        raise PromotionError("--override needs a reason, e.g. --override \"forward test of the HMM regime\"")
    failed = _failed_gates(row)
    if not row["gates_passed"] and override_reason is None:
        detail = f": {', '.join(failed)}" if failed else ""
        raise PromotionError(
            f"{model_id} failed its gates{detail}. Promote it anyway only with --override \"<reason>\" (logged)"
        )
    if not Path(row["model_path"]).exists():
        raise PromotionError(f"the model file {row['model_path']} is missing; retrain to recreate it")

    role = row["role"]
    with engine.begin() as conn:
        previous = conn.execute(
            select(models.c.model_id).where(models.c.role == role, models.c.status == ACTIVE)
        ).scalar_one_or_none()
        if previous is not None:
            conn.execute(models.update().where(models.c.model_id == previous).values(status=RETIRED))
        conn.execute(models.update().where(models.c.model_id == model_id).values(status=ACTIVE))

    stale = None
    if role == "REGIME":
        predictor = queries.active_model(engine, "PREDICTOR")
        if predictor is not None and predictor["depends_on"] and predictor["depends_on"] != model_id:
            stale = predictor["model_id"]
    elif row["depends_on"]:
        regime = queries.active_model(engine, "REGIME")
        in_use = regime["model_id"] if regime is not None else settings.regime.rule_version
        if row["depends_on"] != in_use:
            stale = model_id

    parts = [f"{model_id} ACTIVE for {role}"]
    if previous:
        parts.append(f"{previous} RETIRED")
    overridden = override_reason is not None and not row["gates_passed"]
    if overridden:
        parts.append(f"OVERRIDE of failed gates ({', '.join(failed) or 'gates not passed'}): {override_reason.strip()}")
    queries.record_pipeline_run(engine, command=COMMAND, stage="promote", status="OK", message="; ".join(parts))
    return Promotion(model_id, role, row["model_type"], previous,
                     override_reason.strip() if overridden else None, failed, stale)


__all__ = ["CANDIDATE", "Promotion", "PromotionError", "promote_model"]
