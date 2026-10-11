"""Pre-defined pass criteria for a PREDICTOR candidate (proposal sections 15.4 and 16).

The gates are fixed in ``retrain.gates.predictor`` before a run and are never
changed after seeing results. Each is reported as PASS, FAIL, DISABLED (set to
false in the config) or NOT EVALUATED (the holdout gate, unless the holdout was
explicitly evaluated). A candidate passes when no gate FAILs; `promote` refuses
a candidate that did not pass unless overridden with a logged reason.

The leak gate does not trust earlier code: it re-checks the actual folds and
predictions (every boundary purged by at least ``embargo_days`` sessions, and
every prediction made out of sample).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd

from quant.config.settings import PREDICTOR_GATES
from quant.models.folds import Fold
from quant.models.ml_backtest import MLBacktest, check_out_of_sample
from quant.models.walk_forward import WalkForwardResult

PASS, FAIL, DISABLED, NOT_EVALUATED = "PASS", "FAIL", "DISABLED", "NOT EVALUATED"


@dataclass(frozen=True)
class LeakCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class PredictorEvidence:
    """Everything the predictor gates are computed from."""

    leak_checks: list[LeakCheck]
    calibration_error: float  # mean over the walk-forward folds
    sharpe: float  # walk-forward ML backtest, after costs
    baseline_sharpe: float
    max_drawdown: float
    baseline_max_drawdown: float
    folds: int
    folds_higher_sharpe: int
    holdout_sharpe: float | None = None
    holdout_baseline_sharpe: float | None = None


@dataclass(frozen=True)
class GateResult:
    name: str
    status: str
    observed: str
    required: str


@dataclass(frozen=True)
class GateReport:
    results: list[GateResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(result.status != FAIL for result in self.results)

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([vars(result) for result in self.results]).set_index("name")


def leak_checks(
    folds: list[Fold],
    result: WalkForwardResult,
    sessions: pd.DatetimeIndex,
    holdout_start: pd.Timestamp,
    embargo: int,
) -> list[LeakCheck]:
    """Re-check every boundary and every prediction of a walk-forward run (section 21)."""
    position = {stamp: i for i, stamp in enumerate(sessions)}
    checks = []

    gaps = [position[f.validation_start] - position[f.train_end] - 1 for f in folds]
    overlap = any((f.train >= f.validation_start).any() for f in folds)
    checks.append(LeakCheck(
        "training purged before validation",
        min(gaps) >= embargo and not overlap,
        f"smallest gap {min(gaps)} sessions (need {embargo})",
    ))

    holdout_gap = sessions.get_indexer([holdout_start])[0] - position[folds[-1].validation_end] - 1
    touches_holdout = any((f.train >= holdout_start).any() or (f.validation >= holdout_start).any() for f in folds)
    checks.append(LeakCheck(
        "development purged before the holdout",
        holdout_gap >= embargo and not touches_holdout,
        f"gap {holdout_gap} sessions (need {embargo}); folds touch the holdout: {touches_holdout}",
    ))

    calibration_gaps = [
        position[pd.Timestamp(row.calibration_start)] - position[pd.Timestamp(row.fit_end)] - 1
        for row in result.folds.itertuples()
    ]
    checks.append(LeakCheck(
        "model fit purged before calibration",
        min(calibration_gaps) >= embargo,
        f"smallest gap {min(calibration_gaps)} sessions (need {embargo})",
    ))

    try:
        check_out_of_sample(result.predictions, folds, sessions, embargo)
        duplicated = bool(result.predictions.index.duplicated().any())
        checks.append(LeakCheck(
            "predictions out of sample",
            not duplicated,
            "every prediction from the fold whose validation window contains it"
            + ("; duplicated predictions" if duplicated else ""),
        ))
    except ValueError as exc:
        checks.append(LeakCheck("predictions out of sample", False, str(exc)))
    return checks


def collect_evidence(
    name: str,
    result: WalkForwardResult,
    backtest: MLBacktest,
    folds: list[Fold],
    sessions: pd.DatetimeIndex,
    holdout_start: pd.Timestamp,
    embargo: int,
    holdout: Mapping[str, float] | None = None,
) -> PredictorEvidence:
    """``holdout`` (optional): {'sharpe': ..., 'baseline_sharpe': ...} from an explicit holdout evaluation."""
    comparison = backtest.versus_baseline(name)
    return PredictorEvidence(
        leak_checks=leak_checks(folds, result, sessions, holdout_start, embargo),
        calibration_error=float(result.folds["calibration_error"].mean()),
        sharpe=comparison["sharpe"],
        baseline_sharpe=comparison["baseline_sharpe"],
        max_drawdown=comparison["max_drawdown"],
        baseline_max_drawdown=comparison["baseline_max_drawdown"],
        folds=comparison["folds"],
        folds_higher_sharpe=comparison["folds_higher_sharpe"],
        holdout_sharpe=None if holdout is None else float(holdout["sharpe"]),
        holdout_baseline_sharpe=None if holdout is None else float(holdout["baseline_sharpe"]),
    )


def _status(ok: bool) -> str:
    return PASS if ok else FAIL


def evaluate_predictor_gates(gates: Mapping[str, Any], evidence: PredictorEvidence) -> GateReport:
    """Evaluate ``retrain.gates.predictor`` against the evidence, in config order."""
    unknown = sorted(set(gates) - set(PREDICTOR_GATES))
    if unknown:
        raise ValueError(f"no evaluator for gate(s): {', '.join(unknown)}")
    e = evidence
    results = []
    for name, setting in gates.items():
        if setting is False:
            results.append(GateResult(name, DISABLED, "-", "disabled in retrain.gates.predictor"))
            continue
        if name == "leak_tests_pass":
            failed = [check.name for check in e.leak_checks if not check.passed]
            results.append(GateResult(
                name, _status(not failed),
                f"{len(e.leak_checks) - len(failed)}/{len(e.leak_checks)} checks passed"
                + (f" (failed: {', '.join(failed)})" if failed else ""),
                "every leak check passes",
            ))
        elif name == "beats_baseline_after_costs":
            results.append(GateResult(
                name, _status(e.sharpe > e.baseline_sharpe),
                f"Sharpe {e.sharpe:.2f} vs regime_switch {e.baseline_sharpe:.2f}",
                "higher Sharpe than regime_switch after costs",
            ))
        elif name == "min_fraction_of_folds_beating_baseline":
            fraction = e.folds_higher_sharpe / e.folds
            results.append(GateResult(
                name, _status(fraction >= float(setting)),
                f"{e.folds_higher_sharpe}/{e.folds} folds = {fraction:.0%}",
                f">= {float(setting):.0%} of folds with a higher Sharpe",
            ))
        elif name == "max_drawdown_not_worse_than_baseline":
            results.append(GateResult(
                name, _status(e.max_drawdown >= e.baseline_max_drawdown),
                f"{e.max_drawdown:.1%} vs regime_switch {e.baseline_max_drawdown:.1%}",
                "drawdown no deeper than regime_switch",
            ))
        elif name == "max_calibration_error":
            results.append(GateResult(
                name, _status(e.calibration_error <= float(setting)),
                f"{e.calibration_error:.3f} (mean over folds)",
                f"<= {float(setting):.3f}",
            ))
        elif name == "holdout_not_worse_than_baseline":
            if e.holdout_sharpe is None:
                results.append(GateResult(name, NOT_EVALUATED, "-", "evaluated only with --evaluate-holdout"))
            else:
                results.append(GateResult(
                    name, _status(e.holdout_sharpe >= e.holdout_baseline_sharpe),
                    f"holdout Sharpe {e.holdout_sharpe:.2f} vs regime_switch {e.holdout_baseline_sharpe:.2f}",
                    "holdout Sharpe no lower than regime_switch",
                ))
    return GateReport(results)
