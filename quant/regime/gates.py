"""Pre-defined pass criteria for a REGIME candidate (proposal sections 13.1 and 16.1).

Each key of ``retrain.gates.regime`` is reported as PASS, FAIL or DISABLED
(``false`` in the config), in config order; a candidate passes when no gate
FAILs. As for the predictor, the point-in-time gate does not trust earlier
tests: it re-runs the walk-forward on the real observations cut at given
sessions and requires identical regimes up to each cut.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from quant.config.settings import REGIME_GATES, Settings
from quant.models.folds import Fold
from quant.models.gates import DISABLED, FAIL, PASS, GateReport, GateResult, LeakCheck
from quant.regime.comparison import RegimeComparison, portfolio_name
from quant.regime.hmm_walk_forward import MIN_MAPPING_AGREEMENT, HMMWalkForward, run_hmm_walk_forward
from quant.regime.rule_based import average_dwell

BASELINE_STRATEGY = "regime_switch"


@dataclass(frozen=True)
class RegimeEvidence:
    point_in_time_checks: list[LeakCheck]
    agreement: pd.Series  # per consecutive pair of fits
    average_dwell: float  # sessions, after the minimum-dwell rule
    sharpe: float  # BASELINE_STRATEGY traded with the candidate regime, after costs
    baseline_sharpe: float  # ... with the rule-based regime


def point_in_time_checks(
    observations: pd.DataFrame,
    folds: list[Fold],
    settings: Settings,
    result: HMMWalkForward,
    cuts: Sequence[pd.Timestamp],
) -> list[LeakCheck]:
    """Re-run the walk-forward on observations ending at each cut; regimes up to it must match."""
    checks = []
    for cut in cuts:
        active = [fold for fold in folds if fold.validation_start <= cut]
        truncated = run_hmm_walk_forward(observations.loc[:cut], active, settings)
        common = truncated.raw_regime.index
        same_raw = truncated.raw_regime.equals(result.raw_regime.loc[common])
        same_regime = truncated.regime.equals(result.regime.loc[common])
        same_probabilities = np.allclose(
            truncated.probabilities.to_numpy(), result.probabilities.loc[common].to_numpy(), rtol=0, atol=1e-9
        )
        checks.append(LeakCheck(
            f"regimes unchanged when history ends {pd.Timestamp(cut).date()}",
            same_raw and same_regime and same_probabilities,
            f"{len(common)} sessions compared; probabilities, raw and dwelled regimes identical: "
            f"{same_probabilities}, {same_raw}, {same_regime}",
        ))
    return checks


def default_cuts(folds: list[Fold]) -> list[pd.Timestamp]:
    """Two cut sessions: inside an early fold and inside the last one."""
    early = folds[min(3, len(folds) - 1)].validation
    late = folds[-1].validation
    return [early[len(early) // 2], late[len(late) // 2]]


def collect_regime_evidence(
    observations: pd.DataFrame,
    folds: list[Fold],
    settings: Settings,
    result: HMMWalkForward,
    comparison: RegimeComparison,
    candidate: str = "hmm",
    baseline: str = "rule",
) -> RegimeEvidence:
    versus = comparison.versus(
        portfolio_name(BASELINE_STRATEGY, candidate), portfolio_name(BASELINE_STRATEGY, baseline)
    )
    return RegimeEvidence(
        point_in_time_checks=point_in_time_checks(observations, folds, settings, result, default_cuts(folds)),
        agreement=result.agreement,
        average_dwell=float(average_dwell(result.regime)),
        sharpe=versus["sharpe"],
        baseline_sharpe=versus["baseline_sharpe"],
    )


def _status(ok: bool) -> str:
    return PASS if ok else FAIL


def evaluate_regime_gates(gates: Mapping[str, Any], evidence: RegimeEvidence) -> GateReport:
    unknown = sorted(set(gates) - set(REGIME_GATES))
    if unknown:
        raise ValueError(f"no evaluator for regime gate(s): {', '.join(unknown)}")
    e = evidence
    results = []
    for name, setting in gates.items():
        if setting is False:
            results.append(GateResult(name, DISABLED, "-", "disabled in retrain.gates.regime"))
        elif name == "filtered_only_test_pass":
            failed = [c.name for c in e.point_in_time_checks if not c.passed]
            results.append(GateResult(
                name, _status(not failed and len(e.point_in_time_checks) > 0),
                f"{len(e.point_in_time_checks) - len(failed)}/{len(e.point_in_time_checks)} cuts identical"
                + (f" (failed: {', '.join(failed)})" if failed else ""),
                "regimes unchanged when later sessions are removed",
            ))
        elif name == "state_mapping_stable_across_refits":
            lowest = float(e.agreement.min()) if len(e.agreement) else float("nan")
            results.append(GateResult(
                name, _status(len(e.agreement) > 0 and lowest >= MIN_MAPPING_AGREEMENT),
                f"lowest agreement {lowest:.1%} over {len(e.agreement)} consecutive pairs",
                f">= {MIN_MAPPING_AGREEMENT:.0%} for every consecutive pair of fits",
            ))
        elif name == "min_avg_regime_dwell_days":
            results.append(GateResult(
                name, _status(e.average_dwell >= float(setting)),
                f"{e.average_dwell:.1f} sessions",
                f">= {float(setting):g} sessions",
            ))
        elif name == "beats_rule_based_regime_after_costs":
            results.append(GateResult(
                name, _status(e.sharpe > e.baseline_sharpe),
                f"{BASELINE_STRATEGY} Sharpe {e.sharpe:.4f} vs {e.baseline_sharpe:.4f} with the rule-based regime",
                "higher Sharpe after costs than with the rule-based regime",
            ))
    return GateReport(results)
