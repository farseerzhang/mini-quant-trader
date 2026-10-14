"""`quant altdata research`: stage 1 of A4 round 1 (altdata_proposal.md section 8.1).

Per feature: the daily cross-sectional rank IC against the label's 5-day excess return over
the sessions of the 11 development validation windows (holdout start fixed at 2025-10-10),
the t-statistic on non-overlapping sessions, and the number of folds whose mean IC has the
overall sign. Pass = |t| >= 2 and at least 7 of 11 folds. Nothing here touches the holdout.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from quant.models.feature_research import ic_summary, rank_ic
from quant.models.folds import Fold, walk_forward_folds

HOLDOUT_START = pd.Timestamp("2025-10-10")  # fixed by the pre-registration (8.1)
MIN_ABS_T = 2.0
MIN_FOLDS_SAME_SIGN = 7


@dataclass(frozen=True)
class FeatureResult:
    name: str
    mean_ic: float
    t_stat: float
    folds_same_sign: int
    folds: int
    sessions: int
    fold_ic: list[float]
    coverage: list[float]  # per fold: mean share of pool stocks with a value

    @property
    def passed(self) -> bool:
        return (not np.isnan(self.t_stat) and abs(self.t_stat) >= MIN_ABS_T
                and self.folds_same_sign >= MIN_FOLDS_SAME_SIGN)


def development_folds(labelled_sessions: pd.DatetimeIndex, settings) -> list[Fold]:
    return walk_forward_folds(labelled_sessions, HOLDOUT_START, settings.retrain.validation,
                              settings.model.embargo_days)


def stage1(features: dict[str, pd.DataFrame], target: pd.DataFrame, folds: list[Fold], horizon: int
           ) -> list[FeatureResult]:
    validation = pd.DatetimeIndex(sorted(set().union(*(set(f.validation) for f in folds))))
    results = []
    for name, feature in features.items():
        ic = rank_ic(feature.reindex(index=validation), target.reindex(index=validation))
        summary = ic_summary(ic, horizon) if len(ic) else {"mean_ic": np.nan, "t_stat": np.nan}
        sign = np.sign(summary["mean_ic"])
        fold_ic, coverage = [], []
        for fold in folds:
            mine = ic[ic.index.isin(fold.validation)]
            fold_ic.append(float(mine.mean()) if len(mine) else float("nan"))
            values = feature.reindex(index=fold.validation, columns=target.columns)
            coverage.append(float(values.notna().mean().mean()))
        same = sum(1 for value in fold_ic if not np.isnan(value) and np.sign(value) == sign and sign != 0)
        results.append(FeatureResult(name, float(summary["mean_ic"]), float(summary["t_stat"]), same, len(folds),
                                     int(len(ic)), fold_ic, coverage))
    return results


def write_report(results: list[FeatureResult], folds: list[Fold], reports_dir: str | Path) -> Path:
    now = datetime.now(timezone.utc)
    path = Path(reports_dir) / "altdata" / f"a4-round1-{now:%Y%m%dT%H%M%SZ}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# A4 round 1, stage 1: Form 4 features ({now:%Y-%m-%d %H:%M} UTC)",
        "",
        f"Pre-registration: altdata_proposal.md section 8.1. Rank IC against the 5-day excess return over the "
        f"validation sessions of {len(folds)} folds ({folds[0].validation_start.date()} .. "
        f"{folds[-1].validation_end.date()}), holdout start fixed at {HOLDOUT_START.date()}; no holdout data. "
        f"Pass = |t| >= {MIN_ABS_T:g} and >= {MIN_FOLDS_SAME_SIGN} of {len(folds)} folds with the overall sign.",
        "",
        "| Feature | Mean IC | t | Folds same sign | Sessions | Result |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r.name} | {r.mean_ic:+.4f} | {r.t_stat:+.2f} | {r.folds_same_sign}/{r.folds} | "
                     f"{r.sessions} | {'PASS' if r.passed else 'fail'} |")
    lines += ["", "## Per fold: mean IC (coverage = share of pool stocks with a value)", "",
              "| Fold | Validation | " + " | ".join(r.name for r in results) + " |",
              "|---|---|" + "---|" * len(results)]
    for i, fold in enumerate(folds):
        cells = [f"{r.fold_ic[i]:+.4f} ({r.coverage[i]:.0%})" if not np.isnan(r.fold_ic[i]) else "-" for r in results]
        lines.append(f"| {fold.number} | {fold.validation_start.date()} .. {fold.validation_end.date()} | "
                     + " | ".join(cells) + " |")
    passed = [r.name for r in results if r.passed]
    lines += ["", f"**Stage 1 outcome:** {', '.join(passed) + ' pass' if passed else 'no feature passes'}. "
              "With 4 features about 0.2 passes are expected by chance; a pass is a candidate for stage 2, "
              "not a finding.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
