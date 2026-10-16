"""`quant altdata research`: A4 rounds 1 (Form 4, section 8.1) and 2 (analysts, section 8.2) of
altdata_proposal.md; stages 1 and 2 follow the same protocol in both rounds.

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

HOLDOUT_START = pd.Timestamp("2025-10-10")  # fixed by the pre-registration (8.1, 8.2)
ROUNDS = {1: ("Form 4 features", "8.1", "ml_lr_insider"), 2: ("analyst features", "8.2", "ml_lr_analysts")}
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


STAGE2_MODEL = "logistic_regression"
STAGE2_PARAMETERS = {"C": 1.0}  # the 15.4.6 / 15.4.7 configuration (8.1); no new search


@dataclass
class Stage2Result:
    passing: list[str]
    folds: list[Fold]
    rows_trial: int
    rows_full: int
    first_session: pd.Timestamp
    overall: pd.DataFrame  # the backtest's overall table (trial, reference, baseline, benchmarks)
    gates: object  # GateReport of the trial
    reference_gates: object  # GateReport of the same model without the features (descriptive)
    evidence: object
    reference_evidence: object
    trial_auc: float
    reference_auc: float
    round_no: int = 1


def trial_dataset(dataset, features: dict[str, pd.DataFrame], passing: list[str]):
    """The dataset with the passing features as extra columns; rows without them dropped (8.1)."""
    from quant.models.feature_research import with_candidates

    return with_candidates(dataset, {name: features[name] for name in passing})


def trial_folds(folds: list[Fold], sessions: pd.DatetimeIndex) -> list[Fold]:
    """The same folds with their training windows cut to the sessions the trial rows have (the
    validation windows are unchanged; the trial's rows start later, see 8.1)."""
    out = []
    for fold in folds:
        missing = fold.validation.difference(sessions)
        if len(missing):
            raise ValueError(f"fold {fold.number}: {len(missing)} validation session(s) have no trial rows")
        out.append(Fold(fold.number, fold.train.intersection(sessions), fold.validation))
    return out


def stage2(engine, settings, features: dict[str, pd.DataFrame], passing: list[str], round_no: int = 1) -> Stage2Result:
    """One trial (8.1): the 35 inputs plus the passing features, everything else unchanged, judged by
    the gates against regime_switch. The same model without the features on the same rows is run
    alongside as a descriptive reference; it decides nothing."""
    from quant.backtest.data import load_prices
    from quant.models.dataset import INPUT_COLUMNS, load_dataset
    from quant.models.gates import collect_evidence, evaluate_predictor_gates
    from quant.models.ml_backtest import run_ml_backtest
    from quant.models.walk_forward import run_walk_forward
    from quant.persistence import queries
    from quant.regime.pipeline import RULE_METHOD
    from quant.strategies import PANEL_COLUMNS

    stocks = list(settings.market.stocks)
    embargo = settings.model.embargo_days
    dataset = load_dataset(engine, settings)
    # The 11 folds come from the full dataset (min_train_years would cut folds if they were built
    # from the shorter trial rows); dropping rows only shortens the training windows.
    trial = trial_dataset(dataset, features, passing)
    folds = trial_folds(development_folds(dataset.labelled().sessions, settings), trial.sessions)
    columns = [*INPUT_COLUMNS, *passing]
    trial_name, reference_name = ROUNDS[round_no][2], "ml_lr_same_rows"
    trial_result = run_walk_forward(trial, folds, STAGE2_MODEL, STAGE2_PARAMETERS, settings, input_columns=columns)
    reference = run_walk_forward(trial, folds, STAGE2_MODEL, STAGE2_PARAMETERS, settings)
    prices = load_prices(engine, [*stocks, "SPY"])
    panel = queries.load_feature_panel(engine, settings.features.feature_version, stocks, PANEL_COLUMNS)
    regimes = queries.load_regimes(engine, RULE_METHOD, settings.regime.rule_version)
    backtest = run_ml_backtest({trial_name: trial_result, reference_name: reference}, folds, prices, panel,
                               regimes, settings)
    evidence = collect_evidence(trial_name, trial_result, backtest, folds, prices.close.index, HOLDOUT_START, embargo)
    reference_evidence = collect_evidence(reference_name, reference, backtest, folds, prices.close.index,
                                          HOLDOUT_START, embargo)
    gate_settings = settings.retrain.gates["predictor"]
    return Stage2Result(
        passing=passing, folds=folds, rows_trial=len(trial.frame), rows_full=len(dataset.frame),
        first_session=trial.sessions[0], overall=backtest.overall,
        gates=evaluate_predictor_gates(gate_settings, evidence),
        reference_gates=evaluate_predictor_gates(gate_settings, reference_evidence),
        evidence=evidence, reference_evidence=reference_evidence,
        trial_auc=float(trial_result.folds["auc"].mean()), reference_auc=float(reference.folds["auc"].mean()),
        round_no=round_no)


def write_stage2_report(result: Stage2Result, reports_dir: str | Path) -> Path:
    now = datetime.now(timezone.utc)
    title, section, trial_name = ROUNDS[result.round_no]
    path = Path(reports_dir) / "altdata" / f"a4-round{result.round_no}-stage2-{now:%Y%m%dT%H%M%SZ}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    e, r = result.evidence, result.reference_evidence
    lines = [
        f"# A4 round {result.round_no}, stage 2: {', '.join(result.passing)} ({now:%Y-%m-%d %H:%M} UTC)",
        "",
        f"Pre-registration: altdata_proposal.md section {section}. Logistic regression C = 1.0, the 35 inputs plus "
        f"{', '.join(result.passing)}; {len(result.folds)} folds {result.folds[0].validation_start.date()} .. "
        f"{result.folds[-1].validation_end.date()} (holdout start fixed at {HOLDOUT_START.date()}; no holdout); "
        f"`ml_select` as configured; after costs. Rows without the feature are dropped: {result.rows_trial:,} of "
        f"{result.rows_full:,} rows, from {result.first_session.date()}.",
        "",
        "| Portfolio | CAGR | Sharpe | Max drawdown | Turnover/yr |",
        "|---|---|---|---|---|",
    ]
    for name, row in result.overall.iterrows():
        lines.append(f"| {name} | {row['cagr']:.1%} | {row['sharpe']:.2f} | {row['max_drawdown']:.1%} | "
                     f"{row['turnover']:.1f}x |")
    lines += [
        "",
        f"`{trial_name}` is the trial; `ml_lr_same_rows` is the same model without the feature on the same rows "
        "(descriptive only: it isolates what the feature adds and decides nothing).",
        "",
        "| | Trial | Same rows, no feature |",
        "|---|---|---|",
        f"| Mean AUC | {result.trial_auc:.4f} | {result.reference_auc:.4f} |",
        f"| Sharpe vs regime_switch | {e.sharpe:.2f} vs {e.baseline_sharpe:.2f} | {r.sharpe:.2f} vs {r.baseline_sharpe:.2f} |",
        f"| Max drawdown vs regime_switch | {e.max_drawdown:.1%} vs {e.baseline_max_drawdown:.1%} | "
        f"{r.max_drawdown:.1%} vs {r.baseline_max_drawdown:.1%} |",
        f"| Years with a higher Sharpe | {e.folds_higher_sharpe}/{e.folds} | {r.folds_higher_sharpe}/{r.folds} |",
        f"| Gates | {'PASSED' if result.gates.passed else 'FAILED'} | "
        f"{'PASSED' if result.reference_gates.passed else 'FAILED'} |",
        "",
        "## Gates of the trial",
        "",
        "| Gate | Status | Observed | Required |",
        "|---|---|---|---|",
    ]
    for gate in result.gates.results:
        lines.append(f"| {gate.name} | {gate.status} | {gate.observed} | {gate.required} |")
    lines += ["", f"**Stage 2 outcome:** gates {'PASSED: candidate for A5' if result.gates.passed else 'FAILED'}.",
              ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_report(results: list[FeatureResult], folds: list[Fold], reports_dir: str | Path, round_no: int = 1) -> Path:
    now = datetime.now(timezone.utc)
    title, section, _ = ROUNDS[round_no]
    path = Path(reports_dir) / "altdata" / f"a4-round{round_no}-{now:%Y%m%dT%H%M%SZ}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# A4 round {round_no}, stage 1: {title} ({now:%Y-%m-%d %H:%M} UTC)",
        "",
        f"Pre-registration: altdata_proposal.md section {section}. Rank IC against the 5-day excess return over the "
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
              f"With {len(results)} features about {0.05 * len(results):.2g} passes are expected by chance; a pass "
              "is a candidate for stage 2, not a finding.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
