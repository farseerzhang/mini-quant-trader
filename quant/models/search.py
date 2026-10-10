"""Hyperparameter search over retrain.search.space (proposal sections 15.1 and 15.4).

Candidates: every combination of the configured values when there are at most
``max_trials`` of them, otherwise ``max_trials`` combinations drawn at random
(seed ``retrain.search.seed``). Each candidate runs the full walk-forward.

Selection: the highest *mean minus one standard deviation* of the per-fold
log-loss skill (log-loss relative to that fold's base-rate reference). Skill
makes folds of different difficulty comparable; subtracting the spread prefers
a candidate that is good in most years over one that is very good in a few.
Trading results are judged by the gates, not used here, so the backtest is not
tuned on. Every candidate tried is counted in the trial log.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quant.config.settings import Settings
from quant.models.dataset import Dataset
from quant.models.folds import Fold
from quant.models.walk_forward import WalkForwardResult, run_walk_forward


@dataclass(frozen=True)
class SearchResult:
    model_type: str
    # One row per candidate: parameters, fold-metric summary and selection score, in trial order.
    trials: pd.DataFrame
    best_parameters: dict[str, Any]
    best: WalkForwardResult

    @property
    def trial_count(self) -> int:
        return int(len(self.trials))


def candidates(space: dict[str, tuple], max_trials: int, seed: int) -> list[dict[str, Any]]:
    """The parameter sets to try, in a deterministic order."""
    names = sorted(space)
    grid = [dict(zip(names, values)) for values in itertools.product(*(space[name] for name in names))]
    if len(grid) <= max_trials:
        return grid
    chosen = np.random.default_rng(seed).choice(len(grid), size=max_trials, replace=False)
    return [grid[i] for i in sorted(chosen)]


def selection_score(fold_log_loss_skill: pd.Series) -> float:
    """Mean minus one (population) standard deviation of the per-fold log-loss skill."""
    return float(fold_log_loss_skill.mean() - fold_log_loss_skill.std(ddof=0))


def search(dataset: Dataset, folds: list[Fold], model_type: str, settings: Settings) -> SearchResult:
    """Run every candidate for ``model_type`` and pick the best by ``selection_score``."""
    space = settings.retrain.search.space.get(model_type)
    if not space:
        raise ValueError(f"retrain.search.space has no entry for {model_type!r}")
    rows, results = [], []
    for trial, parameters in enumerate(candidates(space, settings.retrain.search.max_trials, settings.retrain.search.seed), 1):
        result = run_walk_forward(dataset, folds, model_type, parameters, settings)
        summary = result.summary()
        rows.append(
            {
                "trial": trial,
                **{f"param_{name}": value for name, value in parameters.items()},
                **summary.to_dict(),
                "score": selection_score(result.folds["log_loss_skill"]),
            }
        )
        results.append(result)
    trials = pd.DataFrame(rows).set_index("trial")
    best = int(np.argmax(trials["score"].to_numpy()))  # first of equal scores: deterministic
    return SearchResult(
        model_type=model_type,
        trials=trials,
        best_parameters=dict(results[best].parameters),
        best=results[best],
    )
