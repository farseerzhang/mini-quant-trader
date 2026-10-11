"""The final refit of a chosen predictor configuration (proposal section 15.5).

After the search has chosen the model type and hyperparameters, the model is
refit with those fixed settings on every usable development session (purged
before the holdout), sampled like the folds. A refit on all data leaves nothing
unseen to calibrate on, so the calibrator is fit on the walk-forward
out-of-sample predictions instead: raw scores from models that had not seen
those sessions, paired with what actually happened. Those scores come from the
fold models rather than the refit model, which is the price of using all the
data for the model itself.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from quant.config.settings import Settings
from quant.models.calibration import Calibrator
from quant.models.dataset import INPUT_COLUMNS, Dataset
from quant.models.folds import development_sessions, sample_sessions
from quant.models.predictors import build_model
from quant.models.walk_forward import WalkForwardResult


@dataclass
class FinalModel:
    model_type: str
    parameters: dict[str, Any]
    input_columns: tuple[str, ...]
    model: Any
    calibrator: Calibrator
    training_start: pd.Timestamp
    training_end: pd.Timestamp
    training_rows: int
    # XGBoost is stored in its own format next to the joblib file (section 6.10).
    native_model_file: str | None = None

    def predict(self, inputs: pd.DataFrame) -> pd.Series:
        """Calibrated probability per row; ``inputs`` must have the training input columns."""
        missing = [column for column in self.input_columns if column not in inputs]
        if missing:
            raise ValueError(f"inputs are missing: {', '.join(missing)}")
        raw = self.model.predict_proba(inputs.loc[:, list(self.input_columns)].to_numpy(dtype=float))[:, 1]
        return pd.Series(self.calibrator.transform(raw), index=inputs.index, name="probability")

    def save(self, path: Path) -> Path:
        """joblib file with everything; XGBoost's booster goes to ``<path>.ubj`` in its native format."""
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.model_type == "xgboost":
            native = path.with_suffix(".ubj")
            self.model.save_model(native)
            joblib.dump(replace(self, model=None, native_model_file=native.name), path)
        else:
            joblib.dump(self, path)
        return path


def load_final_model(path: Path) -> FinalModel:
    path = Path(path)
    model = joblib.load(path)
    if not isinstance(model, FinalModel):
        raise ValueError(f"{path} does not hold a FinalModel")
    if model.native_model_file is not None:
        from xgboost import XGBClassifier

        booster = XGBClassifier()
        booster.load_model(path.parent / model.native_model_file)
        model = replace(model, model=booster)
    return model


def refit_final(
    dataset: Dataset,
    holdout_start: pd.Timestamp,
    walk_forward: WalkForwardResult,
    settings: Settings,
) -> FinalModel:
    """Refit ``walk_forward``'s configuration on all development data; calibrate out of sample."""
    labelled = dataset.labelled()
    sessions = development_sessions(labelled.sessions, holdout_start, settings.model.embargo_days)
    rows = labelled.rows_on(sample_sessions(sessions, settings.retrain.sample_every_sessions, dataset.sessions))
    x = rows.frame.loc[:, list(INPUT_COLUMNS)].to_numpy(dtype=float)
    y = rows.frame["label"].to_numpy(dtype=float)
    model = build_model(walk_forward.model_type, walk_forward.parameters, settings.retrain.search.seed).fit(x, y)

    oos = walk_forward.predictions
    calibrator = Calibrator(settings.model.calibration).fit(oos["raw"].to_numpy(), oos["label"].to_numpy())
    dates = rows.frame.index.get_level_values("date")
    return FinalModel(
        model_type=walk_forward.model_type,
        parameters=dict(walk_forward.parameters),
        input_columns=tuple(INPUT_COLUMNS),
        model=model,
        calibrator=calibrator,
        training_start=dates.min(),
        training_end=dates.max(),
        training_rows=int(len(y)),
    )


def explain(final: FinalModel) -> pd.Series:
    """What the model leans on: LR coefficients per standardized input, or RF importances."""
    if final.model_type == "logistic_regression":
        coefficients = final.model.named_steps["model"].coef_[0]
        return pd.Series(coefficients, index=final.input_columns, name="coefficient").reindex(
            pd.Series(np.abs(coefficients), index=final.input_columns).sort_values(ascending=False).index
        )
    if hasattr(final.model, "feature_importances_"):
        return pd.Series(final.model.feature_importances_, index=final.input_columns, name="importance").sort_values(
            ascending=False
        )
    raise NotImplementedError(f"no explanation for {final.model_type}")
