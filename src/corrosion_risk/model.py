"""The submitted model: a LightGBM regressor on 0/1 labels, shrunk toward 0.5.

Regressing with squared error on a 0/1 target minimises the Brier score directly,
so no separate calibration step is needed in-distribution. Out of distribution the
model is over-confident; shrinking toward the base rate (0.5 on the balanced scored
set) is a one-parameter correction that cannot reorder predictions.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

ALPHA = 0.7  # shrinkage factor used for the final submission
N_JOBS = 3

PARAMS: dict = {
    "n_estimators": 300,
    "learning_rate": 0.04,
    "max_depth": 5,
    "num_leaves": 31,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "objective": "regression",
    "verbosity": -1,
}


def make_model(seed: int = 42, **overrides) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(**{**PARAMS, "random_state": seed, "n_jobs": N_JOBS, **overrides})


def shrink(p: np.ndarray, alpha: float) -> np.ndarray:
    """Pull clipped probabilities toward 0.5: p -> 0.5 + alpha * (p - 0.5)."""
    return 0.5 + alpha * (np.clip(p, 0.0, 1.0) - 0.5)


def fit_predict(
    X_train: pd.DataFrame, y_train: np.ndarray, X_pred: pd.DataFrame, seed: int = 42
) -> np.ndarray:
    """Raw (unshrunk) predictions clipped to [0, 1]."""
    model = make_model(seed).fit(X_train, y_train)
    return np.clip(model.predict(X_pred), 0.0, 1.0)
