"""Validation tools for a test set made of unseen, differently distributed aircraft.

The key idea: a random (or grouped) cross-validation measures in-distribution error.
When the test aircraft are distinguishable from the training aircraft, that number
is optimistic. We therefore (1) measure the shift with adversarial validation,
(2) build an out-of-distribution (OOD) split that holds out the training aircraft
that look most like the test fleet, and (3) report uncertainty with an aircraft-level
bootstrap, since rows of one aircraft are not independent.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from corrosion_risk.model import N_JOBS


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def cluster_bootstrap(
    values: np.ndarray, groups: np.ndarray, n_boot: int = 2000, seed: int = 0, level: float = 0.95
) -> tuple[float, float, float]:
    """Row mean of `values` with a percentile CI that resamples whole aircraft.

    Rows of one aircraft share the same history, so resampling rows would understate
    the uncertainty; resampling aircraft keeps that dependence.
    """
    frame = pd.DataFrame({"v": np.asarray(values, float), "g": np.asarray(groups)})
    per = frame.groupby("g")["v"].agg(["sum", "count"]).to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per), size=(n_boot, len(per)))
    means = per[idx, 0].sum(axis=1) / per[idx, 1].sum(axis=1)
    tail = (1 - level) / 2 * 100
    return (
        float(frame["v"].mean()),
        float(np.percentile(means, tail)),
        float(np.percentile(means, 100 - tail)),
    )


@dataclass(frozen=True)
class Adversarial:
    auc: float
    p_test: np.ndarray  # out-of-fold P(test) for each row, train rows first then test rows
    importance: pd.Series


def adversarial_validation(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    grouped: bool = True,
    seed: int = 42,
) -> Adversarial:
    """Out-of-fold classifier that tells training rows from test rows.

    AUC near 0.5 means the two fleets are indistinguishable. With `grouped=True`,
    folds split by aircraft so the classifier cannot win by memorising an aircraft
    it has already seen in another fold; this is the honest version.
    """
    both = pd.concat([train[features], test[features]], ignore_index=True)
    both = both.replace([np.inf, -np.inf], 0).fillna(0)
    is_test = np.r_[np.zeros(len(train), int), np.ones(len(test), int)]
    aircraft = np.r_[train["aircraft_id"].to_numpy(), test["aircraft_id"].to_numpy()]
    if grouped:
        splits = GroupKFold(n_splits=5, shuffle=True, random_state=seed).split(both, is_test, aircraft)
    else:
        splits = StratifiedKFold(n_splits=5).split(both, is_test)

    def clf() -> lgb.LGBMClassifier:
        return lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5, random_state=seed, verbosity=-1, n_jobs=N_JOBS
        )

    oof = np.zeros(len(both))
    for tr_idx, va_idx in splits:
        oof[va_idx] = clf().fit(both.iloc[tr_idx], is_test[tr_idx]).predict_proba(both.iloc[va_idx])[:, 1]
    importance = pd.Series(clf().fit(both, is_test).feature_importances_, index=features)
    return Adversarial(roc_auc_score(is_test, oof), oof, importance.sort_values(ascending=False))


def ood_split(aircraft_p_test: pd.Series, holdout_fraction: float = 0.45) -> tuple[set, set]:
    """Train on the least test-like aircraft, validate on the most test-like ones."""
    ranked = aircraft_p_test.sort_values().index
    cut = round((1 - holdout_fraction) * len(ranked))
    return set(ranked[:cut]), set(ranked[cut:])


def grouped_oof(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, predict, n_splits: int = 5) -> np.ndarray:
    """Out-of-fold predictions with folds split by aircraft."""
    oof = np.zeros(len(y))
    for tr_idx, va_idx in GroupKFold(n_splits=n_splits).split(X, y, groups):
        oof[va_idx] = predict(X.iloc[tr_idx], y[tr_idx], X.iloc[va_idx])
    return oof


def optimal_alpha(y: np.ndarray, p: np.ndarray) -> float:
    """Closed-form Brier-optimal shrinkage: least squares of (y - 0.5) on (p - 0.5)."""
    centred = np.asarray(p, float) - 0.5
    denom = float(np.dot(centred, centred))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(centred, np.asarray(y, float) - 0.5) / denom)


def leaderboard_standard_error(squared_errors: np.ndarray, n_rows: int) -> float:
    """Standard error of a Brier score computed on `n_rows` rows drawn like these."""
    return float(np.std(squared_errors, ddof=1) / np.sqrt(n_rows))
