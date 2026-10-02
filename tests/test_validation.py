import numpy as np
import pandas as pd
import pytest

from corrosion_risk.validation import (
    adversarial_validation,
    brier,
    cluster_bootstrap,
    leaderboard_standard_error,
    ood_split,
    optimal_alpha,
)


def test_brier_hand_computed():
    assert brier([1, 0, 1, 0], [0.9, 0.2, 0.5, 0.5]) == pytest.approx((0.01 + 0.04 + 0.25 + 0.25) / 4)


def test_optimal_alpha_recovers_known_overconfidence():
    """Ground truth: the true probability is the raw score shrunk with alpha = 0.6."""
    rng = np.random.default_rng(1)
    raw = rng.uniform(0, 1, 200_000)
    y = rng.uniform(size=raw.size) < 0.5 + 0.6 * (raw - 0.5)
    assert optimal_alpha(y, raw) == pytest.approx(0.6, abs=0.01)


def test_optimal_alpha_is_one_for_calibrated_scores():
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1, 200_000)
    assert optimal_alpha(rng.uniform(size=p.size) < p, p) == pytest.approx(1.0, abs=0.01)


def test_cluster_bootstrap_point_and_interval():
    values = np.array([0.1, 0.3, 0.2, 0.2, 0.9, 0.7])
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    est, lo, hi = cluster_bootstrap(values, groups, n_boot=500)
    assert est == pytest.approx(values.mean())
    assert 0.2 - 1e-12 <= lo <= est <= hi <= 0.8 + 1e-12  # bounded by the extreme aircraft means
    assert cluster_bootstrap(np.full(6, 0.25), groups)[1:] == (0.25, 0.25)


def test_leaderboard_standard_error_matches_formula():
    e = np.array([0.0, 1.0] * 50)
    assert leaderboard_standard_error(e, 100) == pytest.approx(np.std(e, ddof=1) / 10)


def test_ood_split_holds_out_the_most_test_like_aircraft():
    p = pd.Series(np.linspace(0, 1, 20), index=[f"a{i}" for i in range(20)])
    fit, hold = ood_split(p, holdout_fraction=0.25)
    assert len(hold) == 5 and not fit & hold
    assert min(p[list(hold)]) > max(p[list(fit)])


def _fleet(n_aircraft: int, months: int, shift: float, prefix: str, rng) -> pd.DataFrame:
    ac = np.repeat([f"{prefix}{i}" for i in range(n_aircraft)], months)
    level = np.repeat(rng.normal(shift, 1.0, n_aircraft), months)
    return pd.DataFrame(
        {"aircraft_id": ac, "x": level + rng.normal(0, 0.3, len(ac)), "z": rng.normal(size=len(ac))}
    )


def test_adversarial_validation_detects_a_known_shift():
    rng = np.random.default_rng(3)
    train, test = _fleet(80, 10, 0.0, "tr", rng), _fleet(40, 10, 2.5, "te", rng)
    assert adversarial_validation(train, test, ["x", "z"]).auc > 0.9


def test_adversarial_validation_is_near_chance_without_shift():
    rng = np.random.default_rng(4)
    train, test = _fleet(80, 10, 0.0, "tr", rng), _fleet(40, 10, 0.0, "te", rng)
    assert 0.35 < adversarial_validation(train, test, ["x", "z"]).auc < 0.65
