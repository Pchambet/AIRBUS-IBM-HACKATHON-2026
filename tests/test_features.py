import numpy as np
import pandas as pd

from corrosion_risk.features import design_matrix, engineer, feature_columns


def test_features_are_causal(competition):
    """Changing an aircraft's future must not change any feature in its past."""
    env = competition.env_train
    cut = pd.Timestamp("2018-06-01", tz="UTC")
    future = (env["aircraft_id"] == "tr000") & (pd.to_datetime(env["month_start_date"]) > cut)
    perturbed = env.copy()
    numeric = perturbed.select_dtypes("number").columns
    perturbed.loc[future, numeric] = perturbed.loc[future, numeric] * 50 + 3

    base, alt = engineer(env), engineer(perturbed)
    past = (base["aircraft_id"] == "tr000") & (base["month_start_date"] <= cut)
    cols = feature_columns(base, base)
    pd.testing.assert_frame_equal(base.loc[past, cols], alt.loc[past, cols])


def test_backfill_would_leak_but_forward_fill_does_not(competition):
    env = competition.env_train.copy()
    env.loc[0, "metar_relative_humidity"] = np.nan  # first month of tr000
    out = engineer(env)
    assert np.isnan(out.loc[0, "metar_relative_humidity"])  # not filled from the future


def test_clock_and_dose_features(competition):
    out = engineer(competition.env_train)
    a = out[out["aircraft_id"] == "tr001"]
    assert a["months_observed"].tolist() == list(range(1, len(a) + 1))
    assert a["aircraft_age_months"].tolist() == list(range(len(a)))
    assert np.allclose(a["cum_total_parking_minutes"], a["total_parking_minutes"].cumsum())
    assert a["cum_total_sea_salt_aerosol"].is_monotonic_increasing


def test_design_matrix_has_no_missing_or_infinite_values(competition):
    env = competition.env_train.copy()
    env.loc[5, "total_parking_minutes"] = np.inf
    out = engineer(env)
    X = design_matrix(out, feature_columns(out, out))
    assert np.isfinite(X.to_numpy()).all()
