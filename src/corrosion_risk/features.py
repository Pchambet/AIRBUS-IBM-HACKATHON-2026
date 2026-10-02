"""Per-aircraft, strictly causal exposure features.

Only two rows per aircraft are scored: the corrosion month T and T-24. Anything
constant within an aircraft cancels between those two rows, so the useful signal is
how exposure *accumulates* over time. Every feature at month m uses months <= m only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

AEROSOLS = [
    "total_sea_salt_aerosol",
    "sulphate_aerosol_mixing_ratio",
    "carbon_monoxide_mass_mixing_ratio",
    "ozone_mass_mixing_ratio",
    "sulphur_dioxide_mass_mixing_ratio",
    "nitrogen_dioxide_mass_mixing_ratio",
]
WEATHER = ["metar_relative_humidity", "metar_temperature_c", "metar_dew_point_c"]
DOSES = [
    "total_sea_salt_aerosol",
    "sulphate_aerosol_mixing_ratio",
    "acid_risk_index",
    "total_parking_minutes",
]
NON_FEATURES = {"dy", "dm", "corrosion_event", "month_start_date", "aircraft_id", "year_month"}
EWMA_HALFLIFE_MONTHS = 12


def engineer(env: pd.DataFrame) -> pd.DataFrame:
    """Add exposure features to a monthly environment table (one row per aircraft-month)."""
    env = env.copy()
    env["month_start_date"] = pd.to_datetime(env["month_start_date"])

    # The first observed month stands in for delivery: the test set has no delivery date.
    first = env.groupby("aircraft_id")["month_start_date"].min().reset_index()
    first["dy"] = first["month_start_date"].dt.year
    first["dm"] = first["month_start_date"].dt.month
    env = env.merge(first.drop(columns="month_start_date"), on="aircraft_id", how="left")
    env = env.sort_values(["aircraft_id", "month_start_date"]).reset_index(drop=True)
    env["aircraft_age_months"] = (env["month_start_date"].dt.year - env["dy"]) * 12 + (
        env["month_start_date"].dt.month - env["dm"]
    )

    env["total_sea_salt_aerosol"] = (
        env["sea_salt_aerosol_003_05_mixing_ratio"]
        + env["sea_salt_aerosol_05_5_mixing_ratio"]
        + env["sea_salt_aerosol_5_20_mixing_ratio"]
    )

    # Exponential decay mimics washing and rain: recent exposure counts more.
    for f in AEROSOLS:
        env[f"ewma_{f}"] = env.groupby("aircraft_id")[f].transform(
            lambda x: x.ewm(halflife=EWMA_HALFLIFE_MONTHS, ignore_na=True).mean()
        )

    # Forward-fill only (never backward) so no future value leaks into the past.
    for f in WEATHER:
        env[f] = env.groupby("aircraft_id")[f].transform(lambda x: x.ffill())
        for w in (6, 12, 24):
            env[f"rolling_{w}m_{f}"] = env.groupby("aircraft_id")[f].transform(
                lambda x, w=w: x.rolling(w, min_periods=1).mean()
            )

    env["stress_log_interaction"] = np.log1p(
        env["total_parking_minutes"] * env["metar_relative_humidity"] * env["total_sea_salt_aerosol"]
    )
    env["acid_risk_index"] = (
        env["sulphur_dioxide_mass_mixing_ratio"] + env["nitrogen_dioxide_mass_mixing_ratio"]
    ) * env["metar_relative_humidity"]

    # Accumulated dose is monotone in time, hence systematically higher at T than at T-24.
    g = env.groupby("aircraft_id")
    env["months_observed"] = g.cumcount() + 1
    for f in DOSES:
        env[f"cum_{f}"] = g[f].cumsum()
        env[f"cummean_{f}"] = env[f"cum_{f}"] / env["months_observed"]
        env[f"rel_{f}"] = env[f] / (env[f"cummean_{f}"] + 1e-12)
    env["cum_stress"] = g["stress_log_interaction"].cumsum()
    return env


def feature_columns(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    return [c for c in train.columns if c not in NON_FEATURES and c in test.columns]


def design_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return df[features].replace([np.inf, -np.inf], 0).fillna(0)
