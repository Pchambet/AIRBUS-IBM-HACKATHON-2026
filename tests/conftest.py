"""Synthetic competition tables with the real schema (the Kaggle data is not redistributable)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from corrosion_risk.data import Competition

RAW_COLUMNS = [
    "total_parking_minutes",
    "metar_temperature_c",
    "metar_relative_humidity",
    "metar_dew_point_c",
    "sea_salt_aerosol_003_05_mixing_ratio",
    "sea_salt_aerosol_05_5_mixing_ratio",
    "sea_salt_aerosol_5_20_mixing_ratio",
    "sulphate_aerosol_mixing_ratio",
    "carbon_monoxide_mass_mixing_ratio",
    "ozone_mass_mixing_ratio",
    "sulphur_dioxide_mass_mixing_ratio",
    "nitrogen_dioxide_mass_mixing_ratio",
]


def make_env(aircraft: list[str], start: str, months: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for ac in aircraft:
        dates = pd.date_range(start, periods=months, freq="MS", tz="UTC")
        block = pd.DataFrame(rng.lognormal(0.0, 0.3, size=(months, len(RAW_COLUMNS))), columns=RAW_COLUMNS)
        block.insert(0, "month_start_date", dates.strftime("%Y-%m-%dT%H:%M:%SZ"))
        block.insert(0, "year_month", dates.strftime("%Y-%m"))
        block.insert(0, "aircraft_id", ac)
        rows.append(block)
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def competition() -> Competition:
    rng = np.random.default_rng(0)
    train_ac = [f"tr{i:03d}" for i in range(40)]
    test_ac = [f"te{i:03d}" for i in range(10)]
    env_train = make_env(train_ac, "2016-01-01", 60, rng)
    env_test = make_env(test_ac, "2015-01-01", 66, rng)
    t_offsets = rng.integers(30, 60, size=len(train_ac))
    corrosions = pd.DataFrame(
        {
            "observation_date": [
                (pd.Timestamp("2016-01-01") + pd.DateOffset(months=int(k)) + pd.Timedelta(days=9)).strftime(
                    "%Y-%m-%d"
                )
                for k in t_offsets
            ],
            "aircraft_delivery_year": 2015,
            "aircraft_delivery_month": 6,
            "aircraft_id": train_ac,
        }
    )
    sample = env_test[["aircraft_id", "year_month"]].copy()
    sample.insert(0, "id", sample["aircraft_id"] + "_" + sample["year_month"])
    sample["corrosion_risk"] = 0.5
    sample = sample.sample(frac=1.0, random_state=0).reset_index(drop=True)
    return Competition(env_train=env_train, env_test=env_test, corrosions=corrosions, sample=sample)
