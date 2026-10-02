"""Label construction that mirrors the competition scoring rule.

For each training aircraft the scored rows are the month of the corrosion
observation T (label 1) and the month 24 months earlier (label 0). Every other
month is unlabelled (-1) and dropped from training.
"""

from __future__ import annotations

import pandas as pd

HORIZON_MONTHS = 24


def build_target(env: pd.DataFrame, corrosions: pd.DataFrame) -> pd.Series:
    month = _month_index(env["month_start_date"])
    observed = _month_index(corrosions["observation_date"])
    t_month = env["aircraft_id"].map(dict(zip(corrosions["aircraft_id"], observed, strict=True)))
    label = pd.Series(-1, index=env.index, dtype="int64")
    label[month == t_month] = 1
    label[month == t_month - HORIZON_MONTHS] = 0
    return label


def _month_index(dates: pd.Series) -> pd.Series:
    """Months since year 0, so that month arithmetic is plain integer arithmetic."""
    d = pd.to_datetime(dates)
    return d.dt.year * 12 + d.dt.month - 1
