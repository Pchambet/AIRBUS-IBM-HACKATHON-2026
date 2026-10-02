import pandas as pd

from corrosion_risk.target import build_target


def test_labels_t_and_t_minus_24_only():
    months = pd.date_range("2019-01-01", "2023-12-01", freq="MS", tz="UTC")
    env = pd.DataFrame(
        {
            "aircraft_id": ["a"] * len(months) + ["b"] * len(months),
            "month_start_date": list(months) * 2,
        }
    )
    corrosions = pd.DataFrame({"aircraft_id": ["a"], "observation_date": ["2022-06-17"]})
    label = build_target(env, corrosions)
    a = label[env["aircraft_id"] == "a"]
    month_a = env.loc[a.index, "month_start_date"].dt.strftime("%Y-%m")
    assert set(month_a[a == 1]) == {"2022-06"}
    assert set(month_a[a == 0]) == {"2020-06"}
    assert (a == -1).sum() == len(months) - 2
    assert (label[env["aircraft_id"] == "b"] == -1).all()  # no corrosion record: never scored


def test_t_minus_24_outside_history_is_simply_missing():
    months = pd.date_range("2021-01-01", periods=24, freq="MS")
    env = pd.DataFrame({"aircraft_id": "a", "month_start_date": months})
    corrosions = pd.DataFrame({"aircraft_id": ["a"], "observation_date": ["2022-03-01"]})
    label = build_target(env, corrosions)
    assert label.tolist().count(1) == 1
    assert label.tolist().count(0) == 0
