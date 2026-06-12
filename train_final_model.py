"""
train_final_model.py — Winning model for the Airbus x IBM x AWS 2026 corrosion hackathon.
Final private leaderboard: Brier 0.18, 6th place.

Approach (what actually won):
  - Target: the scored set is the corrosion month T (=1) and the month 24 months earlier
    T-24 (=0) per aircraft -> a balanced 50/50 within-aircraft discrimination problem.
  - Model: LightGBM regressor with objective='regression' (MSE == Brier score).
  - Features: per-aircraft causal exposure history (EWMA aerosols, rolling weather,
    cumulative dose / age / months observed). Aircraft-level constants cancel by design,
    so only within-aircraft temporal change is informative.
  - The decisive lever: shrinkage of predictions toward 0.5 (alpha=0.7). The train and test
    aircraft are disjoint with a strong covariate shift (adversarial AUC ~0.82), so the raw
    model is over-confident out-of-distribution. alpha=0.7 was tuned on an OOD validation
    (train on least-test-like aircraft, evaluate on most-test-like) that reproduced the real
    public score (0.218 vs 0.22 observed). See analysis/ for the full validation.

Run:  python train_final_model.py   ->  writes data/final_submission_best.csv
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from dateutil.relativedelta import relativedelta

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ALPHA = 0.7  # shift-aware shrinkage toward 0.5 (OOD-validated optimum)


def engineer(env: pd.DataFrame) -> pd.DataFrame:
    """Per-aircraft, strictly causal feature engineering (no future leakage)."""
    env = env.copy()
    env["month_start_date"] = pd.to_datetime(env["month_start_date"])

    # delivery proxy = first observed month -> aircraft age in months
    first = env.groupby("aircraft_id")["month_start_date"].min().reset_index()
    first["dy"] = first["month_start_date"].dt.year
    first["dm"] = first["month_start_date"].dt.month
    env = env.merge(first.drop(columns="month_start_date"), on="aircraft_id", how="left")
    env = env.sort_values(["aircraft_id", "month_start_date"]).reset_index(drop=True)
    env["aircraft_age_months"] = (env["month_start_date"].dt.year - env["dy"]) * 12 + \
                                 (env["month_start_date"].dt.month - env["dm"])

    # combined chloride load
    env["total_sea_salt_aerosol"] = (env["sea_salt_aerosol_003_05_mixing_ratio"]
                                     + env["sea_salt_aerosol_05_5_mixing_ratio"]
                                     + env["sea_salt_aerosol_5_20_mixing_ratio"])

    # exponentially-weighted exposure (recent months weighted more, decay = washing/maintenance)
    for f in ["total_sea_salt_aerosol", "sulphate_aerosol_mixing_ratio",
              "carbon_monoxide_mass_mixing_ratio", "ozone_mass_mixing_ratio",
              "sulphur_dioxide_mass_mixing_ratio", "nitrogen_dioxide_mass_mixing_ratio"]:
        env[f"ewma_{f}"] = env.groupby("aircraft_id")[f].transform(
            lambda x: x.ewm(halflife=12, ignore_na=True).mean())

    # weather: causal forward-fill + trailing rolling means
    for f in ["metar_relative_humidity", "metar_temperature_c", "metar_dew_point_c"]:
        env[f] = env.groupby("aircraft_id")[f].transform(lambda x: x.ffill())
        for w in [6, 12, 24]:
            env[f"rolling_{w}m_{f}"] = env.groupby("aircraft_id")[f].transform(
                lambda x: x.rolling(w, min_periods=1).mean())

    # corrosion-stress interactions
    env["stress_log_interaction"] = np.log1p(
        env["total_parking_minutes"] * env["metar_relative_humidity"] * env["total_sea_salt_aerosol"])
    env["acid_risk_index"] = (env["sulphur_dioxide_mass_mixing_ratio"]
                              + env["nitrogen_dioxide_mass_mixing_ratio"]) * env["metar_relative_humidity"]

    # accumulated dose to date (monotone in time -> systematically higher at T than T-24)
    g = env.groupby("aircraft_id")
    env["months_observed"] = g.cumcount() + 1
    for f in ["total_sea_salt_aerosol", "sulphate_aerosol_mixing_ratio",
              "acid_risk_index", "total_parking_minutes"]:
        env[f"cum_{f}"] = g[f].cumsum()
        env[f"cummean_{f}"] = env[f"cum_{f}"] / env["months_observed"]
        env[f"rel_{f}"] = env[f] / (env[f"cummean_{f}"] + 1e-12)
    env["cum_stress"] = g["stress_log_interaction"].cumsum()
    return env


def build_target(env: pd.DataFrame, corr: pd.DataFrame) -> pd.Series:
    """Label = 1 at the corrosion month T, 0 at T-24, -1 (ignored) otherwise."""
    obs = dict(zip(corr["aircraft_id"], pd.to_datetime(corr["observation_date"])))

    def lab(row):
        ac = row["aircraft_id"]
        if ac not in obs:
            return -1
        o, m = obs[ac], row["month_start_date"]
        if m.year == o.year and m.month == o.month:
            return 1
        t = o - relativedelta(months=24)
        if m.year == t.year and m.month == t.month:
            return 0
        return -1

    return env.apply(lab, axis=1)


def main():
    env_tr = engineer(pd.read_csv(os.path.join(DATA_DIR, "environment_training.csv")))
    env_te = engineer(pd.read_csv(os.path.join(DATA_DIR, "environment_test.csv")))
    corr = pd.read_csv(os.path.join(DATA_DIR, "corrosions_training.csv"))
    sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission-2.csv"))

    env_tr["corrosion_event"] = build_target(env_tr, corr)
    drop = ["dy", "dm", "corrosion_event", "month_start_date", "aircraft_id", "year_month"]
    feats = [c for c in env_tr.columns if c not in drop and c in env_te.columns]

    train = env_tr[env_tr["corrosion_event"].isin([0, 1])].copy()
    X = train[feats].replace([np.inf, -np.inf], 0).fillna(0)
    y = train["corrosion_event"].astype(int).values
    print(f"Training on {len(train)} balanced rows ({dict(pd.Series(y).value_counts())}), {len(feats)} features.")

    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.04, max_depth=5,
                              num_leaves=31, colsample_bytree=0.8, reg_lambda=1.0,
                              objective="regression", random_state=42, verbosity=-1)
    model.fit(X, y)

    X_te = env_te[feats].replace([np.inf, -np.inf], 0).fillna(0)
    pred = np.clip(model.predict(X_te), 0, 1)
    pred = 0.5 + ALPHA * (pred - 0.5)  # shift-aware shrinkage

    env_te["corrosion_risk"] = np.clip(pred, 0, 1)
    env_te["id"] = env_te["aircraft_id"] + "_" + env_te["month_start_date"].dt.strftime("%Y-%m")
    out = sample[["id"]].merge(env_te[["id", "corrosion_risk"]], on="id", how="left")
    out["corrosion_risk"] = out["corrosion_risk"].fillna(out["corrosion_risk"].median())
    assert (out["id"].values == sample["id"].values).all() and out["corrosion_risk"].isna().sum() == 0

    path = os.path.join(DATA_DIR, "final_submission_best.csv")
    out.to_csv(path, index=False)
    print(f"Saved {path} | rows={len(out)} | mean={out['corrosion_risk'].mean():.3f} "
          f"| range=[{out['corrosion_risk'].min():.3f}, {out['corrosion_risk'].max():.3f}]")


if __name__ == "__main__":
    main()
