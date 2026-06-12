"""
Adversarial validation: can a model tell TRAIN aircraft from TEST aircraft using the
SAME features we model with? If AUC ~ 0.5, train/test are indistinguishable -> the
GroupKFold CV is a trustworthy proxy for the leaderboard, and the public 0.24 is noise.
If AUC is high -> distribution shift -> CV is optimistic and must be trusted less.
Built on the V2 feature set.
"""
import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

data_dir = "/Users/pierre/Desktop/GitHub/AIRBUS-IBM/data"

def build(path, is_test):
    env = pd.read_csv(os.path.join(data_dir, path))
    env['month_start_date'] = pd.to_datetime(env['month_start_date'])
    fd = env.groupby('aircraft_id')['month_start_date'].min().reset_index()
    fd['dy'] = fd['month_start_date'].dt.year; fd['dm'] = fd['month_start_date'].dt.month
    env = env.merge(fd.drop(columns='month_start_date'), on='aircraft_id', how='left')
    env = env.sort_values(['aircraft_id', 'month_start_date']).reset_index(drop=True)
    env['aircraft_age_months'] = (env['month_start_date'].dt.year - env['dy'])*12 + (env['month_start_date'].dt.month - env['dm'])
    env['total_sea_salt_aerosol'] = (env['sea_salt_aerosol_003_05_mixing_ratio'] +
                                     env['sea_salt_aerosol_05_5_mixing_ratio'] + env['sea_salt_aerosol_5_20_mixing_ratio'])
    g = env.groupby('aircraft_id')
    env['months_observed'] = g.cumcount() + 1
    for f in ['total_sea_salt_aerosol', 'sulphate_aerosol_mixing_ratio', 'total_parking_minutes']:
        env[f'cum_{f}'] = g[f].cumsum()
        env[f'cummean_{f}'] = env[f'cum_{f}'] / env['months_observed']
    env['is_test'] = is_test
    return env

tr = build("environment_training.csv", 0)
te = build("environment_test.csv", 1)
print(f"train aircraft={tr['aircraft_id'].nunique()}  test aircraft={te['aircraft_id'].nunique()}")
print(f"train delivery years: {sorted(tr['dy'].unique())[:6]}...{sorted(tr['dy'].unique())[-3:]}")
print(f"test  delivery years: {sorted(te['dy'].unique())}")

feats = ['aircraft_age_months', 'months_observed', 'total_parking_minutes',
         'total_sea_salt_aerosol', 'sulphate_aerosol_mixing_ratio',
         'metar_relative_humidity', 'metar_temperature_c',
         'cum_total_parking_minutes', 'cummean_total_parking_minutes',
         'cum_total_sea_salt_aerosol', 'ozone_mass_mixing_ratio', 'sulphur_dioxide_mass_mixing_ratio']
both = pd.concat([tr, te], ignore_index=True).fillna(0)
X = both[feats]; y = both['is_test']

model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, verbosity=-1, random_state=42)
p = cross_val_predict(model, X, y, cv=5, method='predict_proba')[:, 1]
auc = roc_auc_score(y, p)
print(f"\n=== ADVERSARIAL AUC = {auc:.4f} ===")
print("~0.5  -> train and test indistinguishable -> CV trustworthy, public 0.24 is noise")
print(">0.75 -> strong distribution shift -> CV is optimistic, trust it less")

model.fit(X, y)
imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
print("\nFeatures that most separate train from test (= where the shift lives):")
print(imp.head(8).to_string())

# how different are the key features?
print("\nKey feature medians (train vs test):")
for f in ['aircraft_age_months', 'months_observed', 'total_parking_minutes', 'total_sea_salt_aerosol']:
    print(f"  {f:28s} train={tr[f].median():.3g}  test={te[f].median():.3g}")
