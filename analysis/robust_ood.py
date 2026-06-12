"""
Attack the real bottleneck: the generalization gap (in-dist ~0.14 vs public ~0.22).
Build a STABLE OOD validation: train on the least test-like aircraft, evaluate balanced
Brier on the most test-like aircraft (mimicking the real shifted test), with a bootstrap
95% CI over evaluation aircraft to kill the noise agent-3 flagged.
Then test levers that could close the gap:
  L0  full V2 features (baseline)
  L1  drop the top-K shift-driving features (by adversarial importance)
  L2  ensemble (mse + binary, multi-seed) for variance reduction
Lower OOD-balanced-Brier with non-overlapping CI = a real, transferable gain.
"""
import pandas as pd, numpy as np, os
import lightgbm as lgb
from sklearn.model_selection import cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from dateutil.relativedelta import relativedelta

d = "/Users/pierre/Desktop/GitHub/AIRBUS-IBM/data"
rng = np.random.default_rng(0)

def engineer(env):
    env = env.copy(); env['month_start_date'] = pd.to_datetime(env['month_start_date'])
    fd = env.groupby('aircraft_id')['month_start_date'].min().reset_index()
    fd['dy']=fd['month_start_date'].dt.year; fd['dm']=fd['month_start_date'].dt.month
    env = env.merge(fd.drop(columns='month_start_date'), on='aircraft_id', how='left')
    env = env.sort_values(['aircraft_id','month_start_date']).reset_index(drop=True)
    env['aircraft_age_months']=(env['month_start_date'].dt.year-env['dy'])*12+(env['month_start_date'].dt.month-env['dm'])
    env['total_sea_salt_aerosol']=(env['sea_salt_aerosol_003_05_mixing_ratio']+env['sea_salt_aerosol_05_5_mixing_ratio']+env['sea_salt_aerosol_5_20_mixing_ratio'])
    for f in ['total_sea_salt_aerosol','sulphate_aerosol_mixing_ratio','carbon_monoxide_mass_mixing_ratio','ozone_mass_mixing_ratio','sulphur_dioxide_mass_mixing_ratio','nitrogen_dioxide_mass_mixing_ratio']:
        env[f'ewma_{f}']=env.groupby('aircraft_id')[f].transform(lambda x:x.ewm(halflife=12,ignore_na=True).mean())
    for f in ['metar_relative_humidity','metar_temperature_c','metar_dew_point_c']:
        env[f]=env.groupby('aircraft_id')[f].transform(lambda x:x.ffill())
        for w in [6,12,24]:
            env[f'rolling_{w}m_{f}']=env.groupby('aircraft_id')[f].transform(lambda x:x.rolling(w,min_periods=1).mean())
    env['stress_log_interaction']=np.log1p(env['total_parking_minutes']*env['metar_relative_humidity']*env['total_sea_salt_aerosol'])
    env['acid_risk_index']=(env['sulphur_dioxide_mass_mixing_ratio']+env['nitrogen_dioxide_mass_mixing_ratio'])*env['metar_relative_humidity']
    g=env.groupby('aircraft_id'); env['months_observed']=g.cumcount()+1
    for f in ['total_sea_salt_aerosol','sulphate_aerosol_mixing_ratio','acid_risk_index','total_parking_minutes']:
        env[f'cum_{f}']=g[f].cumsum(); env[f'cummean_{f}']=env[f'cum_{f}']/env['months_observed']; env[f'rel_{f}']=env[f]/(env[f'cummean_{f}']+1e-12)
    env['cum_stress']=g['stress_log_interaction'].cumsum()
    return env

tr=engineer(pd.read_csv(f"{d}/environment_training.csv")); te=engineer(pd.read_csv(f"{d}/environment_test.csv"))
corr=pd.read_csv(f"{d}/corrosions_training.csv"); corr['observation_date']=pd.to_datetime(corr['observation_date'])
cd=dict(zip(corr['aircraft_id'],corr['observation_date']))
def target(r):
    ac=r['aircraft_id']
    if ac not in cd: return -1
    o=cd[ac]; m=r['month_start_date']
    if m.year==o.year and m.month==o.month: return 1
    t=o-relativedelta(months=24)
    if m.year==t.year and m.month==t.month: return 0
    return -1
tr['corrosion_event']=tr.apply(target,axis=1)
EX=['dy','dm','corrosion_event','month_start_date','aircraft_id','year_month','year_month_str']
feats=[c for c in tr.columns if c not in EX and c in te.columns]

# adversarial p_test per row + feature importances (where shift lives)
comb=pd.concat([tr.assign(_t=0),te.assign(_t=1)],ignore_index=True); comb[feats]=comb[feats].replace([np.inf,-np.inf],0).fillna(0)
advm=lgb.LGBMClassifier(n_estimators=200,learning_rate=0.05,max_depth=5,verbosity=-1,random_state=42)
p=cross_val_predict(advm,comb[feats],comb['_t'],cv=5,method='predict_proba')[:,1]
advm.fit(comb[feats],comb['_t'])
shift_imp=pd.Series(advm.feature_importances_,index=feats).sort_values(ascending=False)
tr=tr.reset_index(drop=True); tr['p_test']=p[(comb['_t']==0).values]

train=tr[tr['corrosion_event'].isin([0,1])].copy()
ac_ptest=train.groupby('aircraft_id')['p_test'].mean().sort_values()
# OOD split: train on least test-like 55% of aircraft, evaluate on most test-like 45%
cut=int(0.55*len(ac_ptest)); train_ac=set(ac_ptest.index[:cut]); val_ac=set(ac_ptest.index[cut:])
m_tr=train['aircraft_id'].isin(train_ac).values; m_va=train['aircraft_id'].isin(val_ac).values
y=train['corrosion_event'].astype(int).values

def fit_pred(feature_set, kind, seed, Xtr_idx, Xva_idx):
    X=train[feature_set].replace([np.inf,-np.inf],0).fillna(0)
    if kind=='mse':
        m=lgb.LGBMRegressor(n_estimators=300,learning_rate=0.04,max_depth=5,num_leaves=31,colsample_bytree=0.8,reg_lambda=1.0,objective='regression',random_state=seed,verbosity=-1)
        m.fit(X.iloc[Xtr_idx],y[Xtr_idx]); return np.clip(m.predict(X.iloc[Xva_idx]),0,1)
    m=lgb.LGBMClassifier(n_estimators=300,learning_rate=0.04,max_depth=5,num_leaves=31,colsample_bytree=0.8,reg_lambda=1.0,objective='binary',random_state=seed,verbosity=-1)
    m.fit(X.iloc[Xtr_idx],y[Xtr_idx]); c=CalibratedClassifierCV(estimator=m,method='sigmoid',cv=3); c.fit(X.iloc[Xtr_idx],y[Xtr_idx])
    return c.predict_proba(X.iloc[Xva_idx])[:,1]

tr_idx=np.where(m_tr)[0]; va_idx=np.where(m_va)[0]; y_va=y[va_idx]; val_acs=train['aircraft_id'].values[va_idx]

def ood_brier(pred):
    # balanced per-aircraft: mean over aircraft of that aircraft's mean row-Brier, then bootstrap CI over aircraft
    dfp=pd.DataFrame({'ac':val_acs,'e':(pred-y_va)**2})
    per_ac=dfp.groupby('ac')['e'].mean()
    point=per_ac.mean()
    boot=[per_ac.sample(len(per_ac),replace=True,random_state=int(s)).mean() for s in range(200)]
    return point, np.percentile(boot,2.5), np.percentile(boot,97.5)

print(f"OOD val: train {len(train_ac)} least-test-like aircraft, eval {len(val_ac)} most-test-like")
print(f"{'lever':28s} {'OOD-Brier':>10s} {'95% CI':>20s}")

# L0 baseline (mse, single seed)
pr=fit_pred(feats,'mse',42,tr_idx,va_idx); pt,lo,hi=ood_brier(pr)
print(f"{'L0 full V2 (mse)':28s} {pt:10.4f}   [{lo:.4f},{hi:.4f}]")

# L1 drop top-K shift features
for K in [5,10,15]:
    keep=[f for f in feats if f not in set(shift_imp.index[:K])]
    pr=fit_pred(keep,'mse',42,tr_idx,va_idx); pt,lo,hi=ood_brier(pr)
    print(f"{f'L1 drop top-{K} shift feats':28s} {pt:10.4f}   [{lo:.4f},{hi:.4f}]  ({len(keep)} feats)")

# L2 ensemble mse+binary, multi-seed
preds=[]
for kind in ['mse','binary']:
    for s in [1,7,42,123,2024]:
        preds.append(fit_pred(feats,kind,s,tr_idx,va_idx))
pr=np.mean(preds,axis=0); pt,lo,hi=ood_brier(pr)
print(f"{'L2 ensemble mse+bin x5seed':28s} {pt:10.4f}   [{lo:.4f},{hi:.4f}]")

print(f"\nTop shift-driving features (candidates to drop):\n{shift_imp.head(10).to_string()}")
print("\nA lever wins only if its OOD-Brier CI is clearly below L0's CI.")
