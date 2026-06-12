"""
Demolish-or-confirm test of the 'revolutionary' central claim:
  "objective=regression (MSE) GUARANTEES Brier 0.10-0.12 vs 0.246 binary baseline (51-59%)."
Head-to-head on IDENTICAL V2 features + identical splits:
  A) binary + Platt calibration            (our approach)
  B) regression/MSE, clip[0,1], NO calib   (their approach: n_estimators=500, max_depth=7)
Reported on naive CV, adversarially-weighted, and oldest-aircraft holdout.
"""
import pandas as pd, numpy as np, os
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from dateutil.relativedelta import relativedelta

d = "/Users/pierre/Desktop/GitHub/AIRBUS-IBM/data"

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
        env[f]=env.groupby('aircraft_id')[f].transform(lambda x:x.ffill().bfill())
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
comb=pd.concat([tr.assign(_t=0),te.assign(_t=1)],ignore_index=True); comb[feats]=comb[feats].replace([np.inf,-np.inf],0).fillna(0)
adv=lgb.LGBMClassifier(n_estimators=200,learning_rate=0.05,max_depth=5,verbosity=-1,random_state=42)
p=cross_val_predict(adv,comb[feats],comb['_t'],cv=5,method='predict_proba')[:,1]
tr=tr.reset_index(drop=True); tr['p_test']=p[(comb['_t']==0).values]; tr['w']=(tr['p_test']/(1-tr['p_test']).clip(lower=1e-3)).clip(upper=20)
train=tr[tr['corrosion_event'].isin([0,1])].copy()
y=train['corrosion_event'].astype(int).values; groups=train['aircraft_id']; w=train['w'].values
X=train[feats].replace([np.inf,-np.inf],0).fillna(0)
age=train.groupby('aircraft_id')['aircraft_age_months'].max(); old=set(age[age>=age.quantile(0.7)].index)
mask=train['aircraft_id'].isin(old).values

def run(kind):
    gkf=GroupKFold(n_splits=5); oof=np.zeros(len(y))
    for tri,vai in gkf.split(X,y,groups=groups):
        if kind=='binary_cal':
            m=lgb.LGBMClassifier(n_estimators=200,learning_rate=0.04,max_depth=5,num_leaves=31,subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,objective='binary',random_state=42,verbosity=-1)
            m.fit(X.iloc[tri],y[tri]); cal=CalibratedClassifierCV(estimator=m,method='sigmoid',cv=3); cal.fit(X.iloc[tri],y[tri])
            oof[vai]=cal.predict_proba(X.iloc[vai])[:,1]
        else:  # their MSE recipe
            m=lgb.LGBMRegressor(n_estimators=500,learning_rate=0.05,max_depth=7,objective='regression',random_state=42,verbosity=-1)
            m.fit(X.iloc[tri],y[tri]); oof[vai]=np.clip(m.predict(X.iloc[vai]),0,1)
    # old-holdout
    if kind=='binary_cal':
        m=lgb.LGBMClassifier(n_estimators=200,learning_rate=0.04,max_depth=5,num_leaves=31,subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,objective='binary',random_state=42,verbosity=-1)
        m.fit(X[~mask],y[~mask]); c=CalibratedClassifierCV(estimator=m,method='sigmoid',cv=3); c.fit(X[~mask],y[~mask]); op=c.predict_proba(X[mask])[:,1]
    else:
        m=lgb.LGBMRegressor(n_estimators=500,learning_rate=0.05,max_depth=7,objective='regression',random_state=42,verbosity=-1)
        m.fit(X[~mask],y[~mask]); op=np.clip(m.predict(X[mask]),0,1)
    return brier_score_loss(y,oof), np.average((oof-y)**2,weights=w), brier_score_loss(y[mask],op)

print(f"{'approach':>22s} | {'naive':>6s} | {'weighted':>8s} | {'old':>6s}")
for kind,name in [('binary_cal','A) binary + Platt calib'),('mse','B) MSE/regr (their recipe)')]:
    nv,wt,ol=run(kind)
    print(f"{name:>22s} | {nv:.4f} | {wt:8.4f} | {ol:.4f}")
print("\nClaim under test: B should reach 0.10-0.12 (51-59% better than 0.246). Does it?")
