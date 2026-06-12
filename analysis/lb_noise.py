"""
Sampling noise of a Brier score measured on N=143 public observations.
Brier = mean of per-row squared errors e_i=(p_i-y_i)^2. We get realistic e_i from our
balanced T/T-24 OOF predictions, then quantify:
  - SE of a single Brier on 143 rows
  - margin of error (95% CI half-width)
  - the gap between two scores that is 'significant' at various levels
Reported for both independent scores and paired (same 143 rows) comparisons.
"""
import pandas as pd, numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from dateutil.relativedelta import relativedelta
import warnings; warnings.filterwarnings("ignore")

d="/Users/pierre/Desktop/GitHub/AIRBUS-IBM/data"
def engineer(env):
    env=env.copy(); env['month_start_date']=pd.to_datetime(env['month_start_date'])
    fd=env.groupby('aircraft_id')['month_start_date'].min().reset_index()
    fd['dy']=fd['month_start_date'].dt.year; fd['dm']=fd['month_start_date'].dt.month
    env=env.merge(fd.drop(columns='month_start_date'),on='aircraft_id',how='left').sort_values(['aircraft_id','month_start_date']).reset_index(drop=True)
    env['aircraft_age_months']=(env['month_start_date'].dt.year-env['dy'])*12+(env['month_start_date'].dt.month-env['dm'])
    env['salt']=env['sea_salt_aerosol_003_05_mixing_ratio']+env['sea_salt_aerosol_05_5_mixing_ratio']+env['sea_salt_aerosol_5_20_mixing_ratio']
    g=env.groupby('aircraft_id'); env['months_observed']=g.cumcount()+1
    for f in ['salt','total_parking_minutes','metar_relative_humidity']:
        env[f]=g[f].transform(lambda x:x.ffill())
        env[f'cum_{f}']=g[f].cumsum(); env[f'cummean_{f}']=env[f'cum_{f}']/env['months_observed']
    return env
tr=engineer(pd.read_csv(f"{d}/environment_training.csv"))
corr=pd.read_csv(f"{d}/corrosions_training.csv"); corr['observation_date']=pd.to_datetime(corr['observation_date'])
cd=dict(zip(corr['aircraft_id'],corr['observation_date']))
def tgt(r):
    ac=r['aircraft_id']
    if ac not in cd: return -1
    o=cd[ac]; m=r['month_start_date']
    if m.year==o.year and m.month==o.month: return 1
    t=o-relativedelta(months=24)
    if m.year==t.year and m.month==t.month: return 0
    return -1
tr['e']=tr.apply(tgt,axis=1)
EX=['dy','dm','e','month_start_date','aircraft_id','year_month']
feats=[c for c in tr.columns if c not in EX]
train=tr[tr['e'].isin([0,1])].copy(); y=train['e'].astype(int).values
X=train[feats].replace([np.inf,-np.inf],0).fillna(0); groups=train['aircraft_id']
oof=np.zeros(len(y))
for ti,vi in GroupKFold(5).split(X,y,groups=groups):
    m=lgb.LGBMRegressor(n_estimators=200,max_depth=5,learning_rate=0.04,random_state=42,verbosity=-1)
    m.fit(X.iloc[ti],y[ti]); oof[vi]=np.clip(0.5+0.7*(m.predict(X.iloc[vi])-0.5),0,1)
err=(oof-y)**2
B=err.mean(); sd=err.std(ddof=1)
print(f"Realistic per-row squared errors e_i on the balanced scored set:")
print(f"  Brier (mean e_i) = {B:.4f} | sd(e_i) = {sd:.4f}")
for N in [143]:
    se=sd/np.sqrt(N)
    print(f"\n=== Public LB on N={N} observations ===")
    print(f"  SE of a single Brier        = {se:.4f}")
    print(f"  95% CI half-width (1.96*SE) = {1.96*se:.4f}  -> a score is only meaningful +/-{1.96*se:.3f}")
    print(f"  ~68% (1*SE)                 = +/-{se:.4f}")
    # bootstrap to confirm (non-normal e_i)
    rng=np.random.default_rng(0)
    boots=[rng.choice(err,N,replace=True).mean() for _ in range(20000)]
    print(f"  bootstrap SE                = {np.std(boots):.4f} (confirms the formula)")
    print(f"\n  --- Gap between TWO teams that is 'significant' ---")
    se_ind=np.sqrt(2)*se   # two independent scores
    for name,z in [("at 20% level (z=1.28)",1.281),("at 10% (z=1.64)",1.645),("at 5% (z=1.96)",1.960)]:
        print(f"    independent scores, {name:22s}: gap > {z*se_ind:.4f}")
    print(f"    (paired, same 143 rows: smaller if models similar, but still ~{0.7*se_ind:.3f}-{se_ind:.3f})")
