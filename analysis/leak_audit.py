"""
Leak / structure audit. In TRAIN we know the corrosion month T (label=1) and T-24 (label=0).
We hunt for any artefact that betrays T, then check it also exists in TEST.
"""
import pandas as pd, numpy as np, os
from dateutil.relativedelta import relativedelta
pd.set_option('display.width', 160)
d = "/Users/pierre/Desktop/GitHub/AIRBUS-IBM/data"
env_tr = pd.read_csv(f"{d}/environment_training.csv"); env_te = pd.read_csv(f"{d}/environment_test.csv")
corr = pd.read_csv(f"{d}/corrosions_training.csv")
env_tr['month_start_date'] = pd.to_datetime(env_tr['month_start_date'])
env_te['month_start_date'] = pd.to_datetime(env_te['month_start_date'])
corr['observation_date'] = pd.to_datetime(corr['observation_date'])
cd = dict(zip(corr['aircraft_id'], corr['observation_date']))
num_cols = env_tr.select_dtypes(include=[np.number]).columns.tolist()

# label train rows
def lab(r):
    ac=r['aircraft_id']
    if ac not in cd: return -1
    o=cd[ac]; m=r['month_start_date']
    if m.year==o.year and m.month==o.month: return 1
    t=o-relativedelta(months=24)
    if m.year==t.year and m.month==t.month: return 0
    return -1
env_tr['lbl']=env_tr.apply(lab,axis=1)
T=env_tr[env_tr['lbl']==1]; T24=env_tr[env_tr['lbl']==0]
print(f"T rows={len(T)}  T-24 rows={len(T24)}")

print("\n=== A. NaN / sentinel patterns: does any column differ in missingness at T? ===")
for c in num_cols:
    na_T=T[c].isna().mean(); na_o=env_tr[env_tr['lbl']!=1][c].isna().mean()
    if abs(na_T-na_o)>0.05:
        print(f"  {c}: NaN@T={na_T:.2f} vs other={na_o:.2f}")
print("  (nothing printed = no missingness leak)")

print("\n=== B. Is total_parking_minutes at T an extreme for that aircraft? ===")
def pct_within_ac(rows, col):
    out=[]
    for _,r in rows.iterrows():
        s=env_tr[env_tr['aircraft_id']==r['aircraft_id']][col]
        out.append((r[col]>=s).mean())
    return np.array(out)
for col in ['total_parking_minutes','total_parking_minutes']:
    pass
pp_T = pct_within_ac(T.sample(min(150,len(T)),random_state=1),'total_parking_minutes')
pp_T24 = pct_within_ac(T24.sample(min(150,len(T24)),random_state=1),'total_parking_minutes')
print(f"  parking percentile-within-aircraft  @T:   mean={pp_T.mean():.2f}  (1.0=always the max)")
print(f"  parking percentile-within-aircraft  @T-24: mean={pp_T24.mean():.2f}")
print(f"  % of T rows where parking is the aircraft MAX: {(pp_T>=0.999).mean():.2f}")

print("\n=== C. Duplicate-row check (are filler rows copies?) ===")
dup_te=env_te.duplicated(subset=num_cols).mean(); dup_tr=env_tr.duplicated(subset=num_cols).mean()
print(f"  duplicated numeric rows: train={dup_tr:.3f}  test={dup_te:.3f}")

print("\n=== D. TEST structure: rows & month span per aircraft ===")
g=env_te.groupby('aircraft_id')
rpa=g.size()
span=g['month_start_date'].agg(lambda s:(s.max()-s.min()).days/30.4)
print(f"  rows/aircraft: min={rpa.min()} max={rpa.max()} median={rpa.median()}")
print(f"  month span:    min={span.min():.0f} max={span.max():.0f} median={span.median():.0f}")
# are all months contiguous?
def gaps(s):
    months=sorted(s.dt.to_period('M'))
    full=pd.period_range(months[0],months[-1],freq='M')
    return len(full)-len(set(months))
gp=g['month_start_date'].apply(gaps)
print(f"  aircraft with missing months in their span: {(gp>0).sum()}/{len(gp)}  (gaps could mark scored rows)")

print("\n=== E. Which calendar months appear in TEST ids? (sampling structure) ===")
sample=pd.read_csv(f"{d}/sample_submission-2.csv")
sample['ym']=sample['id'].str.split('_').str[1]
print("  distinct year-months in submission:", sample['ym'].nunique())
print("  range:", sample['ym'].min(), "->", sample['ym'].max())
cnt=sample.groupby(sample['id'].str.split('_').str[0]).size()
print(f"  ids per aircraft in submission: min={cnt.min()} max={cnt.max()} median={cnt.median()}")

print("\n=== F. Does the LAST observed month align with corrosion in TRAIN? ===")
last_is_T=0; tot=0
for ac,o in cd.items():
    s=env_tr[env_tr['aircraft_id']==ac]['month_start_date']
    if len(s)==0: continue
    tot+=1
    last=s.max()
    if last.year==o.year and last.month==o.month: last_is_T+=1
print(f"  corrosion month == last observed month: {last_is_T}/{tot} ({100*last_is_T/tot:.1f}%)")
# distribution of (months from last obs back to T)
deltas=[]
for ac,o in cd.items():
    s=env_tr[env_tr['aircraft_id']==ac]['month_start_date']
    if len(s)==0: continue
    last=s.max(); deltas.append((last.year-o.year)*12+(last.month-o.month))
deltas=pd.Series(deltas)
print(f"  months from corrosion T to last-observed: median={deltas.median():.0f} (0=corrosion is last)")
print(deltas.describe())
