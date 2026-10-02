"""Every number quoted in the README is computed here and written to `results/`.

Run with `corrosion-risk analyze` (about a minute on a laptop). The outputs are small
CSV/JSON tables that the figure and report builders read, so the narrative can never
drift from the code.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from corrosion_risk import data, features, model, target, validation
from corrosion_risk.data import ROOT

RESULTS = ROOT / "results"
ALPHAS = np.round(np.arange(0.0, 1.0001, 0.05), 2)
PUBLIC_LB_ROWS = 143  # size of the public leaderboard sample reported by the organisers
SEEDS = (1, 7, 42, 123, 2024)
SUBMITTED = "LightGBM squared error (submitted)"
PLATT = "LightGBM log-loss + Platt scaling"
CLOCK = "LightGBM on the two clock features"

Predictor = Callable[[pd.DataFrame, np.ndarray, pd.DataFrame], np.ndarray]


@dataclass
class Prepared:
    train: pd.DataFrame  # engineered training environment, all months
    test: pd.DataFrame  # engineered test environment, all months
    labelled: pd.DataFrame  # scored training rows only (T and T-24)
    features: list[str]
    X: pd.DataFrame
    y: np.ndarray
    groups: np.ndarray


def prepare(comp: data.Competition) -> Prepared:
    train = features.engineer(comp.env_train)
    test = features.engineer(comp.env_test)
    train["corrosion_event"] = target.build_target(train, comp.corrosions)
    cols = features.feature_columns(train, test)
    labelled = train[train["corrosion_event"].isin([0, 1])].reset_index(drop=True)
    return Prepared(
        train=train,
        test=test,
        labelled=labelled,
        features=cols,
        X=features.design_matrix(labelled, cols),
        y=labelled["corrosion_event"].to_numpy(),
        groups=labelled["aircraft_id"].to_numpy(),
    )


def submission(prep: Prepared, sample: pd.DataFrame, alpha: float = model.ALPHA) -> pd.DataFrame:
    """Fit on all scored training rows and score every test row, in sample order."""
    raw = model.fit_predict(prep.X, prep.y, features.design_matrix(prep.test, prep.features))
    scored = pd.DataFrame(
        {
            "id": prep.test["aircraft_id"] + "_" + prep.test["month_start_date"].dt.strftime("%Y-%m"),
            "corrosion_risk": np.clip(model.shrink(raw, alpha), 0, 1),
        }
    )
    out = sample[["id"]].merge(scored, on="id", how="left")
    out["corrosion_risk"] = out["corrosion_risk"].fillna(out["corrosion_risk"].median())
    if not (out["id"].to_numpy() == sample["id"].to_numpy()).all():
        raise ValueError("Submission ids are not aligned with the sample submission.")
    return out


# --- candidate models (each returns raw probabilities in [0, 1]) -------------------


def _mse(**overrides) -> Predictor:
    def predict(Xa, ya, Xb):
        return np.clip(model.make_model(**overrides).fit(Xa, ya).predict(Xb), 0, 1)

    return predict


def _binary_platt(Xa, ya, Xb, seed: int = 42):
    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.04,
        max_depth=5,
        num_leaves=31,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        verbosity=-1,
        n_jobs=model.N_JOBS,
    )
    cal = CalibratedClassifierCV(estimator=clf, method="sigmoid", cv=3).fit(Xa, ya)
    return cal.predict_proba(Xb)[:, 1]


def _logistic(Xa, ya, Xb):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    return pipe.fit(Xa, ya).predict_proba(Xb)[:, 1]


def _restricted(cols: list[str], inner: Predictor) -> Predictor:
    return lambda Xa, ya, Xb: inner(Xa[cols], ya, Xb[cols])


def _ensemble(Xa, ya, Xb):
    preds = [model.fit_predict(Xa, ya, Xb, seed=s) for s in SEEDS]
    preds += [_binary_platt(Xa, ya, Xb, seed=s) for s in SEEDS]
    return np.mean(preds, axis=0)


# --- analysis -----------------------------------------------------------------------


def _summary(y, p, groups, alpha: float, reference: np.ndarray | None = None) -> dict:
    """Brier at `alpha` with an aircraft-bootstrap CI and, given a reference model's raw
    predictions on the same rows, the paired difference to it (negative = better)."""
    err = (model.shrink(p, alpha) - y) ** 2
    est, lo, hi = validation.cluster_bootstrap(err, groups)
    out = {
        "brier_raw": validation.brier(y, p),
        "brier_at_alpha": est,
        "ci_low": lo,
        "ci_high": hi,
        "auc": float(roc_auc_score(y, p)),
        "best_alpha": validation.optimal_alpha(y, p),
    }
    if reference is not None:
        diff = err - (model.shrink(reference, alpha) - y) ** 2
        out["delta_vs_submitted"], out["delta_ci_low"], out["delta_ci_high"] = validation.cluster_bootstrap(
            diff, groups
        )
    return out


def run(comp: data.Competition, out_dir: Path = RESULTS) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    prep = prepare(comp)
    X, y, g = prep.X, prep.y, prep.groups
    metrics: dict = {
        "n_train_aircraft": int(prep.train["aircraft_id"].nunique()),
        "n_test_aircraft": int(prep.test["aircraft_id"].nunique()),
        "n_train_rows": len(prep.train),
        "n_test_rows": len(prep.test),
        "n_scored_train_rows": len(y),
        "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
        "n_features": len(prep.features),
        "alpha_submitted": model.ALPHA,
    }

    # 1. Covariate shift between the two fleets.
    adv = validation.adversarial_validation(prep.train, prep.test, prep.features, grouped=True)
    adv_rows = validation.adversarial_validation(prep.train, prep.test, prep.features, grouped=False)
    metrics["adversarial_auc_grouped"] = adv.auc
    metrics["adversarial_auc_row_folds"] = adv_rows.auc
    adv.importance.head(15).rename("split_importance").to_csv(out_dir / "shift_features.csv")
    first_year = {
        name: df.groupby("aircraft_id")["dy"].first().value_counts().sort_index()
        for name, df in (("train", prep.train), ("test", prep.test))
    }
    shift = pd.DataFrame(first_year).fillna(0).astype(int).rename_axis("first_observed_year")
    shift.to_csv(out_dir / "fleet_first_observed_year.csv")
    # Aircraft whose history starts in the first year of the record were most likely delivered
    # earlier, so their "age" is left-censored.
    first = int(shift.index.min())
    metrics["first_record_year"] = first
    metrics["share_test_first_record_year"] = float(shift.loc[first, "test"] / shift["test"].sum())
    metrics["share_train_first_record_year"] = float(shift.loc[first, "train"] / shift["train"].sum())

    # 2. In-distribution (grouped CV) versus out-of-distribution (test-like holdout).
    p_test = pd.Series(adv.p_test[: len(prep.train)]).groupby(prep.train["aircraft_id"].to_numpy()).mean()
    fit_ac, holdout_ac = validation.ood_split(p_test[p_test.index.isin(set(g))])
    m_fit, m_ood = np.isin(g, list(fit_ac)), np.isin(g, list(holdout_ac))
    metrics["ood_fit_aircraft"], metrics["ood_holdout_aircraft"] = len(fit_ac), len(holdout_ac)
    metrics["ood_holdout_rows"] = int(m_ood.sum())

    def evaluate(predict: Predictor) -> tuple[np.ndarray, np.ndarray]:
        oof = validation.grouped_oof(X, y, g, predict)
        ood = predict(X[m_fit], y[m_fit], X[m_ood])
        return oof, ood

    oof, ood = evaluate(_mse())
    y_ood, g_ood = y[m_ood], g[m_ood]
    # Split the CV-to-holdout gap: grouped-CV predictions on the holdout aircraft come from
    # models that did see other test-like aircraft, so they measure how hard these aircraft
    # are in themselves; the rest of the gap comes from leaving them out of training.
    metrics |= {
        "id_brier_raw_holdout_aircraft": validation.brier(y_ood, oof[m_ood]),
        "id_best_alpha_holdout_aircraft": validation.optimal_alpha(y_ood, oof[m_ood]),
        "id_brier_raw_other_aircraft": validation.brier(y[~m_ood], oof[~m_ood]),
    }
    rows = []
    for a in ALPHAS:
        id_est, id_lo, id_hi = validation.cluster_bootstrap((model.shrink(oof, a) - y) ** 2, g)
        od_est, od_lo, od_hi = validation.cluster_bootstrap((model.shrink(ood, a) - y_ood) ** 2, g_ood)
        rows.append([a, id_est, id_lo, id_hi, od_est, od_lo, od_hi])
    sweep = pd.DataFrame(rows, columns=["alpha", "id", "id_low", "id_high", "ood", "ood_low", "ood_high"])
    sweep.to_csv(out_dir / "alpha_sweep.csv", index=False)
    at = sweep.set_index("alpha")
    metrics |= {
        "id_brier_raw": float(at.loc[1.0, "id"]),
        "id_brier_alpha": float(at.loc[model.ALPHA, "id"]),
        "id_auc": float(roc_auc_score(y, oof)),
        "id_best_alpha": validation.optimal_alpha(y, oof),
        "ood_brier_raw": float(at.loc[1.0, "ood"]),
        "ood_brier_raw_ci": [float(at.loc[1.0, "ood_low"]), float(at.loc[1.0, "ood_high"])],
        "ood_brier_alpha": float(at.loc[model.ALPHA, "ood"]),
        "ood_brier_alpha_ci": [float(at.loc[model.ALPHA, "ood_low"]), float(at.loc[model.ALPHA, "ood_high"])],
        "ood_auc": float(roc_auc_score(y_ood, ood)),
        "ood_best_alpha": validation.optimal_alpha(y_ood, ood),
    }
    # Paired effect of shrinkage on the same holdout aircraft.
    delta = (model.shrink(ood, model.ALPHA) - y_ood) ** 2 - (ood - y_ood) ** 2
    d_est, d_lo, d_hi = validation.cluster_bootstrap(delta, g_ood)
    metrics["ood_shrinkage_gain"] = -d_est
    metrics["ood_shrinkage_gain_ci"] = [-d_hi, -d_lo]

    # 3. Reliability on the holdout: raw versus shrunk.
    edges = np.linspace(0, 1, 11)
    rel = []
    for name, p in (("raw", ood), ("shrunk", model.shrink(ood, model.ALPHA))):
        bins = np.clip(np.digitize(p, edges) - 1, 0, 9)
        for b in np.unique(bins):
            sel = bins == b
            rel.append([name, b, float(p[sel].mean()), float(y_ood[sel].mean()), int(sel.sum())])
    pd.DataFrame(rel, columns=["model", "bin", "mean_predicted", "observed_rate", "n"]).to_csv(
        out_dir / "reliability_ood.csv", index=False
    )

    # 4. Model comparison: same features, same splits.
    clock = ["aircraft_age_months", "months_observed"]
    candidates: dict[str, Predictor] = {
        SUBMITTED: _mse(),
        PLATT: _binary_platt,
        "LightGBM squared error, 500 trees depth 7": _mse(
            n_estimators=500, learning_rate=0.05, max_depth=7, colsample_bytree=1.0, reg_lambda=0.0
        ),
        "Logistic regression": _logistic,
        CLOCK: _restricted(clock, _mse()),
    }
    comp_rows, ood_preds = [], {}
    for name, predict in {"Constant 0.5": None, **candidates}.items():
        if predict is None:
            o, d = np.full(len(y), 0.5), np.full(len(y_ood), 0.5)
        else:
            o, d = (oof, ood) if name == SUBMITTED else evaluate(predict)
        ood_preds[name] = d
        comp_rows.append({"model": name, "split": "in-distribution", **_summary(y, o, g, model.ALPHA, oof)})
        comp_rows.append(
            {"model": name, "split": "out-of-distribution", **_summary(y_ood, d, g_ood, model.ALPHA, ood)}
        )
    comparison = pd.DataFrame(comp_rows)
    comparison.to_csv(out_dir / "model_comparison.csv", index=False)
    ood_rows = comparison[comparison["split"] == "out-of-distribution"].set_index("model")["brier_at_alpha"]
    gain_full = ood_rows["Constant 0.5"] - ood_rows[SUBMITTED]
    gain_clock = ood_rows["Constant 0.5"] - ood_rows[CLOCK]
    metrics["clock_only_ood_brier"] = float(ood_rows[CLOCK])
    metrics["clock_share_of_ood_gain"] = float(gain_clock / gain_full)

    # 5. Levers against the shift, judged on the holdout only.
    lever_rows = [{"lever": "Full feature set (submitted)", **_summary(y_ood, ood, g_ood, model.ALPHA, ood)}]
    for k in (5, 10, 15):
        keep = [f for f in prep.features if f not in set(adv.importance.index[:k])]
        p = _restricted(keep, _mse())(X[m_fit], y[m_fit], X[m_ood])
        lever_rows.append(
            {
                "lever": f"Drop the {k} most shift-revealing features",
                **_summary(y_ood, p, g_ood, model.ALPHA, ood),
            }
        )
    p = _ensemble(X[m_fit], y[m_fit], X[m_ood])
    lever_rows.append(
        {"lever": "10-model ensemble (5 seeds x 2 objectives)", **_summary(y_ood, p, g_ood, model.ALPHA, ood)}
    )
    pd.DataFrame(lever_rows).to_csv(out_dir / "levers_ood.csv", index=False)

    # 6. How noisy is a public leaderboard score?
    sq = (model.shrink(ood, model.ALPHA) - y_ood) ** 2
    se = validation.leaderboard_standard_error(sq, PUBLIC_LB_ROWS)
    metrics["public_lb_rows"] = PUBLIC_LB_ROWS
    metrics["public_lb_se"] = se
    metrics["public_lb_ci_halfwidth"] = 1.96 * se
    # Gap needed to separate two *independent* scores (iid rows, as if from different row samples).
    metrics["public_lb_significant_gap"] = 1.96 * np.sqrt(2) * se
    # Two submissions scored on the same rows are correlated: a paired test is much tighter.
    # Illustrated with two similar models (submitted versus log-loss + Platt).
    paired = sq - (model.shrink(ood_preds[PLATT], model.ALPHA) - y_ood) ** 2
    metrics["public_lb_paired_gap_example"] = 1.96 * validation.leaderboard_standard_error(
        paired, PUBLIC_LB_ROWS
    )
    noise = pd.DataFrame({"n_rows": [50, 100, 143, 284, 500, 1000, 2000]})
    noise["standard_error"] = [validation.leaderboard_standard_error(sq, int(n)) for n in noise["n_rows"]]
    noise.to_csv(out_dir / "leaderboard_noise.csv", index=False)

    # 7. Leak audit on the raw training table.
    metrics |= leak_audit(comp, prep)

    # 8. What the final model relies on.
    final = model.make_model().fit(X, y)
    gain = pd.Series(final.booster_.feature_importance("gain"), index=prep.features)
    (gain / gain.sum()).sort_values(ascending=False).head(15).rename("gain_share").to_csv(
        out_dir / "feature_importance.csv"
    )

    # 9. The submission itself.
    sub = submission(prep, comp.sample)
    metrics["submission_rows"] = len(sub)
    metrics["submission_mean"] = float(sub["corrosion_risk"].mean())
    metrics["submission_min"] = float(sub["corrosion_risk"].min())
    metrics["submission_max"] = float(sub["corrosion_risk"].max())

    (out_dir / "metrics.json").write_text(json.dumps(_rounded(metrics), indent=2) + "\n")
    return metrics


def leak_audit(comp: data.Competition, prep: Prepared) -> dict:
    """Look for artefacts that reveal the corrosion month and would not exist at prediction time."""
    raw = comp.env_train.copy()
    label = target.build_target(raw, comp.corrosions)
    numeric = raw.select_dtypes("number").columns
    at_t, elsewhere = raw[label == 1], raw[label != 1]
    missing_gap = (at_t[numeric].isna().mean() - elsewhere[numeric].isna().mean()).abs().max()
    dup = raw.duplicated(subset=list(numeric), keep=False)
    months = raw["month_start_date"].pipe(pd.to_datetime)
    last = months.groupby(raw["aircraft_id"]).transform("max")
    t_rows = label == 1
    gap = (last[t_rows].dt.year * 12 + last[t_rows].dt.month) - (
        months[t_rows].dt.year * 12 + months[t_rows].dt.month
    )
    return {
        "leak_max_missingness_gap": float(missing_gap),
        "leak_duplicate_rate_at_t": float(dup[t_rows].mean()),
        "leak_duplicate_rate_elsewhere": float(dup[~t_rows].mean()),
        "leak_t_is_last_month_share": float((gap == 0).mean()),
        "leak_months_from_t_to_last_median": float(gap.median()),
    }


def _rounded(obj):
    if isinstance(obj, float | np.floating):
        return round(float(obj), 4)
    if isinstance(obj, dict):
        return {k: _rounded(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rounded(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    return obj
