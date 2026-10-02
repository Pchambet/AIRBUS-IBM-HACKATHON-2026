"""Build the static report `site/index.html` from the tables in `results/`.

Every number in the page is read from `results/`, so the report cannot disagree
with the code. Charts are Plotly (loaded from jsDelivr) with the shared palette.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

import pandas as pd

from corrosion_risk.data import ROOT
from corrosion_risk.experiments import CLOCK, PLATT, RESULTS, SUBMITTED
from corrosion_risk.figures import REPORTED_PRIVATE, REPORTED_PUBLIC

SITE = ROOT / "site"
SHORT = {
    "Constant 0.5": "constant 0.5",
    SUBMITTED: "submitted",
    PLATT: "log-loss + Platt",
    "LightGBM squared error, 500 trees depth 7": "larger model",
    "Logistic regression": "logistic",
    CLOCK: "two clock features",
}
PLOTLY = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"
NOISE_TICKS = (50, 100, 200, 500, 1000, 2000)


def _charts(m: dict) -> dict:
    sweep = pd.read_csv(RESULTS / "alpha_sweep.csv")
    sweep = sweep[sweep["alpha"] >= 0.3]
    shift = pd.read_csv(RESULTS / "fleet_first_observed_year.csv", index_col=0)
    share = (shift / shift.sum() * 100).round(1)
    comp = pd.read_csv(RESULTS / "model_comparison.csv")
    ood = comp[comp["split"] == "out-of-distribution"].sort_values("brier_at_alpha", ascending=False)
    ind = comp[comp["split"] == "in-distribution"].set_index("model").loc[ood["model"]].reset_index()
    rel = pd.read_csv(RESULTS / "reliability_ood.csv")
    noise = pd.read_csv(RESULTS / "leaderboard_noise.csv")

    def band(col: str, name: str, color: str) -> list[dict]:
        x = sweep["alpha"].tolist()
        return [
            {
                "x": x + x[::-1],
                "y": sweep[f"{col}_high"].tolist() + sweep[f"{col}_low"].tolist()[::-1],
                "fill": "toself",
                "fillcolor": color + "22",
                "line": {"width": 0},
                "hoverinfo": "skip",
                "showlegend": False,
            },
            {
                "x": x,
                "y": sweep[col].round(4).tolist(),
                "name": name,
                "line": {"color": color, "width": 2.5},
                "hovertemplate": "α=%{x:.2f}<br>Brier %{y:.4f}<extra>" + name + "</extra>",
            },
        ]

    a = m["alpha_submitted"]
    return {
        "sweep": {
            "data": band("id", "In-distribution CV", "#64748b")
            + band("ood", "Test-like holdout", "#0d9488")
            + [
                {
                    "x": [a, a],
                    "y": [REPORTED_PUBLIC, REPORTED_PRIVATE],
                    "mode": "markers+text",
                    "marker": {"symbol": "diamond", "size": 10, "color": "INK"},
                    "text": [
                        f"reported public LB {REPORTED_PUBLIC}",
                        f"reported private LB {REPORTED_PRIVATE}",
                    ],
                    "textposition": "middle right",
                    "name": "Kaggle (reported)",
                    "hovertemplate": "%{text}<extra></extra>",
                }
            ],
            # Phones: the labels go left of the markers, where the plot has room.
            "narrow": {
                "traces": {
                    "Kaggle (reported)": {
                        "textposition": "middle left",
                        "text": [f"public LB {REPORTED_PUBLIC}", f"private LB {REPORTED_PRIVATE}"],
                    }
                }
            },
            "layout": {
                "xaxis": {"title": "shrinkage factor α (1 = raw model)"},
                "yaxis": {"title": "Brier score (lower is better)"},
                "shapes": [
                    {
                        "type": "line",
                        "x0": a,
                        "x1": a,
                        "yref": "paper",
                        "y0": 0,
                        "y1": 1,
                        "line": {"color": "#d97706", "dash": "dash", "width": 1.5},
                    }
                ],
            },
        },
        "shift": {
            "data": [
                {
                    "type": "bar",
                    "x": share.index.tolist(),
                    "y": share["train"].tolist(),
                    "name": "train fleet",
                    "marker": {"color": "#64748b"},
                    "hovertemplate": "%{x}: %{y}%<extra>train</extra>",
                },
                {
                    "type": "bar",
                    "x": share.index.tolist(),
                    "y": share["test"].tolist(),
                    "name": "test fleet",
                    "marker": {"color": "#0d9488"},
                    "hovertemplate": "%{x}: %{y}%<extra>test</extra>",
                },
            ],
            "layout": {
                "barmode": "group",
                "bargap": 0.25,
                "xaxis": {"title": "first year with data", "dtick": 1},
                "yaxis": {"title": "share of fleet (%)"},
            },
        },
        "models": {
            "data": [
                {
                    "type": "scatter",
                    "mode": "markers",
                    "name": name,
                    "y": d["model"].map(SHORT).tolist(),
                    "x": d["brier_at_alpha"].round(4).tolist(),
                    "marker": {"color": color, "size": 10},
                    "error_x": {
                        "type": "data",
                        "symmetric": False,
                        "color": color,
                        "array": (d["ci_high"] - d["brier_at_alpha"]).round(4).tolist(),
                        "arrayminus": (d["brier_at_alpha"] - d["ci_low"]).round(4).tolist(),
                    },
                    "hovertemplate": "%{y}<br>Brier %{x:.4f}<extra>" + name + "</extra>",
                }
                for d, name, color in (
                    (ind, "in-distribution CV", "#64748b"),
                    (ood, "test-like holdout", "#0d9488"),
                )
            ],
            "layout": {
                "xaxis": {"title": f"Brier at α = {a} (95% aircraft-bootstrap CI)"},
                "yaxis": {"automargin": True},
            },
            "narrow": {
                "layout": {
                    "xaxis": {"title": f"Brier at α = {a}<br>(95% aircraft-bootstrap CI)"},
                    "margin": {"t": 20, "r": 20, "b": 70, "l": 60},
                }
            },
        },
        "reliability": {
            "data": [
                {
                    "x": [0, 1],
                    "y": [0, 1],
                    "mode": "lines",
                    "name": "perfect calibration",
                    "line": {"color": "#94a3b8", "dash": "dot", "width": 1},
                    "hoverinfo": "skip",
                }
            ]
            + [
                {
                    "x": d["mean_predicted"].round(3).tolist(),
                    "y": d["observed_rate"].round(3).tolist(),
                    "customdata": d["n"].tolist(),
                    "mode": "lines+markers",
                    "name": label,
                    "line": {"color": color, "width": 2.5},
                    "marker": {"size": 8},
                    "hovertemplate": "predicted %{x:.2f}<br>observed %{y:.2f}<br>n = %{customdata}<extra></extra>",
                }
                for d, label, color in (
                    (rel[rel["model"] == "raw"], "raw model", "#64748b"),
                    (rel[rel["model"] == "shrunk"], f"shrunk, α = {a}", "#0d9488"),
                )
            ],
            "layout": {
                "xaxis": {"title": "mean predicted probability", "range": [0, 1]},
                "yaxis": {"title": "observed corrosion rate", "range": [0, 1]},
            },
        },
        "noise": {
            "data": [
                {
                    "x": noise["n_rows"].tolist(),
                    "y": (1.96 * noise["standard_error"]).round(4).tolist(),
                    "mode": "lines+markers",
                    "name": "95% margin of error",
                    "line": {"color": "#0d9488", "width": 2.5},
                    "hovertemplate": "%{x} rows: ±%{y:.3f}<extra></extra>",
                }
            ],
            "layout": {
                "xaxis": {
                    "title": "rows used to compute the score (log scale)",
                    "type": "log",
                    "tickvals": NOISE_TICKS,
                    "ticktext": [f"{t:,}" for t in NOISE_TICKS],
                },
                "yaxis": {"title": "95% margin of error"},
                "annotations": [
                    {
                        "x": math.log10(m["public_lb_rows"]),
                        "y": 1,
                        "yref": "paper",
                        "text": f"{m['public_lb_rows']} public rows",
                        "showarrow": False,
                        "xanchor": "left",
                        "yanchor": "top",
                        "xshift": 4,
                        "font": {"color": "#d97706"},
                    }
                ],
                "shapes": [
                    {
                        "type": "line",
                        "x0": m["public_lb_rows"],
                        "x1": m["public_lb_rows"],
                        "yref": "paper",
                        "y0": 0,
                        "y1": 1,
                        "line": {"color": "#d97706", "dash": "dash", "width": 1.5},
                    }
                ],
            },
        },
    }


def _table(df: pd.DataFrame, first: str) -> str:
    head = f"<tr><th>{first}</th><th>Brier at α = 0.7</th><th>95% CI</th><th>Δ vs submitted</th><th>AUC</th></tr>"
    rows = []
    for _, r in df.iterrows():
        delta = (
            ""
            if r[first.lower()].endswith("(submitted)")
            else (f"{r['delta_vs_submitted']:+.4f} [{r['delta_ci_low']:+.4f}, {r['delta_ci_high']:+.4f}]")
        )
        rows.append(
            f"<tr><td>{html.escape(r[first.lower()])}</td><td>{r['brier_at_alpha']:.4f}</td>"
            f"<td>[{r['ci_low']:.4f}, {r['ci_high']:.4f}]</td><td>{delta}</td><td>{r['auc']:.3f}</td></tr>"
        )
    return f"<table>{head}{''.join(rows)}</table>"


def _lever_reading(levers: pd.DataFrame) -> str:
    better = levers[levers["delta_ci_high"] < 0]
    tried = len(levers) - 1
    if better.empty:
        return f"None of the {tried} levers against the shift beats the submitted model with a paired interval excluding zero."
    names = "; ".join(html.escape(n) for n in better["lever"])
    count = "only one has" if len(better) == 1 else f"only {len(better)} have"
    return (
        f"Of the {tried} levers against the shift, {count} a paired interval that excludes zero, and by a small "
        f"margin ({better['delta_vs_submitted'].iloc[0]:+.4f}): {names}. With {tried} levers tried, treat it as "
        "suggestive; it was not part of the submission."
    )


def build() -> Path:
    m = json.loads((RESULTS / "metrics.json").read_text())
    comp = pd.read_csv(RESULTS / "model_comparison.csv")
    ood_table = _table(comp[comp["split"] == "out-of-distribution"], "Model")
    lever_df = pd.read_csv(RESULTS / "levers_ood.csv")
    levers = _table(lever_df, "Lever")
    charts = json.dumps(_charts(m))
    a = m["alpha_submitted"]
    clock = comp[(comp["model"] == CLOCK) & (comp["split"] == "out-of-distribution")].iloc[0]
    labelled = m["ood_fit_aircraft"] + m["ood_holdout_aircraft"]
    # Upper bound on the private board if only T and T-24 rows are scored.
    max_scored = 2 * m["n_test_aircraft"]
    private_rows = max_scored - m["public_lb_rows"]
    body = f"""
<header>
  <p class="kicker">Airbus × IBM × AWS hackathon 2026 · aircraft corrosion risk</p>
  <h1>When the test fleet is not your training fleet</h1>
  <p class="lede">A corrosion-risk model built for a Kaggle-scored hackathon, and the validation work that decided
  its single most important parameter. Reported result: <strong>6th on the Brier leaderboard, 2nd overall</strong>
  (model + business pitch), climbing from 22nd on the public leaderboard to 6th on the private one. Both boards are
  small, so that move is consistent with the approach rather than proof of it (section 5). Team project: the event-time
  code is in <a href="https://github.com/Yixian-ch/AIRBUS-IBM">the team repository</a>; this is a post-event rebuild
  and re-analysis of the pipeline I owned as ML lead.</p>
</header>

<section class="kpis">
  <div><span>{m["adversarial_auc_grouped"]:.2f}</span>adversarial AUC: test aircraft are easy to tell apart</div>
  <div><span>{m["id_brier_raw"]:.3f} → {m["id_brier_raw_holdout_aircraft"]:.3f} → {m["ood_brier_raw"]:.3f}</span>raw-model
  Brier: CV overall → CV on test-like aircraft → test-like aircraft left out of training</div>
  <div><span>α = {m["ood_best_alpha"]:.2f}</span>Brier-optimal shrinkage on test-like aircraft (CV says 1.00)</div>
  <div><span>±{m["public_lb_ci_halfwidth"]:.3f}</span>95% margin of error of a {m["public_lb_rows"]}-row public score</div>
</section>

<h2>1. The problem, stated precisely</h2>
<p>Each of {m["n_train_aircraft"]} training aircraft has a monthly environment history (weather, aerosols, gases,
parking time) and one corrosion observation date T. The leaderboard scores only two months per aircraft: T (label 1)
and T − 24 months (label 0). That gives {m["n_scored_train_rows"]} labelled rows ({m["n_positive"]} positives,
{m["n_negative"]} negatives) and turns the task into a balanced, <em>within-aircraft</em> question: is this aircraft
more exposed now than two years ago? Anything constant per aircraft cancels out. The model is a LightGBM regressor
on 0/1 labels (squared error is the Brier score) over {m["n_features"]} strictly causal exposure features.</p>

<h2>2. The test fleet is a different population</h2>
<p>A classifier trained to tell training rows from test rows, with folds split by aircraft, reaches
AUC {m["adversarial_auc_grouped"]:.2f}. The clearest symptom: {m["share_test_first_record_year"]:.0%} of test aircraft
have data from the first year of the record ({m["first_record_year"]}), against
{m["share_train_first_record_year"]:.1%} of training aircraft. Their histories are left-censored, so "age" means
something else for them.</p>
<div class="chart" id="shift"></div>

<h2>3. A validation split that looks like the test set</h2>
<p>Grouped cross-validation answers "how good is the model on aircraft like the training ones?". To answer the
question that matters, the training aircraft were ranked by how test-like the adversarial model finds them; the model
is fitted on the {m["ood_fit_aircraft"]} least test-like and scored on the {m["ood_holdout_aircraft"]} most
test-like ({m["ood_holdout_rows"]} rows). Uncertainty comes from a bootstrap over aircraft, not rows.</p>
<p>The gap between the two validations has two parts. Grouped CV scores {m["id_brier_raw"]:.3f} overall but already
{m["id_brier_raw_holdout_aircraft"]:.3f} on the test-like aircraft (against {m["id_brier_raw_other_aircraft"]:.3f} on
the others): these aircraft are harder in themselves. Leaving them out of training, as the test set does, raises it
to {m["ood_brier_raw"]:.3f}. Part of that last step is the smaller training set ({m["ood_fit_aircraft"]} aircraft
instead of about {labelled * 4 // 5} per CV fold).</p>
<p>The two validations disagree on the one decision that matters. In-distribution, shrinking predictions toward
0.5 only hurts (best α = {m["id_best_alpha"]:.2f}). On test-like aircraft the raw model is over-confident and the
Brier-optimal shrinkage is α = {m["ood_best_alpha"]:.2f}. The submission used α = {a}; on the holdout that improves
the Brier by {m["ood_shrinkage_gain"]:.4f} (paired 95% CI {m["ood_shrinkage_gain_ci"][0]:.4f} to
{m["ood_shrinkage_gain_ci"][1]:.4f}). That gain is in-sample for α. With α cross-fitted on random halves of the
holdout aircraft (fitted α from {m["ood_crossfit_alpha_range"][0]:.2f} to {m["ood_crossfit_alpha_range"][1]:.2f}),
the gain is {m["ood_shrinkage_gain_crossfit"]:.4f} (95% CI {m["ood_shrinkage_gain_crossfit_ci"][0]:.4f} to
{m["ood_shrinkage_gain_crossfit_ci"][1]:.4f}).</p>
<div class="chart" id="sweep"></div>
<p class="note">Diamonds are the scores reported on the Kaggle leaderboard; they cannot be recomputed without the
hidden labels. The holdout estimate at α = {a} is {m["ood_brier_alpha"]:.3f}
(95% CI {m["ood_brier_alpha_ci"][0]:.3f}–{m["ood_brier_alpha_ci"][1]:.3f}); in-distribution CV would have promised
{m["id_brier_raw"]:.3f}.</p>
<div class="chart small" id="reliability"></div>

<h2>4. What else was tried, judged on the holdout</h2>
<p>All candidates share features and splits; differences are paired (same aircraft), negative is better.</p>
{ood_table}
<div class="chart" id="models"></div>
<p>Two honest readings. First, a LightGBM that sees only the two clock features (months observed, age) captures
most of the full model's gain over a constant forecast on test-like aircraft (point estimate
{m["clock_share_of_ood_gain"]:.0%}; its paired difference to the full model, {clock["delta_vs_submitted"]:+.4f}
[{clock["delta_ci_low"]:+.4f}, {clock["delta_ci_high"]:+.4f}], is not significant); the environmental features add
little that transfers. Second, the choice of objective (squared error versus log-loss
with Platt scaling) is within noise on the holdout.</p>
{levers}
<p>{_lever_reading(lever_df)}</p>

<h2>5. The leaderboards were too small to rank on</h2>
<p>With {m["public_lb_rows"]} rows, a single public Brier score carries a 95% margin of error of
±{m["public_lb_ci_halfwidth"]:.3f}, treating rows as independent. Two independent scores would need to differ by more
than {m["public_lb_significant_gap"]:.3f}. Submissions scored on the same rows are correlated, so a paired comparison
is tighter: for two similar models (submitted versus log-loss + Platt) the threshold is about
{m["public_lb_paired_gap_example"]:.3f}. Public rows probably include T / T − 24 pairs from the same aircraft, which
these row-level formulas ignore. Either way, small public gaps carried little information, which is why the
submission was chosen on the holdout rather than on public feedback.</p>
<p>The private board is probably no better. Its row count is not known here, but if only T and T − 24 rows are scored,
the {m["n_test_aircraft"]} test aircraft give at most {max_scored} scored rows, leaving at most about {private_rows}
for the private board: the same margin of error. The move from 22nd to 6th is consistent with choosing α on a
shift-aware holdout, but it does not prove the approach better than the teams ranked nearby.</p>
<div class="chart small" id="noise"></div>

<h2>6. Leak audit</h2>
<p>No artefact betrays the corrosion month: missingness at T differs from other months by at most
{m["leak_max_missingness_gap"] * 100:.2f} percentage points; duplicate rows are about as frequent at T ({m["leak_duplicate_rate_at_t"]:.1%})
as elsewhere ({m["leak_duplicate_rate_elsewhere"]:.1%}); T is the last observed month for only
{m["leak_t_is_last_month_share"]:.1%} of aircraft (median gap {m["leak_months_from_t_to_last_median"]:.0f} months).</p>

<h2>Limitations</h2>
<ul>
  <li>The holdout is one split of {m["ood_holdout_aircraft"]} aircraft; it is deliberately pessimistic (the model sees
  only {m["ood_fit_aircraft"] / labelled:.0%} of the {labelled} labelled training aircraft) and served to choose α, not to
  forecast the final score.</li>
  <li>The benchmark contrasts T with T − 24 within each aircraft, so elapsed time is informative by construction. A good
  Brier here does not by itself show value for scheduling inspections, which would need every-month labels and a cost
  model.</li>
  <li>Leaderboard scores and ranks are as reported by the organisers and cannot be reproduced here.</li>
  <li>The first observed month stands in for delivery because the test set has no delivery date; it is wrong for
  aircraft whose history starts with the record.</li>
  <li>The competition data cannot be redistributed; reproducing requires accepting the Kaggle rules.</li>
</ul>
"""
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Corrosion risk under shift</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 16 16%22%3E%3Crect width=%2216%22 height=%2216%22 rx=%224%22 fill=%22%230d9488%22/%3E%3C/svg%3E">
<meta name="description" content="Validation study of a corrosion-risk model whose test fleet differs from its training fleet.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<script src="{PLOTLY}"></script>
<style>
:root {{ --bg:#ffffff; --ink:#0f172a; --muted:#64748b; --line:#e2e8f0; --card:#f8fafc; --accent:#0d9488; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#0b1120; --ink:#e2e8f0; --muted:#94a3b8;
  --line:#1e293b; --card:#111827; --accent:#2dd4bf; }} }}
:root[data-theme="dark"] {{ --bg:#0b1120; --ink:#e2e8f0; --muted:#94a3b8; --line:#1e293b; --card:#111827; --accent:#2dd4bf; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:16px/1.65 Inter, system-ui, -apple-system, sans-serif; }}
main {{ max-width:860px; margin:0 auto; padding:48px 16px 80px; }}
.kicker {{ color:var(--accent); font-weight:600; font-size:.85rem; letter-spacing:.04em; text-transform:uppercase; margin:0; }}
h1 {{ font-size:2.1rem; line-height:1.2; margin:.3em 0 .4em; }}
h2 {{ font-size:1.3rem; margin:2.2em 0 .5em; }}
.lede {{ font-size:1.1rem; color:var(--muted); }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin:28px 0; }}
.kpis div {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px; font-size:.85rem; color:var(--muted); }}
.kpis span {{ display:block; font-size:1.35rem; font-weight:700; color:var(--ink); }}
.chart {{ height:420px; margin:16px 0; }}
.chart.small {{ height:360px; }}
.note {{ font-size:.88rem; color:var(--muted); }}
table {{ width:100%; border-collapse:collapse; font-size:.85rem; margin:16px 0; display:block; overflow-x:auto; }}
th, td {{ text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); white-space:nowrap; }}
th {{ color:var(--muted); font-weight:600; }}
footer {{ margin-top:48px; padding-top:16px; border-top:1px solid var(--line); color:var(--muted); font-size:.85rem; }}
a {{ color:var(--accent); }}
</style>
</head>
<body>
<main>
{body}
<footer>Code and data pipeline: <a href="https://github.com/Pchambet/AIRBUS-IBM-HACKATHON-2026">github.com/Pchambet/AIRBUS-IBM-HACKATHON-2026</a>.
Built by <a href="https://github.com/Pchambet">Pierre Chambet</a> — decision science for operations under uncertainty.</footer>
</main>
<script>
const charts = {charts};
const dark = matchMedia("(prefers-color-scheme: dark)").matches && document.documentElement.dataset.theme !== "light";
const ink = dark ? "#e2e8f0" : "#0f172a", grid = dark ? "#1e293b" : "#e2e8f0";
const narrow = matchMedia("(max-width: 600px)").matches;
for (const [id, spec] of Object.entries(charts)) {{
  if (narrow && spec.narrow) {{
    for (const t of spec.data) Object.assign(t, (spec.narrow.traces || {{}})[t.name]);
    for (const [k, v] of Object.entries(spec.narrow.layout || {{}})) spec.layout[k] = Object.assign({{}}, spec.layout[k], v);
  }}
  for (const t of spec.data) if (t.marker && t.marker.color === "INK") t.marker.color = ink;
  const layout = Object.assign({{
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: {{ family: "Inter, system-ui, sans-serif", color: ink, size: 12 }},
    margin: {{ t: 20, r: 20, b: 50, l: 60 }}, legend: {{ orientation: "h", y: 1.12 }},
    hoverlabel: {{ font: {{ family: "Inter, system-ui, sans-serif" }} }},
  }}, spec.layout);
  for (const ax of ["xaxis", "yaxis"]) layout[ax] = Object.assign({{ gridcolor: grid, zerolinecolor: grid }}, layout[ax] || {{}});
  Plotly.newPlot(id, spec.data, layout, {{ displayModeBar: false, responsive: true }});
}}
</script>
</body>
</html>
"""
    SITE.mkdir(exist_ok=True)
    path = SITE / "index.html"
    path.write_text(page)
    return path
