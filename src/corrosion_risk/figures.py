"""Static figures for the README, drawn only from the tables in `results/`."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from corrosion_risk.data import ROOT
from corrosion_risk.experiments import RESULTS

FIGURES = ROOT / "docs" / "figures"
INK, TEAL, AMBER, SLATE, GRID = "#0f172a", "#0d9488", "#d97706", "#64748b", "#e2e8f0"
# Scores published on the Kaggle leaderboard; they cannot be recomputed without the hidden labels.
REPORTED_PUBLIC, REPORTED_PRIVATE = 0.215, 0.18


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": SLATE,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": SLATE,
            "ytick.color": SLATE,
            "text.color": INK,
            "font.size": 10,
            "legend.frameon": False,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def hero(metrics: dict) -> Path:
    sweep = pd.read_csv(RESULTS / "alpha_sweep.csv")
    sweep = sweep[sweep["alpha"] >= 0.3]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for col, color, label in (
        ("id", SLATE, "In-distribution CV\n(aircraft-grouped folds)"),
        ("ood", TEAL, f"Test-like holdout\n({metrics['ood_holdout_aircraft']} most test-like aircraft)"),
    ):
        ax.fill_between(
            sweep["alpha"], sweep[f"{col}_low"], sweep[f"{col}_high"], color=color, alpha=0.13, lw=0
        )
        ax.plot(sweep["alpha"], sweep[col], color=color, lw=2.2)
        last = sweep.iloc[-1]
        ax.annotate(
            label,
            (1.0, last[col]),
            xytext=(8, 0),
            textcoords="offset points",
            color=color,
            va="center",
            fontsize=9.5,
            fontweight="bold",
        )
    a = metrics["alpha_submitted"]
    ax.axvline(a, color=AMBER, lw=1.4, ls="--")
    ax.annotate(
        f"submitted α = {a}",
        (a, 0.236),
        xytext=(-6, 0),
        textcoords="offset points",
        ha="right",
        color=AMBER,
        fontsize=9.5,
        fontweight="bold",
    )
    ood = metrics["ood_brier_alpha"]
    ax.scatter([a], [ood], s=46, color=TEAL, zorder=5, edgecolor="white", linewidth=1.5)
    ax.annotate(
        f"{ood:.3f}",
        (a, ood),
        xytext=(0, -16),
        textcoords="offset points",
        ha="center",
        color=TEAL,
        fontsize=9.5,
        fontweight="bold",
    )
    for score, name in ((REPORTED_PUBLIC, "public LB"), (REPORTED_PRIVATE, "private LB")):
        ax.scatter([a], [score], marker="D", s=34, color=INK, zorder=5)
        ax.annotate(
            f"reported {name} {score:.3f}",
            (a, score),
            xytext=(9, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK,
        )
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0.13, 0.24)
    ax.set_xlabel("Shrinkage factor α in  p → 0.5 + α (p − 0.5)    (α = 1: raw model)")
    ax.set_ylabel("Brier score (lower is better)")
    ax.set_title(
        f"In-distribution CV says “don't shrink” (best α = {metrics['id_best_alpha']:.2f});\n"
        f"on test-like aircraft the optimum is α = {metrics['ood_best_alpha']:.2f}",
        pad=12,
    )
    return _save(fig, "hero_shrinkage.png")


def fleet_shift(metrics: dict) -> Path:
    shift = pd.read_csv(RESULTS / "fleet_first_observed_year.csv", index_col=0)
    share = shift / shift.sum()
    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = share.index.to_numpy()
    w = 0.38
    ax.bar(
        x - w / 2,
        share["train"] * 100,
        width=w,
        color=SLATE,
        label=f"train fleet ({metrics['n_train_aircraft']} aircraft)",
    )
    ax.bar(
        x + w / 2,
        share["test"] * 100,
        width=w,
        color=TEAL,
        label=f"test fleet ({metrics['n_test_aircraft']} aircraft)",
    )
    t = metrics["share_test_first_record_year"] * 100
    ax.annotate(
        f"{t:.0f}% of test aircraft have data\nfrom the first year of the record\n(train: {metrics['share_train_first_record_year'] * 100:.1f}%)",
        (metrics["first_record_year"] + w / 2, t),
        xytext=(28, -8),
        textcoords="offset points",
        color=TEAL,
        fontweight="bold",
        fontsize=9.5,
        va="top",
    )
    ax.set_xticks(x)
    ax.set_xlabel("First month with environment data (year)")
    ax.set_ylabel("Share of fleet (%)")
    ax.legend(loc="upper right")
    ax.set_title(
        f"The test fleet is a different population: adversarial AUC {metrics['adversarial_auc_grouped']:.2f}",
        pad=10,
    )
    return _save(fig, "fleet_shift.png")


def reliability(metrics: dict) -> Path:
    rel = pd.read_csv(RESULTS / "reliability_ood.csv")
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.plot([0, 1], [0, 1], color=SLATE, lw=1, ls=":")
    ax.annotate("perfect calibration", (0.04, 0.07), rotation=42, color=SLATE, fontsize=8.5)
    for name, color, label in (
        ("raw", SLATE, "raw model"),
        ("shrunk", TEAL, f"shrunk, α = {metrics['alpha_submitted']}"),
    ):
        d = rel[rel["model"] == name]
        ax.plot(d["mean_predicted"], d["observed_rate"], color=color, lw=2, marker="o", ms=5, label=label)
    low = rel[(rel["model"] == "raw")].iloc[0]
    ax.annotate(
        f"raw says {low['mean_predicted']:.0%},\nreality is {low['observed_rate']:.0%}",
        (low["mean_predicted"], low["observed_rate"]),
        xytext=(14, 30),
        textcoords="offset points",
        fontsize=9,
        color=INK,
        arrowprops={"arrowstyle": "-", "color": SLATE, "lw": 0.8},
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted corrosion probability (bin)")
    ax.set_ylabel("Observed corrosion rate")
    ax.legend(loc="upper left")
    ax.set_title(
        "On test-like aircraft the raw model is over-confident\nat the low end; shrinkage removes the worst of it",
        pad=10,
    )
    return _save(fig, "reliability_ood.png")


def model_comparison(metrics: dict) -> Path:
    comp = pd.read_csv(RESULTS / "model_comparison.csv")
    order = comp[comp["split"] == "out-of-distribution"].sort_values("brier_at_alpha", ascending=False)[
        "model"
    ]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, name in enumerate(order):
        for split, color, off in (("in-distribution", SLATE, 0.16), ("out-of-distribution", TEAL, -0.16)):
            r = comp[(comp["model"] == name) & (comp["split"] == split)].iloc[0]
            ax.plot([r["ci_low"], r["ci_high"]], [i + off] * 2, color=color, lw=2, solid_capstyle="round")
            ax.scatter([r["brier_at_alpha"]], [i + off], color=color, s=40, zorder=3, edgecolor="white")
            if split == "out-of-distribution":
                ax.annotate(
                    f"{r['brier_at_alpha']:.3f}",
                    (r["ci_high"], i + off),
                    xytext=(5, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=8.5,
                    color=INK,
                )
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.tick_params(axis="y", colors=INK)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Brier score at α = {metrics['alpha_submitted']} with 95% aircraft-bootstrap CI")
    ax.scatter([], [], color=SLATE, label="in-distribution CV")
    ax.scatter([], [], color=TEAL, label="test-like holdout")
    ax.legend(loc="lower left", ncol=2, bbox_to_anchor=(0, 1.0))
    ax.set_title(
        f"On test-like aircraft, two clock features deliver {metrics['clock_share_of_ood_gain']:.0%}\n"
        "of the full model's gain over a constant 0.5 forecast",
        pad=30,
    )
    return _save(fig, "model_comparison.png")


def leaderboard_noise(metrics: dict) -> Path:
    noise = pd.read_csv(RESULTS / "leaderboard_noise.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(noise["n_rows"], 1.96 * noise["standard_error"], color=TEAL, lw=2.2, marker="o", ms=4)
    n, half = metrics["public_lb_rows"], metrics["public_lb_ci_halfwidth"]
    ax.scatter([n], [half], s=60, color=AMBER, zorder=4)
    ax.annotate(
        f"public leaderboard ({n} rows):\n±{half:.3f} on a single score",
        (n, half),
        xytext=(18, 4),
        textcoords="offset points",
        color=AMBER,
        fontweight="bold",
        fontsize=9.5,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Rows used to compute the score (log scale)")
    ax.set_ylabel("95% margin of error on the Brier score")
    ax.set_title(
        f"Two teams' public scores must differ by > {metrics['public_lb_significant_gap']:.3f} to be told apart",
        pad=10,
    )
    return _save(fig, "leaderboard_noise.png")


def feature_importance(metrics: dict) -> Path:
    imp = pd.read_csv(RESULTS / "feature_importance.csv", index_col=0)["gain_share"].head(12)[::-1]
    clock = {"months_observed", "aircraft_age_months"}
    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = [AMBER if f in clock else TEAL for f in imp.index]
    ax.barh(imp.index, imp * 100, color=colors, height=0.66)
    for i, v in enumerate(imp * 100):
        ax.annotate(f"{v:.1f}%", (v, i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8.5)
    ax.tick_params(axis="y", colors=INK)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share of total split gain in the final model (%)")
    share = imp[imp.index.isin(clock)].sum()
    ax.set_title(
        f"The model mostly reads the clock: two time features (amber) carry {share:.0%} of the gain", pad=10
    )
    return _save(fig, "feature_importance.png")


def build_all() -> list[Path]:
    _style()
    metrics = json.loads((RESULTS / "metrics.json").read_text())
    return [
        f(metrics)
        for f in (hero, fleet_shift, reliability, model_comparison, leaderboard_noise, feature_importance)
    ]
