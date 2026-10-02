import json

import numpy as np

from corrosion_risk import experiments, model
from corrosion_risk.model import ALPHA


def test_submission_is_aligned_and_bounded(competition):
    prep = experiments.prepare(competition)
    assert set(prep.y) == {0, 1}
    out = experiments.submission(prep, competition.sample)
    assert out["id"].tolist() == competition.sample["id"].tolist()
    assert out["corrosion_risk"].between(0.5 - ALPHA / 2, 0.5 + ALPHA / 2).all()


def test_full_analysis_runs_end_to_end(competition, tmp_path, monkeypatch):
    # A plumbing check, not a benchmark: fewer trees keep it fast.
    monkeypatch.setitem(model.PARAMS, "n_estimators", 40)
    metrics = experiments.run(competition, out_dir=tmp_path)
    assert {"alpha_sweep.csv", "model_comparison.csv", "levers_ood.csv", "metrics.json"} <= {
        p.name for p in tmp_path.iterdir()
    }
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["n_scored_train_rows"] == metrics["n_scored_train_rows"]
    assert np.isfinite(saved["ood_brier_alpha"])
