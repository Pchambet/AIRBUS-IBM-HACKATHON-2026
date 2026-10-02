"""Command-line entry point: `uv run corrosion-risk <command>`."""

from __future__ import annotations

import argparse
from pathlib import Path

from corrosion_risk import data
from corrosion_risk.data import ROOT

SUBMISSION = ROOT / "data" / "submission" / "final_submission_best.csv"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="corrosion-risk", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("data", help="put the competition CSVs in data/raw/")
    fetch.add_argument("--source", type=Path, help="copy from this directory instead of Kaggle")
    sub.add_parser("train", help="fit on all scored rows and write the submission file")
    sub.add_parser("analyze", help="run every validation experiment and write results/")
    sub.add_parser("figures", help="draw docs/figures/ from results/")
    sub.add_parser("report", help="build site/index.html from results/")
    args = parser.parse_args(argv)

    if args.command == "data":
        print(f"Competition files ready in {data.fetch(args.source)}")
    elif args.command == "train":
        from corrosion_risk.experiments import prepare, submission

        comp = data.load()
        out = submission(prepare(comp), comp.sample)
        SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(SUBMISSION, index=False)
        risk = out["corrosion_risk"]
        print(
            f"Wrote {SUBMISSION.relative_to(ROOT)}: {len(out)} rows, mean {risk.mean():.3f}, "
            f"range [{risk.min():.3f}, {risk.max():.3f}]"
        )
    elif args.command == "analyze":
        from corrosion_risk.experiments import RESULTS, run

        metrics = run(data.load())
        print(f"Wrote {RESULTS.relative_to(ROOT)}/ ({len(metrics)} headline metrics)")
    elif args.command == "figures":
        from corrosion_risk.figures import build_all

        for path in build_all():
            print(f"Wrote {path.relative_to(ROOT)}")
    elif args.command == "report":
        from corrosion_risk.report import build

        print(f"Wrote {build().relative_to(ROOT)}")


if __name__ == "__main__":
    main()
