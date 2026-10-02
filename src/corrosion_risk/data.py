"""Locate and load the competition files.

The Kaggle competition data cannot be redistributed, so it lives in `data/raw/`
(gitignored). `CORROSION_DATA_DIR` overrides the location.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

COMPETITION = "haks-airbus-x-ibm-x-aws-2026"
FILES = {
    "env_train": "environment_training.csv",
    "env_test": "environment_test.csv",
    "corrosions": "corrosions_training.csv",
    "sample": "sample_submission-2.csv",
}
ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return Path(os.environ.get("CORROSION_DATA_DIR", ROOT / "data" / "raw"))


def missing_files(directory: Path | None = None) -> list[str]:
    directory = directory or data_dir()
    return [name for name in FILES.values() if not (directory / name).exists()]


@dataclass(frozen=True)
class Competition:
    env_train: pd.DataFrame
    env_test: pd.DataFrame
    corrosions: pd.DataFrame
    sample: pd.DataFrame


def load(directory: Path | None = None) -> Competition:
    directory = directory or data_dir()
    missing = missing_files(directory)
    if missing:
        raise FileNotFoundError(f"Missing {missing} in {directory}. Run `make data` (see data/README.md).")
    return Competition(**{key: pd.read_csv(directory / name) for key, name in FILES.items()})


def fetch(source: Path | None = None) -> Path:
    """Put the four competition CSVs in `data/raw/`; idempotent.

    Copies them from `source` when given, otherwise downloads them with the Kaggle
    CLI (requires accepted competition rules and Kaggle credentials).
    """
    target = data_dir()
    target.mkdir(parents=True, exist_ok=True)
    if not missing_files(target):
        return target
    if source is not None:
        for name in missing_files(target):
            shutil.copy2(source / name, target / name)
    else:
        if shutil.which("kaggle") is None:
            raise RuntimeError(
                "Kaggle CLI not found. Install it (`uv tool install kaggle`) or pass "
                "--source DIR pointing to the downloaded competition files."
            )
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(target)],
            check=True,
        )
        for archive in target.glob("*.zip"):
            shutil.unpack_archive(archive, target)
            archive.unlink()
    still_missing = missing_files(target)
    if still_missing:
        raise FileNotFoundError(f"Still missing after fetch: {still_missing}")
    return target
