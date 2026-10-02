# Data

The Airbus × IBM × AWS 2026 competition data is **not included**: the Kaggle rules do not
allow redistribution. Results tables derived from it (`results/`) and the figures are committed.

Expected files in `data/raw/` (gitignored):

| File | Content |
|---|---|
| `environment_training.csv` | 63,524 aircraft-months for 758 training aircraft: weather (METAR), aerosols, gases, parking time |
| `environment_test.csv` | 14,303 aircraft-months for 142 test aircraft |
| `corrosions_training.csv` | one corrosion observation date per training aircraft |
| `sample_submission-2.csv` | submission template (one row per test aircraft-month) |

## Getting them

1. Accept the competition rules on Kaggle (`haks-airbus-x-ibm-x-aws-2026`).
2. Either `make data` (uses the Kaggle CLI and your Kaggle credentials), or download the
   files by hand and run `make data SOURCE=/path/to/folder` to copy them into place.

`CORROSION_DATA_DIR` overrides the location.
