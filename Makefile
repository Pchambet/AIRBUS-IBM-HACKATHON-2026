.PHONY: setup data train run report test lint

setup:  ## install the locked environment
	uv sync --locked

data:  ## put the Kaggle competition files in data/raw/ (see data/README.md)
	uv run corrosion-risk data $(if $(SOURCE),--source $(SOURCE),)

train:  ## fit on all scored rows, write data/submission/final_submission_best.csv
	uv run corrosion-risk train

run: train  ## submission + every validation experiment + figures
	uv run corrosion-risk analyze
	uv run corrosion-risk figures

report:  ## static report in site/index.html
	uv run corrosion-risk report

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .
