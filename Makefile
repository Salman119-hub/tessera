PY ?= python
CONFIG ?= examples/fraud/tessera.yaml

.PHONY: install lint typecheck test cov data demo plan apply backfill dataset train serve compose clean

install:
	$(PY) -m pip install -e ".[dev]"

lint:
	ruff check src tests examples pipelines scripts
	ruff format --check src tests examples pipelines scripts

typecheck:
	mypy

test:
	pytest

cov:
	pytest --cov --cov-report=term-missing

data:
	$(PY) examples/fraud/generate_data.py

demo: data
	$(PY) examples/fraud/run_demo.py --workdir .tessera-demo

plan:
	tessera --config $(CONFIG) plan

apply:
	tessera --config $(CONFIG) apply

backfill:
	tessera --config $(CONFIG) validate
	tessera --config $(CONFIG) backfill

dataset:
	tessera --config $(CONFIG) build-dataset --service fraud_v1 \
		--spine examples/fraud/data/spine.csv --out .tessera/datasets/fraud_v1.csv

train: dataset
	tessera --config $(CONFIG) train --name card_fraud --service fraud_v1 \
		--dataset .tessera/datasets/fraud_v1.csv --label is_fraud --promote

serve:
	tessera --config $(CONFIG) serve --port 8080

compose: data
	docker compose up --build

clean:
	rm -rf .tessera .tessera-demo .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml
