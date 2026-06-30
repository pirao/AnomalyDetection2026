.PHONY: help run stop test inference-test notebooks demo

help:
	@echo "make run             Build and start the local API on http://localhost:8000"
	@echo "make test            Run the fast test suite (unit, contract, performance); excludes the benchmark"
	@echo "make inference-test  Run the private benchmark gate (test_evaluation.py); can take about 15 minutes"
	@echo "make notebooks       Start JupyterLab on http://localhost:8888 with notebooks/cache mounted"
	@echo "make demo            Replay a sensor through the running API and regenerate the deployment GIF"
	@echo "make stop            Stop Compose services"

run:
	docker compose up --build api

stop:
	docker compose down

test:
	docker compose run --rm --build test

inference-test:
	docker compose run --rm --build inference-test

notebooks:
	docker compose up --build notebooks

# Replays sensor 9's pred split through the running API's POST /predict, then
# renders reports/figures/mlflow/deploy_demo.gif. Runs inside the `notebooks`
# image (matplotlib + analysis stack, PYTHONPATH=/app/src already set) so it does
# not depend on the host Python environment. Reaches the API over the Compose
# network as http://api:8000 and writes the GIF to the mounted ./reports volume.
#
# Prerequisite: start the API first in another terminal with `make run` (it brings
# up the api + mlflow services this target connects to).
demo:
	docker compose run --rm --build notebooks \
	  uv run --no-sync python -m analysis.mlflow.deploy_demo --sensor 9 --http http://api:8000
