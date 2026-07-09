.PHONY: help run stop test inference-test notebooks demo demo-sensor

help:
	@echo "make run                  Build and start the local API on http://localhost:8000"
	@echo "make test                 Run the fast test suite (unit, contract, performance); excludes the benchmark"
	@echo "make inference-test       Run the private benchmark gate (test_evaluation.py); can take about 15 minutes"
	@echo "make notebooks            Start JupyterLab on http://localhost:8888 with notebooks/cache mounted"
	@echo "make demo                 Start the API (if needed) and replay sensor 9 to regenerate the deployment GIF"
	@echo "make demo-sensor SENSOR=N Start the API (if needed) and replay a specific sensor (e.g. make demo-sensor SENSOR=5)"
	@echo "make stop                 Stop Compose services"

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

# Replays sensor 9's pred split through the API's POST /predict, then renders
# reports/figures/mlflow/deploy_demo.gif. Self-contained: `up -d --wait api` starts
# the api + mlflow services and blocks until the API is healthy (the @production
# bundle is loaded), so no separate `make run` terminal is needed. The stream runs
# inside the `notebooks` image (matplotlib + analysis stack, PYTHONPATH=/app/src set),
# reaches the API over the Compose network as http://api:8000, and writes the GIF to
# the mounted ./reports volume. Services stay up afterward; run `make stop` to tear down.
demo:
	docker compose up -d --build --wait api
	docker compose run --rm --build notebooks \
	  uv run --no-sync python -m pipelines.deploy_demo --sensor 9 --http http://api:8000

# Replay any sensor: make demo-sensor SENSOR=5  (also self-contained; run make stop after).
demo-sensor:
	docker compose up -d --build --wait api
	docker compose run --rm --build notebooks \
	  uv run --no-sync python -m pipelines.deploy_demo --sensor $(SENSOR) --http http://api:8000
