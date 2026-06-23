.PHONY: help run stop test inference-test notebooks

help:
	@echo "make run             Build and start the local API on http://localhost:8000"
	@echo "make test            Run the Dockerized pytest suite with data/labels mounted read-only"
	@echo "make inference-test  Run the private benchmark gate; this can take about 15 minutes"
	@echo "make notebooks       Start JupyterLab on http://localhost:8888 with notebooks/cache mounted"
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
