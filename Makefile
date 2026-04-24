run:
	docker compose up --build

stop:
	docker compose down

inference-test:
	docker compose run --rm --build inference-test

test:
	docker compose run --rm --build test
