FROM python:3.10-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY . .
RUN uv sync --no-dev

CMD ["uv", "run", "uvicorn", "sample_processing.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
