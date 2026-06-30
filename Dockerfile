# UV_VERSION is pinned instead of using `latest` so Docker cache behavior is
# reproducible. If you intentionally upgrade uv, change the ARG and rebuild.
ARG UV_VERSION=0.11.24
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv_tool

FROM python:3.10-slim AS base

COPY --from=uv_tool /uv /usr/local/bin/uv

WORKDIR /app
ENV PYTHONPATH=/app/src

# Copy only the dependency manifests before source. README.md is required by
# pyproject.toml as package metadata, but only for the project-install step
# (uv sync without --no-install-project). Deferring it to each target means
# editing docs never invalidates the third-party dep layer.
COPY pyproject.toml uv.lock ./

# Named targets:
# - api: local runtime API, no tests, notebooks, analysis package, data, labels,
#   or cached model artifacts copied into the image.
# - test: pytest helper used by `make test` and `make inference-test`.
# - notebooks: analysis environment; notebooks, data, labels, and cache are
#   mounted by compose.yaml at runtime.
#
# Each target installs third-party packages with --no-install-project before
# source is copied. After the narrow source copy, uv installs only this project.

FROM base AS api

# Runtime dependencies only. pyarrow, pytest, httpx, Jupyter, MLflow, and
# analysis-only packages stay out of the API image.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project --link-mode=copy

COPY src/sample_processing ./src/sample_processing
COPY README.md ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --link-mode=copy

# --no-sync keeps container startup deterministic: the image must already contain
# the installed environment instead of modifying .venv when the container starts.
CMD ["uv", "run", "--no-sync", "uvicorn", "sample_processing.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS test

# Test dependencies are kept in the test target so the API image does not carry
# packages used only by pytest or benchmark evaluation.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --group test --no-install-project --link-mode=copy

COPY src/sample_processing ./src/sample_processing
# src/analysis is copied so the benchmark test can import the canonical
# evaluator (analysis.evaluation.summarize_inference_test_metrics), the single
# source of truth for precision/recall/F1 shared with the notebooks.
COPY src/analysis ./src/analysis
COPY src/tests ./src/tests
COPY README.md ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --group test --link-mode=copy

CMD ["uv", "run", "--no-sync", "pytest", "src/tests", "-v"]

FROM base AS notebooks

# Notebook dependencies include the analysis stack. The notebook files and
# generated images are mounted from the host, not baked into this image.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra notebooks --no-install-project --link-mode=copy

COPY src/sample_processing ./src/sample_processing
COPY src/analysis ./src/analysis
COPY README.md ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra notebooks --link-mode=copy

CMD ["uv", "run", "--no-sync", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token=", "--ServerApp.root_dir=/app"]

# Keep `docker build .` deployment-shaped even though compose.yaml uses explicit
# targets. Without this final alias, Docker would build the last target above.
FROM api AS final
