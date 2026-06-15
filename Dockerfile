FROM python:3.10-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY . .

# Default build is the lean API/test image. Set INSTALL_NOTEBOOKS=true (the
# `notebooks` compose service does this) to also install the analysis/notebook
# stack from the `notebooks` optional-dependency group.
ARG INSTALL_NOTEBOOKS=false
RUN if [ "$INSTALL_NOTEBOOKS" = "true" ]; then \
        uv sync --no-dev --extra notebooks; \
    else \
        uv sync --no-dev; \
    fi

CMD ["uv", "run", "uvicorn", "sample_processing.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
