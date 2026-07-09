"""Offline analysis (ANALYZE) - evaluation metrics and EDA/scoring plotting.

Not shipped in the API image. Works entirely from local artifacts: it loads
fitted models from ``cache/`` (never the MLflow registry) and raw data from
``data/``, so it runs without the API or MLflow up.

- ``evaluation`` - benchmark metrics, report tables, and diagnostics (``.venv``-safe)
- ``plotting``   - EDA and scoring widgets (needs the ``notebooks`` extra)

See ``ARCHITECTURE.md`` for the offline-vs-online boundary.
"""
