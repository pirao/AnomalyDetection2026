"""Deployable anomaly-detection service - the ONLINE serving layer.

The only package shipped in the API image. Contains the FastAPI app (``api``),
the MLflow-registry bundle + loader (``registry``), and the detector models
(``model``). At startup the service loads the ``@production`` bundle once via
``registry.bundle.load_for_serving`` and scores every request from memory, so
request handling never calls MLflow.

See ``ARCHITECTURE.md`` for how this relates to ``pipelines`` (build) and
``offline_analysis`` (analyze).
"""
