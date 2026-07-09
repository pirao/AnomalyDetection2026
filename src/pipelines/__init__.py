"""Model lifecycle (BUILD) - training, tracking, registry ops, and the deploy demo.

Not shipped in the API image. Turns data into a promoted model:

- ``model_cache``        - fit + version per-sensor models into ``cache/``
- ``mlflow_experiments`` - log the baseline-vs-current comparison
- ``mlflow_registry``    - register the bundle and promote it to ``@production``
- ``deploy_demo``        - replay a sensor through the live API and render the GIF

Depends on ``anomaly_detection`` (the served detector + registry bridge) and on
``offline_analysis.evaluation`` (metrics). Submodules are imported directly
(e.g. ``from pipelines.model_cache import fit_and_save``). See ``ARCHITECTURE.md``.
"""
