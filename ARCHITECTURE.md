# Architecture

This page explains **why** the code is organized the way it is, so you know which package new logic
belongs in. For *what* the service does or *how* to run it, see the [README](README.md).

## The three packages

| Package | Role | In the API image? | Loads models from |
|---|---|---|---|
| `anomaly_detection` | **Serve** — FastAPI app, registry loader, detector models | Yes | MLflow registry (`@production`) |
| `pipelines` | **Build** — training/caching, experiment tracking, registry ops, demo | No | `cache/` (to register) + registry (to demo) |
| `offline_analysis` | **Analyze** — evaluation metrics, EDA/scoring plots | No | `cache/` only |


## The offline-vs-online boundary

The most important rule: **where does a model come from?**

- **Offline** loads fitted models from the committed `cache/models/` via
  `pipelines.model_cache.load_fitted_models`. No API, no MLflow server needed. Never imports the
  registry loader.
- **Online / as-served** loads the promoted model from the MLflow registry at `@production`, through
  the single bridge `anomaly_detection.registry.bundle.load_for_serving` (or by calling the running
  API).

Quick test: reading the cache is offline; loading `@production` or calling `/predict` is online.

## How the packages hand off

```text
data/raw/  --fit-->  cache/models/v{N}          pipelines.model_cache.fit_and_save
                          |
                          |- evaluate --> baseline-vs-current runs   pipelines.mlflow_experiments
                          |
                          `- register --> registry version + @production   pipelines.mlflow_registry
                                                |
                                                v
                          anomaly_detection API loads @production once at startup
                                                |
                                                v
                          pipelines.deploy_demo replays a sensor through /predict -> GIF
```

(README's [Model lifecycle and serving](README.md#model-lifecycle-and-serving) narrates the same path
from a user's point of view.)

A model's **identity** is a fingerprint of `data + config + MODEL_CODE_VERSION`. The git sha is
provenance only, not part of the fingerprint, so an unrelated commit never mints a spurious new
version — `register_bundle` reuses an existing version whenever the fingerprint matches and only moves
the alias.

