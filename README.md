# Industrial Sensor Anomaly Detection API

A FastAPI service that detects faults in industrial vibration sensors. For each sensor it learns normal behavior from a private `fit` stream, scores a private `pred` stream in 2-hour windows with a 1-hour stride, and raises alarms for real fault windows without reacting to every isolated spike. The project is migrating toward model deployment with MLflow tracking, registry promotion, and Dockerized runtime/test environments.

> This public repo ships without private datasets, labels, or fitted artifacts. Reference figures are included as generated reports only.

## Repository Structure

```text
AnomalyDetection2026/
|-- src/
|   |-- sample_processing/              # deployable service package
|   |   |-- api/                        # FastAPI: /fit, /predict, /health
|   |   `-- model/                      # baseline/current detectors and alert engines
|   |-- analysis/                       # offline-only evaluation, MLflow, plotting helpers
|   `-- tests/                          # contract, model, evaluation, performance tests
|-- notebooks/
|   |-- 0.01-acp-exploratory-data-analysis.ipynb
|   `-- 3.01-acp-model-debugging.ipynb
|-- data/
|   |-- raw/                            # private immutable parquet files and labels
|   |   `-- labels/                     # private incident windows
|   |-- interim/                        # ignored temporary transformed data
|   `-- processed/                      # ignored final derived data products
|-- reports/
|   `-- figures/                        # generated plots, screenshots, GIFs, report assets
|-- cache/                              # local fitted model artifacts, ignored by Git
|-- Dockerfile
|-- compose.yaml
|-- Makefile
`-- pyproject.toml
```

This layout follows Cookiecutter Data Science more closely: raw inputs live under `data/raw`, generated report graphics live under `reports/figures`, and notebooks are named with phase, author initials, and description.

## Problem And Evaluation

Each scenario has a private `fit` split used to estimate normal behavior and a private `pred` split replayed as the evaluation stream. Incident labels define private fault windows. The API receives `pred` in overlapping batches and returns alarms; the evaluator scores them by event window:

- **True positive:** an alarm overlaps a labelled fault window.
- **False negative:** no alarm overlaps the window.
- **False positive:** an alarm fires in a no-event scenario.

Precision, recall, and F1 summarize these. No-event scenarios matter as much as faults because frequent false alarms make an alerting system untrustworthy.

## Results: Baseline vs Current

Measured on the private benchmark. The current model replaces a single global velocity z-score with group-specific residual-space detectors plus a tiered alert engine.

| Metric | Baseline | Current |
|---|---:|---:|
| Precision | 0.286 | **1.000** |
| Recall | 0.273 | **0.909** |
| F1 | 0.279 | **0.952** |

The baseline emitted 21 alarms with several false positives across no-event scenarios. The current model emits 29 alarms at about 0.79 alert efficiency with zero false positives. Remaining tuning leads: scenarios 6 and 27 missed, scenarios 7 and 29 partially covered.

## Model

The detector is intentionally small and inspectable:

1. **Self baseline per sensor.** Each scenario's `fit` split defines its healthy mean/std; residuals are measured in those units.
2. **Scoring.** Residuals pass through group-tuned sigmoid functions; the strongest samples in each 2-hour batch are aggregated into one fusion score.
3. **Alarm selection.** A tiered engine turns the noisy detection stream into a few well-timed alarms using per-channel confirmation, grouped-channel promotion, cooldown, and reset rules.

| Aspect | Baseline | Current |
|---|---|---|
| Detector | One global velocity-norm z-score | Group-specific residual detectors |
| Features | Velocity RMS collapsed to one norm | Residual-space scoring on all RMS channels |
| Aggregation | Fraction of anomalous samples | Top-K occupancy on the 2h batch |
| Alert state | Single lock | Tiered ownership: confirmation, cooldown, holdback, reset |

![Sigmoid scoring example](reports/figures/widget_exports/sigmoid_scoring/scenario_2.png)

The model produces many anomaly markers, but the API does not alert on every one. The alert layer decides when movement is strong and coordinated enough to report.

![Scenario 2 offline replay](reports/figures/widget_exports/offline_replay/scenario_2.png)

![Alert hierarchy](reports/figures/alert_hierarchy/alert-hierarchy-demo.svg)

More exported scenarios are in [reports/figures/widget_exports](reports/figures/widget_exports).

## MLflow Tracking, Registry And Deployment

The model lifecycle runs locally on MLflow tracking and registry storage backed by `mlflow.db`.

- **Track.** Baseline and current are evaluated through comparable runs in the `baseline-vs-current` experiment with metrics, parameters, and dataset fingerprints.
- **Register.** The 29 per-sensor fitted models are packaged into one pyfunc bundle registered as `anomaly-detector-current` because the sensors are calibrations of the same detector routed by `sensor_id`.
- **Promote.** Deployment is an alias move, so promotion or rollback is a registry update rather than a code change.
- **Serve.** The FastAPI service can load the promoted bundle once at startup; if the registry is unavailable it can still fall back to runtime fit/predict behavior.

![MLflow experiment comparison](reports/figures/mlflow/experiments.png)
![MLflow model registry and production alias](reports/figures/mlflow/registry.png)

Below, the deployed service replays `sensor_9` through `/predict`; the promoted model raises a single alert inside the labelled incident window.

![Deployed model serving a live sensor stream](reports/figures/mlflow/deploy_demo.gif)

## How To Run

```bash
make help            # show Docker shortcuts and their roles
make run             # build and start only the API on localhost:8000
make test            # run the Dockerized pytest suite
make inference-test  # run the private benchmark gate; can take about 15 minutes
make notebooks       # start JupyterLab on localhost:8888
make stop            # stop Docker Compose services

mlflow ui --backend-store-uri sqlite:///mlflow.db
uv run --extra notebooks python -m analysis.mlflow.deploy_demo --sensor 9
```

Restore private files under [data/raw/README.md](data/raw/README.md) and [data/raw/labels/README.md](data/raw/labels/README.md) before running `make test`, `make inference-test`, or the notebooks. During the migration, the code still falls back to the previous local ignored layout if the canonical `data/raw` files are not present.

## Docker Image Layout

| Service | Dockerfile target | What is copied into the image | What is mounted at runtime |
|---|---|---|---|
| `api` | `api` | `src/sample_processing` only | Nothing private; callers send data over HTTP |
| `test` | `test` | `src/sample_processing`, `src/tests` | `./data:/app/data:ro` |
| `inference-test` | `test` | Same image as `test` | Same read-only private-data mount |
| `notebooks` | `notebooks` | `src/sample_processing`, `src/analysis` | `./notebooks`, `./reports`, `./data:ro`, `./cache` |

Important Docker rules for this project:

- The API image does not contain private data, notebooks, tests, analysis code, generated figures, or cached model artifacts.
- Private data is mounted read-only into test and notebook containers.
- `reports/` is mounted into the notebook container so exported figures are generated outside `notebooks/`.
- `/app/.venv` is created inside the Linux image by `uv sync`; the host `.venv` is ignored.
- `uv run --no-sync` is used in container commands because image builds already created the environment.

## Reproducibility And License

Hyperparameters are versioned under `src/sample_processing/model/.../hyperparameters/`; fitted models are cached locally under `cache/models/`; MLflow tracking uses `mlflow.db`; generated visual summaries live under `reports/figures/`.

Released under the MIT License. The license applies to the code and documentation here, not to private datasets, labels, or fitted artifacts.