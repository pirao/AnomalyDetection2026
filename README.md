# Industrial Sensor Anomaly Detection API

[![CI](https://github.com/pirao/AnomalyDetection2026/actions/workflows/ci.yml/badge.svg)](https://github.com/pirao/AnomalyDetection2026/actions/workflows/ci.yml)

A FastAPI service that detects faults in industrial vibration sensors. For each sensor it learns normal behavior from a private `fit` stream, scores a private `pred` stream in 2-hour windows with a 1-hour stride, and raises alarms for real fault windows without reacting to every isolated spike. The model is served from the MLflow `@production` registry alias; `/ready` and `/metadata` make exactly which model is serving an observable fact.

> This public repo ships without private datasets, labels, or fitted artifacts. Reference figures are included as generated reports only.

## Repository Structure

```text
AnomalyDetection2026/
|-- src/
|   |-- sample_processing/              # deployable service package
|   |   |-- api/                        # FastAPI: /fit, /predict, /health, /ready, /metadata, /metrics
|   |   |-- serving/                    # registry bundle + @production loader (unpickled in the API image)
|   |   `-- model/                      # baseline/current detectors and alert engines
|   |-- analysis/                       # offline-only evaluation, MLflow, plotting helpers
|   `-- tests/                          # contract, model, evaluation, performance tests
|-- notebooks/
|   |-- 0.01-acp-exploratory-data-analysis.ipynb
|   `-- 1.01-acp-model-debugging.ipynb
|-- data/
|   `-- raw/                            # private immutable parquet files and labels
|-- reports/
|   `-- figures/                        # generated plots, screenshots, GIFs, report assets
|-- cache/                              # local fitted model artifacts, ignored by Git
|-- Dockerfile
|-- compose.yaml
|-- Makefile
`-- pyproject.toml
```

## Problem And Evaluation

Each scenario has a private `fit` split used to estimate normal behavior and a private `pred` split replayed as the evaluation stream. Incident labels define private fault windows. The API receives `pred` in overlapping batches and returns alarms; the evaluator scores them by event window:

- **True positive:** an alarm overlaps a labelled fault window.
- **False negative:** no alarm overlaps the window.
- **False positive:** an alarm fires in a no-event scenario.

## Results: Baseline vs Current

| Metric | Baseline | Current |
|---|---:|---:|
| Precision | 0.286 | **1.000** |
| Recall | 0.273 | **0.909** |
| F1 | 0.279 | **0.952** |

The baseline emitted 21 alarms with several false positives across no-event scenarios. The current model emits 29 alarms at about 0.79 alert efficiency with zero false positives.

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

![Scenario 2 offline replay](reports/figures/widget_exports/offline_replay/scenario_2.png)

![Alert hierarchy](reports/figures/alert_hierarchy/alert-hierarchy-demo.svg)

## MLflow Tracking And Deployment

- **Track.** Baseline and current are evaluated in the `baseline-vs-current` experiment with metrics, parameters, and dataset fingerprints.
- **Register.** The 29 per-sensor fitted models are packaged into one pyfunc bundle registered as `anomaly-detector-current`.
- **Promote.** `@production` points at one version; promotion or rollback is a registry alias move.
- **Serve.** The API loads `models:/anomaly-detector-current@production` once at startup.

## Serving

Two-service Compose stack. The API loads the promoted bundle once at startup; request scoring never calls MLflow.

```text
          docker compose
  +------------+   load @production    +-----------------------+
  |    api     | --------------------> |  mlflow server :5000  |
  |  FastAPI + |                       |  backend: mlflow.db   |
  |  mlflow    | <-------------------- |  artifacts: mlruns/   |
  |   :8000    |     pyfunc bundle     +-----------------------+
  +------------+
```

| Endpoint | Question | Behavior |
|---|---|---|
| `GET /health` | Is the process alive? | always `200 {"status": "ok"}` |
| `GET /ready` | Can it serve the registered model? | `200` once bundle loads, else `503` |
| `GET /metadata` | Which model, with what provenance? | registry version, fingerprint, git sha, F1 |
| `GET /metrics` | Prometheus scrape target | request counter + latency histogram |

**No silent fallback.** If the registry is unreachable, `/ready` returns `503` and `/metadata` shows `bundle_loaded: false` — a degraded container is held out of rotation rather than silently serving a stale model.

![Promoted bundle replaying a sensor stream](reports/figures/mlflow/deploy_demo.gif)

## How To Run

```bash
make run             # build and start the API on localhost:8000 (also starts mlflow)
make test            # fast suite: unit, contract, performance
make inference-test  # private benchmark gate (~15 min); requires private data
make notebooks       # start JupyterLab on localhost:8888
make stop            # stop all services
make help            # show all targets

# Serving lab
docker compose up --build mlflow api
curl localhost:8000/ready       # 503 until bundle loads, then 200
curl localhost:8000/metadata    # which model is serving
curl localhost:8000/metrics     # Prometheus exposition
```

Restore private files under [data/raw/README.md](data/raw/README.md) and [data/raw/labels/README.md](data/raw/labels/README.md) before running `make test`, `make inference-test`, or the notebooks.

`make test` runs fast with synthetic data and no private files. `make inference-test` replays every scenario and takes ~15 minutes; it is the source of the precision/recall/F1 reported above. A few per-scenario assertions are expected to fail by design (known coverage gaps on scenarios 6, 7, 27, 29); the aggregate F1 gate clearing 0.85 is what matters.

Released under the MIT License.
