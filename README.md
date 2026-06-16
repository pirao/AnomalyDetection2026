# Industrial Sensor Anomaly Detection API

A FastAPI service that detects faults in industrial vibration sensors. For each sensor it learns "normal" from a private `fit` file, then scores a private `pred` stream in 2-hour windows (1-hour stride) and raises alarms for real fault windows — without reacting to every isolated spike. The full lifecycle (track → register → promote → serve) is managed with MLflow.

> This public repo ships without the private datasets, labels, or fitted artifacts. Reference figures are included as visual summaries only.

## Problem And Evaluation

Each scenario has a private `fit` split (used only to estimate normal behavior) and a `pred` split that is replayed as the evaluation stream. Labels are private fault windows. The API receives `pred` in overlapping batches and returns alarms; the evaluator scores them **by event window**:

- **True positive** — an alarm overlaps a labelled fault window.
- **False negative** — no alarm overlaps the window.
- **Partial** — an alarm fires but does not cover enough of the window.
- **False positive** — an alarm fires in a no-event scenario.

Precision, recall, and F1 summarize these. The model must clear the benchmark's (private) precision, recall, and F1 gates. No-event scenarios matter as much as faults: frequent false alarms make an alerting system untrustworthy.

## Results: Baseline vs Current

Measured on the private benchmark. The current model replaces a single global velocity z-score with four group-specific residual-space detectors plus a tiered alert engine.

| Metric | Baseline | Current |
|---|---:|---:|
| Precision | 0.286 | **1.000** |
| Recall | 0.273 | **0.909** |
| F1 | 0.279 | **0.952** |

**Alarm quality:** the baseline emitted 21 alarms — only ~6 useful, with 4 false positives across the 7 no-event scenarios. The current model emits 29 alarms at ~0.79 efficiency with **zero false positives**. Remaining tuning leads: scenarios 6 and 27 (missed), 7 and 29 (partial coverage).

## Model

The detector is intentionally small and inspectable:

1. **Baseline per sensor.** Each scenario's `fit` split defines its healthy mean/std; residuals are measured in those units.
2. **Scoring.** Residuals pass through a group-tuned sigmoid; the strongest samples in each 2-hour batch are aggregated (top-K occupancy) into one fusion score. Group settings live in [norm_model_hyperparams.yaml](src/sample_processing/model/current/hyperparameters/norm_model_hyperparams.yaml).
3. **Alarm selection.** A tiered engine turns the noisy detection stream into a few well-timed alarms using per-channel confirmation, grouped-channel promotion, cooldown, and reset rules.

| Aspect | Baseline | Current |
|---|---|---|
| Detector | One global velocity-norm z-score | Four group-specific residual detectors |
| Features | Velocity RMS collapsed to one norm | Residual-space scoring on all RMS channels |
| Aggregation | Fraction of anomalous samples | Top-K occupancy on the 2h batch |
| Alert state | Single lock | Tiered ownership: confirmation, cooldown, holdback, reset |

![Sigmoid scoring example](notebooks/_images/widget_exports/sigmoid_scoring/scenario_2.png)

The model produces many anomaly markers, but the API does not alert on every one — the alert layer decides when movement is strong and coordinated enough to report.

![Scenario 2 API replay](notebooks/_images/widget_exports/api_replay/scenario_2.png)

![Alert hierarchy](notebooks/assets/alert_hierarchy/alert-hierarchy-demo.svg)

More exported scenarios are in [notebooks/_images/widget_exports](notebooks/_images/widget_exports).

## MLflow Tracking, Registry And Deployment

The model lifecycle runs end to end on MLflow (tracking + registry, SQLite-backed at `mlflow.db`).

- **Track.** Baseline and current are evaluated through the exact FastAPI path and logged as comparable runs in the `baseline-vs-current` experiment (metrics, params, dataset) — so shipping the current model is evidence-backed (see Results).
- **Register.** The 29 per-sensor fitted models are packaged into **one** pyfunc bundle registered as `anomaly-detector-current` — one model, not 29, because the sensors are calibrations of the same detector routed by `sensor_id`. Each version carries the data/config/git **fingerprint** tying it to the exact run that validated it.
- **Promote.** Deployment is an **alias** move (`@production`) — promotion or rollback is one line, no code change.
- **Serve.** The FastAPI service loads the `@production` bundle once at startup (pre-fitted — **no runtime training**); if the registry is unavailable it degrades to runtime-fit (`/fit` + `/predict`).

![MLflow experiment comparison](notebooks/_images/mlflow/experiments.png)
![MLflow model registry and @production alias](notebooks/_images/mlflow/registry.png)

Below, the deployed service replays `sensor_9` through `/predict`; the `@production` model raises a single alert that lands inside the labelled incident window — produced entirely from the registry, with no runtime fit.

![Deployed model serving a live sensor stream](notebooks/_images/mlflow/deploy_demo.gif)

## Repository Structure

```text
src/
  sample_processing/            # deployable service (lean runtime image)
    api/main.py                 # FastAPI: /fit, /predict, /health; loads @production at startup
    model/
      baseline/                 # baseline detector + simple alert engine
      current/                  # current detector
        alerting/               #   tiered alert engine
        hyperparameters/        #   norm_model + alert YAML
        anomaly_model.py, sensor_model.py, normalization.py, preprocessing.py
      shared/                   # pipeline_hyperparams.yaml
      scenario_groups.py        # sensor -> scenario-group routing
  analysis/                     # offline only (notebook stack; not used by the runtime service)
    evaluation/                 # API-replay benchmark evaluation
    mlflow/                     # tracking, registry, model cache, deploy_demo
    plotting/                   # notebook viz widgets + shared style
  tests/
notebooks/                      # 01_eda, 02_model_debugging + exported figures (_images/)
data/  labels/  cache/          # private placeholders (see their READMEs)
Dockerfile  compose.yaml  Makefile  pyproject.toml
```

## How To Run

```bash
make run             # build + start the API on localhost:8000
make inference-test  # private benchmark gate and source of the reported metrics
make stop            # stop the Docker services

mlflow ui --backend-store-uri sqlite:///mlflow.db        # browse experiments + registry
python -m analysis.mlflow.deploy_demo --sensor 9         # regenerate the deployment GIF
```

Restore the private files in [data/README.md](data/README.md) and [labels/README.md](labels/README.md) before running the benchmark. Notebook `02_model_debugging.ipynb` uses the same criteria.

## Reproducibility And License

Hyperparameters are versioned under `src/sample_processing/model/.../hyperparameters/`; fitted models are cached locally under `cache/models/v{N}/` (`.pkl` ignored); MLflow runs and registry live in `mlflow.db`. Reference figures in `notebooks/_images/` are kept in the repo.

Released under the MIT License — applies to the code and documentation here, not to private datasets, labels, or fitted artifacts.
