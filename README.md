# Industrial Sensor Anomaly Detection API

Anomaly detection and alerting pipeline for vibration-style industrial sensor streams. The system learns a per-sensor healthy baseline from a `fit` window, scores a `pred` window in 2-hour batches with 1-hour stride, and emits alerts through a three-tier priority engine.

This public repository does not include private datasets, private labels, external source materials, local assistant configuration, or fitted model artifacts. Generated reference figures are included as visual summaries of the private benchmark and do not include raw source data.

## Private Benchmark Results

The metrics below were produced on a private benchmark dataset that is not distributed with this repository.

| Metric | Value | Threshold | Status |
|---|---:|---:|---|
| Precision | **1.000** | 0.50 | Pass |
| Recall | **0.909** | 0.30 | Pass |
| F1 | **0.952** | 0.35 | Pass |

Zero false positives were observed across 7 no-event sensor scenarios. Four scenarios remained as tuning leads: 6 and 27 as missed detections, and 7 and 29 as partial event-window coverage.

## Repository Structure

```text
.
|-- src/
|   |-- sample_processing/
|   |   |-- api/                    # FastAPI endpoints and request contracts
|   |   |-- hyperparameters/        # Versioned YAML configuration
|   |   `-- model/
|   |       |-- anomaly_model.py     # Runtime model orchestration and param loading
|   |       |-- sensor_model.py      # Per-batch scoring pipeline
|   |       |-- baselines.py         # Runtime baseline fit/score helpers
|   |       |-- preprocessing.py     # Runtime spike clipping helper
|   |       |-- scenario_groups.py   # Shared sensor-scenario group mapping
|   |       `-- alerting/            # Alert engine internals
|   `-- analysis/
|       |-- api_replay/              # Offline replay and benchmark-style evaluation
|       |-- plotting/                # Notebook-facing plotting and widgets
|       `-- model_cache.py           # Versioned fitted-model cache helpers
|-- notebooks/
|   |-- 01_eda.ipynb
|   |-- 02_model_debugging.ipynb
|   |-- assets/                     # Maintained notebook source assets
|   `-- _generated/                 # Reference image exports
|-- data/                           # Private data placeholder
|-- labels/                         # Private label placeholder
|-- cache/                          # Local fitted-model cache placeholder
|-- compose.yaml
|-- Dockerfile
|-- Makefile
`-- README.md
```

The runtime path lives in `src/sample_processing`. The `src/analysis` package contains notebook and offline evaluation tooling layered on top of that stable core.

## Private Data Layout

The public repo ships without private benchmark files. To run the full private workflow locally, restore the files documented in [data/README.md](data/README.md) and [labels/README.md](labels/README.md).

Expected private inputs:

- `data/vibe_data_fit_{1..29}.parquet`
- `data/vibe_data_pred_{1..29}.parquet`
- `labels/incidents.yaml`

If these files are missing, the private benchmark tests cannot run. Restore the private files locally before running the benchmark command.

## How To Run

```bash
make run             # build and start the API on localhost:8000
make stop            # stop the Docker services
make inference-test  # main private benchmark gate and source of reported metrics
```

The primary evaluation command is:

```bash
make inference-test
```

Notebook `02_model_debugging.ipynb` uses the same private benchmark criteria and metrics. The other `make` commands are operational Docker helpers.

## Methodology

The pipeline is split into five stages:

1. **Data loading and labelling.** Per-scenario fit / pred parquet files are concatenated, `uptime` is used as the operational gate, and `pred` rows are labelled against private event windows.
2. **Preprocessing.** The retained signal-level transform is `clip_rms_spikes(vel=100, accel=10)`, which clips sparse gross outliers without redefining the signal shape.
3. **Per-sensor baseline.** Each scenario's `fit` split defines its own healthy baseline. Residuals are measured in fit-healthy standard deviations.
4. **Group-specific sigmoid scoring.** Sensor scenarios are partitioned into four visual archetype groups. Each group has its own `(alpha, beta, threshold, window_top_k, fusion_threshold)` in [norm_model_hyperparams.yaml](src/sample_processing/hyperparameters/norm_model_hyperparams.yaml).
5. **Three-tier alert engine.** Priority flows from individual channel to group-3 to group-6 ownership. Confirmation gates, holdback windows, exclusive-individual mode, and a pending-priority queue suppress lower-priority events while higher-priority ownership is forming.

## Model And Alerting Improvements

| Aspect | Baseline | Current |
|---|---|---|
| Detector config | Single global velocity-norm z-score model | Four group-specific configurations |
| Features | Velocity RMS collapsed to one norm | Residual-space scoring on raw RMS channels |
| Scoring rule | Absolute z-score against one healthy mean/std | Group-tuned sigmoid on residuals |
| Window aggregation | Fraction of anomalous samples in the batch | Top-K occupancy on the outer 2h batch |
| Alert state | Single lock | Tiered ownership with confirmation, cooldown, holdback, and reset logic |

The current hierarchy is designed to suppress transient single-axis ticks while preserving detection of sustained multi-channel degradation.

## Reproducibility

- Hyperparameters are versioned in `src/sample_processing/hyperparameters/`.
- Fitted models are written locally under `cache/models/v{N}/`; these `.pkl` artifacts are ignored.
- `METRICS.md` is ignored for local/private report experiments.
- Reference visualizations in `notebooks/_generated/` are intentionally kept in the public repo.

## License

This project is released under the MIT License. The license applies to the code and documentation in this repository, not to private datasets, labels, external source materials, or fitted model artifacts.
