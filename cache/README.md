# Local Model Cache

This directory holds the per-sensor fitted-model artifacts used for offline replay
and as the source from which the registered MLflow bundle is packaged.

Committed layout:

```text
models/
  meta.json
  v1/
    meta.json
    1.pkl
    ...
```

These `.pkl` files are committed so the offline notebooks, the deploy demo, and
`make run` work without the private dataset. Re-running the private benchmark
workflow regenerates this cache and overwrites the artifacts in place.
