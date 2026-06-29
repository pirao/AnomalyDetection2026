# Data Directory

This directory holds only the data stages the project actually uses. Derived products are written to `reports/figures/` or `cache/`, so intermediate staging directories are intentionally omitted.

```text
data/
`-- raw/          # private immutable benchmark inputs
```

The public repository keeps only documentation and a lightweight manifest here. Private datasets are supplied locally and mounted into Docker containers at runtime; they are not copied into images and are not committed to Git.

See [raw/README.md](raw/README.md) for the expected private parquet files and [raw/labels/README.md](raw/labels/README.md) for the incident-label file.