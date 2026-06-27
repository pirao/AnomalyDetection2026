# Data Directory

This project follows the Cookiecutter Data Science data lifecycle.

```text
data/
|-- raw/          # private immutable benchmark inputs
|-- interim/      # temporary transformed data, ignored by Git
`-- processed/    # final modeling/evaluation data products, ignored by Git
```

The public repository keeps only documentation and lightweight manifests here. Private datasets are supplied locally and mounted into Docker containers at runtime; they are not copied into images and are not committed to Git.

See [raw/README.md](raw/README.md) for the expected private parquet files and [raw/labels/README.md](raw/labels/README.md) for the incident-label file.