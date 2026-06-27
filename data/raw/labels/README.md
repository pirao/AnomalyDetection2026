# Raw Private Labels

Place the private benchmark event-window file here:

```text
incidents.yaml
```

Expected shape:

```yaml
1:
  - start: "YYYY-MM-DDTHH:MM:SS+00:00"
    end: "YYYY-MM-DDTHH:MM:SS+00:00"
2: []
```

Labels describe the raw benchmark stream, so they live under `data/raw/labels/` with the private raw parquet files. The YAML file is ignored by Git and should not be committed.