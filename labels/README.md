# Private Labels Placeholder

This directory is intentionally kept without private benchmark labels.

To run the full local benchmark workflow, place the private event-window file here:

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

Private labels are ignored by Git and should not be committed.
