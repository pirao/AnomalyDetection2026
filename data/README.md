# Private Data Placeholder

This directory is intentionally kept without private benchmark data.

To run the full local benchmark workflow, place the private parquet files here:

```text
sensor_data_fit_1.parquet
sensor_data_pred_1.parquet
...
sensor_data_fit_29.parquet
sensor_data_pred_29.parquet
```

Expected columns:

- `sampled_at`: timestamp
- `uptime`: boolean operating-state flag
- `vel_rms_x`, `vel_rms_y`, `vel_rms_z`: velocity RMS channels
- `accel_rms_x`, `accel_rms_y`, `accel_rms_z`: acceleration RMS channels

Private data files are ignored by Git and should not be committed.
