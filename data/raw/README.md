# Raw Private Data

Place the immutable private benchmark parquet files here:

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

These files are ignored by Git. Treat this folder as immutable input: downstream cleaning, feature creation, or exports should write to `reports/figures/` or `cache/`, not back into `data/raw/`.