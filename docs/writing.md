# Write options

`WriteOpts` fields:

| Field | Description | Default |
|---|---|---|
| `zstd_level` | Zstd compression level | 5 |
| `max_levels` | maximum number of levels | 15 |
| `min_bins` | fewest bins a level may hold | 1024 |
| `inner_len` | inner Zarr chunk length in samples | 8192 |
| `target_shard_bytes` | target outer shard size in bytes | 16 MiB |
| `write_var` | include per-bin variance | `False` |
| `voltage_policy` | `"convert"` or `"require_uv"` | `"convert"` |

`max_levels=0` writes the raw samples and the event arrays with no levels or count
levels.

`write_var=True` uses more memory. The writer holds one shard of statistics per level
while it writes.

`voltage_policy="convert"` scales a voltage channel to microvolts and labels it `uV`.
`"require_uv"` raises `ValueError` for a voltage channel in any other unit.
`timeseries_zarr.UNIT_TO_UV` maps each voltage unit name to its factor to microvolts.
The writer keeps every other unit as given.
