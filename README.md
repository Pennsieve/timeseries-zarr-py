# timeseries-zarr-py

Converts a neurophysiology recording into a TS-Zarr 1.0 bundle. The bundle is a static
directory of Zarr v3 arrays that a browser reads over HTTP range requests to render any
time window interactively.

Raw recordings can be terabytes large, but a view is about 2000 pixels wide. The writer
precomputes per-bin statistics at several resolutions. A reader then fetches only the bins
the visible window needs. A continuous channel holds its raw samples and levels of
per-bin minimum, maximum, mean, and count. Spikes, intervals, and annotations become event
channels.

This repository is a reference implementation of a bundle writer. The format is specified
in [docs/bundle-format.md](./docs/bundle-format.md). The reference reader is
[`@pennsieve/timeseries-zarr-reader`](https://github.com/Pennsieve/timeseries-zarr-reader).

## Usage

NWB is currently the only supported input format.
[docs/architecture.md](./docs/architecture.md) explains how additional formats plug in.

Read one NWB file and write one bundle:

```bash
python -m timeseries_zarr.main recording.nwb /data/recording.zarr
```

With no arguments the writer takes the directory convention instead: it reads the single
`.nwb` file in `INPUT_DIR` and publishes to `OUTPUT_DIR/<input-stem>.zarr`. This is how
the container runs.

The writer reads every `ElectricalSeries` and every other numeric `TimeSeries` in the
acquisition as continuous channels, one per column. It reads `Units` tables, interval
tables (epochs, trials, invalid times, and any other `TimeIntervals`), `EventsTable`
tables, and `AnnotationSeries` as event channels. Voltage samples are stored in
microvolts. A series in any other unit keeps its unit.

The subject, the session, and the wall-clock start time go into `recording.json` at the
bundle root and into no other metadata. Delete that file to de-identify a bundle. Free text
in event channels can still hold identity.

Every run also writes a sibling properties file next to the bundle, `asset-properties.json`
by default, or whatever `ASSET_PROPERTIES_FILE` names instead. It records the bundle's
directory name under the key `root_path`.

```bash
make run        # docker-compose build + up, against data/input and data/output
```

Writer settings come from the environment under the `ZARR_WRITER_` prefix. Every
setting is optional.

| Variable | Governs | Default |
|---|---|---|
| `ZARR_WRITER_STAGING_DIR` | scratch path for the atomic publish | alongside the output |
| `ZARR_WRITER_ZSTD_LEVEL` | Zstd compression level | 5 |
| `ZARR_WRITER_MAX_LEVELS` | most levels above the raw samples | 15 |
| `ZARR_WRITER_MIN_BINS` | bin threshold for keeping a coarser level | 1024 |
| `ZARR_WRITER_INNER_LEN` | inner Zarr chunk length in samples | 8192 |
| `ZARR_WRITER_TARGET_SHARD_BYTES` | target outer shard size in bytes | 16 MiB |
| `ZARR_WRITER_VAR` | write the per-bin variance member, 0 or 1 | 0 |

`ZARR_WRITER_VAR=1` adds a variance array to every level of every continuous channel. The
writer then folds every level from the raw samples in one pass. That pass holds one shard
of statistics per level in memory.

## Development

Python 3.12, fully typed under `mypy --strict`, with a strict `ruff` ruleset. Tests in
`tests/` mirror `timeseries_zarr/` one to one.

```bash
make venv        # create the virtualenv and install deps
source venv/bin/activate

make test        # pytest
make typecheck   # mypy --strict
make lint        # ruff check --fix + ruff format (rewrites files)
make check       # the gate: ruff check + format check + mypy + pytest
make pre-commit  # install the git pre-commit hook
```

`make check` must stay green.

Ruff enforces absolute imports through its `TID` rules. It does not enforce that `zarr`
is imported only in `zarr_io.py`. That boundary holds by convention alone.

The module layout and data flow are in [docs/architecture.md](./docs/architecture.md).

## Dependencies

Runtime: `zarr>=3`, `numcodecs`, `numpy`, `pynwb`, `h5py`. Development: `pytest`,
`pytest-cov`, `pytest-mock`, `mypy`, `ruff`, `pre-commit`.

## License

Apache-2.0. See [LICENSE](LICENSE).
