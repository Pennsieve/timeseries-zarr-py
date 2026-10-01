# timeseries-zarr-py

Converts a neurophysiology recording into a pyramid Zarr v3 viewer bundle. The bundle is a
static directory that a browser reads over HTTP range requests to render any time window
interactively.

Raw recordings can be terabytes large, but only a tiny fraction of that data can
be represented in a ~2000-pixel-wide view. This writer precomputes multi-resolution
min/max envelopes at ingest so readers only fetch the data needed to render the recording
over the visible time range.

This repository is reference implementation of a bundle writer. The format is specified in
[bundle-format.md](https://github.com/Pennsieve/timeseries-zarr-paper/blob/main/bundle-format.md), in the paper repository, which is its only
source of truth. The reference reader is
[`@pennsieve/timeseries-zarr-reader`](https://github.com/Pennsieve/timeseries-zarr-reader).

The writer does not implement the current spec yet; that work is in progress.

## Usage

NWB is currently the only supported input format.
[docs/architecture.md](./docs/architecture.md) explains how a additional formats could plug in.

Read one NWB file and write one bundle:

```bash
python -m timeseries_zarr.main recording.nwb /data/recording.zarr
```

With no arguments the writer takes the directory convention instead: it reads the single
`.nwb` file in `INPUT_DIR` and publishes to `OUTPUT_DIR/<input-stem>.zarr`. This is how
the container runs.

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
| `ZARR_WRITER_MAX_LEVELS` | most level groups a channel can hold, above raw | 7 |
| `ZARR_WRITER_MIN_BINS` | bin threshold for keeping a coarser pyramid level | 1024 |
| `ZARR_WRITER_INNER_LEN` | inner Zarr chunk length in samples | 8192 |
| `ZARR_WRITER_TARGET_SHARD_BYTES` | target outer shard size in bytes | 16 MiB |

The last four override the pyramid and chunk parameters that the
[format spec](https://github.com/Pennsieve/timeseries-zarr-paper/blob/main/bundle-format.md) specifies.

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
