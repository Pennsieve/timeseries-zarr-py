# timeseries-zarr-py

Python library and reference writer for TS-Zarr bundles. A bundle is a static
directory of Zarr v3 arrays. A browser reads it over HTTP range requests to render any
time window of a neurophysiology recording. The writer takes channel sources,
precomputes per-bin statistics at several levels, and publishes the bundle in one
atomic rename.

The format is specified in [docs/bundle-format.md](./docs/bundle-format.md). The
reference reader is
[`@pennsieve/timeseries-zarr-reader`](https://github.com/Pennsieve/timeseries-zarr-reader).
The NWB converter built on this library is
[timeseries-zarr-nwb](https://github.com/Pennsieve/timeseries-zarr-nwb).

## Installation

```bash
pip install git+https://github.com/Pennsieve/timeseries-zarr-py.git
```

## Usage

A caller implements `ContinuousChannelSource` or `EventChannelSource` from
`timeseries_zarr.protocols` and passes its sources to `write_bundle`. This example
writes one channel holding a minute of a 10 Hz sine sampled at 1000 Hz:

```python
from pathlib import Path

import numpy as np

from timeseries_zarr.bundle import write_bundle
from timeseries_zarr.types import WriteOpts


class SineSource:
    id = "sine-0"
    name = "sine"
    unit = "uV"

    def __init__(self):
        t = np.arange(60_000) / 1000.0
        self._samples = (100.0 * np.sin(2 * np.pi * 10 * t)).astype(np.float32)

    def rate_hz(self):
        return 1000.0

    def start_us(self):
        return 0

    def num_samples(self):
        return self._samples.shape[0]

    def read_samples(self, start, stop):
        return self._samples[start:stop]


write_bundle(
    [SineSource()],
    [],
    staging_dir=Path("sine.zarr.staging"),
    final_dir=Path("sine.zarr"),
    opts=WriteOpts(),
)
```

Every time in a bundle is int64 microseconds. A source reports `start_us` as wall-clock
microseconds since the Unix epoch. The bundle counts from the earliest start of any
source or of the session. The writer stores samples as the source gives them and
declares the source's `unit`. Voltage sources report microvolts and `"uV"`.

`staging_dir` and `final_dir` must sit on one filesystem. The staged bundle replaces
an existing bundle at `final_dir`. A `RecordingMeta` passed as `recording` must carry a
UTC offset in its start. Its subject, session, and source objects go into
`recording.json` at the bundle root and nowhere else. Deleting that file removes
structured identity and the date. Free text in event channels can still hold identity.

`WriteOpts` fields:

| Field | Governs | Default |
|---|---|---|
| `zstd_level` | Zstd compression level | 5 |
| `max_levels` | most levels above the raw samples | 15 |
| `min_bins` | bin threshold for keeping a coarser level | 1024 |
| `inner_len` | inner Zarr chunk length in samples | 8192 |
| `target_shard_bytes` | target outer shard size in bytes | 16 MiB |
| `write_var` | write the per-bin variance member | False |

`write_var=True` adds a variance array to every level of every continuous channel. The
writer then folds every level from the raw samples in one pass. That pass holds one
shard of statistics per level in memory.

## Development

Python 3.12, fully typed under `mypy --strict`, with a strict `ruff` ruleset. Tests in
`tests/` mirror `timeseries_zarr/` one to one.

```bash
make venv        # create the virtualenv and install the package with its extras
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

Runtime: `zarr>=3`, `numpy`. Development: `pytest`, `pytest-cov`, `pytest-mock`,
`mypy`, `ruff`, `pre-commit`.

## License

Apache-2.0. See [LICENSE](LICENSE).
