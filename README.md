# timeseries-zarr-py

Python library and reference implementation for the
[TS-Zarr specification](./docs/bundle-format.md).

## Installation

Requires Python 3.12 or later. The runtime dependencies are `zarr>=3` and `numpy`.

```bash
pip install git+https://github.com/Pennsieve/timeseries-zarr-py.git
```

## Usage

`write_bundle` reads channel sources, computes per-bin statistics up
to the specified maximum number of levels of time resolution, and writes the
bundle to disk.

Each source implements `ContinuousChannelSource` or `EventChannelSource` from
[`timeseries_zarr.protocols`](./timeseries_zarr/protocols.py).

This example writes a single channel holding a minute of a 10 Hz sine sampled at 1000 Hz:

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
    output_path=Path("sine.zarr"),
    opts=WriteOpts(),
)
```

Passing a `RecordingMeta` as `recording` adds a `recording.json` with the session
start time and subject, session, and source details. Its `start` must be a
timezone-aware `datetime`. The [recording metadata](./docs/bundle-format.md#recording-metadata)
section of the specification lists what the file can hold.

Sources must provide times in microseconds since the Unix epoch.
The writer stores them as offsets from the earliest start time in the recording.

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

`write_var=True` uses more memory. The writer holds one shard of statistics per level
while it writes.

`voltage_policy` specifies whether the writer should convert voltage channels to
microvolts and label them `uV` or raise `ValueError` for voltage channels not
already in `uV`. `UNIT_TO_UV` in [`constants.py`](./timeseries_zarr/constants.py)
lists which units are considered voltage units. All other units are stored as given.

## Development

```bash
make venv        # create the virtualenv and install the package with its extras
source venv/bin/activate

make test        # pytest
make typecheck   # mypy --strict
make lint        # ruff check --fix + ruff format (rewrites files)
make check       # the gate: ruff check + format check + mypy + pytest
make pre-commit  # install the git pre-commit hook
```

## Related projects

- [timeseries-zarr-nwb](https://github.com/Pennsieve/timeseries-zarr-nwb) uses this
  library to convert NWB recordings to TS-Zarr bundles.
- [`@pennsieve/timeseries-zarr-reader`](https://github.com/Pennsieve/timeseries-zarr-reader)
  is the TypeScript reference reader.

## License

Apache-2.0. See [LICENSE](LICENSE).
