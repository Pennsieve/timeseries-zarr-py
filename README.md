# timeseries-zarr-py

Python library and reference implementation for the
[TS-Zarr specification](https://github.com/Pennsieve/timeseries-zarr-py/blob/main/docs/spec.md).

## Installation

Requires Python 3.12 or later. The runtime dependencies are `zarr>=3` and `numpy`.

```bash
pip install timeseries-zarr
```

## Usage

`write_bundle` reads channel sources, computes per-bin statistics up
to the specified maximum number of levels of time resolution, and writes the
bundle to disk.

Each source implements `ContinuousChannelSource` or `EventChannelSource`, which
[`timeseries_zarr.protocols`](https://github.com/Pennsieve/timeseries-zarr-py/blob/main/timeseries_zarr/protocols.py)
defines.

This example writes a single channel holding a minute of a 10 Hz sine sampled at 1000 Hz:

```python
from pathlib import Path

import numpy as np

from timeseries_zarr import WriteOpts, write_bundle


class SineSource:
    id = "sine-0"
    name = "sine"
    unit = "uV"

    def __init__(self):
        t = np.arange(60_000) / 1000.0
        self._samples = (100.0 * np.sin(2 * np.pi * 10 * t)).astype(np.float32)

    def provenance(self):
        return {"software": "sine-example 1.0"}

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

Sources must provide times in microseconds since the Unix epoch. The writer stores
each time as an offset from the reference time, the earlier of the session start you
declare and the first sample. A channel or event before it has a negative time.

Passing a `RecordingMeta` as `recording` adds a `recording.json`. It holds
`reference_time` as a wall-clock time and the subject, session, and source details
you give. `session_start_time` is the acquisition time, written as
`session.start_time`; it places no data. Pass `reference_time` to keep a bundle's own
t = 0 when you write it again. Both must be timezone-aware `datetime` values. Set
`wall_clock=False` when your times do not count from the Unix epoch, and
`recording.json` then holds no `reference_time`. The
[recording metadata](https://github.com/Pennsieve/timeseries-zarr-py/blob/main/docs/spec.md#recording-metadata)
section of the specification lists what the file can hold.

| Field | Description | Default |
|---|---|---|
| `zstd_level` | Zstd compression level | 5 |
| `max_levels` | maximum number of levels; 0 writes raw samples only | 15 |
| `min_bins` | fewest bins a level may hold | 1024 |
| `inner_len` | inner Zarr chunk length in samples | 8192 |
| `target_shard_bytes` | target outer shard size in bytes | 16 MiB |
| `write_var` | include per-bin variance | `False` |
| `voltage_policy` | `"convert"` or `"require_uv"` | `"convert"` |

`max_levels=0` writes only raw samples, without pyramid levels or statistics.

`write_var=True` uses more memory. The writer holds one shard of statistics per level
while it writes.

`voltage_policy` specifies whether the writer should convert voltage channels to
microvolts and label them `uV` or raise `ValueError` for voltage channels not
already in `uV`. `timeseries_zarr.UNIT_TO_UV` maps each unit name treated as a voltage
unit to its factor to microvolts. All other units are stored as given.

## Development

```bash
make venv        # create the virtualenv and install the package with its extras
source venv/bin/activate

make test        # pytest
make typecheck   # mypy --strict
make lint        # ruff check --fix + ruff format (rewrites files)
make check       # ruff check + format check + mypy + pytest
make pre-commit  # install the git pre-commit hook
```

[Releasing](https://github.com/Pennsieve/timeseries-zarr-py/blob/main/docs/releasing.md)
describes how to publish a release.

## Related projects

- [timeseries-zarr-nwb](https://github.com/Pennsieve/timeseries-zarr-nwb) uses this
  library to convert NWB recordings to TS-Zarr bundles.
- [`@pennsieve/timeseries-zarr-reader`](https://github.com/Pennsieve/timeseries-zarr-reader)
  is a TypeScript implementation of a TS-Zarr bundle reader.

## License

Apache-2.0. See [LICENSE](https://github.com/Pennsieve/timeseries-zarr-py/blob/main/LICENSE).
