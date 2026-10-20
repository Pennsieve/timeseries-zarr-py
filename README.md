# timeseries-zarr-py

Python library that reads and writes Zarr bundles of neurophysiology time series.

It servers as a reference writer and reader implementation for the
[TS-Zarr specification](https://github.com/Pennsieve/timeseries-zarr-py/blob/main/docs/spec.md).

## Installation

Requires Python 3.12 or later. The runtime dependencies are `zarr>=3.2.1` and `numpy`.

```bash
pip install timeseries-zarr
```

## Usage

### Writing

`write_bundle` reads channel sources, computes per-bin statistics up to the specified
maximum number of levels of time resolution, and writes the bundle to disk.

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
`wall_clock=False` when your times do not count from the Unix epoch, and leave
`reference_time` unset; `recording.json` then holds no `reference_time`. The
[recording metadata](https://github.com/Pennsieve/timeseries-zarr-py/blob/main/docs/spec.md#recording-metadata)
section of the specification lists what the file can hold.

[Write options](https://github.com/Pennsieve/timeseries-zarr-py/blob/main/docs/writing.md)
lists the `WriteOpts` fields and their defaults.

### Reading

```python
from timeseries_zarr import open_bundle

with open_bundle("sine.zarr") as bundle:
    channel = bundle.continuous[0]
    print(channel.name, channel.unit, channel.rate_hz, channel.num_samples)
    samples = channel.read(0, 2_000_000)
    print(samples.time_s[:3], samples.values[:3])
    level = channel.level_for(0, 60_000_000, bins=2000)
    bins = channel.read(0, 60_000_000, level=level)
    print(level, len(bins), bins.min[:3], bins.max[:3])
```

The `start_us` and `end_us` arguments of `read` and `level_for` are integer
microseconds from the reference time.

`to_dataframe` and `to_series` requires the `pandas` extra, `to_xarray` requires the `xarray`
extra, and a remote URL requires the `remote` extra.

[Reading bundles](https://github.com/Pennsieve/timeseries-zarr-py/blob/main/docs/reading.md)
covers windows, levels, event channels, stacking, remote stores, and rewriting a bundle.

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
