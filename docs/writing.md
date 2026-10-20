# Writing bundles

## Write options

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

## Multi-channel sources

A `ContinuousChannelSource` serves one channel, and `write_bundle` finishes a channel
before it starts the next. A format that stores every channel in one stream, such as EDF,
decodes the same bytes for every channel's window. Reading it through one
`ContinuousChannelSource` per channel decodes the whole stream once per channel. A source
that implements `MultiChannelSource` lets the writer decode the stream once and fill every
channel as it goes. Use `ContinuousChannelSource` for a format that stores each channel in
its own file.

A `MultiChannelSource` has two methods.

`channels()` returns the channels in channel order, the same on every call. Return a
stored list. The writer calls `channels()` more than once per write. Each channel
implements `ContinuousChannelMetadata`, which holds `id`, `name`, `unit`, `rate_hz()`,
`start_us()`, and `num_samples()`. A channel needs no `read_samples`. It may define
`provenance()`.

`iter_windows()` yields one window at a time, and the writer calls it once per write. A
window is a sequence with one rank-1 float32 block per channel, in channel order. The
blocks of one window may differ in length, and a block may be empty. A channel whose
samples end before the others gives an empty block in each later window. For each
channel, the blocks concatenate to its samples, and their lengths total `num_samples()`.
The columns of a `(rows, channels)` array are the blocks of a window, so
`list(data[start:stop].T)` is a window.

Blocks may differ in length, so one source can hold channels of different sample rates.
A one-second EDF record, for example, holds 256 samples of an EEG channel beside 1 sample
of an SpO2 channel.

This example writes four channels of one 256 Hz stream in windows of 4096 rows:

```python
from pathlib import Path

import numpy as np

from timeseries_zarr import WriteOpts, write_bundle

RATE_HZ = 256.0
WINDOW_ROWS = 4096


class Channel:
    unit = "uV"

    def __init__(self, index, num_samples):
        self.id = f"eeg-{index}"
        self.name = f"EEG {index}"
        self._num_samples = num_samples

    def rate_hz(self):
        return RATE_HZ

    def start_us(self):
        return 0

    def num_samples(self):
        return self._num_samples


class Stream:
    def __init__(self, data):
        self._data = data
        self._channels = [
            Channel(index, data.shape[0]) for index in range(data.shape[1])
        ]

    def channels(self):
        return self._channels

    def iter_windows(self):
        for start in range(0, self._data.shape[0], WINDOW_ROWS):
            yield list(self._data[start : start + WINDOW_ROWS].T)


rng = np.random.default_rng(0)
data = rng.normal(0.0, 50.0, size=(100_000, 4)).astype(np.float32)

write_bundle(
    [Stream(data)],
    [],
    output_path=Path("stream.zarr"),
    opts=WriteOpts(),
)
```

Each channel has a key that names its group in the bundle. Keys count up from 0 through
the continuous sources in list order and then the event sources. A multi-channel source
takes one key for each channel, in channel order. A list of one single-channel source and
one three-channel source, with one event source, uses the keys 0 through 3 for the
continuous channels and the key 4 for the event channel.

While it sweeps, the writer holds one shard of `raw` for each channel, plus the current
window. A shard is at most 16 MiB at the default `target_shard_bytes`, so 256 channels
hold up to 4 GiB. Lower `target_shard_bytes` for a wide recording. After the sweep, the
writer folds the levels of each channel from its stored `raw` array, one channel at a
time.

The writer raises `ValueError` when a source breaks the window contract in one of three
ways:

- A window does not hold one block for each channel. The message gives the window index
  and both counts.
- A block is not rank 1. The message names the channel.
- The blocks of a channel do not total its `num_samples()`. The message names the
  channel and gives both counts.

Each error comes before the writer folds any level. A failed write leaves a partial
bundle that a reader rejects.
