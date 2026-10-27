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
| `max_workers` | threads that fold and encode the channels of a window | available CPUs |

`max_levels=0` writes the raw samples and the event arrays with no levels or count
levels.

`max_workers` is 1 or more. `None` uses the CPUs that the process may run on: the CPU
affinity where the platform reports it, and `os.cpu_count()` elsewhere. The bytes written
do not depend on it.

`write_var=True` adds the `var` member to every level and costs no extra memory. The
writer folds every level from float64 statistics whatever the setting, so the variance
stays exact on a channel with a large DC offset.

The writer holds about 5.8 MiB per channel at the defaults, which write 15 levels,
whatever the length of the recording. That is one partial chunk of `raw` and, for each
level, up to `inner_len` rows of float64 statistics, plus the index of each open shard
and up to 1 MiB of encoded chunks that wait to be written. `max_levels=8` lowers the
figure to about 3.6 MiB.
`timeseries_zarr.channel_writer_bytes(opts)` returns the estimate in bytes for the
`max_levels` of `opts`. A `MultiChannelSource` also holds two windows, the one that the
writer folds and the one that it reads.

`voltage_policy="convert"` scales a voltage channel to microvolts and labels it `uV`.
`"require_uv"` raises `ValueError` for a voltage channel in any other unit.
`timeseries_zarr.UNIT_TO_UV` maps each voltage unit name to its factor to microvolts.
The writer keeps every other unit as given.

## Multi-channel sources

A `ContinuousChannelSource` serves one channel, and `write_bundle` finishes a channel
before it starts the next. Some formats, such as EDF, store every channel in one
stream, so reading one channel's window decodes the bytes of every channel. Reading
such a format through one `ContinuousChannelSource` per channel decodes the whole
stream once per channel. A source that implements `MultiChannelSource` lets the writer
decode the stream once and fill every channel as it goes. Use `ContinuousChannelSource`
for a format that stores each channel in its own file.

A `MultiChannelSource` has two methods.

`channels()` returns the channels in channel order. Each channel implements
`ContinuousChannelMetadata`, which holds `id`, `name`, `unit`, `rate_hz()`,
`start_us()`, and `num_samples()`. A channel needs no `read_samples`. It may define
`provenance()`.

`iter_windows()` yields one window at a time, and the writer calls it once per write. The
writer reads the next window while it writes the previous one, so a source must not
overwrite the arrays of a window it has yielded.

A window is a `(rows, channels)` array, or a sequence with one rank-1 block per channel in
channel order. The columns of the array are the blocks, so `data[start:stop]` is a window.
The writer copies an array into channel-major order once, so each channel reads contiguous
rows. It casts each block to float32, so an int16 array works. The sequence form takes
strided column views, such as `list(data[start:stop].T)`. In the sequence form, the
blocks of one window may differ in length, and a block may be empty. A channel whose
samples end before the others gives an empty block in each later window. For each
channel, the blocks concatenate to its samples, and their lengths total `num_samples()`.

Blocks may differ in length, so one source can hold channels of different sample rates.
A one-second EDF record, for example, holds 256 samples of an EEG channel beside 1 sample
of an SpO2 channel.

Before it clears or writes anything, `write_bundle` checks that every entry of `continuous`
implements `ContinuousChannelSource` or `MultiChannelSource`, that every channel of a
`MultiChannelSource` implements `ContinuousChannelMetadata`, and that every entry of
`events` implements `EventChannelSource`. A source with a member missing raises
`ValueError` with the list position and the missing member names.

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
            yield self._data[start : start + WINDOW_ROWS]


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

While it sweeps, the writer holds `channel_writer_bytes(opts)` for each channel, plus two
windows. At the defaults, 256 channels hold about 1.4 GiB of channel state.
Each shard file is written once, as its chunks complete, and nothing is read back.

The writer raises `ValueError` when a source breaks the window contract in one of three
ways:

- A window does not hold one block for each channel, or one column for each channel in the
  array form. The message gives the window index and both counts.
- A block is not rank 1. The message names the channel.
- The blocks of a channel do not total its `num_samples()`. The message names the
  channel and gives both counts.

Each error comes before the writer folds any level. A failed write leaves a partial
bundle that a reader rejects.
