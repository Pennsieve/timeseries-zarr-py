# Reading bundles

`open_bundle` opens a TS-Zarr bundle for reading. Each channel exposes its zarr arrays
for direct slicing, and its `read` methods return results that hold numpy arrays and
compute the time axis. The [specification](./spec.md) defines the format.

## Example bundle

The examples on this page read `example.zarr`, which the code below writes to the
working directory. It holds five minutes of five channels. Channels 0 and 1 are 5 Hz
and 10 Hz tones sampled at 1000 Hz on one grid. Channel 2 is a 1 Hz tone sampled at
256 Hz. It starts 0.5 s after the reference time and pauses from 120.5 s. It resumes
1 ms before 125 s, off its sample grid, and holds two runs.
Channel 3 holds six marks with durations, labels, text, and the channels each mark
applies to. Channel 4 holds the spike times of three units, 70,000 events in all, and
the mean waveform of each unit. `Tone` and `Events` implement the source protocols
that `write_bundle` reads.

```python
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from timeseries_zarr import (
    RecordingMeta,
    Run,
    WriteOpts,
    text_width,
    write_bundle,
)

SESSION_START = datetime(
    2024, 3, 14, 9, 15, tzinfo=timezone(timedelta(hours=-5))
)
SESSION_START_US = (
    SESSION_START - datetime(1970, 1, 1, tzinfo=UTC)
) // timedelta(microseconds=1)
SECONDS = 300


def unix_us(seconds):
    return SESSION_START_US + np.round(np.asarray(seconds) * 1e6).astype(
        np.int64
    )


class Tone:
    unit = "uV"

    def __init__(self, hertz, rate_hz=1000.0, delay_s=0.0, pause_s=None):
        self.id = f"tone-{hertz}"
        self.name = f"{hertz} Hz"
        self._rate_hz = rate_hz
        self._start_us = int(unix_us(delay_s))
        seconds = np.arange(round(SECONDS * rate_hz)) / rate_hz
        samples = 100.0 * np.sin(2 * np.pi * hertz * seconds)
        self.samples = samples.astype(np.float32)
        self._runs = [Run(self._start_us, self.samples.shape[0])]
        if pause_s is not None:
            stop, resume = (round(edge * rate_hz) for edge in pause_s)
            resume_us = int(unix_us(delay_s + pause_s[1]))
            self._runs = [
                Run(self._start_us, stop),
                Run(resume_us, self.samples.shape[0] - resume),
            ]
            self.samples = np.concatenate(
                [self.samples[:stop], self.samples[resume:]]
            )

    def runs(self):
        return self._runs

    def rate_hz(self):
        return self._rate_hz

    def start_us(self):
        return self._start_us

    def num_samples(self):
        return self.samples.shape[0]

    def read_samples(self, start, stop):
        return self.samples[start:stop]


class Events:
    unit = None

    def __init__(self, id, name, seconds, labels, label_names, **columns):
        self.id = id
        self.name = name
        self.times = unix_us(seconds)
        self.labels = np.asarray(labels, dtype=np.uint16)
        self.names = label_names
        self.durations = columns.get("durations")
        self.text = columns.get("text")
        self.channel_ids = columns.get("channel_ids")
        self.mean = columns.get("waveform_mean")

    def num_events(self):
        return self.times.shape[0]

    def columns(self):
        present = {
            "labels": self.labels,
            "durations": self.durations,
            "text": self.text,
            "channels": self.channel_ids,
            "waveform_mean": self.mean,
        }
        return frozenset(
            name for name, value in present.items() if value is not None
        )

    def label_names(self):
        return self.names

    def num_labels(self):
        return len(self.names)

    def max_duration_us(self):
        if self.durations is None:
            return None
        return int(max(self.durations))

    def text_width(self):
        return 0 if self.text is None else text_width(self.text)

    def read_events(self, start, stop):
        return self.times[start:stop]

    def read_durations(self, start, stop):
        return np.asarray(self.durations[start:stop], dtype=np.int64)

    def read_labels(self, start, stop):
        return self.labels[start:stop]

    def read_text(self, start, stop):
        return self.text[start:stop]

    def read_channel_ids(self, start, stop):
        return self.channel_ids[start:stop]

    def waveform_mean(self):
        return self.mean

    def waveform_period_us(self):
        return 1e6 / 32_000


marks = Events(
    "marks",
    "curated marks",
    [10.0, 45.0, 120.5, 200.0, 250.0, 290.0],
    [1, 1, 2, 0, 1, 2],
    ["seizure", "spike", "note"],
    durations=[0, 0, 4_500_000, 60_000_000, 0, 0],
    text=[
        "interictal spike",
        "interictal spike",
        "acquisition dropout",
        "electrographic seizure",
        "interictal spike",
        "electrode check",
    ],
    channel_ids=[
        ["tone-5"],
        ["tone-5"],
        ["tone-1"],
        ["tone-5", "tone-10", "tone-1"],
        ["tone-5"],
        ["tone-1"],
    ],
)
rng = np.random.default_rng(0)
spike_seconds = np.sort(rng.uniform(0, SECONDS, 70_000))
spikes = Events(
    "spikes",
    "spikes",
    spike_seconds,
    rng.integers(0, 3, spike_seconds.shape[0]),
    ["unit-0", "unit-1", "unit-2"],
    waveform_mean=rng.standard_normal((3, 32)).astype(np.float32),
)
write_bundle(
    [
        Tone(5),
        Tone(10),
        Tone(1, rate_hz=256.0, delay_s=0.5, pause_s=(120.0, 124.499)),
    ],
    [marks, spikes],
    output_path=Path("example.zarr"),
    opts=WriteOpts(),
    recording=RecordingMeta(
        session_start_time=SESSION_START, subject={"id": "sub-01"}
    ),
)
```

## Opening a bundle

`open_bundle` takes a local path as a `str` or a `Path`, a `file://` URL, a zarr
`Store`, or a zarr `StorePath`.

A string that holds `://` or an fsspec `::` chain is a URL. Any other string is a local path.
A relative path resolves against the current working directory.

A remote URL needs the `remote` extra, described in [remote stores](#remote-stores).

```python
from pathlib import Path

from zarr.storage import LocalStore, StorePath

from timeseries_zarr import open_bundle

path = Path("example.zarr").absolute()
for location in [
    "example.zarr",
    path,
    path.as_uri(),
    LocalStore(path, read_only=True),
    StorePath(LocalStore(path.parent, read_only=True), path.name),
]:
    with open_bundle(location) as bundle:
        print(bundle.location)
```

`open_bundle` reads only the root `zarr.json` and `recording.json`.
The consolidated metadata in the root `zarr.json` describes every channel,
level, and array. A read method loads samples or events from the store.

```python
from timeseries_zarr import open_bundle

with open_bundle("example.zarr") as bundle:
    print(bundle.version)
    print(bundle.describe())
    print(bundle.channel("1 Hz"))
    print(bundle.recording.subject["id"])
```

`bundle.channels` maps each channel key to its channel in key order.

`bundle.continuous` and `bundle.events` map the key of each channel of one kind
to its channel, in key order.

`bundle.channel` finds a channel by name, or by id when no name matches.
It raises `KeyError` when neither matches and `ValueError` when several
channels share the name or the id.

`bundle.recording` holds the parsed `recording.json` as a `Recording`,
or `None` when the bundle has no such file. A `Recording` has `reference_time` and
`session_start_time`, each a timezone-aware `datetime`, or `None` when the file
omits it. `session_start_time` is the `start_time` of the `session` object.
`subject`, `session`, and `source` are read-only mappings and are empty when
the file omits them.

`Bundle` is a context manager. Use it in a `with` statement, or call `close` to
close its store.

`open_bundle` raises `ValueError` when the root `zarr.json` carries no consolidated
metadata. The writer consolidates after the last channel, so a bundle from an
interrupted write does not open.

A channel group that does not parse, for example because of a missing attribute or an
invalid array shape, raises `ValueError` naming the channel. A channel group of unknown
`kind` is left out of `channels` with a `UserWarning`.

A location without a root `zarr.json` raises `FileNotFoundError`, and a location
whose root is an array raises `ValueError`.

## Times

Every window bound is an integer count of microseconds from the reference time,
and a time before it is negative. The [numeric conventions](./spec.md#23-numeric-conventions) section of
the specification defines the reference time. A window bound, `level`, `bins`,
and `gap_us` each take an `int` or a numpy integer. A `float` raises `TypeError`.

A continuous read stores `offset_us`, `first`, and `period_us` in its result.
Entry `i` of the result starts at `offset_us + (first + i) * period_us` microseconds
on the sample grid, and `time_us` computes that for every entry. A raw read of a
channel that has [runs](#runs) gives the exact time of each recorded sample instead.

`time_us` is float64. A sample period of `1e6 / rate_hz` microseconds
is rarely a whole number, and at 256 Hz it is 3906.25 microseconds.

Event times are int64 microseconds, as stored.

```python
from timeseries_zarr import open_bundle

with open_bundle("example.zarr") as bundle:
    channel = bundle.continuous[2]
    samples = channel.read(round(100.0 * 1e6), round(100.01 * 1e6))
    print(samples.first, samples.period_us)
    print(samples.time_us)
    print(samples.time_s)
```

`bundle.datetimes(time_us)` returns each time as a UTC `datetime64[us]`.
It rounds each time to the nearest microsecond and adds it to the `reference_time`.
A NaN gives NaT.

```python
from timeseries_zarr import open_bundle

with open_bundle("example.zarr") as bundle:
    print(bundle.recording.reference_time)
    samples = bundle.continuous[0].read(1_500_000, 1_503_000)
    print(bundle.datetimes(samples.time_us))
```

## Continuous channels

A `ContinuousChannel` carries `name`, `id`, `unit`, `rate_hz`, `offset_us`, and
`provenance` from its group attributes. Its `key` is the number that names its group.

`provenance` is a read-only mapping, or `None` when the group has none.
`num_samples` is the length of `raw`, which counts the NaN samples of every pause.
`period_us` is the spacing of the raw samples, and `start_us` is the time of sample 0.
`end_us` is the end of the last sample, rounded up to a whole microsecond. A window
from `start_us` to `end_us` holds every sample.

`raw` is the zarr array of samples.
`levels` maps each level number, from 1, to a `Level`. The `samples_per_bin` of a
`Level` is the number of raw samples that one bin folds.

The [levels](./spec.md#43-levels) section of the
specification defines the samples each bin covers.

A `Level` holds the zarr arrays `minmax`, `mean`, `var`, and `count`, with
`None` for a member the level lacks. Slicing a zarr array, for example
`channel.raw[1000:2000]`, reads that slice from the store.

`read(start_us=None, end_us=None, *, level=None, members=None)` returns `Samples`
for level `None` or 0, and `Bins` for a level number. The window is half-open,
`[start_us, end_us)`, and a `None` bound leaves that side open.
The read widens the window outward to whole samples or bins.
The result holds every entry that overlaps the window, and `first` is the index
of its first entry. A window that reaches past either end of the channel reads
the entries that exist, and a window outside the channel gives an empty result.

`Bins` splits `minmax` into `min` and `max`.
It also holds `mean`, `var`, and `count`, each `None` when the level lacks it or
the read left it out.

`members` names the members of the level to read, from `minmax`, `mean`, `var`,
and `count`. `None` reads every member the level holds. The read always holds
`min` and `max`, so `members=()` reads them alone. A name the level lacks raises
`ValueError`, and `members` on a raw read raises `ValueError`.

`level_for(start_us=None, end_us=None, *, bins)` returns the coarsest level at
which `read` returns at least `bins` bins for the window. It returns 0, for the
raw samples, when no level returns that many bins.

```python
from timeseries_zarr import open_bundle

with open_bundle("example.zarr") as bundle:
    channel = bundle.continuous[0]
    print(channel)
    print({number: level.bins for number, level in channel.levels.items()})
    for start_us, end_us in [
        (0, 300_000_000),
        (100_000_000, 130_000_000),
        (100_000_000, 106_000_000),
        (100_000_000, 101_000_000),
    ]:
        level = channel.level_for(start_us, end_us, bins=1200)
        result = channel.read(start_us, end_us, level=level)
        print(level, type(result).__name__, len(result), result.first)
    envelope = channel.read(0, 300_000_000, level=1, members=())
    print(len(envelope), envelope.mean)
```

Channel 2 of `example.zarr` starts 500,000 microseconds after the reference time.
A read of a window that starts before the channel returns bins from bin 0.
A window that ends at the channel start is empty.
A window that runs past the channel end returns the samples that exist,
128 in the last read.

```python
from timeseries_zarr import open_bundle

with open_bundle("example.zarr") as bundle:
    channel = bundle.continuous[2]
    print(channel.offset_us, channel.end_us)
    bins = channel.read(0, 1_000_000, level=1)
    print(bins.first, len(bins), bins.time_us[0])
    print(len(channel.read(0, 500_000, level=1)))
    print(len(channel.read(300_000_000, 400_000_000)))
```

A NaN sample is a gap. The [statistics](./spec.md#44-statistics)
section of the specification defines how a gap folds into each level, and the
reader passes the statistics through unchanged. A raw read of a channel with runs
checks that every sample outside a run is NaN and raises `ValueError` otherwise.

`spans(start_us=None, end_us=None, *, level=None, gap_us=0)` returns the
half-open spans of the window that hold data, as pairs of integer microseconds.
It reads the coarsest level when `level` is `None`, or the raw samples
on a channel without levels. A bin holds data when its `count` is above 0.
On a level without `count`, a bin holds data when a `minmax` value is finite.
A span that starts at most `gap_us` after the previous span ends joins it. A span on
level 0 of a channel with runs starts and ends at exact sample times. A span on a level
starts and ends at bin edges on the grid.

```python
from timeseries_zarr import open_bundle

with open_bundle("example.zarr") as bundle:
    channel = bundle.continuous[2]
    bins = channel.read(118_000_000, 128_000_000, level=3)
    print(bins.mean)
    print(bins.count)
    print(channel.spans())
    print(channel.spans(level=1, gap_us=5_000_000))
```

## Runs

A channel whose recording paused and resumed off its sample grid holds a `runs` array.
Each row describes one run, a span of samples recorded at the sample rate. The
[sample times](./spec.md#42-sample-times) section of the specification
defines the rows and the rules that place a run on the grid. The grid keeps a NaN
sample for each sample period of a pause, and `runs` holds the exact time that each
run starts.

`channel.runs` is `None` for a channel without a `runs` array. Otherwise it is a
read-only int64 array of shape `(R, 3)`. Column 0 is the index of the first sample of a
run, column 1 its number of samples, and column 2 the time of its first sample in
microseconds from the reference time. The first access reads the array and checks every
row against the specification. A row that breaks a rule raises `ValueError`, and the
message names the channel and the row.

A raw read of a channel with runs uses the exact times:

- `Samples.time_us` gives the exact time of each sample inside a run. A NaN sample
  between two runs gets its grid time.
- A window maps to samples through the exact times of the runs. A window inside a pause
  maps to the NaN samples on the grid.
- `end_us` is the exact end of the last run.

`Bins.time_us` stays on the grid at every level, so channels that share a grid still
stack by index. A reader that ignores `runs` places each sample within one sample
period of its time.

Channel 2 of `example.zarr` pauses from 120.5 s and resumes 1 ms before 125 s. Sample
31,872 is the first of the second run. The window below holds three NaN samples of the
pause and three samples of the second run. The printed difference is 0 for a NaN
sample and -1000 microseconds for a sample of the run.

```python
import numpy as np

from timeseries_zarr import open_bundle

with open_bundle("example.zarr") as bundle:
    channel = bundle.continuous[2]
    print(channel.runs)
    samples = channel.read(124_990_000, 125_010_000)
    print(samples.first, samples.values)
    indices = samples.first + np.arange(len(samples))
    print(samples.time_us - (channel.offset_us + indices * channel.period_us))
    print(channel.spans(level=0))
    print(channel.spans(level=1))
```

## Event channels

An `EventChannel` carries `name`, `id`, `unit`, `label_names`, `max_duration_us`,
and `provenance` from its group attributes. Its `key` is the number that names
its group. Each attribute except `name` is `None` when the group omits it, and
`provenance` is a read-only mapping.

`events` is the zarr array of event times. `durations`, `labels`, `values`, `text`,
and `channels` are zarr arrays, or `None` when the channel lacks them.

`columns` names the per-event columns the channel holds, in the order `durations`,
`labels`, `values`, `text`, `channels`, and `waveforms`.

`read(start_us=None, end_us=None, *, columns=None)` returns an `EventTable` with one row
per event. `time_us` holds the event times. `duration_us`, `label`, `label_name`,
`value`, `text`, `channel_keys`, and `waveform` hold the other columns. Each is `None`
when the channel lacks it or the read left it out. `columns` names the stored columns to
read, and `None` reads every one. A name the channel lacks raises `ValueError`.

A point event, with no duration or a duration of 0, is in the window when
`start_us <= time < end_us`. An interval is in the window when it starts before
`end_us` and ends after `start_us`. To find the events that begin before the window,
the read searches from `max_duration_us` before `start_us`. On a channel with
`durations` and no `max_duration_us`, it searches from the first event.

`text` holds the decoded UTF-8 string of each event. `channel_keys` holds the channel
keys each event applies to, and an empty list means the event applies to no channel. `label_name`
holds the name of each label when the channel has `label_names`. A label at or above the
length of `label_names` raises `ValueError`.

`decode_text_rows(channel.text[0:10])` and
`decode_channels_rows(channel.channels[0:10])` decode rows sliced from the `text` and
`channels` arrays, as `read` does.

`applies_to(key)` returns a boolean mask of the events that apply to channel `key`. An
event applies when its `channel_keys` names `key`. On a channel without a `channels`
column, `applies_to` returns `True` for every event. A table read from such a
channel has `whole_recording` set to `True`. On a table read
without `channels` from a channel that has the column, `applies_to` raises `ValueError`.

```python
from timeseries_zarr import open_bundle

with open_bundle("example.zarr") as bundle:
    marks = bundle.events[3]
    print(marks.columns, marks.label_names, marks.max_duration_us)
    table = marks.read(100_000_000, 300_000_000)
    print(table.time_us)
    print(table.duration_us)
    print(table.label_name)
    print(table.text[-1])
    print(table.channel_keys)
    print(table.applies_to(2))
    print(len(marks.read(259_999_999, 270_000_000)))
    print(len(marks.read(260_000_000, 270_000_000)))
```

The seizure starts at 200 s and lasts 60 s. It covers the span up to but not
including 260 s, so the read from 1 microsecond before 260 s holds it and the read
from 260 s does not.

`waveform` is a `Waveform` with one snippet row per event of the table.
The column at index `event_sample` holds the sample at the event time,
and `time_us` counts microseconds from the event time. `time_s` gives the same
times in seconds, and `len` gives the number of samples in each snippet.

`channel.waveforms` and `channel.waveform_mean` are `Snippets` that wrap the
zarr arrays, and `read()` returns every row as a `Waveform`.

Each row of `waveform_mean` is the mean snippet of one label. `rows` is `event` for the
snippets of a table and `label` for `waveform_mean`. `write_bundle` writes
`waveform_mean` and no per-event `waveforms`, so the `spikes` channel of `example.zarr`
has no `waveform` column.

`count_levels` maps each count level number, from 1, to a `CountLevel` with
`period_us` and `offset_us`. `offset_us` is the time of its bin 0. The
[count levels](./spec.md#57-count-levels) section of the specification defines
the bins of a count level. `read_counts(start_us=None, end_us=None, *, level)`
returns the bins that overlap the window as `Counts`. The bins widen the window,
so their counts can include events outside it. A window past the last bin returns
no bins.

`level_for(start_us=None, end_us=None, *, bins)` on an event channel returns the
coarsest count level at which `read_counts` returns at least `bins` bins for the
window. It returns 0 when no count level does. An open `start_us` stands for the first
event, and an open `end_us` for 1 microsecond after the last event. `read_counts`
rejects level 0, so when `level_for` returns 0, use `read` to read the events.

```python
import numpy as np

from timeseries_zarr import open_bundle

with open_bundle("example.zarr") as bundle:
    spikes = bundle.events[4]
    table = spikes.read(100_000_000, 110_000_000)
    print(len(table), np.bincount(table.label), table.waveform)
    mean = spikes.waveform_mean.read()
    print(mean.values.shape, mean.event_sample, mean.time_us[mean.event_sample])
    level = spikes.level_for(100_000_000, 110_000_000, bins=100)
    if level == 0:
        print(len(spikes.read(100_000_000, 110_000_000)))
    else:
        counts = spikes.read_counts(100_000_000, 110_000_000, level=level)
        print(level, counts.first, len(counts), counts.count.sum(axis=0))
```

## Stacking channels

`bundle.stack(keys, start_us=None, end_us=None, *, level=0, member="mean", max_workers=1)`
reads one window of several continuous channels into one array. The channels must
share `rate_hz` and `offset_us`. Level 0 reads the raw samples, and the `member`
of the result is `None`. A level of at least 1 reads `member`, which is
`min`, `max`, `mean`, `var`, or `count`. Every channel must hold that member
at that level.

`Stack.values` has one row per key, in the order of `keys`. The rows span every
entry that overlaps the window on any channel. A row holds NaN past the end of
its channel, or 0 when `member` is `count`. `names` and `units` follow the order
of `keys`.

A stack reads one array per channel. At level 0 it reads `raw`, and at a level of
at least 1 it reads the array that holds `member`. `max_workers` is the number of
threads that read the channels. At 1 the call reads them in order. A larger value
reads up to that many channels at once. On a remote store, a stack of many
channels then takes less time. The result does not depend on `max_workers`. Each
thread holds its own zarr request budget, so `max_workers` threads can keep that
many times `async.concurrency` requests in flight.

```python
from timeseries_zarr import open_bundle

with open_bundle("example.zarr") as bundle:
    stack = bundle.stack([0, 1], 0, 10_000_000, level=1, member="mean")
    print(stack.keys, stack.names, stack.values.shape, stack.period_us)
    raw = bundle.stack([0, 1], 0, 3_000, max_workers=2)
    print(raw.level, raw.member)
    print(raw.values)
```

## Conversions and extras

Every result has `columns()`, which returns its columns as a dict of numpy arrays
in column order. It requires no extra.

`to_dataframe` and `to_series` require the pandas extra, and `to_xarray` requires
the xarray extra. Install them with `pip install 'timeseries-zarr[pandas]'` or
`pip install 'timeseries-zarr[xarray]'`. The xarray extra also installs pandas.

A conversion imports pandas or xarray when it runs. When the package is missing,
it raises `ImportError` with a message that names the extra:
`pandas is not installed. Install the pandas extra: pip install 'timeseries-zarr[pandas]'`.

| Result | pandas | xarray |
|---|---|---|
| `Samples` | `to_series(*, name=None)`: Series of the values indexed by `time_us` | `to_xarray(*, name=None, unit=None)`: DataArray along `time_us`, with attributes `period_us`, `offset_us`, `units` when `unit` is given, and `long_name` when `name` is given |
| `Bins` | `to_dataframe()`: indexed by `time_us`, with columns `min`, `max`, then `mean`, `var`, and `count` when present | `to_xarray()`: Dataset of the same variables along `time_us`, with attributes `level` and `period_us` |
| `EventTable` | `to_dataframe()`: one row per event under a numbered index, with columns `time_us`, then `duration_us`, `label`, `label_name`, `value`, `text`, and `channel_keys` when present | none |
| `Counts` | `to_dataframe()`: indexed by `time_us`, with one column per label named by `label_names` or by the label index | `to_xarray()`: DataArray with dimensions `time_us` and `label` |
| `Waveform` | `to_dataframe()`: indexed by `time_us` from the event time, with one column per row named by the row index | `to_xarray()`: DataArray with dimensions `event` and `time_us`, or `label` and `time_us` for `waveform_mean`, with attributes `period_us` and `event_sample` |
| `Stack` | `to_dataframe()`: indexed by `time_us`, with one column per channel key, and `attrs` that hold `names`, `units`, `level`, and `member` | `to_xarray()`: DataArray with dimensions `channel` and `time_us`, coordinates `name` and `unit` along `channel`, and attributes `level`, `member` above level 0, `period_us`, `offset_us`, and `units` when the channels share one unit |

The `EventTable` frame leaves out `waveform`. Its `label_name` and `text` columns hold
strings, and `channel_keys` holds a Python list per event. The xarray attribute that
holds a unit is `units`, which xarray plotting puts in the axis label.

```python
from timeseries_zarr import open_bundle

with open_bundle("example.zarr") as bundle:
    channel = bundle.continuous[2]
    bins = channel.read(0, 20_000_000, level=3)
    print(list(bins.columns()))
    print(bins.to_dataframe().head())
    samples = channel.read(1_000_000, 1_020_000)
    print(samples.to_xarray(name=channel.name, unit=channel.unit))
    print(bundle.events[3].read(columns=["labels"]).to_dataframe())
```

## Rewriting a bundle

`bundle_sources(bundle, keys=None, *, epoch_us=None, drop_columns=frozenset())` wraps
channels of an open bundle as writer sources. It returns the lists of continuous and
event sources that `write_bundle` takes. `keys` names the channels to keep, and `None`
keeps every channel. `recording_meta(bundle)` returns the `RecordingMeta` that rewrites
`recording.json`, or `None` when the bundle has no `recording.json`.

A writer source gives times in microseconds since the Unix epoch.

`bundle_sources` adds `epoch_us` to every stored time.

`epoch_us` defaults to `reference_time` in microseconds since the Unix
epoch, or to 0 when the bundle has no `reference_time`.

When `recording_meta` carries `reference_time`, the writer keeps it as given.
The copy then keeps every offset and event time.
A kept or added channel that begins before the reference time gets a negative offset.

A rewrite without a `RecordingMeta` places the reference time of the copy at
the first kept sample, and every stored time shifts by the same amount.

`recording_meta` also carries `session.start_time` as `session_start_time`, and the
writer puts it back. A bundle without `reference_time` gives a `RecordingMeta` with
`wall_clock=False`, which marks times that count from an unknown instant. The copy
then has no `reference_time` either.

The writer numbers the kept channels from 0, continuous channels first. A channel
without an id takes its key, as a decimal string, as its id. `bundle_sources` raises
`ValueError` when two kept channels share an id.

An event source names the channels of each event by id, and it maps the key of every
kept channel to an id. A kept event that names a channel left out of `keys` makes
`read_channel_ids` raise `ValueError` during `write_bundle`. Pass
`drop_columns={"channels"}` to leave the column out. The new channel then has no
`channels` column.

A rewrite leaves out `values` and `waveforms`, which the writer takes no input for. It
keeps the `event_sample` of `waveform_mean`. A rewrite also leaves out a `text` column of width
0, and `waveform_mean` when it carries no `labels`. The writer computes the levels and
count levels again with the `WriteOpts` it gets. A channel with `runs` keeps its runs and
the exact time of each one.

Write the new bundle to a path other than the bundle being read. `write_bundle` clears
`output_path` before it reads a source, so a rewrite in place leaves only fill values.

```python
from pathlib import Path

from timeseries_zarr import (
    WriteOpts,
    bundle_sources,
    open_bundle,
    recording_meta,
    write_bundle,
)

with open_bundle("example.zarr") as bundle:
    continuous, events = bundle_sources(
        bundle, [2, 3], drop_columns={"channels"}
    )
    write_bundle(
        continuous,
        events,
        output_path=Path("subset.zarr"),
        opts=WriteOpts(),
        recording=recording_meta(bundle),
    )

with open_bundle("subset.zarr") as subset:
    print(subset.describe())
    print(subset.continuous[0].offset_us)
    print(subset.events[1].columns)
```

## Remote stores

The `remote` extra installs fsspec, s3fs, and aiohttp.
Install the extra with `pip install 'timeseries-zarr[remote]'`.

With the extra installed, `open_bundle` takes an fsspec URL such as
`https://`, `s3://`, or `memory://`, or a chain such as
`simplecache::s3://bucket/recording.tszarr`.

`storage_options` passes options to the filesystem, as in
`open_bundle("s3://bucket/recording.tszarr", storage_options={"anon": True})`.

Without fsspec, a remote URL raises `ImportError` with a message that names the extra.

A `memory://` URL reads the fsspec memory filesystem only when fsspec is installed.
Without fsspec, zarr opens the URL as an in-process store of its own, and `open_bundle`
raises `FileNotFoundError` when that store holds no bundle.

`storage_options` works only with a remote URL. With a local path, a store,
or a memory URL, `open_bundle` raises `TypeError` for it.

For a `memory://` bundle, `bundle.location` gives the URL with three slashes, as in
`memory:///demo/example.zarr`. That URL reopens the bundle.
To pass options to the memory filesystem, build an `FsspecStore` and open the
store.

An HTTP server must honor `Range` requests. The reader fetches byte ranges of each
shard. Over a server that ignores the `Range` header, `open_bundle` succeeds.
A read that needs only some of the inner chunks of a shard then raises `ValueError`.

```python
import fsspec
from zarr.storage import FsspecStore

from timeseries_zarr import open_bundle

fsspec.filesystem("memory").put(
    "example.zarr/", "/demo/example.zarr", recursive=True
)

with open_bundle("memory://demo/example.zarr") as bundle:
    print(bundle.location)

store = FsspecStore.from_url(
    "memory://demo/example.zarr",
    storage_options={"skip_instance_cache": True},
    read_only=True,
)
with open_bundle(store) as bundle:
    print(bundle.continuous[0].read(0, 3_000).values)
```
