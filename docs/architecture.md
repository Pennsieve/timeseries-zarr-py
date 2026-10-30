# Architecture

Data flows one way through the writer: read, plan, stream and fold, write. No stage
reaches back into an earlier one.

## Input adapters

`protocols.py` declares `ContinuousChannelSource`, `MultiChannelSource`, and
`EventChannelSource`. A continuous source exposes one channel's metadata and windowed
sample reads. A multi-channel source exposes the metadata of every channel through
`channels()` and yields one window from `iter_windows()`. A window is a `(rows, channels)`
array or a sequence with one block per channel.
`ContinuousChannelMetadata` declares the metadata that a continuous source and each
channel of a multi-channel source share. An event source exposes its metadata, its event
times, and the optional columns it provides. Its `columns()` method names them, and the
writer calls a column's reader only for a column named there. Every channel may also
define `provenance()` and `runs()`, which the protocols leave optional. `RunsSource`
declares `runs()` and `source_runs` reads it. A channel without it is one run. An event
source with `waveform_mean` may define `waveform_event_sample()`, which
`WaveformEventSampleSource` declares and `source_waveform_event_sample` reads. A source
without it places the event at sample 0.

Everything downstream depends on the protocols. An adapter for an input format lives in
its own package and implements them. The NWB adapter is
[timeseries-zarr-nwb](https://github.com/Pennsieve/timeseries-zarr-nwb). The writer
scales a voltage channel to microvolts and keeps any other unit, so every channel
carries the unit its group attribute names. The writer is tested against the in-memory
sources in `tests/conftest.py`.

## Decision layer

`planning.py`, `sizing.py`, `counts.py`, `columns.py`, and `units.py` are pure functions
with no I/O. `planning.py` decides how many levels a continuous channel gets, numbered from 1, and each
level's bin count and `period_us`. `sizing.py` decides the inner chunk and outer shard
shapes for one array. Every member of a level takes the row counts of its `minmax` array.

`counts.py` plans the count levels of a dense event channel. Level 1 takes an integer
period from a ladder of `1000 * decimation_factor^j` microseconds, and each level above
it spans `decimation_factor` times the time. Every count level carries an `offset_us`
that is a multiple of its period.
`columns.py` encodes the fixed-width `text` rows and the `channels` bitmask rows. An
event that names no channel gets a row of zero bytes.
`units.py` resolves the factor and stored unit of a continuous channel.
`runs.py` holds `Run`, the span of samples that a source states, and `place_runs`, which
puts the runs of a channel on its sample grid. It picks the grid index of each run in exact
fractions and checks every bound of the table in binary64. It returns the offset,
the grid length, and the table of the `runs` array. The same module checks the table
and the NaN samples outside the runs. The reader checks the table on the first access of
`runs` and the samples outside the runs on each raw read.

Keeping these separate from the write path means the format's arithmetic is unit-testable
without touching a store.

## Numeric core

`fold.py` reduces one level to the next over disjoint blocks of `decimation_factor` rows.
`Statistics` holds the minimum, maximum, mean, population variance, and count of the
finite samples of consecutive bins, as one float64 array each. A bin with no finite
sample has a count of 0 and NaN in the other four. `fold_raw` and
`fold_statistics` fold it on column views, and `Statistics.members` converts it to the
member arrays a level stores. A `LevelFolder` holds the state of one channel's pyramid.
It folds raw samples into every level as they arrive and emits one chunk of every member
each time a level fills `inner_len` rows. A level never holds more than one chunk of
pending rows. A level folds from the children
with a count above 0. See [the format spec](./spec.md#44-statistics) for the exact rules.

`streaming.py` reads a source in blocks and re-cuts block streams. The count levels of an
event channel use it.

## Write path

`write_continuous.py` writes continuous channels in one pass. A `ChannelWriter` holds the
state of one channel. Each block of samples is scaled to the stored unit, appended to
`raw`, and folded into every level. Each chunk that the folder emits is appended to its
member array. Nothing is read back from the store. A channel with several runs also gets the NaN
samples of each pause, pushed through the same path before the first sample of the next
run. `_channel_writer` writes the `runs` array of a channel with two or more runs as one shard
through `zarr_io.write_region`. `write_var` decides only whether the
`var` member is stored, because every level folds from float64 statistics and the variance
stays exact on a channel with a large DC offset.

`write_continuous_channels` writes every channel of a multi-channel source. It places the
runs of each channel, builds the metadata of every channel group, `raw` array, level group, and level member, and
`zarr_io.create_nodes` creates them all in one concurrent call. `zarr_io.array_metadata`
and `zarr_io.group_metadata` build the metadata, and a test pins that they store the same
`zarr.json` as `create_array` and `create_group_with_attrs`. The writer then makes one
pass over the windows of the source. The loop gathers consecutive windows into a batch of
`WINDOW_BATCH_ROWS` rows or `WINDOW_BATCH_BYTES`, whichever comes first. Each batch runs one
job per channel on a pool of `WriteOpts.max_workers` threads, and each job joins its
channel's blocks and pushes them to the `ChannelWriter` of the channel as one block. The
loop transposes a `(rows, channels)` window once into contiguous rows for each channel. It
gathers the next batch while the jobs run and waits for them before it dispatches the
next. A window with fewer channels than threads also encodes the chunks of
each job in groups on the idle threads. `write_continuous_channel` builds the same nodes
and writer for one source, pushes blocks of one `raw` shard, and encodes their chunks in
groups on the pool. A channel is written by one job at a time, so the bytes on disk do not
depend on the thread count.

`shards.py` writes the shards of one array as its rows arrive. A `ChunkEncoder` turns the
rows of an inner chunk into little-endian bytes compressed with Zstd. A `ShardWriter`
appends encoded chunks to a `ShardSink`, then writes the shard index and its CRC-32C. An
`ArrayWriter` cuts blocks of any length into chunks and holds at most one partial chunk.
A `ChannelWriter` gives all its arrays one `AppendBudget` of `CHANNEL_APPEND_BYTES`. Each
shard sink then holds encoded chunks and writes them in one append when the budget runs
out or the shard closes. The sink that holds the most is written first.
The module imports `numcodecs` and `google_crc32c` and never zarr, and its shards match
the bytes that zarr writes for the same array. `zarr_io.shard_layout` reads the layout and
the shard sinks of an existing array from its metadata. It raises `ValueError` for any
codec chain other than the one `shards.py` encodes. zarr creates the nodes of a
continuous channel and reads them back. It does not encode a chunk of one.

`write_event.py` writes one event channel: the `events` array, each optional column the
source names, and the count levels of a dense channel. Event times are stored from the
reference time. Every write covers a whole shard. A narrower write makes the sharding
codec read the shard back, re-encode every inner chunk, and rewrite it. That costs about
10x the store traffic on a 16-chunk shard.

`zarr_io.py` is the only module that imports `zarr`. Other modules handle zarr groups
and arrays through the `ZarrGroup` and `ZarrArray` aliases.

`attrs.py` builds the attribute dicts. The format keeps them all under one `tsz` key.
`recording.py` writes `recording.json`, the one plain file that holds identity and
datetimes.

## Orchestration

`bundle.py` runs the whole job. It assigns each channel an index. It finds the reference
time, the earlier of the session start and the first sample, and maps channel ids to
channel indices. Event channels use that map to name the channels they apply to. It checks each
channel's unit and provenance before it clears `output_path`. It then writes each
channel, consolidates the metadata, and writes `recording.json`.

A converter calls `write_bundle` with its sources, a `WriteOpts`, and an optional
`RecordingMeta`.

## Read path

`read_bundle.py` opens a bundle. `open_bundle` opens the root group through
`zarr_io.py`, and `parse_bundle` builds every channel record from the consolidated
metadata in the root `zarr.json`. `read_recording.py` parses `recording.json` into a
`Recording`.

`read_continuous.py` and `read_event.py` each parse one kind of channel group and hold
the window reads of that kind. `windows.py` maps a time window to a range of samples
or bins in exact fractions and computes the float64 time axis of a result. For a channel
with a `runs` table it maps windows and indices through the exact times of the runs. The parsers
also read channel and level numbers from group names through it. `attrs.py` parses the
`tsz` attributes beside its builders and applies the time bound `TIME_LIMIT_US`.
`read_continuous.py` checks the period of each level against the sample rate within
`LEVEL_PERIOD_TOLERANCE`. `columns.py` decodes the `text` and `channels` rows.

`convert.py` turns read results into pandas and xarray objects. `extras.py` imports
each package when a conversion runs and raises `ImportError` naming the extra when the
package is missing. `stack.py` reads one window of several continuous
channels that share a grid into one array. `sources.py` wraps the channels of an open
bundle as writer sources for `write_bundle`, which then rewrites the bundle or a subset
of it.

## Validation

`validator.py` holds `validate_bundle`, which checks a bundle against sections 8.3 and 8.4
of the specification and returns a list of `Finding` records. It reads each `zarr.json`
and `recording.json` as plain JSON and reports every violation it finds instead of
raising on the first. It raises `ValueError` for a store that cannot list its keys. `validate_arrays.py` checks the shape, dtype, fill value, chunk grid, and codec chain of
one array, and reads the shard indexes to find missing chunks. `validate_full.py`
recomputes the statistics from `raw`, the events, and the waveforms in blocks. It runs
only when `full` is set. `validate.py` is the entry point of `python -m
timeseries_zarr.validate`. It stays out of `__init__.py`.

`scripts/build_fixtures.py` writes the conformance fixtures into `conformance/`, with a
`manifest.json` that lists what each fixture exercises, the values a reader can assert,
and the findings that full validation reports. `make fixtures` runs it, and
Git tracks only the manifest and ignores the bundles. `tests/test_conformance.py` builds
the bundles in a temporary directory, checks that two runs write the same bytes, and
checks that the committed manifest is what the script writes.
