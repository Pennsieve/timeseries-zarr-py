# Architecture

Data flows one way through the writer: read, plan, stream and fold, write. No stage
reaches back into an earlier one.

## Input adapters

`protocols.py` declares `ContinuousChannelSource`, `MultiChannelSource`, and
`EventChannelSource`. A continuous source exposes one channel's metadata and windowed
sample reads. A multi-channel source exposes the metadata of every channel through
`channels()` and yields one window, with one block per channel, from `iter_windows()`.
`ContinuousChannelMetadata` declares the metadata that a continuous source and each
channel of a multi-channel source share. An event source exposes its metadata, its event
times, and the optional columns it provides. Its `columns()` method names them, and the
writer calls a column's reader only for a column named there. Every channel may also
define `provenance()`, which the protocols leave optional.

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

`counts.py` plans the count levels of a dense event channel. Level 1 takes its period
from a ladder of `1000 * 4^j` microseconds, and each level above it spans 4x the time.
`columns.py` encodes the fixed-width `text` rows and the `channels` bitmask rows.
`units.py` resolves the factor and stored unit of a continuous channel.

Keeping these separate from the write path means the format's arithmetic is unit-testable
without touching a store.

## Numeric core

`fold.py` reduces one level to the next over disjoint blocks of 4. A folded block is a
float64 statistics block with one column each for the minimum, maximum, mean, population
variance, and count of finite samples. `fold.py` also splits a block into the member
arrays a level stores and assembles one from them. `Statistics` holds the same five
statistics as one float64 array each. `fold_raw` and `fold_statistics` fold it on
column views, and `Statistics.members` converts it to the member arrays. See
[the format spec](./spec.md) for the exact rules and the NaN behavior.

`streaming.py` drives the folds one block at a time. It buffers across block boundaries,
so a run of 4 that straddles two blocks folds correctly. `regroup` re-cuts an iterator of
blocks into whole shards, and `Regrouper` re-cuts blocks that arrive one push at a time.
Memory stays bounded however long the recording is.

## Write path

`write_continuous.py` writes one continuous channel: the `raw` array, then a group per
level with its `minmax`, `mean`, `count`, and optional `var` arrays. By default each level
folds from the stored members of the level below. With `write_var` on, every level folds
from the raw samples in one pass through float64 statistics. That keeps the variance
exact on a channel with a large DC offset.

`write_continuous_channels` writes every channel of a multi-channel source. It creates
each channel group and its `raw` array, then makes one pass over the windows of the
source. Each channel has a `Regrouper`, so every write to `raw` covers a whole shard
whatever the length of the blocks. After the pass, the writer folds the levels of each
channel from its stored `raw` array.

`shards.py` writes the shards of one array as its rows arrive. A `ChunkEncoder` turns the
rows of an inner chunk into little-endian bytes compressed with Zstd. A `ShardWriter`
appends encoded chunks to a `ShardSink`, then writes the shard index and its CRC-32C. An
`ArrayWriter` cuts blocks of any length into chunks and holds at most one partial chunk.
The module imports `numcodecs` and `google_crc32c` and never zarr, and its shards match
the bytes that zarr writes for the same array.

`write_event.py` writes one event channel: the `events` array, each optional column the
source names, and the count levels of a dense channel. Event times are stored from the
reference time.

Both writers cover a whole shard with every write. A narrower write makes the sharding
codec read the shard back, re-encode every inner chunk, and rewrite it. That costs about
10x the store traffic on a 16-chunk shard.

`zarr_io.py` is the only module that imports `zarr`. Other modules handle zarr groups
and arrays through the `ZarrGroup` and `ZarrArray` aliases.

`attrs.py` builds the attribute dicts. The format keeps them all under one `tsz` key.
`recording.py` writes `recording.json`, the one plain file that holds identity and
datetimes.

## Orchestration

`bundle.py` runs the whole job. It assigns each channel a key. It finds the reference
time, the earlier of the session start and the first sample, and maps channel ids to
keys. Event channels use that map to name the channels they apply to. It checks each
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
or bins in exact fractions and computes the float64 time axis of a result. The parsers
also read channel and level numbers from group names through it. `attrs.py` parses the
`tsz` attributes beside its builders, and `columns.py` decodes the `text` and
`channels` rows.

`convert.py` turns read results into pandas and xarray objects. `extras.py` imports
each package when a conversion runs and raises `ImportError` naming the extra when the
package is missing. `stack.py` reads one window of several continuous
channels that share a grid into one array. `sources.py` wraps the channels of an open
bundle as writer sources for `write_bundle`, which then rewrites the bundle or a subset
of it.
