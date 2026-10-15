# Architecture

Data flows one way through the writer: read, plan, stream and fold, write. No stage
reaches back into an earlier one.

## Input adapters

`protocols.py` declares `ContinuousChannelSource` and `EventChannelSource`. A continuous
source exposes one channel's metadata and windowed sample reads. An event source
exposes its metadata, its event times, and the optional columns it provides. Its
`columns()` method names them, and the writer calls a column's reader only for a column
named there. Either kind of source may also define `provenance()`, which the protocols
leave optional.

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
arrays a level stores and assembles one from them. See [the format spec](./spec.md)
for the exact rules and the NaN behavior.

`streaming.py` drives the folds one block at a time. It buffers across block boundaries,
so a run of 4 that straddles two blocks folds correctly. It also re-cuts a stream into
whole shards. Memory stays bounded however long the recording is.

## Write path

`write_continuous.py` writes one continuous channel: the `raw` array, then a group per
level with its `minmax`, `mean`, `count`, and optional `var` arrays. By default each level
folds from the stored members of the level below. With `write_var` on, every level folds
from the raw samples in one pass through float64 statistics. That keeps the variance
exact on a channel with a large DC offset.

`write_event.py` writes one event channel: the `events` array, each optional column the
source names, and the count levels of a dense channel. Event times are stored from the
bundle origin.

Both writers read shard-sized blocks. Every write then covers a whole shard. A narrower
write makes the sharding codec read the shard back, re-encode every inner chunk, and
rewrite it. That costs about 10x the store traffic on a 16-chunk shard.

`zarr_io.py` is the only module that imports `zarr`. Everything else is Zarr-agnostic. The
Zarr v3 API surface this package depends on sits in one file.

`attrs.py` builds the attribute dicts. The format keeps them all under one `tsz` key.
`recording.py` writes `recording.json`, the one plain file that holds identity and the
wall-clock start.

## Orchestration

`bundle.py` runs the whole job. It assigns each channel a key and finds the origin, the
earliest start of any channel or of the session. It maps channel ids to keys. Event
channels use that map to name the channels they apply to. It checks each channel's unit
and provenance before it clears `output_path`. It then writes each channel, consolidates
the metadata, and writes `recording.json`.

A converter calls `write_bundle` with its sources, a `WriteOpts`, and an optional
`RecordingMeta`.
