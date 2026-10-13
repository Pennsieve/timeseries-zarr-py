# Architecture

Data flows one way through the writer: read, plan, stream and fold, write, publish. No
stage reaches back into an earlier one.

## Input adapters

`protocols.py` declares `ContinuousChannelSource` and `EventChannelSource`. A continuous
source exposes one channel's metadata and windowed sample reads. An event source exposes
its event times and the optional columns it provides. Its `columns()` method names them,
and the writer calls a column's reader only for a column named there.

`nwb_reader.py` and `nwb_events.py` hold the concrete NWB adapters. Everything downstream
depends on the protocols, not on NWB. The core is testable against in-memory sources, and
a second input format costs new adapters and no changes elsewhere.

Two continuous adapters live in `nwb_reader.py`. `NwbContinuousSource` reads one column of
an `ElectricalSeries` and normalizes it to microvolts. `NwbTimeSeriesSource` reads one
column of any other numeric `TimeSeries` in the acquisition. It normalizes the samples to
microvolts when the unit is in the volts family and keeps any other unit.

Four event adapters present NWB structures as event channels. `NwbUnitSource` flattens a
`Units` table into spike times, unit labels, and the per-unit `waveform_mean` snippets.
`NwbIntervalSource` reads any `TimeIntervals` table, with its tags as labels, its other
columns as JSON text, and its timeseries references as channels. `NwbEventsTableSource`
reads an `EventsTable`, and `NwbAnnotationSeriesSource` reads an `AnnotationSeries`.

`build_sources_from_nwb` finds every source in a fixed order and returns them with the
recording metadata that `recording_meta` reads. Waveforms carry no unit metadata in NWB
and are stored unscaled.

## Decision layer

`planning.py`, `sizing.py`, `counts.py`, and `columns.py` are pure functions with no I/O.
`planning.py` decides how many levels a continuous channel gets, numbered from 1, and each
level's bin count and `period_us`. `sizing.py` decides the inner chunk and outer shard
shapes for one array. Every member of a level takes the row counts of its `minmax` array.

`counts.py` plans the count levels of a dense event channel. Level 1 takes its period
from a ladder of `1000 * 4^j` microseconds, and each level above it spans 4x the time.
`columns.py` encodes the fixed-width `text` rows and the `channels` bitmask rows.

Keeping these separate from the write path means the format's arithmetic is unit-testable
without touching a store.

## Numeric core

`fold.py` reduces one level to the next over disjoint blocks of 4. A folded block is a
float64 statistics block with one column each for the minimum, maximum, mean, population
variance, and count of finite samples. `fold.py` also splits a block into the member
arrays a level stores and assembles one from them. See [the format spec](./bundle-format.md)
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
channels use that map to name the channels they apply to. It then writes each channel, consolidates
the metadata, writes `recording.json`, and publishes the bundle atomically.

`main.py` and `config.py` are the CLI and environment-config shell around it.
`config.py` resolves both invocation forms, positional arguments and the
`INPUT_DIR`/`OUTPUT_DIR` convention. After publishing, `properties.py` writes the
`asset-properties.json` sidecar that records the bundle's directory name.
