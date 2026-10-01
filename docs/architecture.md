# Architecture

Data flows one way through the writer: read, plan, stream and fold, write, publish. No
stage reaches back into an earlier one.

## Input adapter

`protocols.py` declares `ContinuousChannelSource` and `UnitChannelSource`. Each exposes
one channel's metadata and windowed reads (`read_samples`, `read_events`, and the rest).

NWB is the only input format so far. Everything downstream depends on the protocols, not
on NWB. The core is testable against in-memory sources, and a second input format costs
one new adapter and no changes elsewhere.

`nwb_series.py` holds the readers every adapter shares: the rate and start time of a
series, one channel's window as float64 or as microvolts, an electrode's id and display
name. Keeping them here lets each adapter live in its own module without importing
another.

`nwb_reader.py` holds the rate-sampled adapters and the discovery that picks them.
`NwbContinuousSource` reads one column of an `ElectricalSeries` and normalizes it to
microvolts. `NwbTimeSeriesSource` reads one column of any other numeric `TimeSeries` in
the acquisition, normalized to microvolts when its unit is in the volts family and kept
in its own unit otherwise. Unit-channel waveforms carry no unit metadata in NWB and are
stored unscaled. `build_sources_from_nwb` chooses an adapter per series: a rate picks
these, timestamps pick the one below.

`nwb_timestamped.py` holds `NwbTimestampedSource`, which presents a recording with breaks
in it as a uniform grid. A series sampled by timestamps rather than a rate is how NWB has
to express one, since a rate asserts unbroken regularity. The adapter reports the grid's
length as its sample count and NaN wherever nothing was recorded, so nothing downstream
needs gap awareness. See [gapped recordings](./gapped-recordings.md).

## Decision layer

`planning.py`, `sizing.py` and `grid.py` are pure functions with no I/O. `planning.py`
decides how many levels a channel gets and each level's shape and `period_us`.
`sizing.py` decides the inner chunk and outer shard shapes for one array. `grid.py` maps
a recording's timestamps onto the uniform grid they occupy, returning one row per
contiguous run rather than per sample, and measures the rate when none is supplied; it
imports neither NWB nor Zarr.

Keeping these separate from the write path means the format's arithmetic is unit-testable
without touching a store.

## Numeric core

`fold.py` reduces one level to the next over disjoint blocks of 4. See
[the format spec](https://github.com/Pennsieve/timeseries-zarr-paper/blob/main/bundle-format.md) for the exact rule and the NaN behavior.

A folded level travels as one float64 array of five columns: min, max, mean, the count of
raw samples behind the bin, and how many of those were finite. One array is what lets the streaming machinery carry
every statistic in a single pass without knowing what the columns mean, so raw is read
once however many statistics a level holds.

The count and the valid column are different numbers and both are needed. Count is time
support, the slots a bin spans, and it exists because a trailing partial bin holds fewer
than 4 samples and a plain mean of means would over-weight it. It is the one column that
never reaches disk: a level read back rebuilds it with `planning.bin_counts` from its own
number. Valid is how many of those slots held a finite sample, which is data rather than
arithmetic, so it is written and read back like the rest.

The arithmetic is float64 and narrows to float32 at the write. A mean is a sum, and
summing thousands of samples of a signal riding on a large DC offset is where float32
loses the part you wanted. The same concern is why a channel's `offset_uv` is subtracted
before anything is folded: `raw` keeps the offset, the statistics are relative to it, and
a reader adds it back in float64.

`streaming.py` drives the fold over a source one block at a time and buffers across block
boundaries so a run of 4 that straddles two blocks still folds correctly. Memory stays
bounded no matter how long the recording is.

## Write path

`write_continuous.py`, `write_unit.py` and `write_annotation.py` each write one channel by
composing the stages above. They hold the per-channel logic and no Zarr specifics.

Spikes and annotations are both event channels; the kind does not say which, and a view
reads the columns present to decide what it can draw. A spike channel carries events,
labels and waveforms; an annotation channel carries whichever of durations, labels,
values, bodies and channel refs its source offers. Bodies and channel refs are stored
Arrow-style, a flat array plus n+1 offsets, because the count per mark varies from zero
upward and sharing the boundary between neighbours makes a gap or an overlap
unrepresentable rather than merely invalid. `counts.py` holds the count pyramid a dense
event channel carries, streamed because its finest level does not fit in memory.

A continuous channel is the raw samples under `raw/`, then level groups keyed `1/`, `2/`
and so on, each carrying `period_us` and holding one array per statistic over a shared
bin axis: `env`, `mean` and `valid` today. All three are sized from one row geometry,
because a mean row is half an env row and a valid row half of that, and sizing them apart
would put them on different shard boundaries, where only one could be written a whole
shard at a time. Raw is not a level:
it carries no bin
arithmetic and no `period_us`, since the sample period is the channel's `rate_hz`.
Keeping numeric keys for levels alone is what lets a reader find them without inspecting
array shapes, and it is why a channel can omit `raw` and still be readable.

Both pick their read block so that every write covers a whole shard. A narrower write
makes the sharding codec read the shard back, re-encode every inner chunk, and rewrite
it, which costs about 10x the store traffic on a 16-chunk shard.

`zarr_io.py` is the only module that imports `zarr`. Everything else is Zarr-agnostic, so
the Zarr v3 API surface this package depends on sits in one file. It takes each array's
fill value from its dtype, NaN for floats and 0 for integers, rather than from the
caller: the fill is what Zarr serves for a chunk nobody wrote, and one float array
created without it would read back as a zero-volt flatline where a gap belongs.

`attrs.py` builds the attribute dicts, the format's only custom surface.

## Orchestration

`bundle.py` runs the whole job: assign channel indices, take the bundle's onset as the
earliest channel start, write each channel relative to it, consolidate metadata, write
`meta/`, publish atomically.

The order of the last three is load-bearing. A bundle's timeline is onset-relative and the
one wall-clock instant lives in `meta/session.start_us`, so deleting that directory
de-identifies the bundle. That only holds while `meta/` stays out of the root's
consolidated metadata, which is why it is written after consolidation and straight to the
filesystem rather than through the Group API.

`main.py` and `config.py` are the CLI and environment-config shell around it.
`config.py` resolves both invocation forms, positional arguments and the
`INPUT_DIR`/`OUTPUT_DIR` convention. After publishing, `properties.py` writes the
`asset-properties.json` sidecar that records the bundle's directory name.
