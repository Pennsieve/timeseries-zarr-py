# Gapped recordings

How the writer handles a recording with breaks in it.

## What fails today

`nwb_reader.py` rejects any series without a fixed rate:

```
ValueError: irregular sampling is not supported:
ElectricalSeries has timestamps and no rate
```

An NWB `TimeSeries` carries its timing one of two ways, and they are mutually exclusive:
`starting_time` + `rate`, or one `timestamps` value per sample. A recording with breaks
in it has to use timestamps, because `rate` asserts unbroken regularity. Claim it and
every sample after the first break sits at the wrong time.

So the writer handles clean continuous recordings and refuses everything else. Clinical
recordings are not clean. A real MEF session measured on this pipeline:

| | |
|---|---|
| Samples | 251,999,744 across 4 channels |
| Rate | 512 Hz exactly |
| Wall-clock span | 6.1 days |
| Recorded | 5.7 days |
| Breaks | 8, from 5 s to 4.8 h |
| Total gap | 9.6 h |

Eight amplifier stops across a six-day admission: four short disconnects, a seven-minute
intervention, two multi-hour absences. That is the normal shape of an epilepsy
monitoring unit stay, and none of it can be written today.

## The approach

`ContinuousChannelSource` in `protocols.py` is a Protocol. The writer consumes anything
exposing `rate_hz()`, `start_us()`, `num_samples()` and `read_samples(start, stop)`.

So this is a new source adapter, not a change to the write path. The adapter wraps a
timestamped series and presents it as a regular one: `num_samples()` returns the length
of the uniform grid rather than the sample count, and `read_samples()` returns a grid
window with NaN wherever nothing was recorded.

Nothing downstream changes. `planning.py`, `sizing.py`, `streaming.py`,
`write_continuous.py` and `fold.py` already know how to write a source with a rate and N
samples, and `fold.py` already propagates NaN into the bin holding it.

For the measured recording above, the grid is about 269 M slots holding 252 M samples;
the 17 M slots in between are the eight breaks.

## New modules

```
grid.py               pure. no NWB, no zarr, no I/O.
nwb_series.py         readers both adapters share
nwb_timestamped.py    the source adapter
nwb_reader.py         one branch in build_sources_from_nwb
```

`nwb_series.py` exists because the timestamped adapter reads a series exactly as the
rate-sampled one does; only where a sample sits on the time axis differs. Without it the
two adapters would have to import each other.

### `grid.py`

Holds the arithmetic, importable without opening an NWB file.

```python
@dataclass(frozen=True, slots=True)
class Segment:
    """One contiguous run of recorded samples on the grid."""
    grid_start: int
    grid_stop: int
    sample_start: int
    sample_stop: int


def build_segments(timestamps, rate_hz) -> list[Segment]:
    """Map each sample to a grid slot and group the result into runs."""


def grid_length(segments) -> int:
    """Total slots spanned, gaps included."""


def derive_rate_hz(timestamps) -> float:
    """Rate from the span of the longest gap-free run."""
```

`derive_rate_hz` lives here and is never called from here. It is a tool the caller may
choose; see below.

### `nwb_timestamped.py`

```python
class NwbTimestampedSource:
    def __init__(
        self,
        electrical_series: ElectricalSeries,
        channel_index: int,
        session_start_time: datetime,
        *,
        rate_hz: float,
    ) -> None:
```

`rate_hz` is keyword-only and has no default. Every construction site states the rate, so
where the number came from is greppable.

`read_samples` must never allocate the grid. It fills a window with NaN, finds the
segments overlapping it, translates each into a sample range, reads those and places
them:

```python
def read_samples(self, start, stop):
    out = np.full(stop - start, np.nan, dtype=np.float32)
    for seg in self._segments_overlapping(start, stop):
        ...
    return out
```

The segment table has one row per contiguous run: nine rows for the recording above,
and nine for a six-month one. It is proportional to the number of breaks, not the number
of samples, so `_segments_overlapping` is a bisect over a short list. Memory stays
bounded by the window, which is what `streaming.py` already assumes.

## Where the rate comes from

The adapter does not infer it. In descending order of trust:

1. **The source format's header.** MEF carries the sampling frequency; the MEF converter
   has it. Authoritative when the chain is MEF.
2. **Configuration**, for a pipeline whose rate is known.
3. **`derive_rate_hz(timestamps)`**, called by name at the call site so the fallback is
   visible in the code and in the log line.

`build_sources_from_nwb` picks, records which it used, and passes the number down.

### Do not derive the rate from the median interval

Measured on the recording above, a gap-free stretch alternates between exactly two
intervals in equal numbers:

```
dt = 0.001953000 s    999,999 times    -> 512.032770 Hz
dt = 0.001953250 s  1,000,000 times    -> 511.967234 Hz
```

Their mean is `0.001953125`, which is exactly 1/512. The timestamps are dithered between
two neighboring representable values whose average is the true interval.

The median picks one of them and returns 511.967234 Hz. Over the longest segment that
misplaces samples by about 5 seconds; across the full span, about 34. Invisible on a
waveform, fatal for an annotation.

Take the rate as the span of a gap-free run divided by its sample count less one. On the
first two million samples of that file it returns 512.0 exactly.

## Reject, do not resample

Two conditions fail the run rather than producing a bundle:

- **Two samples landing on the same grid slot.** The rate is too low. This also catches
  genuinely irregular data, whose jittered intervals collide.
- **The timestamps implying a different rate than the one given.** The recording is
  irregular, or the rate is wrong.

A gap needs no threshold once the rate is right: it is a sample whose grid slot is more
than one past its predecessor.

The second check earns its keep only when the rate comes from outside the timestamps. A
rate too low is caught by collisions; a rate too *high* is not, because every sample then
lands two or more slots on, which segmentation alone cannot tell from a recording that is
nothing but gaps. While `build_sources_from_nwb` derives the rate from the timestamps it
is comparing a number with itself and always passes. Supplying the rate from the source
format's header, which MEF carries and the converter currently drops, is what turns it
into a real cross-check.

## Tests

`grid.py` takes plain arrays, so most of these need no NWB fixture.

- No gaps: pass-through; grid length equals sample count.
- One gap: grid longer than the sample count, hole reads back NaN, two segments.
- A window wholly inside a gap: all NaN, no underlying read.
- A window straddling a boundary: split correctly.
- Dithered timestamps: `derive_rate_hz` returns 512.0 where the median returns
  511.967234. This is the regression test for the easiest mistake to reintroduce.
- Irregular spacing: raises.
- Two samples on one slot: raises.

## Out of scope

The adapter writes NaN as ordinary stored values. It does not depend on `fill_value` or
on `write_empty_chunks`, and it changes neither.

Omitting the chunks that fall wholly inside a gap is a separate change with its own
argument. That is the sparse storage the format spec describes, where an absent chunk is
served as the fill value. It needs two things this one does not:

- `fill_value` declared as NaN. `zarr_io.create_array` passes none, so float arrays take
  zarr-python's default of `0.0`; an omitted chunk would read back as a zero-volt
  flatline rather than as no data.
- The `write_empty_chunks: True` decision revisited. It is there because an object store
  that answers a missing key with an authorization error rather than a not-found status
  turns an absent chunk into a failed read.

Those two were a matched pair. `fill_value` is now declared as NaN, so the first half is
done and only the `write_empty_chunks` decision is left.

**What it is waiting on is not code.** `write_empty_chunks: True` is there because an
object store that answers a missing key with an authorization error rather than a
not-found turns an absent chunk into a failed read. On S3 a `GetObject` for a key that
does not exist returns 403 unless the caller holds `s3:ListBucket` on the bucket, and 404
only if it does. A browser reading a public bundle through a CDN is exactly that case, and
it fails hard rather than reading as a gap.

So the open question is: **on the bucket these bundles are served from, does a missing key
answer 404 or 403?** Once that is 404, the flip is one line in `zarr_io.create_array`.
Worth checking at the same time: how `write_empty_chunks=False` interacts with the
sharding codec, since whether an all-fill inner chunk is dropped inside a shard, or only a
wholly empty shard is skipped, decides what the saving actually is.

For the measured recording the saving would be a few kilobytes: 9.6 h of gap in a 146 h
span, and Zstd flattens a constant NaN run to almost nothing. Sparse storage earns
its keep on a genuinely sparse timeline, not on this one.
