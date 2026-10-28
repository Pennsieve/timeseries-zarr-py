# TS-Zarr specification

Version 1.0 (draft)

## Abstract

This document specifies TS-Zarr, a format for storing one time-series recording,
such as EEG, intracranial EEG, or sorted spikes, as a Zarr v3 hierarchy with
precomputed statistics at several time resolutions. A browser-based viewer opens a
TS-Zarr bundle over HTTP range requests and renders any window at any zoom without
downloading the recording and without a server-side service. The format defines how
samples, per-bin statistics, events, and annotations are laid out, and the metadata a
reader needs to find and interpret them. Discussion of the specification takes place
in the issue tracker of this repository.

## Status of this document

This is the first versioned release of the specification. It supersedes the
unversioned format written by `timeseries-zarr-py` 1.x. The reference producer
provides migration tooling between numbered versions. Implementations of the released
version need not read data written to a draft of this version.

## Document conventions

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD",
"SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be
interpreted as described in RFC 2119.

Some of the JSON examples in this document include comments. They exist for clarity
only, and comments MUST NOT be included in JSON objects.

Dtypes are written with their Zarr v3 names: `float32`, `float64`, `int64`,
`uint16`, `uint32`, `uint8`. Every time value in a Zarr node is a count of
microseconds, and every key that holds one ends in `_us`. The datetimes in
`recording.json` are RFC 3339 strings. A *channel key* is the name of a channel
group: a decimal integer without leading zeros. Keys are opaque, their order carries
no meaning, and they need not be contiguous.

## Storage format

A bundle is one Zarr v3 group and its descendants, plus one optional plain JSON file.
The hierarchy is shown here as it would appear on a local disk, but it can equally be
stored on a web server and accessed over HTTP, or in object storage such as S3 or
GCS.

Zarr arrays hold every value a reader renders: samples, per-bin statistics, event
times, and per-event columns, including annotation text. Zarr attributes hold the
metadata that describes them. The one plain file, `recording.json`, holds the
recording's identity and its datetimes, which no reader needs in order to render.

### Bundles

```
<bundle>                     # One TS-Zarr bundle. A Zarr v3 group.
|-- zarr.json                # Root group metadata: the "tsz" attributes (see below)
|                            # and Zarr consolidated metadata for every descendant.
|-- recording.json           # Optional. Plain JSON, not a Zarr node: identity and
|                            # datetimes. Delete it to de-identify.
|-- 0                        # One group per channel, named by its channel key.
|   `-- ...                  # A continuous channel; see the next section.
|-- 1
|   `-- ...
`-- 7
    `-- ...                  # An event channel; see the section after.
```

One group per channel lets channels differ in sample rate, lets a producer write one
channel at a time, and lets a reader fetch only the channels on screen.

### Continuous channels

A continuous channel holds a regularly sampled signal of `N` samples and a ladder of
levels, each level coarser than the one before it by a whole number of samples per bin.

```
0                            # Channel group. Attributes: kind = "continuous", name,
|-- zarr.json                # id, rate_hz, offset_us, unit, provenance.
|-- raw                      # Array, shape (N,), float32. The samples.
|   |-- zarr.json
|   `-- c                    # Chunk objects: c/0, c/1, ... under Zarr's default
|                            # chunk key encoding, one object per shard.
|-- 1                        # Level 1 group. Attribute: period_us.
|   |-- zarr.json
|   |-- minmax               # Array, shape (bins, 2), float32. Per-bin min and max.
|   |   |-- zarr.json
|   |   `-- c                # c/0/0, c/1/0, ... The pair axis is not chunked
|   |                        # (see Array storage).
|   |-- mean                 # Array, shape (bins,), float32. Optional.
|   |-- var                  # Array, shape (bins,), float32. Optional.
|   `-- count                # Array, shape (bins,), uint32. Optional.
|-- 2                        # Level 2 group. Attribute: period_us.
`-- K                        # Level K group. Attribute: period_us.
```

| array | requirement | shape | dtype | description |
|---|---|---|---|---|
| `raw` | MUST | `(N,)` | float32 | The samples. Sample `i` is at time `offset_us + i * 1e6 / rate_hz`. A non-finite sample is a gap. |
| `<k>/minmax` | MUST per level | `(bins, 2)` | float32 | Column 0 the minimum and column 1 the maximum of the samples in bin `i`. |
| `<k>/mean` | MAY | `(bins,)` | float32 | Arithmetic mean of the samples in bin `i`. |
| `<k>/var` | MAY | `(bins,)` | float32 | Population variance of the samples in bin `i`. |
| `<k>/count` | MAY | `(bins,)` | uint32 | Number of finite samples in bin `i`. |

Level `k`, for `k = 1, 2, ...`, is a child group of the channel named `k`. The name is
an ordinal. The `period_us` attribute defines the level (see
[Level metadata](#level-metadata)). The bin width of level `k` is
`s_k = period_us * rate_hz / 1e6` samples. It MUST be an integer of at least 2 and less
than `2^32`, so that `count` fits in a uint32. Bin `i` of level `k` covers the samples
with indices in `[s_k * i, s_k * (i + 1))` and starts at time `offset_us + i * period_us`.
A level has `ceil(N / s_k)` bins, and the last bin may cover fewer than `s_k` samples.

`s_(k+1)` MUST be an integer multiple of `s_k` and at least `2 * s_k`. The bins of
level `k + 1` then nest exactly over the bins of level `k`, and a reader derives the
factor between two levels from their `period_us` values, never from the level names. A
factor of 4 between every pair of levels is RECOMMENDED, which gives
`s_k = 4^k`. The factor MAY differ from level to level.

Levels MUST be numbered `1` through `K` with no gaps. A producer SHOULD add levels
until the next level would hold fewer bins than a threshold of its choice; a threshold
of 1024 bins is RECOMMENDED. A channel with no samples holds a zero-length `raw` and no
levels.

Each statistic is defined over the raw samples of its bin. A producer MAY compute
level `k + 1` from level `k` instead: the minimum of minimums, the maximum of
maximums, the count-weighted mean of means, the sum of counts, and for `var` the
count-weighted mean over the children of the child variance plus the squared
difference between the child mean and the parent mean. The results are identical in
exact arithmetic. Producers SHOULD accumulate in float64 and store float32.

If any sample in a bin is non-finite, `minmax`, `mean`, and `var` of that bin MUST be
NaN, and `count` holds the number of finite samples. A reader treats NaN as a gap. A
fold that skipped non-finite samples would average different time supports across
channels and break the montage identity.

`minmax` serves envelope rendering. `mean` serves montage and filtered views, which need
it. `var` serves intensity trends such as RMS envelopes and variance checks; the RMS
of a bin is `sqrt(var + mean^2)`.

`count` is the bin's coverage. Because one non-finite sample makes a bin NaN at every
level above it, `minmax`, `mean`, and `var` cannot say how much of a NaN bin was
recorded. `count` can: a bin with zero finite samples is a dropout, and a bin short by
one is a single bad sample, which a reader may resolve from a finer level or from
`raw`. Read at the coarsest level and divided by the bin width, `count` is the
fraction of each bin that was recorded. A coverage view of a long timeline costs one
request. Unwritten chunks read as zero, which is the correct coverage for a span that
was never recorded.

With a constant factor `f` between levels, `raw` plus `minmax` stores
`N * (f + 1) / (f - 1)` values, and each of `mean`, `var`, and `count` adds
`N / (f - 1)`. At the recommended factor of 4 that is `5N / 3` and `N / 3`.

### Event channels

An event channel holds a sorted series of event times and optional per-event
columns. It covers spike times with cluster labels and waveforms, detector output
with scores, and clinical marks with durations and free text. Every column has length
`n`, and index `i` refers to the same event in every column.

```
7                            # Channel group. Attributes: kind = "event", name, id,
|-- zarr.json                # unit, label_names, max_duration_us, provenance.
|-- events                   # Array, shape (n,), int64. Event times.
|-- durations                # Array, shape (n,), int64. Optional.
|-- labels                   # Array, shape (n,), uint16. Optional.
|-- values                   # Array, shape (n,), float32. Optional.
|-- text                     # Array, shape (n, W), uint8. Optional.
|-- channels                 # Array, shape (n, M), uint8. Optional.
|-- waveforms                # Array, shape (n, P), float32. Optional.
|                            # Attributes: period_us, event_sample.
|-- waveform_mean            # Array, shape (L, P), float32. Optional.
|                            # Attributes: period_us, event_sample.
|-- 1                        # Count level 1 group. Optional. Attributes: period_us,
|                            # offset_us.
|   |-- zarr.json
|   `-- count                # Array, shape (bins, L), uint32.
`-- J                        # Count level J group.
```

| array | requirement | shape | dtype | description |
|---|---|---|---|---|
| `events` | MUST | `(n,)` | int64 | Time of each event in microseconds from the reference time, non-decreasing. May be negative. |
| `durations` | MAY | `(n,)` | int64 | Duration of each event in microseconds, at least 0. 0 is a point event. |
| `labels` | MAY | `(n,)` | uint16 | Category of each event, such as a cluster id or an event type. |
| `values` | MAY | `(n,)` | float32 | One measurement per event, in the channel's `unit`. |
| `text` | MAY | `(n, W)` | uint8 | Free text of each event as UTF-8, zero-padded to width `W`. |
| `channels` | MAY | `(n, M)` | uint8 | Bitmask of the channel keys each event applies to. |
| `waveforms` | MAY | `(n, P)` | float32 | `P` samples around each event. |
| `waveform_mean` | MAY | `(L, P)` | float32 | Mean snippet of the events whose label is `j`, for each label `j`. Equals the mean of the `waveforms` rows when `waveforms` exists. Requires `labels`. |
| `<k>/count` | MUST per count level | `(bins, L)` | uint32 | Number of events in bin `i`, per label category. |

Equal times are allowed. A reader finds a window by binary search over `events`.
When `durations` exists, the channel's `max_duration_us` attribute MUST bound every
duration, and a reader extends its search back by that bound to find intervals that
start before the window and are still open. The bound MAY be loose.

`labels` values index the channel's `label_names` attribute when it exists. `L`, the
number of categories, is the length of `label_names` when it exists and otherwise one
more than the largest label. `L` is 1 for a channel without `labels`.

Row `i` of `text` holds the UTF-8 bytes of event `i`'s text followed by zero bytes up
to width `W`. The producer sets `W` to the length of the longest text in the channel.
A row of zero bytes is an empty text. The content is free: a clinical note, or a JSON
object a detector emits per event.

Row `i` of `channels` names the channels event `i` applies to. Bit `k mod 8` of byte
`floor(k / 8)`, least significant bit first, is set when channel key `k` is included.
`M` is `ceil((max_key + 1) / 8)`, where `max_key` is the largest channel key in the
bundle. A row of zero bytes means the event applies to the whole recording.

A legend or a unit summary reads `waveform_mean`, one small array, in place of
every snippet in the channel.

A dense event channel MAY hold count levels, numbered `1` through `J` with no gaps.
Level 1 has a producer-chosen `period_us`, and `period_us` of level `k + 1` MUST be an
integer multiple of that of level `k`, at least 2 times it. The bins of level `k + 1`
then nest exactly over the bins of level `k`. A factor of 4 between every pair of
levels is RECOMMENDED. When the factor is 4, level 1 SHOULD lie on the ladder
`1000 * 4^j` microseconds (1 ms, 4 ms, 16 ms, ..., 1.024 s, 4.096 s, ...). Event
channels in any bundle then share bin grids. Bin `i` covers the times
`[offset_us + i * period_us, offset_us + (i + 1) * period_us)`, where `offset_us` is
the time of the level's bin 0 (see [Level metadata](#level-metadata)), and a level's
bins cover every event of the channel. An event with a duration is counted in the bin of
its onset only. The count of a bin at level `k + 1` is the sum of the level-`k` bins
it covers. A rate view from any level is exact. A producer SHOULD
write count levels for channels with more than about 64,000 events and SHOULD choose
level 1 as the finest bin a reader needs once the window is too wide to serve from
`events` directly.

## TS-Zarr metadata

TS-Zarr metadata is stored in the `attributes` of the `zarr.json` files throughout the
hierarchy above, and in the plain file `recording.json`. In every `zarr.json` that
carries it, the `attributes` MUST contain a key `tsz`, and the value of the `tsz` key
MUST be a JSON object. The root group's `tsz` object MUST contain a `version` key, the
value of which MUST be a string specifying the version of this specification. No
other node carries `version`.

Readers MUST ignore keys inside `tsz` that they do not recognize, and MUST ignore
child arrays and groups they do not recognize. Other conventions MAY add their own
keys to `attributes` beside `tsz`.

### Timeline

Every time value in a bundle is an offset from the reference time, t = 0, and may be
negative. A producer places the reference time at the earlier of the acquisition
start and the first sample when it first writes a recording, and keeps it when it
writes the recording again, so a channel added later may begin before it. Sample and
event times are `int64` microseconds. Periods are `float64` microseconds.
`reference_time` in `recording.json` gives the reference time as a datetime,
when it is known, and no other object holds one. No array carries a time axis; a
reader computes times from `offset_us`, `rate_hz`, and `period_us`.

### Root metadata

| key | requirement | type | description |
|---|---|---|---|
| `version` | MUST | string | Version of this specification, `"1.0"`. |

The root `zarr.json` MUST also carry consolidated metadata in the form that
zarr-python 3 writes: a `consolidated_metadata` object with `"kind": "inline"`,
`"must_understand": false`, and a `metadata` map from the path of every descendant
node to the content of its `zarr.json`. A group's entry also carries a nested
`consolidated_metadata` object for its own descendants, as zarr-python writes it.
One fetch then yields the whole tree. The Zarr v3 core specification permits this
field with exactly this structure, and a formal extension is in review; zarr-python
and zarrita read it. `recording.json` is not a Zarr node and does not appear in it.

```json
{
  "zarr_format": 3,
  "node_type": "group",
  "attributes": {
    "tsz": {"version": "1.0"}
  },
  "consolidated_metadata": {
    "kind": "inline",
    "must_understand": false,
    "metadata": {
      "0": {},          // elided: the full zarr.json of each node
      "0/raw": {},
      "0/1": {},
      "0/1/minmax": {}
    }
  }
}
```

### Channel metadata

Each channel group's `tsz` object describes the channel.

| key | requirement | type | description |
|---|---|---|---|
| `kind` | MUST | string | `"continuous"` or `"event"`. |
| `name` | MUST | string | Display label, for example an electrode label. |
| `id` | MAY | string | Producer identifier, stable across exports. |
| `unit` | MUST for `continuous`, MAY for `event` | string | Unit of `raw` samples, or of the `values` column. |
| `rate_hz` | MUST for `continuous` | number | Sample rate, greater than 0. |
| `offset_us` | MAY for `continuous` | integer | Time of sample 0 from the reference time. May be negative. Default 0. |
| `label_names` | MAY for `event` | array of string | Name of each label category, indexed by label value. |
| `max_duration_us` | MUST when `durations` exists | integer | Upper bound on every value in `durations`. |
| `provenance` | MAY | object | What produced the channel, such as detector name and version. Content is producer-defined. |

Voltage channels MUST store microvolts and declare `"uV"`. Other channels keep the
unit of the source and declare it. `name`, `id`, and `provenance` MUST NOT identify a
person (see Recording metadata).

```json
{
  "zarr_format": 3,
  "node_type": "group",
  "attributes": {
    "tsz": {
      "kind": "continuous",
      "name": "LA4",
      "id": "N:channel:7f3a",
      "rate_hz": 2000.0,
      "offset_us": 0,
      "unit": "uV"
    }
  }
}
```

```json
{
  "zarr_format": 3,
  "node_type": "group",
  "attributes": {
    "tsz": {
      "kind": "event",
      "name": "seizure marks",
      "label_names": ["onset", "offset"],
      "max_duration_us": 600000000,
      "provenance": {"software": "review-tool 3.2"}
    }
  }
}
```

### Level metadata

Each level group's `tsz` object, in continuous channels and in event channels alike,
holds the bin width. A count level also holds the time of its bin 0.

| key | requirement | type | description |
|---|---|---|---|
| `period_us` | MUST | number | Width of one bin in microseconds. |
| `offset_us` | MUST for count levels, MUST NOT for continuous levels | integer | Time of bin 0 in microseconds from the reference time. May be negative. |

For level `k` of a continuous channel, `period_us` MUST equal `s_k * 1e6 / rate_hz`,
where `s_k` is the integer bin width in samples defined in
[Continuous channels](#continuous-channels). For count levels, `period_us` of level
`k + 1` MUST be an integer multiple of that of level `k`, at least 2 times it, and
`offset_us` MUST be an integer multiple of `period_us` and at most the time of the
channel's first event.

```json
{
  "zarr_format": 3,
  "node_type": "group",
  "attributes": {
    "tsz": {"period_us": 2000.0}
  }
}
```

A count level whose events begin 32 ms before the reference time:

```json
{
  "zarr_format": 3,
  "node_type": "group",
  "attributes": {
    "tsz": {"period_us": 16000.0, "offset_us": -32000}
  }
}
```

### Waveform metadata

The `waveforms` and `waveform_mean` arrays each carry a `tsz` object.

| key | requirement | type | description |
|---|---|---|---|
| `period_us` | MUST | number | Sample period of the snippet in microseconds. |
| `event_sample` | MAY | integer | Index of the snippet sample recorded at the event time. Default 0. |

```json
{
  "zarr_format": 3,
  "node_type": "array",
  "shape": [8192, 48],
  "data_type": "float32",
  "attributes": {
    "tsz": {"period_us": 33.333333333333336, "event_sample": 16}
  }
  // chunk_grid, codecs, fill_value, dimension_names as for any array
}
```

### Recording metadata

A bundle MAY hold a plain JSON file `recording.json` at its root. It is the one place
for structured identity and for datetimes. It is UTF-8 JSON with the following
keys.

| key | requirement | type | description |
|---|---|---|---|
| `reference_time` | MAY | string | The reference time, t = 0 of every time in the bundle, as a datetime, RFC 3339 with a UTC offset. A reader adds a time in the bundle to it to get a datetime. |
| `subject` | MAY | object | Who was recorded. Content is producer-defined. |
| `session` | MAY | object | The recording session. `start_time`, when present, is the datetime acquisition began, RFC 3339 with a UTC offset. Other content is producer-defined. |
| `source` | MAY | object | Acquisition system and converting software. Content is producer-defined. |

`session.start_time` describes the session and places no data. In the usual case it
equals `reference_time`. It is later when the first sample precedes the acquisition
start, and it differs when a recording written again keeps its earlier reference
time.

Producers MUST keep identity and every datetime out of every other object: channel
`name`, `id`, and `provenance` MUST NOT identify a person, and no other key holds a
date. Identity may then appear in two places only, `recording.json` at its fixed path
and the `text` columns of event channels, which the root `zarr.json` enumerates.
Deleting `recording.json` removes structured identity and the date and changes no
other object. Free text is redacted in place: a `text` column stored as the next
section recommends is readable text on disk, and overwriting bytes within a row
changes no other object. A date shift adds the same amount to `reference_time` and
`session.start_time`. Every other time in the bundle is relative to `reference_time`,
so the shift changes no other object.

```json
{
  "reference_time": "2024-03-14T09:15:00.000000-05:00",
  "subject": {"id": "sub-0931", "species": "Homo sapiens", "sex": "F", "age": "P34Y"},
  "session": {"id": "ses-01", "start_time": "2024-03-14T09:15:00.000000-05:00", "description": "EMU admission, day 3"},
  "source": {"system": "Natus Quantum", "converter": "timeseries-zarr-py 2.0.0"}
}
```

## Array storage

`raw` and the float arrays in level groups MUST declare `"NaN"` as their
`fill_value`. Integer arrays MUST declare 0. A producer MAY omit a chunk whose
elements all equal the fill value, and a reader MUST read a missing chunk as fill,
which is what Zarr defines for the uninitialized portion of an array. A sparse
timeline then stores only its recorded spans. A store that answers a missing key with
an authorization error instead of not-found defeats this; a producer that deploys to
such a store writes every chunk.

Producers SHOULD compress with Zstd and shard with the `sharding_indexed` codec. The
inner chunk SHOULD span at most 8192 elements along the first axis. A shard SHOULD
group whole chunks up to about 16 MiB. Trailing axes (the pair of `minmax`, the `L` of
`count`, the `W` of `text`, the `M` of `channels`, the `P` of `waveforms`) SHOULD NOT
be chunked. The inner chunk is the smallest unit a reader fetches, so its width sets
the floor on what a read transfers. A reader that picks the level whose `period_us`
matches one pixel reads a few thousand bins at any zoom. The compression level is a
producer choice and not part of the format.

`text` arrays SHOULD omit the compression codec. The stored object is then readable
text, row after row, with the shard index at its end, and a text tool can audit or
redact it. A producer MAY compress a `text` array whose padding is large, at the cost
of that property for that channel.

`float32` is the dtype of every sample and of every statistic except `count`: a
viewer quantizes to pixels, so float64 precision reaches no screen. Times are
`int64` microseconds.

Arrays SHOULD declare Zarr `dimension_names`: `"time"` for a sample or bin axis,
`"event"` for a per-event axis, `"label"` for the `L` axis of `waveform_mean`, and
`null` for the trailing axes. Generic tools then open a level group as an aligned
per-bin dataset.

```json
{
  "zarr_format": 3,
  "node_type": "array",
  "shape": [900000, 2],
  "data_type": "float32",
  "fill_value": "NaN",
  "dimension_names": ["time", null],
  "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [262144, 2]}},
  "chunk_key_encoding": {"name": "default"},
  "codecs": [{
    "name": "sharding_indexed",
    "configuration": {
      "chunk_shape": [8192, 2],
      "codecs": [{"name": "bytes", "configuration": {"endian": "little"}},
                 {"name": "zstd", "configuration": {"level": 3, "checksum": false}}],
      "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}},
                       {"name": "crc32c"}],
      "index_location": "end"
    }
  }]
}
```

## Validation

A conforming bundle satisfies all of the following, and a validator checks them.

- The root group's `tsz.version` is `"1.0"`, and its `consolidated_metadata` lists
  every Zarr node in the bundle.
- Every child group of the root is a channel group with a valid `kind` and the
  attributes that `kind` requires.
- Level groups are named `1` through `K`, or `1` through `J` for count levels, with
  no gaps, and each carries `period_us`. A count level also carries `offset_us`, and a
  continuous level does not.
  Continuous levels satisfy `period_us = s_k * 1e6 / rate_hz`, where `s_k` is an
  integer of at least 2 and less than `2^32`, `s_(k+1)` is an integer multiple of
  `s_k`, and `s_(k+1) >= 2 * s_k`. Count levels satisfy `period_us(k + 1)` an integer
  multiple of `period_us(k)`, at least 2 times it.
- Every channel holds the arrays its tables require: `raw` for a continuous channel,
  `events` for an event channel, `minmax` in every continuous level, and `count` in
  every count level.
- Every array has the shape and dtype of its table, and all members of one level
  have the same number of bins, `ceil(N / s_k)` at level `k` of a continuous channel.
- `raw` and the float level members declare NaN fill. Integer arrays declare 0.
- `events` is non-decreasing. Every per-event column has length `n`. `durations`
  values lie between 0 and `max_duration_us`. `labels` values are below `L`.
- `waveforms` and `waveform_mean` carry `period_us`. `waveform_mean` requires
  `labels`.
- Every row of `text` is valid UTF-8 followed by zero bytes. `M` equals
  `ceil((max_key + 1) / 8)`, and every set bit in `channels` names a channel group
  in the bundle.
- Each count level's `offset_us` is a multiple of its `period_us` and at most the
  channel's first event.
- `recording.json`, when present, parses as JSON, and `reference_time` and
  `session.start_time` parse as RFC 3339.
- `name`, `id`, and `provenance` values match no identity-like pattern (warning).

## Versioning and compatibility

`version` is `major.minor`. A minor release adds optional attribute keys, level
members, per-event columns, or files. A reader written for `1.0` reads any `1.x`
bundle by ignoring what it does not recognize. A major release may change the
meaning, node type, or shape of an existing path. Readers MUST reject a bundle whose
major version they do not implement.

Under a channel group, names that are positive integers are reserved for level groups.

A bundle is a plain directory, so it fits dataset conventions such as BIDS as a
derivative beside the recording it renders.

## Specification naming style

Multi-word keys use snake_case, as Zarr's own metadata does (`zarr_format`,
`fill_value`, `dimension_names`). Keys that hold a count of microseconds end in
`_us`; keys that hold a rate end in `_hz`. Array and group names are lowercase.

## Implementations

`timeseries-zarr-py` is the reference producer. `@pennsieve/timeseries-zarr-reader` is
the reference reader. Implementations MUST
state the versions of this specification they support.

## Citing

Cite this specification by its version and the repository that publishes it,
`timeseries-zarr-py`.

## Version history

| revision | date | description |
|---|---|---|
| 1.0 (draft) | 2026-10-05 | First versioned release. Namespaced attributes, level groups with `minmax`, `mean`, `var`, `count`, event channels with `text` and `channels` columns, `recording.json`. |

## Conformance

All of the text of this specification is normative except sections explicitly marked
as non-normative, examples, and notes. The key words defined in Document conventions
carry their RFC 2119 meaning throughout.
