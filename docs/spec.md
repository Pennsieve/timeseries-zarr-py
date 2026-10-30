# TS-Zarr specification

Version 1.0 (draft)

## Abstract

This document specifies TS-Zarr, a layout for storing one time-series recording, such
as EEG, intracranial EEG, or sorted spikes, as a Zarr v3 hierarchy with per-bin
statistics at several bin widths. It defines how samples, per-bin statistics,
events, and annotations are laid out, the metadata that describes them, and the
checks that establish whether a hierarchy conforms. Discussion of the specification
takes place in the issue tracker of the repository that publishes it.

## Status of this document

This is the first versioned release of the specification. It supersedes the
unversioned format written by `timeseries-zarr-py` 1.x.

## Table of contents

1. [Introduction](#1-introduction)
2. [Conventions and terminology](#2-conventions-and-terminology)
3. [Bundle structure](#3-bundle-structure)
4. [Continuous channels](#4-continuous-channels)
5. [Event channels](#5-event-channels)
6. [Metadata](#6-metadata)
7. [Storage](#7-storage)
8. [Conformance](#8-conformance)
9. [Versioning and compatibility](#9-versioning-and-compatibility)
10. [Privacy considerations](#10-privacy-considerations)
11. [Implementations](#11-implementations)
12. [References](#12-references)

Appendix A. [Examples](#appendix-a-examples)
Appendix B. [Version history](#appendix-b-version-history)

## 1. Introduction

This section is informative.

A TS-Zarr bundle holds one recording. A recording has continuous channels and event
channels. A continuous channel is a regularly sampled signal together with a ladder of
levels, each level holding the minimum, maximum, and count of the finite samples in
consecutive bins of a fixed width, and optionally their mean and variance. Each level is coarser than the one
before it. An event channel is a time-sorted series of events with optional per-event
columns: a duration, a label, a value, a text, a set of channels, a waveform. An event channel
may hold levels of per-bin event counts.

Samples, and every statistic except `count`, are stored as `float32`. A bundle is a derivative of a
recording made for rendering at any time scale. It is not an archive of the
recording, and a producer that needs the source data keeps the source.

The layout is a Zarr v3 hierarchy and uses two Zarr features beyond the core
specification: consolidated metadata in the root group and the `sharding_indexed`
codec. Any Zarr v3 implementation that supports the codecs in section 7.2 can read
every array of a bundle. The metadata in section 6 gives the arrays their meaning.

## 2. Conventions and terminology

### 2.1. Requirement levels

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD",
"SHOULD NOT", "RECOMMENDED", "NOT RECOMMENDED", "MAY", and "OPTIONAL" in this
document are to be interpreted as described in BCP 14 [RFC2119] [RFC8174] when, and
only when, they appear in all capitals, as shown here.

Some of the JSON examples in this document include comments. They exist for clarity
only. Comments are not part of JSON and MUST NOT appear in a bundle.

### 2.2. Terms

**bundle**: One Zarr v3 group, its descendants, and one optional plain JSON file,
`recording.json`. A bundle holds one recording.

**channel**: A child group of the root group whose name is an integer name. A channel
is continuous or event.

**channel index**: The integer that names a channel group. A bundle with `C` channels
names them `0` through `C - 1`.

**integer name**: A name that matches the pattern `0|[1-9][0-9]*`: a decimal integer
without leading zeros.

**continuous channel**: A channel holding a regularly sampled signal and a ladder of levels (section 4).

**event channel**: A channel holding a time-sorted series of events and their columns (section 5).

**reference time**: The instant from which every time in a bundle is measured, `t = 0`.

**time**: A signed number of microseconds from the reference time. A time stored in
an array or in an attribute is an integer (section 2.3). A grid time and the time of
a sample within a run need not be integers.

**sample period**: `1e6 / rate_hz` microseconds, the spacing of consecutive samples of a continuous
channel.

**grid**: The sequence of times `offset_us + i * 1e6 / rate_hz` for `i = 0, 1, ...`, the
nominal time of each sample of a continuous channel.

**run**: A sequence of consecutive samples of a continuous channel recorded one sample
period apart.

**finite sample**: A sample that is neither NaN nor an infinity.

**level**: A child group of a channel holding arrays with one row per bin. Levels are named
by ordinals `1`, `2`, ..., and level `k + 1` has wider bins than level `k`.

**bin**: A span of consecutive samples (continuous channel) or of time (event channel)
that one row of a level's arrays describes.

**event**: One entry of an event channel: a time and the values of the channel's columns at
the same row index.

**point event**: An event with no duration, or a duration of 0. It occupies the instant of its time.

**interval**: An event with a duration `d > 0`. It covers the half-open span `[t, t + d)`.

**producer**: Software that writes a bundle.

**reader**: Software that reads a bundle.

**validator**: Software that checks a bundle against section 8.

### 2.3. Numeric conventions

Dtypes are written with their Zarr v3 names: `float32`, `int64`, `uint16`, `uint32`,
`uint8`.

Every time stored in a bundle, in an array or in an attribute, is an integer number
of microseconds from the reference time. Keys that hold microseconds end in `_us`. A
time MAY be negative. Every time, every duration, the end `t + d` of every interval,
the time of every sample (section 4.2), and the end of every count-level bin MUST lie
in the open interval `(-2^53, 2^53)`. In an array, a time is `int64`. In an attribute, a
time is a JSON integer. A JSON integer is a JSON number written without a fraction
part and without an exponent. `period_us` of a continuous level and of a waveform
array is a JSON number. Every other `_us` key holds a JSON integer.

Attribute values are JSON values [RFC8259]. NaN and infinities are not JSON values
and MUST NOT appear in an attribute.

A formula over integers only, such as the bin count of section 5.7, is evaluated
exactly, with `floor` and `ceil` applied to the exact quotient. Every other formula
for a time, a bin width, or a comparison is evaluated in IEEE 754 binary64
arithmetic with the usual operator precedence, operators of equal precedence applied
left to right, and one binary64 rounding per operation, so `i * 1e6 / rate_hz` is
`(i * 1e6) / rate_hz` and `offset_us + i * 1e6 / rate_hz` is
`offset_us + ((i * 1e6) / rate_hz)`. Every comparison in this document uses those
results. The statistics of sections 4.4 and 5.6 are defined over the real numbers,
and section 8.4 gives the tolerance within which a stored value matches them. "The
nearest integer to `x`" means the integer `m` that minimizes `|x - m|`; when two
integers are equally near, `m` is the larger. Where a requirement states a
tolerance, `|a - b| <= tol * |b|` is written "`a` is within `tol` relative of `b`".

The datetimes in `recording.json` are [RFC3339] strings with a UTC offset.

### 2.4. Naming

Multi-word attribute keys use `snake_case`, as Zarr's own metadata does
(`zarr_format`, `fill_value`, `dimension_names`). Keys that hold microseconds end in
`_us`. Keys that hold a rate end in `_hz`. Array and group names
are lowercase.

## 3. Bundle structure

### 3.1. Hierarchy

A bundle is one Zarr v3 group, the root, and its descendants, plus one optional plain
JSON file. The hierarchy is shown as it appears on a local disk. It can equally be
stored on a web server and accessed over HTTP, or in object storage.

```
<bundle>                     # One TS-Zarr bundle. A Zarr v3 group.
|-- zarr.json                # Root group metadata: the "tsz" attributes (section 6)
|                            # and Zarr consolidated metadata for every descendant.
|-- recording.json           # Optional. Plain JSON, not a Zarr node: identity and
|                            # datetimes (section 6.6).
|-- 0                        # Channel 0. A continuous channel (section 4).
|   `-- ...
|-- 1                        # Channel 1.
|   `-- ...
`-- 7                        # Channel 7. An event channel (section 5).
    `-- ...
```

Zarr arrays hold every value of the recording: samples, per-bin statistics, event
times, and per-event columns. Zarr attributes hold the metadata that describes them.
The one plain file, `recording.json`, holds the identity of the recording and its
datetimes. It is not a Zarr node and does not appear in the consolidated metadata.

### 3.2. Channel groups

A bundle with `C` channels holds exactly `C` channel groups as children of the root,
with the integer names `0` through `C - 1`. The channel
index carries no meaning beyond naming the group and its bit in the `channels` column
of an event channel (section 5.4). The order of channels is the producer's choice.
No other child node of the root has an integer name. A bundle MAY hold zero
channels.

Each channel group carries a `kind` attribute, `"continuous"` or `"event"`, and the
attributes section 6.3 requires for that kind.

### 3.3. Unknown nodes and keys

A reader MUST ignore a child node of the root that is not a channel group, a child
node of a channel group or of a level group that this document does not define, a
key of a `tsz` object or of `recording.json` that this document does not define, and
a file in the bundle that this document does not define. Under a channel group,
every child whose name matches `[1-9][0-9]*` is a level group, and sections 4.3 and
5.7 apply to it.

## 4. Continuous channels

A continuous channel holds a regularly sampled signal of `N` samples and `K >= 0`
levels.

```
0                            # Channel group. Attributes: kind = "continuous", name,
|-- zarr.json                # id, rate_hz, offset_us, unit, provenance.
|-- raw                      # Array, shape (N,), float32. The samples.
|   |-- zarr.json
|   `-- c                    # Chunk objects: c/0, c/1, ... (section 7.1).
|-- runs                     # Array, shape (R, 3), int64. Optional. The runs.
|-- 1                        # Level 1 group. Attribute: period_us.
|   |-- zarr.json
|   |-- minmax               # Array, shape (bins, 2), float32.
|   |   |-- zarr.json
|   |   `-- c                # c/0/0, c/1/0, ... The pair axis is not chunked.
|   |-- count                # Array, shape (bins,), uint32.
|   |-- mean                 # Array, shape (bins,), float32. Optional.
|   `-- var                  # Array, shape (bins,), float32. Optional.
|-- 2                        # Level 2 group. Attribute: period_us.
`-- K                        # Level K group. Attribute: period_us.
```

### 4.1. Arrays

| array | requirement | shape | dtype | description |
|---|---|---|---|---|
| `raw` | MUST | `(N,)` | float32 | The samples, in the channel's `unit`. Sample `i` is at the time section 4.2 defines. A non-finite sample holds no value and enters no statistic (section 4.4). |
| `runs` | MAY | `(R, 3)` | int64 | Row `r` describes run `r`: column 0 the index of its first sample, column 1 its number of samples, column 2 the time of its first sample. |
| `<k>/minmax` | MUST per level | `(bins, 2)` | float32 | Column 0 the minimum and column 1 the maximum of the finite samples in bin `i`. |
| `<k>/count` | MUST per level | `(bins,)` | uint32 | The number of finite samples in bin `i`. |
| `<k>/mean` | MAY | `(bins,)` | float32 | The arithmetic mean of the finite samples in bin `i`. |
| `<k>/var` | MAY | `(bins,)` | float32 | The population variance of the finite samples in bin `i`. |

### 4.2. Sample times

A channel has one sample rate, `rate_hz`. A recording whose sample rate changes MUST
be stored as one channel per rate.

The grid time of sample `i` is `offset_us + i * 1e6 / rate_hz`, where `offset_us` is
the channel attribute of that name (section 6.3).

A channel without `runs` is one run of `N` samples. Every sample is at its grid time.

A channel with `runs` holds one row per run, in time order. Let `first_r`, `n_r`, and
`time_r` be columns 0, 1, and 2 of row `r`. The table MUST satisfy all of the
following.

- `R >= 1`. Row 0 is `[0, n_0, offset_us]`.
- Every `n_r >= 1`.
- Rows ascend in columns 0 and 2.
- Runs do not overlap: `first_(r+1) >= first_r + n_r`, and the last run ends at `N`:
  `first_(R-1) + n_(R-1) = N`.
- A run starts no earlier than the samples of the run before it end:
  `time_(r+1) >= time_r + n_r * 1e6 / rate_hz`.
- `time_r` lies within one sample period of the grid time of sample `first_r`:
  `|time_r - (offset_us + first_r * 1e6 / rate_hz)| < 1e6 / rate_hz`.

Sample `i` of run `r`, where `first_r <= i < first_r + n_r`, is at time
`time_r + (i - first_r) * 1e6 / rate_hz`. Every sample outside a run MUST be NaN.

Within a run, samples are periodic at `rate_hz`. No object holds the time of an
individual sample. A pause of any length between runs is representable to the
microsecond. The constraints above allow more than one `first_r` for a run that
starts between grid times, so a pause shorter than one sample period need not leave a
NaN sample on the grid.

### 4.3. Levels

Level `k`, for `k = 1, 2, ..., K`, is a child group of the channel named `k`. Levels
MUST be numbered `1` through `K` with no gaps. `K` MAY be 0.

Each level group carries the attribute `period_us` (section 6.4). The bin width of
level `k` in samples, `s_k`, is the nearest integer to `period_us * rate_hz / 1e6`.
The following MUST hold.

- `s_k >= 2` and `s_k < 2^32`.
- `period_us` is within `1e-12` relative of `s_k * 1e6 / rate_hz`.
- `s_(k+1)` is an integer multiple of `s_k` and `s_(k+1) >= 2 * s_k`.

Bin `i` of level `k` covers the samples with indices in `[s_k * i, min(s_k * (i + 1), N))`.
Level `k` has `bins = ceil(N / s_k)` bins, and the last bin MAY cover fewer than `s_k`
samples. Bin `i` starts at the grid time of sample `s_k * i`. The bins of level `k + 1`
nest exactly over the bins of level `k`.

A factor of 4 between consecutive levels, with `s_1 = 4`, is RECOMMENDED. It gives
`s_k = 4^k`. The factor MAY differ from level to level. A reader derives the factor from the `period_us`
values, never from the level names.

### 4.4. Statistics

Each statistic of bin `i` is defined over the finite samples of bin `i`.

- `count` is the number of finite samples.
- `minmax` column 0 is their minimum and column 1 their maximum.
- `mean` is their arithmetic mean.
- `var` is their population variance: the mean of the squared deviations from their
  arithmetic mean.

A bin with `count` 0 holds NaN in `minmax`, `mean`, and `var`.

A producer MAY compute level `k + 1` from level `k` instead of from `raw`. Over the
bins of level `k` that a bin of level `k + 1` covers and that have `count > 0`: the
minimum is the minimum of their minimums, the maximum is the maximum of their
maximums, `count` is the sum of their counts, `mean` is their count-weighted mean of
means, and `var` is the count-weighted mean of `var_child + (mean_child - mean)^2`,
where `mean` is the value computed for the parent bin. These are identities over the
real numbers. A producer that folds SHOULD fold from the binary64 statistics of level
`k` before they are rounded to `float32`. A fold from the stored `float32` members
can exceed the tolerance of section 8.4. Producers SHOULD accumulate in binary64 and
store the `float32` nearest to each value, rounding to nearest with ties to even. A
value that rounds to an infinity under that rule is stored as that infinity. Section
8.4 gives the tolerance a validator applies.

### 4.5. Empty channel

A channel with `N = 0` MUST hold `raw` with shape `(0,)` and MUST NOT hold `runs` or
levels.

## 5. Event channels

An event channel holds `n` events in non-decreasing time order and optional columns.
The arrays `events`, `durations`, `labels`, `values`, `text`, `channels`, and
`waveforms` are the per-event columns. Each has `n` rows, and row `i` of every column
describes event `i`. `n` MUST be less than `2^32`.

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
|   |-- zarr.json            # offset_us.
|   `-- count                # Array, shape (bins, L), uint32.
`-- J                        # Count level J group.
```

### 5.1. Arrays

| array | requirement | shape | dtype | description |
|---|---|---|---|---|
| `events` | MUST | `(n,)` | int64 | Time of each event, non-decreasing. |
| `durations` | MAY | `(n,)` | int64 | Duration of each event in microseconds, `>= 0` and `<= max_duration_us`. |
| `labels` | MAY | `(n,)` | uint16 | Category of each event, `< L` (section 5.3). |
| `values` | MAY | `(n,)` | float32 | One measurement per event, in the channel's `unit`. |
| `text` | MAY | `(n, W)` | uint8 | Text of each event as UTF-8, padded with zero bytes to width `W` (section 5.5). |
| `channels` | MAY | `(n, M)` | uint8 | Bitmask of the channels each event applies to (section 5.4). |
| `waveforms` | MAY | `(n, P)` | float32 | `P` samples around each event, in the channel's `unit` (section 5.6). |
| `waveform_mean` | MAY when `labels` exists | `(L, P)` | float32 | Mean waveform of the events of each label (section 5.6). |
| `<k>/count` | MUST per count level | `(bins, L)` | uint32 | Number of events in bin `i` with label `j` (section 5.7). |

### 5.2. Events and intervals

Event `i` is at time `events[i]`. Equal times are allowed.

Without `durations`, every event is a point event. With `durations`, event `i` is a
point event when `durations[i] = 0` and otherwise an interval covering
`[events[i], events[i] + durations[i])`. When `durations` exists, the channel
attribute `max_duration_us` MUST exist and MUST be at least every value in
`durations`. The bound MAY be loose. When `durations` does not exist,
`max_duration_us` MUST NOT exist.

### 5.3. Labels

`L` is the number of label categories of the channel.

- When the channel attribute `label_names` exists, `L` is its length, and label `j`
  is named `label_names[j]`.
- Otherwise, when `labels` exists and `n > 0`, `L` is one more than the largest value
  in `labels`.
- Otherwise `L` is 1.

Every value in `labels` MUST be less than `L`. When `labels` does not exist, every
event has label 0. When `label_names` exists, it MUST hold at least one string.
`label_names` MAY exist without `labels`; `label_names[0]` then names the one label
in use.

### 5.4. Channels

Row `i` of `channels` names the channels event `i` applies to. Bit `k mod 8` of byte
`floor(k / 8)`, least significant bit first, is set when the event applies to the
channel with index `k`. `M` MUST equal `ceil(C / 8)`, where `C` is the number of
channels in the bundle, and every bit at a position `>= C` MUST be 0. A row with no
set bit names no channel. An event channel without a `channels` column does not name
channels for its events.

### 5.5. Text

Row `i` of `text` holds the UTF-8 encoding [RFC3629] of the text of event `i`
followed by zero bytes up to width `W`. A text MUST NOT contain U+0000, so the first
zero byte of a row, if any, ends the text. `W` MUST be at least 1 and at least the length in bytes
of the longest text in the channel. A row of `W` zero bytes is an empty text. The content
is free: a clinical note, or a JSON object a detector emits per event.

### 5.6. Waveforms

Row `i` of `waveforms` holds `P` samples of one trace around event `i`, in the
channel's `unit`. `P` MUST be at least 1. Sample `j` of the row is at time
`events[i] + (j - event_sample) * period_us`, where `period_us` and `event_sample`
are the attributes of the array (section 6.5). A multi-channel snippet is out of
scope: a row is one trace.

Row `j` of `waveform_mean` holds the mean waveform of the events whose label is `j`,
for `j = 0, ..., L - 1`. Sample `p` of row `j` is the arithmetic mean of sample `p`
over the finite values of the waveforms of those events, whether or not `waveforms`
is stored, and NaN when no such value exists, including when no event has label `j`. `waveform_mean` MUST NOT exist when `labels` does not exist.
When `waveforms` and `waveform_mean` both exist, they MUST carry the same `period_us`
and `event_sample`, have the same `P`, and `waveform_mean` MUST equal the mean of the
`waveforms` rows as defined here, within the tolerance of section 8.4.

### 5.7. Count levels

An event channel MAY hold count levels, numbered `1` through `J` with no gaps. Count
level `k` is a child group of the channel named `k`, carrying the attributes
`period_us` and `offset_us` (section 6.4) and one array, `count`.

`period_us` of a count level is a positive integer. `period_us` of level `k + 1` MUST
be an integer multiple of `period_us` of level `k` and at least 2 times it. A factor
of 4 between consecutive levels is RECOMMENDED. When the factor between levels 1 and
2 is 4, `period_us` of level 1 SHOULD be `1000 * 4^m` microseconds for an integer
`m >= 0`.

`offset_us` of a count level MUST be an integer multiple of its `period_us` and at
most `events[0]`. It MAY be absent, and is then 0, so it MUST be present when
`events[0] < 0`. Bin `i` of the level covers the times
`[offset_us + i * period_us, offset_us + (i + 1) * period_us)`. The level has
`bins = floor((events[n - 1] - offset_us) / period_us) + 1` bins.

`count[i, j]` is the number of events whose time lies in bin `i` and whose label is
`j`. An interval is counted in the bin of its onset only. When `labels` does not exist, every event has label 0, and `count` has the `L`
columns that section 5.3 defines.

A channel with `n = 0` MUST NOT hold count levels.

### 5.8. Empty channel

A channel with `n = 0` holds `events` with shape `(0,)` and any optional column with
`0` rows. `waveform_mean`, when present, holds `L` rows of NaN.

## 6. Metadata

### 6.1. The `tsz` object

TS-Zarr metadata is stored in the `attributes` of the `zarr.json` of the root group,
of each channel group, of each level group, and of the `waveforms` and
`waveform_mean` arrays. The `attributes` of each of those nodes MUST contain a key
`tsz` whose value is a JSON object. No other Zarr node carries a `tsz` key.
`recording.json` is not a Zarr node and has no `tsz` key: its keys are the top-level
keys of its object (section 6.6).

Other conventions MAY add their own keys to `attributes` beside `tsz`. A reader MUST
ignore keys inside `tsz` that it does not recognize (section 3.3).

### 6.2. Root metadata

| key | requirement | type | description |
|---|---|---|---|
| `version` | MUST | string | Version of this specification, `"1.0"`. |

No other node carries `version`.

The root `zarr.json` MUST also carry Zarr consolidated metadata [ZARR-CONSOLIDATED]:
a top-level
`consolidated_metadata` object with `"kind": "inline"`, `"must_understand": false`,
and a `metadata` object that maps the path of every descendant node, relative to the
root and joined with `/`, to the content of that node's `zarr.json`. A group's entry
MAY carry a nested `consolidated_metadata` object for its own descendants, as
zarr-python writes it. A reader MUST NOT require the nested form. A root `zarr.json`
without `consolidated_metadata` is not a bundle.

### 6.3. Channel metadata

Each channel group's `tsz` object describes the channel.

| key | requirement | type | description |
|---|---|---|---|
| `kind` | MUST | string | `"continuous"` or `"event"`. |
| `name` | MUST | string | Display label, for example an electrode label. |
| `id` | MAY | string | Producer identifier, stable across exports. |
| `unit` | MUST for `continuous`, MAY for `event` | string | Unit of `raw`, or of `values`, `waveforms`, and `waveform_mean`. |
| `rate_hz` | MUST for `continuous` | number | Sample rate, finite and greater than 0. |
| `offset_us` | MAY for `continuous` | integer | Time of sample 0. Default 0. When `runs` exists, column 2 of its row 0 holds this value. |
| `label_names` | MAY for `event` | array of string | Name of each label category, indexed by label value. At least one string. |
| `max_duration_us` | MUST when `durations` exists, MUST NOT otherwise | integer | Upper bound on every value in `durations`, `>= 0`. |
| `provenance` | MAY | object | What produced the channel, such as detector name and version. Content is producer-defined. |

A producer whose source declares the unit of a channel as exactly `V`, `mV`, `uV`,
or `nV` MUST store `raw`, `values`, `waveforms`, and `waveform_mean` scaled by `1e6`,
`1e3`, `1`, or `1e-3` respectively, and MUST declare the unit `"uV"`. A producer MAY
treat another spelling of one of these units the same way. Every other channel
declares the unit string of its source unchanged. A continuous channel whose source
has no unit declares the empty string; an event channel whose source has no unit
omits `unit`.

No attribute of a channel identifies a person or holds a date or a datetime
(section 10).

### 6.4. Level metadata

Each level group's `tsz` object holds the bin width. A count level also holds the time
of its bin 0.

| key | requirement | type | description |
|---|---|---|---|
| `period_us` | MUST | number for a continuous level, integer for a count level | Width of one bin in microseconds. For a continuous level, within `1e-12` relative of `s_k * 1e6 / rate_hz` (section 4.3). |
| `offset_us` | MAY for a count level, MUST NOT for a continuous level | integer | Time of bin 0, a multiple of `period_us` and at most `events[0]`. Default 0. MUST be present when `events[0] < 0`. |

A producer SHOULD write the `period_us` of a continuous level as the shortest decimal
that reads back as the binary64 value `s_k * 1e6 / rate_hz`. A value written with
fewer digits can fall outside the tolerance of section 4.3.

### 6.5. Waveform metadata

The `waveforms` and `waveform_mean` arrays each carry a `tsz` object.

| key | requirement | type | description |
|---|---|---|---|
| `period_us` | MUST | number | Sample period of the trace in microseconds, finite and greater than 0. |
| `event_sample` | MAY | integer | Index of the trace sample at the event time, in `[0, P)`. Default 0. |

### 6.6. Recording metadata

A bundle MAY hold a plain JSON file `recording.json` at its root. It is UTF-8 JSON
[RFC8259] holding one object with the following keys.

| key | requirement | type | description |
|---|---|---|---|
| `reference_time` | MAY | string | The reference time as a datetime. A time in the bundle added to it gives a datetime. |
| `subject` | MAY | object | Who was recorded. Content is producer-defined. |
| `session` | MAY | object | The recording session. `start_time`, when present, is the datetime acquisition began. Other content is producer-defined. |
| `source` | MAY | object | Acquisition system and converting software. Content is producer-defined. |

`session.start_time` describes the session and places no data. It need not equal
`reference_time`. No attribute of a Zarr node holds a date or a datetime (section
10).

## 7. Storage

### 7.1. Zarr requirements

A bundle is a Zarr v3 hierarchy [ZARR3]. Every `zarr.json` declares
`"zarr_format": 3`. Every array declares the `regular` chunk grid and the
`default` chunk key encoding with the separator `/`, so the chunk objects of an array `a` are `a/c/0`, `a/c/1`, ... for a
rank-1 array and `a/c/0/0`, `a/c/1/0`, ... for a rank-2 array.

### 7.2. Codecs

Every array MUST declare one of the following codec chains, and a reader MUST
implement all of them.

- `bytes`, optionally followed by `zstd`. For a dtype wider than one byte, `bytes`
  declares `"endian": "little"`. For `uint8`, the `endian` configuration MAY be
  absent, and when present it is `"little"`.
- `sharding_indexed` [ZARR-SHARDING] whose inner `codecs` are the chain above, whose
  `index_codecs` are `bytes` with `"endian": "little"` followed by `crc32c`, and
  whose `index_location` is `"end"`.

The `zstd` level and checksum setting are the producer's choice and not part of the
format. No other codec is permitted in a conforming bundle.

### 7.3. Fill values and chunk presence

`raw`, every `float32` level member, `values`, `waveforms`, and `waveform_mean` MUST
declare `"NaN"` as `fill_value`. Every integer array MUST declare `0`.

A reader MUST read a missing chunk as the fill value, as Zarr defines for the
uninitialized portion of an array. In a sharded array, a chunk is an inner chunk, and
a missing chunk is one whose index entry is empty (offset and length both
`2^64 - 1`). A shard object whose inner chunks are all missing MAY be absent. Only a
chunk that intersects the array's extent is subject to the rules below, and only its
elements within the extent are compared with the fill value, where any NaN equals the
fill value `NaN`. An array with a zero-length axis has no chunks.

A producer MAY omit a chunk of `raw`, of a continuous level member, or of a count
level `count` array when every element of the chunk equals the fill value. Every
chunk of every other array (`runs`, every per-event column, and `waveform_mean`) MUST
be present.

### 7.4. Chunk shapes

Producers SHOULD use `sharding_indexed`. The inner chunk SHOULD span at most 8192
elements along the first axis. A shard SHOULD group whole inner chunks up to about
16 MiB. Trailing axes (the pair axis of `minmax`, the `L` axis of `count`, the `W`
axis of `text`, the `M` axis of `channels`, the `P` axis of `waveforms` and
`waveform_mean`, and the column axis of `runs`) SHOULD NOT be chunked.

### 7.5. Dimension names

Arrays SHOULD declare Zarr `dimension_names`: `"time"` for a sample or bin axis,
`"event"` for a per-event axis, `"label"` for the `L` axis of `waveform_mean` and of
a count level `count`, `"run"` for the `R` axis of `runs`, and `null` for every other
axis.

### 7.6. Stores

A store holding a bundle MUST support reading a byte range of an object and MUST
report a missing object as absent rather than as an error.

## 8. Conformance

### 8.1. Producers

A conforming producer writes bundles that satisfy every requirement on a bundle in
sections 2 through 7, 9, and 10, and that pass full validation (section 8.4). The
following requirements cannot be checked by a validator and rest on the producer:
the requirement of section 10 on the attributes of Zarr nodes; one channel per
sample rate (section 4.2); the conversion to microvolts (section 6.3); `W` at least
the length in bytes of the longest text (section 5.5); and `waveform_mean` of a
channel without `waveforms` (section 5.6).

### 8.2. Readers

A conforming reader reads every conforming bundle whose `version` has the major
version it implements (section 9), ignores what section 3.3 says to ignore, and
rejects a root `zarr.json` that lacks `consolidated_metadata` or carries a major
version it does not implement. A reader MAY accept a bundle that violates another
requirement of this document.

### 8.3. Structural validation

A validator reports every violation it finds. Structural validation reads every
`zarr.json`, `recording.json`, `runs`, the attributes, and the per-event columns
other than `waveforms`. It needs a store that lists its keys, in addition to the
requirements of section 7.6. It checks the following.

- Every `zarr.json` and `recording.json` parses as RFC 8259 JSON. A bare `NaN`,
  `Infinity`, or `-Infinity` token anywhere in a file, including inside the
  consolidated metadata, is a violation. The string `"NaN"` as a `fill_value` is not.
- Every `zarr.json` declares `zarr_format` 3. The root `tsz.version` has the form of
  section 9 and major version 1. A validator reports a minor version it does not
  implement as unsupported (section 9), which is not a violation, and continues with
  the remaining checks. `consolidated_metadata` is present with `"kind": "inline"`
  and `"must_understand": false`, has one entry for every descendant of the root,
  keyed by its path relative to the root (section 6.2), and each entry equals the
  `zarr.json` of its node, ignoring a nested `consolidated_metadata`.
- The `attributes` of the root, of each channel group, of each level group, and of
  `waveforms` and `waveform_mean` contain a `tsz` object. No other Zarr node has a
  `tsz` key. Only the root `tsz` has `version`.
- The channel groups have the integer names `0` through `C - 1`, and no other child
  of the root has an integer name. Each channel group carries a valid `kind` and the
  attributes of section 6.3 that the kind requires, with the stated types and ranges.
  `label_names`, when present, holds at least one string. `max_duration_us` exists
  exactly when `durations` exists.
- `offset_us` of a channel or a count level, `period_us` of a count level, and
  `max_duration_us` are JSON integers in `(-2^53, 2^53)`. `rate_hz` is finite and
  greater than 0. The grid time of sample `N - 1`, the time of the last sample of
  every run, and the end of the last bin of every count level lie in
  `(-2^53, 2^53)`.
- Every child of a channel group whose name matches `[1-9][0-9]*` is a level group
  (section 3.3). Level groups are named `1` through `K`, or `1` through `J` for count
  levels, with no gaps, and each carries `period_us`. A continuous level does not carry
  `offset_us`. Continuous levels satisfy section 4.3. Count levels satisfy section
  5.7: `period_us` is a positive integer, `period_us` of level `k + 1` is an integer
  multiple of that of level `k` and at least 2 times it, `offset_us` is a multiple of
  `period_us`, at most `events[0]`, and present when `events[0] < 0`, and `count` has
  the bin count of section 5.7. The RECOMMENDED factor of 4 and the SHOULD on
  `period_us` of level 1 in section 5.7 are not checked.
- Every array in sections 4.1 and 5.1 that is required is present, and every array
  present has the shape, dtype, fill value, chunk grid, chunk key encoding, and codec
  chain of sections 4.1, 5.1, 7.1, 7.2, and 7.3. `W >= 1` and `P >= 1`. All members
  of one level have the same number of rows, `ceil(N / s_k)` for level `k` of a
  continuous channel. A channel with `N = 0` holds no `runs` and no levels, and a
  channel with `n = 0` holds no count levels.
- When `runs` exists, it satisfies section 4.2, and every value of its column 2 lies
  in `(-2^53, 2^53)`.
- `events` is non-decreasing, and every value in `events`, `durations`, and
  `events[i] + durations[i]` lies in `(-2^53, 2^53)`. `n < 2^32`. Every value in
  `durations` lies in `[0, max_duration_us]`. Every value in `labels` is less than
  `L`. `waveform_mean` has `L` rows and does not exist without `labels`.
- `waveforms` and `waveform_mean` carry `period_us` finite and greater than 0 and
  `event_sample` in `[0, P)`, and agree on `P`, `period_us`, and `event_sample`
  when both exist.
- Every row of `text` is valid UTF-8 containing no zero byte, followed by zero
  bytes. `M = ceil(C / 8)`, and no bit of `channels` at a position `>= C` is set.
- Every chunk of `runs`, of every per-event column, and of `waveform_mean` is
  present (section 7.3).
- `recording.json`, when present, is a JSON object. `subject`, `session`, and
  `source`, when present, are JSON objects. `reference_time` and
  `session.start_time`, when present, are RFC 3339 datetimes with a UTC offset.

### 8.4. Full validation

Full validation performs structural validation and in addition reads `raw`, every
level member, `waveforms`, and `waveform_mean`, and checks the following. Values
compare as numbers, except that NaN equals NaN and `-0.0` equals `+0.0`.

- Every sample of `raw` outside a run is NaN (section 4.2).
- `count` and `minmax` of every continuous level equal the values section 4.4
  defines, exactly.
- Every count level `count` equals the values section 5.7 defines, exactly.
- `mean` and `var` of every continuous level, and `waveform_mean` when `waveforms`
  exists, are within tolerance of the values sections 4.4 and 5.6 define, recomputed
  in binary64. A validator SHOULD sum with an error far below the tolerance, for
  example by pairwise summation. Let `ref` be the recomputed value, `v` the stored
  value, and `max_abs` the largest absolute finite sample among the samples `ref` is
  defined over. When `ref` is NaN, `v` passes if and only if `v` is NaN. When `ref`
  rounds to `+infinity` as a `float32`, `v` passes if and only if `v` is
  `+infinity`. Otherwise `v` passes if and only if it is finite and:
    - for `mean` and `waveform_mean`: `|v - ref| <= 2^-20 * max_abs + 2^-149`;
    - for `var`: `|v - ref| <= 2^-20 * max(ref, 2^-20 * max_abs^2) + 2^-149`.

## 9. Versioning and compatibility

`version` is a string `<major>.<minor>`, each part a decimal integer without leading
zeros. A minor release adds optional attribute keys, level members, per-event
columns, or files. A reader written for `1.0` reads any `1.x`
bundle by ignoring what it does not recognize. A major release may change the
meaning, node type, or shape of an existing path. Readers MUST reject a bundle whose
major version they do not implement.



## 10. Privacy considerations

Information that identifies a person MAY appear in `recording.json`, at its fixed
path, and in the `text` column of an event channel. It MUST NOT appear in the
attributes of any Zarr node, including `name`, `id`, `label_names`, and
`provenance`. No attribute of a Zarr node holds a date or a datetime. A `text`
column MAY hold one.

Deleting `recording.json` removes the identity and datetimes it holds and changes no
other object. It does not remove what a `text` column holds, what a producer put in
an attribute in violation of this section, or what the path or name of the bundle
reveals. Every time in a bundle is relative to `reference_time`, so changing
`reference_time` shifts the datetime of every sample and event and changes no other
object. `session.start_time` places no data (section 6.6). A datetime inside a `text`
column does not shift.

## 11. Implementations

`timeseries-zarr-py` is the reference producer and validator.
`@pennsieve/timeseries-zarr-reader` is the reference reader. Implementations MUST
state the versions of this specification they support. Cite this specification by
its version and the repository that publishes it, `timeseries-zarr-py`.

## 12. References

- [RFC2119] Bradner, S., "Key words for use in RFCs to Indicate Requirement Levels",
  BCP 14, RFC 2119, March 1997. <https://www.rfc-editor.org/rfc/rfc2119>
- [RFC8174] Leiba, B., "Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words",
  BCP 14, RFC 8174, May 2017. <https://www.rfc-editor.org/rfc/rfc8174>
- [RFC3339] Klyne, G. and C. Newman, "Date and Time on the Internet: Timestamps",
  RFC 3339, July 2002. <https://www.rfc-editor.org/rfc/rfc3339>
- [RFC3629] Yergeau, F., "UTF-8, a transformation format of ISO 10646", STD 63,
  RFC 3629, November 2003. <https://www.rfc-editor.org/rfc/rfc3629>
- [RFC8259] Bray, T., Ed., "The JavaScript Object Notation (JSON) Data Interchange
  Format", STD 90, RFC 8259, December 2017.
  <https://www.rfc-editor.org/rfc/rfc8259>
- [ZARR3] "Zarr core specification, version 3".
  <https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html>
- [ZARR-SHARDING] "Sharding codec, version 1".
  <https://zarr-specs.readthedocs.io/en/latest/v3/codecs/sharding-indexed/index.html>
- [ZARR-CONSOLIDATED] zarr-python, "Consolidated metadata".
  <https://zarr.readthedocs.io/en/stable/user-guide/consolidated_metadata.html>

## Appendix A. Examples

This appendix is informative.

### A.1. Root group

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
      "0": {},          // elided: the full zarr.json of each node,
      "0/raw": {},      // and the entries of the other nodes
      "0/1": {},
      "0/1/minmax": {},
      "0/1/count": {}
    }
  }
}
```

### A.2. Continuous channel group

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

### A.3. Event channel group

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

### A.4. Level groups

Level 1 of a continuous channel at 2000 Hz with 4 samples per bin:

```json
{
  "zarr_format": 3,
  "node_type": "group",
  "attributes": {
    "tsz": {"period_us": 2000.0}
  }
}
```

Level 1 of a continuous channel at 30000 Hz with 4 samples per bin. The nearest
integer to `133.33333333333334 * 30000 / 1e6` is 4:

```json
{
  "zarr_format": 3,
  "node_type": "group",
  "attributes": {
    "tsz": {"period_us": 133.33333333333334}
  }
}
```

A count level whose events begin 32 ms before the reference time:

```json
{
  "zarr_format": 3,
  "node_type": "group",
  "attributes": {
    "tsz": {"period_us": 16000, "offset_us": -32000}
  }
}
```

### A.5. Waveform array

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

### A.6. A sharded level member

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

### A.7. Recording metadata

```json
{
  "reference_time": "2024-03-14T09:15:00.000000-05:00",
  "subject": {"id": "sub-0931", "species": "Homo sapiens", "sex": "F", "age": "P34Y"},
  "session": {"id": "ses-01", "start_time": "2024-03-14T09:15:00.000000-05:00", "description": "EMU admission, day 3"},
  "source": {"system": "Natus Quantum", "converter": "timeseries-zarr-py 2.0.0"}
}
```

## Appendix B. Version history

| revision | date | description |
|---|---|---|
| 1.0 (draft) | 2026-10-29 | First versioned release. Namespaced attributes, level groups with `minmax`, `count`, `mean`, `var` over finite samples, event channels with `text` and `channels` columns, count levels, `runs`, `recording.json`, a codec profile, and two conformance levels. |
