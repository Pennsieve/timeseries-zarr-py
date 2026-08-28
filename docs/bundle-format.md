# TS-Zarr bundle format

A bundle is a static directory of Zarr v3 objects holding one recording at
several time resolutions. A browser opens it over HTTP range requests and
renders any window — at any zoom, montaged and filtered — without
downloading the recording and without a server-side service. This document
specifies the format so that other producers can write it and other
readers can consume it.

TS-Zarr is the short name of the format; `timeseries-zarr` is the spelling
used in package and repository names. A complete synthetic reference bundle exercising every feature of this
specification, its seeded generator, and a tutorial notebook reading it
with the standard `zarr` library live in
`Pennsieve/timeseries-zarr-paper`. `timeseries-zarr-py` is the
reference producer. `@pennsieve/timeseries-zarr-reader` is the reference
reader.

The container is stock Zarr v3. There is no sidecar manifest, no schema
language, and no namespace prefix. The only custom surface is fifteen
attribute keys within the standard Zarr `attributes` field (see Metadata
objects). Any Zarr v3 library can open a bundle.

## Why a pyramid

A ten-day stereo-EEG evaluation — two hundred intracranial contacts at
2 kHz — is about 1.4 TB in float32, and a research probe streaming 384
unit-band channels at 30 kHz produces as much in under nine hours. A
viewer draws an overview into roughly 2000 pixels. Fetching raw
samples to compute those 2000 columns is intractable in a browser, so the
producer precomputes per-bin statistics at several resolutions and the
reader fetches only the bins one viewport needs.

Two statistics carry the two halves of review. The min/max **envelope** is
the conservative rendering primitive: it bounds every stored sample, so an
isolated spike survives to the coarsest zoom. The per-bin **mean** is the
linear one: every fixed re-reference (bipolar, common average, linked
ears, Laplacian) computed from stored means equals the fold of the
re-referenced raw signal exactly, at every level, and filtering on
decimated means is governed by a single dimensionless ratio with known
error tiers. Neither statistic substitutes for the other: the montaged
envelope is provably unrecoverable from per-channel statistics, and the
envelope of a filtered signal is not derivable from the unfiltered one.
Event channels get the third statistic, per-bin **counts**, whose sum fold
is exact at every level. Proofs, error bounds, and measurements live in
the accompanying paper and its executable notebooks.

## Layout

```
# dtypes: f4 = float32, i8 = int64, u1/u2/u4/u8 = unsigned 1/2/4/8-byte ints
# a trailing slash = a Zarr node (every array is a directory of
#   zarr.json + chunk objects); it does not imply a nested group
<bundle-root>/
  zarr.json            # root group; consolidated_metadata inlines
                       #   every descendant EXCEPT meta/
  meta/                # recording metadata (optional)
    zarr.json          #   plain JSON text: subject, session, source;
                       #   delete this directory to de-identify
  0/                   # continuous channel (group)
    zarr.json          # attributes: id, offset_us, kind = "continuous",
                       #   name, rate_hz, unit, offset_uv?
    raw/               # raw samples, shape (N,), f4
    1/                 # level group, 4x; attribute: period_us
      env/             #   (ceil(N/4), 2) f4, interleaved (min, max)
      mean/            #   (ceil(N/4),) f4
      ssq/             #   optional: per-bin mean of squares, f4
      valid/           #   optional: valid counts, u2 (gap channels)
    2/ .. 7/           # 16x .. 16384x, same members
  1/                   # continuous channel
  23/                  # event channel (spike-sorted profile)
    zarr.json          # attributes: id, offset_us, kind = "event",
                       #   name, label_names?
    events/            # (n,), i8 µs from bundle onset, non-decreasing
    labels/            # primary label set, u2 (cluster ids)
    waveforms/         # (n, points_per_event) f4; attr period_us
    templates/         # optional: per-label mean waveforms, f4
    labels.<set>/      # optional: alternative label set (attr provenance)
    1/ .. J/           # optional level groups; attribute: period_us
      counts/          #   (bins, n_labels) u4, sum fold
      counts.<set>/    #   per alternative label set
  24/                  # event channel (annotation profile)
    zarr.json          # attributes: id, offset_us, kind = "event",
                       #   name, body_media_type, label_names?,
                       #   max_duration_us (required with durations)
    events/            # shape (n,), i8 µs from bundle onset
    durations/         # optional: shape (n,), i8 microseconds (0 = point)
    bodies/            # concatenated UTF-8 payload bytes, u1
    body_offsets/      # shape (n+1,), u8 monotone offsets into bodies
    labels/            # optional: u2 category per annotation
    1/ .. J/           # optional level groups; attribute: period_us
      counts/          #   (bins, n_labels) u4 -- or (bins,) if unlabeled;
                       #   interval events count in their start bin
```

Channel groups are named by index. The index is opaque and its order is
arbitrary; upstream identifiers never appear in a path. The reader builds
an `id`-to-index map from consolidated metadata. One group per channel
lets channels differ in sample rate, lets a producer write one channel at
a time, and lets a reader fetch only the channels on screen.

## Continuous channels

The `raw` array holds the signal: shape `(N,)`, dtype float32. Each level
k = 1…7 is a numeric-keyed **group** carrying one `period_us` attribute
and per-statistic member arrays over a shared bin axis of length
`ceil(N / 4^k)`:

- `env` — interleaved per-bin (min, max) pairs, shape `(bins, 2)`:
  column 0 the minimum, column 1 the maximum. One interleaved array
  delivers both envelopes in a single fetch.
- `mean` — the per-bin mean, shape `(bins,)`.
- `ssq` (optional) — per-bin mean of squares.
- `valid` (optional) — per-bin valid-sample counts, u2.

Folds are exact and single-pass: level 1 reduces disjoint blocks of 4 raw
samples; level k+1 reduces 4 bins of level k (min of mins, max of maxes,
count-weighted mean of means, sum of counts); a trailing partial block of
1–3 elements forms a final bin. Folds are NaN-propagating — a NaN-aware
mean would average different time supports across channels and break
montage exactness — and a reader treats NaN as a gap.

A bundle holds `raw` plus at most seven level groups (a 16384x range). The
producer stops once the next level would fall below a bin threshold; the
threshold is producer-tunable and the reference producer defaults to 1024
bins. `raw` may be omitted to produce a **viewing-only bundle** — the
statistic levels alone, about the size of the raw signal — in which case
the finest present level is the zoom floor and raw-forcing paths are
unavailable; readers discover this like any other absent member. A
channel with no samples writes a zero-length `raw` and no level
groups.

Samples are stored in microvolts. The bundle timeline is onset-relative:
bin i at level k begins at `offset_us + i * period_us(k)` microseconds
from recording onset. No time axis is stored, and no wall-clock value
appears outside `meta/`.

Storage: raw plus envelope is `N + 2(N/4 + N/16 + ...) = 5N/3`; with the
`mean` members, `2N`; with optional `ssq`, `7N/3`. These are bytes at
rest, not per read: each statistic is its own array, so an envelope read
moves two values per bin and a montage, filter, or band-power read exactly
one — no query fetches members it does not use.

## Level members and extensibility

A level group's members all describe the same bins — one statistic per
member array, one shared bin axis — so a level opens in generic Zarr
tooling as an aligned per-bin dataset. Members are identified by name,
never by shape; readers ignore members they do not recognize, which makes
the member set additive by construction: a future statistic is a new
member in each level group and nothing else changes.

- **`env`.** The conservative rendering primitive. Rendering whole bins
  over-draws by at most one bin at each window edge; it never
  under-draws.
- **`mean`.** Count-weighted, NaN-propagating, computed in the same
  producer pass as `env`. Serves exact linear views (montage, any fixed
  re-reference) and tier-governed filtering. Cost +N/3 over the envelope
  pyramid (+20%).
- **Offset handling.** A per-channel float64 `offset_uv` attribute; the
  producer subtracts it before folding, readers reconcile in float64.
  Advisable for means (float32 catastrophic cancellation reaches 10% of
  an 80 uV montage at a shared 0.5 V offset), mandatory when `ssq` is
  written (the offset enters squared: a 10 mV shared offset already
  corrupts float32-recovered variance by 4.5%).
- **`ssq` (optional).** Per-bin mean of squares under the same fold, for
  broadband quadratic views: RMS/EMG envelopes, within-bin variance
  `q - m^2`, quality checks. Canonical band power does **not** require
  it — delta through gamma power is filter → square → smooth on `mean`.
  Cost a further +N/3.
- **`valid` (optional).** u2, written only for channels containing
  non-finite samples; distinguishes a dropout from an isolated bad sample
  and licenses trust in a montage bin exactly when both sides' counts are
  full.

Member names in event-channel level groups follow the same grammar with an
optional label-set suffix: `<statistic>[.<set>]`, where set labels are
lowercase alphanumerics and hyphens.

## Event channels

An event channel holds a sorted point series with optional per-event
member columns, all of length n; `events` is the only required member, and
column presence — not a channel subtype — determines what a view can
draw. One model covers the two cases neurophysiology actually produces:

**Unit (spike) data.** A microelectrode-array session arrives as spike
times, a cluster id per spike, and a waveform snippet per spike — the
columns `events`, `labels`, and `waveforms`, plus a channel-level
`templates` array of per-cluster mean waveforms. Density motivates the
count level groups (a 24-hour, 100-unit session is a ~170 MB raster read
at overview zoom; the zoom-matched count level is under 1 MB, and exact).
Sorter subjectivity motivates multiple label sets (below): a re-sort or
curation pass shares the events and waveforms and costs 1–2 bytes per
event.

**Annotation data.** A clinician marks a seizure at 14:32 lasting 90
seconds (`events` + `durations`, with the `max_duration_us` bound that
lets a window query find intervals still open at its left edge); an
automated detector emits forty thousand candidates, each with a JSON body
recording score, band, and algorithm version (`bodies` + `body_offsets`,
NDJSON-readable in one array read). A `labels` column categorizes
annotations for filtering and color; dense detector output carries the
same count level groups as spikes.

A detector may legitimately carry all columns — snippet, score, and
structured body — in one channel. Channels whose per-event value is a
measurement rather than a document (beat-to-beat intervals from an R-peak
detector) carry a `values` column, whose level groups reuse the continuous
statistics verbatim. The member definitions:

- **`events` (required).** int64 microseconds from bundle onset,
  non-decreasing; equal timestamps permitted.
- **`durations`.** int64 microseconds, >= 0; absent or zero means a point
  event. A channel with durations must carry a `max_duration_us`
  attribute (an upper bound on every duration): a window query must
  return intervals that started earlier and are still open, and the bound
  lets a reader answer this stabbing query by extending its binary-search
  window backward by `max_duration_us` — one attribute in place of an
  interval index. The bound may be loose (a producer can round up) at the
  cost of a slightly wider backward search.
- **`labels`.** u2 category per event: cluster ids for sorted spikes,
  annotation types for markers. The channel's `label_names` attribute
  optionally names the categories; bare integer ids need no names.
- **`values`.** f4 numeric measurement per event (beat-to-beat interval,
  detection score); its physical unit is the channel's `unit` attribute.
- **`bodies` + `body_offsets`.** Document payloads in the Arrow style:
  `bodies` is u1 concatenated UTF-8; `body_offsets` is u8 of length n+1,
  non-decreasing, with `body_offsets[0] = 0` and
  `body_offsets[n] = len(bodies)`; body i is bytes
  `[body_offsets[i], body_offsets[i+1])`. The channel's `body_media_type`
  attribute declares the payload format (`text/plain` default;
  `application/json` for structured bodies, which the format stores but
  does not schematize). The offsets array costs 8 bytes per event and is
  what makes the byte stream range-addressable; storing per-event lengths
  instead would require the prefix sum offsets already are. Producers must terminate each payload with a
  newline, so the decompressed `bodies` array is directly human-readable:
  valid NDJSON for JSON payloads, plain text lines otherwise — one array
  read and no offset arithmetic for export, while `body_offsets` remains
  the machine path (two range requests per single body). Core dtypes
  only: no variable-length dtype is required, so every Zarr
  implementation can read bodies. Producers should store `bodies`
  uncompressed and unsharded (a single chunk): the chunk object
  `<ch>/bodies/c/0` is then itself the plain NDJSON/text file — openable
  directly from disk or object storage with no tooling at all, with
  `body_offsets` mapping to literal byte ranges of that object — while
  remaining a fully conformant Zarr v3 array. (Very large body stores may
  compress; readers consult the array's codec metadata either way. NDJSON
  is also known as LDJSON or JSON Lines.) Bodies are Zarr arrays rather
  than a JSON or Parquet sidecar deliberately: a sidecar is the one object
  a Zarr library cannot open, invisible to consolidated metadata, and a
  second format stack in every reader.
- **`waveforms`.** f4, shape `(n, points_per_event)`, carrying
  `period_us` for the snippet sample period.

A reader binary-searches `events` to serve any window from one or two
chunks; sparse channels need nothing further. Dense channels — modern
spike sorting on long recordings, automated detectors — may carry optional
numeric-keyed level groups on the same 4x ladder as continuous levels,
on the same onset-relative timeline, with a producer-chosen base
`period_us` on level 1; level k+1 is the 4x sum-fold of level k, and there is no
level 0 — the raw member is `events` itself. Level members:

- **`counts`** — u4 per-bin event counts, shaped `(bins, n_labels)` when a
  `labels` column exists and `(bins,)` otherwise, folded by summation —
  exact at every level, with the label axis unchunked. Interval events
  count in the bin of their **start** time only: overlap-counting would
  count one event once per overlapped bin and break the sum fold's
  exactness under coarsening. (Per-bin occupancy is a distinct future
  member for coverage views.)
- For channels with a `values` column, per-bin statistics of the value
  reuse the continuous member names and folds: `env`, `mean`, `valid`
  (count-weighted mean, NaN-propagating; an empty bin holds NaN with
  count 0).

The producer writes level groups only when a channel's event count makes
raster reads impractical (reference threshold: ~64k events). The
per-label mean-waveform template (`templates`, shape
`(n_labels, points_per_event)`) is a **channel-level** array, not a level
member: it has no time axis and no scale. Waveform matrices themselves are
never pyramided.

### Multiple label sets

The `labels` array is the channel's primary categorization. Because
labeling is algorithm- and curator-dependent — spike sorters disagree,
annotation taxonomies get revised — a channel may carry additional sets as
named arrays `labels.<set>`, each of length n (u2), parallel to the shared
`events`, `waveforms`, and `bodies`. Each label-set array carries a
`provenance` attribute (a JSON object recording, e.g., sorter or curator,
version, and date). Count level members and templates are per-set: the
primary set's are `counts` and `templates`; an alternative's are
`counts.<set>` members and `templates.<set>`. Events, waveforms, and
bodies are never duplicated across sets, so an alternative labeling costs
1–2 bytes per event plus its optional pyramid; readers that recognize only
the primary set ignore the rest.

Adding a label set to an existing bundle is additive at the object level:
it writes only the new arrays and atomically replaces the root
consolidated-metadata object — the event and waveform payload is
untouched. This makes post-hoc curation (sort, review, re-curate, months
apart) an append operation on archived data, the same pattern by which
OME-Zarr adds label layers to an existing image without rewriting pixels.

## Recording metadata

A bundle may carry recording-level metadata — who was recorded, in what
session, from what source — as a `meta/` group with three JSON-object
attributes: `subject` (identifying: subject id, species, sex, age…),
`session` (session id, description), and `source` (acquisition system,
converting software and version). The content of the three objects is
deliberately not schematized: ontology belongs to archival standards, and
a producer converting from NWB can carry its subject table across nearly
verbatim.

`session.start_us` is the bundle's **only wall-clock value**: the
timeline everywhere else is microseconds from recording onset. This is
what makes **date-shifting** — the standard de-identification for dates,
which keeps time-of-day and inter-event intervals but moves the calendar
date — an edit of one JSON field, with no data rewritten anywhere. (Were
event timestamps absolute, as in EDF-style designs, a date shift would
mean rewriting every event array.) A bundle whose `meta/` has been
deleted degrades gracefully to a pure relative timeline and remains fully
reviewable and analyzable.

Two properties are normative. First, `meta/` is **excluded from the
root's consolidated metadata**: a reader that wants it fetches
`meta/zarr.json` directly (one small request), and the root object —
the one every session, cache, and CDN touches — never carries identity.
Second, because a group's `zarr.json` is a plain, uncompressed JSON text
file, the metadata is human-readable and human-editable with no tooling,
and **de-identification is deleting one directory**: `rm -r meta/`
changes no other object in the bundle. (Contrast EDF, where identity
lives in a fixed binary header that must be byte-patched, and NWB, where
stripping it means rewriting HDF5.) A corollary rule applies bundle-wide:
everything outside `meta/` is identity-free **and date-free** by
construction — channel `id` values must be opaque or pseudonymous, and no
absolute wall-clock value may appear outside `meta/`.

The de-identification surface of a bundle is therefore two enumerable,
human-readable locations: `meta/` (remove it, or edit its plain-JSON
attributes to date-shift) and annotation bodies, whose free text can carry
identity the way any clinical note can — and which are stored as plain
NDJSON/text, so they can be audited with `grep` and redacted with a text
tool. Everything else in a bundle is numeric signal data that cannot
embed a name or a date. There is no PHI in any non-human-readable
location, which is what makes systematic de-identification of TS-Zarr
bundles tractable rather than forensic.

## Metadata objects

Every `zarr.json` in a bundle is one of five node shapes; nothing else is
permitted, and readers may validate against this closed list.

1. **Root group.** `node_type: group`. No custom attributes; must carry
   Zarr's `consolidated_metadata` inlining every descendant node except
   `meta/`.
2. **Metadata group.** `meta/`, `node_type: group`, no children;
   attributes `subject`, `session`, `source` (JSON objects, content
   unschematized). Optional; excluded from consolidation; deleting it
   de-identifies the bundle.
3. **Channel group.** `node_type: group`, keyed by an opaque index.
   Attributes per the table below, with requiredness determined by
   `kind ∈ {continuous, event}`.
4. **Level group.** `node_type: group`, keyed by a bare integer k >= 1.
   Exactly one required attribute, `period_us` (float64 > 0), satisfying
   `period_us(k+1) = 4 * period_us(k)` within a channel. Children are
   member arrays only.
5. **Array.** Standard Zarr v3 array metadata (shape, data_type, chunk
   grid, codecs including the sharding codec). Custom attributes appear
   on exactly two array kinds: `period_us` on `waveforms`, and
   `provenance` on alternative label-set arrays.

| Key | Type | On | Meaning and valueset |
|---|---|---|---|
| `id` | string | channel | upstream identifier readers join on; must be opaque/pseudonymous — identity belongs only in `meta/`; required |
| `kind` | string | channel | `continuous` \| `event`; required |
| `name` | string | channel | display label; required |
| `offset_us` | int64 | channel | microseconds from bundle onset of sample/event 0, >= 0; required |
| `rate_hz` | float64 | channel | sample rate, > 0; required for `continuous` only |
| `unit` | string | channel | physical unit of samples, or of an event channel's `values` |
| `offset_uv` | float64 | channel | subtracted from statistic members before folding; continuous only, optional |
| `body_media_type` | string | channel | IANA media type of `bodies`; default `text/plain` |
| `label_names` | JSON array | channel | names for `labels` categories, index-aligned; optional |
| `max_duration_us` | int64 | channel | upper bound on `durations`; required when durations are present |
| `period_us` | float64 | level group; `waveforms` | microseconds per bin (or per waveform sample); > 0 |
| `subject` | JSON object | `meta/` | identifying fields (subject id, species, sex, age…); unschematized |
| `session` | JSON object | `meta/` | session id and description; unschematized |
| `source` | JSON object | `meta/` | acquisition system, converter, versions; unschematized |
| `provenance` | JSON object | `labels.<set>` arrays | who/what produced the label set (suggested keys: `source`, `version`, `date`) |

Data-level constraints a validator checks: `events` non-decreasing;
`durations >= 0` and `<= max_duration_us`; `body_offsets` monotone with
the fixed endpoints above; `labels` values `< len(label_names)` when names
are present; all per-event columns of equal length n; all members of one
level group of equal bin count.

Layout needs no shape heuristics: numeric-keyed children of a channel are
level groups, and every array inside a level group is identified by its
member name (`env`, `mean`, `ssq`, `valid`, `counts`); readers ignore
member names, channel-level array names, and attribute keys they do not
recognize.

Two representative metadata objects, elided to the fields the format
constrains:

```jsonc
// <ch>/1/zarr.json — level group of a 2 kHz continuous channel
{ "zarr_format": 3, "node_type": "group",
  "attributes": { "period_us": 2000.0 } }

// <ch>/1/env/zarr.json — its envelope member
{ "zarr_format": 3, "node_type": "array",
  "shape": [900000, 2], "data_type": "float32",
  "chunk_grid": { "name": "regular",
                  "configuration": { "chunk_shape": [8192, 2] } },
  "codecs": [ { "name": "sharding_indexed",
                "configuration": { /* ... zstd ... */ } } ] }
```

## Storage

Arrays are Zstd-compressed and sharded with the Zarr v3 sharding codec.
The inner chunk spans up to 2^13 (8192) bins along the time axis; the
outer shard groups whole inner chunks up to about 16 MiB, giving one
object per statistic member per level; a single chunk already over that
target forms a shard on its own. Trailing statistic axes (the envelope
pair, the label axis of `counts`, the `points_per_event` axis of
`waveforms`) are never chunked. The compression level is a producer tuning
knob, not part of the format.

The inner chunk is the smallest unit a reader can fetch, so its width sets
the floor on what a read transfers. A reader picks the level whose
`period_us` matches one pixel, which puts a rendered window at a few
thousand bins whatever the zoom. Reading a shard starts with reading its
index from the end of the file; a store should ask for a suffix range
instead of issuing a HEAD followed by an absolute-offset GET.

Float32 is the dtype for signal statistics throughout: a viewer quantizes
to canvas pixels, so float64 precision reaches no screen and costs twice
the storage. (Timestamps and offsets are integer types as specified
above.)

## Consolidated metadata

The root `zarr.json` must carry a Zarr v3 `consolidated_metadata` block
inlining every descendant `zarr.json`. One fetch yields the whole tree —
channels, levels, members, shapes, dtypes, and attributes. Otherwise a
bundle of C channels costs on the order of 10·C metadata requests before
the first chunk fetch.

## Serving filtered and montaged views

Normative for the format is only what is stored; how a reader chooses
levels is guidance, summarized here because every reader needs it. Montage
(any fixed linear combination) is exact from `mean` members at every
level: select by zoom alone. Filtering on decimated means is governed by
ρ = highest filter transition frequency / level Nyquist, with measured
tiers: ρ <= 0.15 serve silently (< 1% on EEG-like signal); 0.15 < ρ <= 0.5
serve with a fidelity indicator; ρ > 0.5 select a finer level. Halve the
constants for broadband (flat-spectrum) channels, and budget one extra
tier of caution for highpass (low-edge) filters. On every filtered fetch,
extend the request backward by a pre-roll of about `2.5 / f_low` seconds
at the serving rate and discard it after filtering; where no earlier data
exists, pad by tiled reflection of a ~50 ms block. Derivations,
measurements, and the maximal-error tables are in the paper and its
notebooks.

## Compatibility

There is no `format_version` attribute. Zarr's own mechanisms carry
compatibility forward: new attribute keys are additive and readers ignore
what they do not recognize; new level members and new channel-level named
arrays are discovered through consolidated-metadata enumeration and
ignored when unrecognized; numeric keys under a channel are reserved for
level groups. Nothing may change the node type or shape contract of an
existing path.
