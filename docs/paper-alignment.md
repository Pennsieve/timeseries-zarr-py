# Aligning the writer with the format spec

What this writer produces today, what the spec now says, and the order in which the
difference gets closed.

The spec lives in
[`Pennsieve/timeseries-zarr-paper`](https://github.com/Pennsieve/timeseries-zarr-paper/blob/main/bundle-format.md)
and nowhere else. This repository no longer carries a copy; the copy is what caused the
drift described below.

## What fails today

The writer implements the format faithfully as it stood at `v0.3.0`. The spec has moved
since, and the local copy of it in `docs/bundle-format.md` did not. The writer matched
the stale copy, so nothing looked wrong: `make check` was green, the reader opened the
bundles, and the two documents disagreed quietly.

A reader written to the current spec cannot open a bundle this writer produces. Not
"misses some members" — it looks for `<ch>/raw/` and `<ch>/1/env/` and finds `<ch>/0/`
and `<ch>/1/`.

Five differences, in plain terms.

**1. Pyramid levels are folders now, not single arrays.** A level holds one array per
statistic over a shared bin axis: `env` (the min/max envelope), `mean`, optionally `ssq`
and `valid`. A reader fetches only the statistic it needs. This writer stores each level
as one rank-2 array that *is* the envelope, with nowhere to put a second statistic.

Raw leaves the numeric ladder in the same change: `<ch>/raw/` rather than `<ch>/0/`, and
without the `period_us` attribute, which was always redundant with the channel's
`rate_hz`. That reserves numeric keys for level groups, so layout needs no shape
heuristic — today a reader has to tell raw from a level by inspecting rank. It also makes
raw omittable, which is what a viewing-only bundle is.

**2. `mean` is never computed.** It is the member the paper's linear claims rest on: a
fixed re-reference computed from stored means equals the fold of the re-referenced signal
exactly, at every level. Proven in the notebooks, unproduced here.

**3. Wall-clock time is stamped across the bundle.** Every channel carries `start_us`, an
absolute Unix timestamp, and unit-channel `events` hold absolute microseconds. The spec
puts one wall-clock value in `meta/session.start_us` and measures everything else from
recording onset, which is what makes date-shifting an edit of one JSON field and
de-identification a `rm -r`. This writer has no `meta/` group, so it has nowhere to put
the date and nothing to strip, and `consolidate()` inlines every descendant with no
exclusion, so the recording date reaches the root object every cache and CDN touches.

Treat this one as a defect, not a missing feature.

**4. Gaps work but are not free.** `NwbTimestampedSource` presents a broken recording as
a uniform grid with NaN in the holes, and the folds propagate it. What is missing is the
storage half: float arrays must declare NaN as their `fill_value` so an absent chunk
reads as no data, and a producer should then skip chunks holding no finite samples. See
the matched pair in [gapped recordings](./gapped-recordings.md), *Out of scope*. Also
missing is `valid`, the per-bin count of real samples, whose stated purpose is telling a
dropout from an isolated bad sample — exactly what the grid now manufactures.

**5. Event channels are half-built.** The spike profile exists but is named wrong:
`units` (u1, 256 clusters) should be `labels` (u2). Missing from it are `templates`,
alternative label sets, and the `counts` level groups that make a dense channel readable
at overview zoom.

The annotation profile does not exist at all — `durations`, `max_duration_us`, `bodies`,
`body_offsets`, `labels`, `values`, `body_media_type`. So `channel_refs` and
`channel_ref_offsets`, which name the electrodes a mark belongs to, have nothing to
attach to yet.

## Compatibility during the branch

This branch breaks the format and bumps the version once, at the end. Bundles it writes
part-way through are not readable by the reader shipping today. The alternative was
building every new statistic into the old array shape and moving it later, which is
rework in `planning.py` and `write_continuous.py` for no lasting gain.

The consequence to schedule, not discover: `@pennsieve/timeseries-zarr-reader` needs its
own alignment work, and nothing produced here reaches a viewer until it lands.

## Order

Each step leaves `make check` green. Steps 1 and 3 are the breaking ones and land first,
so nothing built later gets written twice.

1. **Onset-relative time and `meta/`.** `offset_us` replaces `start_us`; event timestamps
   become onset-relative; a `meta/` group carries `subject`, `session`, `source`, and is
   excluded from consolidation. Small, self-contained, and closes the date leak.
2. **Level groups, `raw/`, and NaN `fill_value`.** The structural change. `plan_levels`
   stops emitting level 0, raw is written as its own step before the ladder, each level
   becomes a group holding `env`. Declaring `fill_value` belongs here because it is a
   property of array creation, and doing it while `zarr_io.create_array` is already open
   is cheaper than a second pass.
3. **`mean`.** A second fold alongside the min/max one, in the same streaming pass. Adds
   `offset_uv` and the float64 reconciliation, without which a shared DC offset eats the
   result.
4. **`valid`, then skipping empty chunks.** Finishes the gap story. Revisit
   `write_empty_chunks: True` here and not before — the note in
   [gapped recordings](./gapped-recordings.md) explains why the two move together.
5. **Event channels.** Rename to the spec's names, widen cluster ids to u2, add `counts`
   level groups, then the annotation profile with `channel_refs` built in from the start.

`ssq` is not in this list. It is optional in the spec, costs a further N/3 on the channels
that carry it, compresses worst of anything in the bundle, and nothing downstream asks for
it yet. Add it when a consumer does.

## Traps

**The Arrow-style offsets pair appears twice.** `bodies`/`body_offsets` and
`channel_refs`/`channel_ref_offsets` are the same mechanic: a flat array plus an n+1
monotone index, where sharing the boundary between neighbours is what makes a gap or an
overlap unrepresentable rather than merely invalid. Write the helper once in step 5 and
use it for both.

**`channel_refs` is not on the spec's main branch yet.** It lives on
`addition/channel-scoped-annotations` in the paper repo. Step 5 depends on that merging,
and the link at the top of this document will not show it until it does.

**Channel indices are the thing `channel_refs` points at.** They are assigned in
`bundle.py::assign_indices` and are opaque and order-dependent. The spec says a producer
writes the refs in the same pass that assigns the keys; anything that renumbers channels
between those two moments silently corrupts every reference.

**Do not let `mean` become NaN-aware.** A NaN-skipping mean would average different time
supports on different channels, and the montage exactness claim dies. The fold propagates
NaN for the same reason `env` does.

**`offset_us` is not `start_us` minus a constant per channel.** It is microseconds from
*bundle* onset, so it needs the earliest start across all channels, which `bundle.py`
knows and a single source does not. That is a signature change, not a subtraction inside
the adapter.

## Tests

The existing suite mirrors the module layout and most of it moves with the code. Three
additions are the regression tests worth naming here, because each guards a mistake that
is easy to reintroduce:

- No absolute wall-clock value appears anywhere outside `meta/`. Assert it over a written
  bundle's consolidated metadata rather than per attribute, so a new attribute cannot slip
  past.
- Float member arrays declare NaN `fill_value`. The failure mode is a silent zero-volt
  flatline, which no waveform assertion catches.
- A montage computed from level-k `mean` equals the fold of the re-referenced raw signal,
  to float tolerance. This is the paper's claim under test in the producer.

## Out of scope

Input formats. MEF carries the exact sample rate that the NWB converter drops, which is
what would turn the rate cross-check in
[gapped recordings](./gapped-recordings.md) into a real one, but that is a converter
change and belongs with the converter.

The reader. It needs matching work and it is a different repository.
