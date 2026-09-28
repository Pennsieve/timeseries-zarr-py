"""Mapping a timestamped recording onto a uniform sample grid.

A recording with breaks in it carries one timestamp per sample, because a
fixed rate asserts unbroken regularity. The bundle stores no time axis, so the
writer places each sample on the grid position its timestamp names and leaves
the untouched positions NaN.

Nothing here imports NWB or Zarr. Timestamps arrive as any object supporting
``len()`` and slicing to a float64 array, so an in-memory array and an HDF5
dataset both work and neither is read whole.
"""

import logging
from bisect import bisect_right
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

READ_BLOCK_SAMPLES = 2**22
"""Timestamps read per block (~4.2M, 32 MiB as float64) when scanning."""

RATE_PROBE_SAMPLES = 2**21
"""Timestamps inspected when deriving a rate (~2.1M)."""

GAP_THRESHOLD = 1.5
"""Multiples of the nominal period that separate a gap from a sample step."""

RATE_REL_TOL = 1e-3
"""Relative tolerance when checking a given rate against the timestamps."""

MIN_TIMESTAMPS_FOR_RATE = 2
"""Timestamps needed before a rate can be measured from their span."""


class Timestamps(Protocol):
    """A sliceable, sized source of float64 timestamps in seconds."""

    def __len__(self) -> int:
        """Return the number of timestamps."""
        ...

    def __getitem__(self, item: slice) -> npt.NDArray[np.float64]:
        """Return the sliced timestamps."""
        ...


@dataclass(frozen=True, slots=True)
class Segment:
    """One contiguous run of recorded samples on the grid.

    ``grid_start`` and ``grid_stop`` are half-open positions on the uniform
    grid; ``sample_start`` and ``sample_stop`` are the half-open indices into
    the recording that fill them. Both spans have the same length.
    """

    grid_start: int
    grid_stop: int
    sample_start: int
    sample_stop: int

    @property
    def length(self) -> int:
        """Return the number of samples in the run."""
        return self.sample_stop - self.sample_start


def _read(
    timestamps: Timestamps, start: int, stop: int
) -> npt.NDArray[np.float64]:
    """Return the half-open [start, stop) timestamps as float64 seconds."""
    return np.asarray(timestamps[start:stop], dtype=np.float64)


def derive_rate_hz(
    timestamps: Timestamps, *, probe: int = RATE_PROBE_SAMPLES
) -> float:
    """Return the sample rate implied by the longest gap-free run of timestamps.

    Measured over the run's whole span rather than from the median interval.
    Timestamps are often dithered between two neighboring representable values
    whose mean is the true period; the median then returns one of them and
    drifts. On a real 512 Hz recording the median gives 511.967234 Hz, which
    misplaces samples by half a minute across six days.

    The longest run is used rather than the first because a recording can
    open with a run too short to measure, such as one sample before a gap.
    The longest run also averages the dither over the most samples.

    Only the first ``probe`` timestamps are read. Raises ValueError when fewer
    than two are available, when they do not increase, or when every run in
    the probe spans no time.
    """
    count = min(len(timestamps), probe)
    if count < MIN_TIMESTAMPS_FOR_RATE:
        raise ValueError("a rate needs at least two timestamps")

    window = _read(timestamps, 0, count)
    steps = np.diff(window)
    nominal = float(np.median(steps))
    if nominal <= 0.0:
        raise ValueError("timestamps are not increasing")

    # A run ends at each gap. Pick the one with the most samples; ties go to
    # the earliest.
    breaks = np.flatnonzero(steps > nominal * GAP_THRESHOLD)
    run_starts = np.concatenate([[0], breaks + 1])
    run_stops = np.concatenate([breaks + 1, [count]])
    longest = int(np.argmax(run_stops - run_starts))
    run_start = int(run_starts[longest])
    run_stop = int(run_stops[longest])

    # DEBUG: suspected cause of "spans no time" on MEF-derived files is a
    # one-sample first run followed by a gap. Log the timestamps whenever the
    # first run isn't the one used, so a real run can confirm it.
    if longest != 0 or run_stop - run_start < MIN_TIMESTAMPS_FOR_RATE:
        logger.warning(
            "derive_rate_hz: first run not usable or not longest. first 10 "
            "timestamps=%s, median step=%r, probed=%d, gaps found=%d, first 5 "
            "gap indices=%s, their steps=%s, first run length=%d, using run "
            "[%d, %d) of length %d",
            window[:10].tolist(),
            nominal,
            count,
            breaks.size,
            breaks[:5].tolist(),
            steps[breaks[:5]].tolist(),
            int(run_stops[0] - run_starts[0]),
            run_start,
            run_stop,
            run_stop - run_start,
        )

    span = float(window[run_stop - 1] - window[run_start])
    if span <= 0.0:
        raise ValueError("no run of timestamps in the probe spans any time")
    return (run_stop - run_start - 1) / span


def build_segments(
    timestamps: Timestamps,
    rate_hz: float,
    *,
    block: int = READ_BLOCK_SAMPLES,
) -> list[Segment]:
    """Group timestamps into the contiguous runs they occupy on the grid.

    Each sample lands on the grid position its timestamp names; a run ends
    where the next sample's position is more than one past the last. The
    timestamps are read a block at a time and scanned with array operations,
    so neither memory nor time grows with the number of samples in a run.

    Raises ValueError when two samples land on the same position or the
    positions run backwards, both of which mean the rate is too low, and when
    the timestamps imply a different rate, which means the recording is
    irregular rather than gapped or the rate is wrong.
    """
    if rate_hz <= 0.0:
        raise ValueError(f"rate must be positive, got {rate_hz!r}")

    total = len(timestamps)
    if total == 0:
        return []

    origin = float(_read(timestamps, 0, 1)[0])
    segments: list[Segment] = []
    run_grid_start = 0
    run_sample_start = 0
    previous_slot: int | None = None

    for offset in range(0, total, block):
        window = _read(timestamps, offset, min(offset + block, total))
        slots = np.rint((window - origin) * rate_hz).astype(np.int64)

        if previous_slot is not None:
            bridge = int(slots[0]) - previous_slot
            _reject_non_advancing(bridge, offset, rate_hz)
            if bridge > 1:
                segments.append(
                    Segment(
                        run_grid_start,
                        previous_slot + 1,
                        run_sample_start,
                        offset,
                    )
                )
                run_grid_start = int(slots[0])
                run_sample_start = offset

        steps = np.diff(slots)
        stalled = np.flatnonzero(steps <= 0)
        if stalled.size:
            index = offset + int(stalled[0]) + 1
            _reject_non_advancing(int(steps[stalled[0]]), index, rate_hz)

        for local in np.flatnonzero(steps > 1):
            sample = offset + int(local) + 1
            segments.append(
                Segment(
                    run_grid_start,
                    int(slots[local]) + 1,
                    run_sample_start,
                    sample,
                )
            )
            run_grid_start = int(slots[local + 1])
            run_sample_start = sample

        previous_slot = int(slots[-1])

    assert previous_slot is not None
    segments.append(
        Segment(run_grid_start, previous_slot + 1, run_sample_start, total)
    )
    if total >= MIN_TIMESTAMPS_FOR_RATE:
        _check_rate(timestamps, rate_hz)
    return segments


def _reject_non_advancing(step: int, sample: int, rate_hz: float) -> None:
    """Raise when a sample does not land past its predecessor on the grid."""
    if step > 0:
        return
    raise ValueError(
        f"samples {sample - 1} and {sample} do not advance on the grid at "
        f"{rate_hz} Hz (step {step}): the rate is too low"
    )


def _check_rate(timestamps: Timestamps, rate_hz: float) -> None:
    """Raise when rate_hz disagrees with the rate the timestamps imply.

    A rate too low for the recording is caught while segmenting, because two
    samples then land on one grid position. A rate too high is not: every
    sample simply lands two or more positions past the last, which is
    indistinguishable from a recording that is nothing but gaps. Comparing
    against the rate the timestamps themselves imply catches that case.
    """
    implied = derive_rate_hz(timestamps)
    if abs(implied - rate_hz) > rate_hz * RATE_REL_TOL:
        raise ValueError(
            f"timestamps imply {implied:.6f} Hz but the rate given is "
            f"{rate_hz:.6f} Hz: the recording is irregular rather than "
            "gapped, or the rate is wrong"
        )


def grid_length(segments: list[Segment]) -> int:
    """Return the total grid positions the runs span, gaps included."""
    return segments[-1].grid_stop if segments else 0


def segments_overlapping(
    segments: list[Segment], grid_start: int, grid_stop: int
) -> list[Segment]:
    """Return the runs intersecting the half-open grid window, in order.

    Bisects the run list, which holds one entry per break rather than per
    sample, so a read costs a lookup over a short list however long the
    recording is.
    """
    if grid_stop <= grid_start or not segments:
        return []
    starts = [segment.grid_start for segment in segments]
    first = max(bisect_right(starts, grid_start) - 1, 0)
    return [
        segment
        for segment in segments[first:]
        if segment.grid_start < grid_stop and segment.grid_stop > grid_start
    ]
