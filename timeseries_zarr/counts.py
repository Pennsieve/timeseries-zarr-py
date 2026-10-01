"""Per-bin event counts: the pyramid a dense event channel carries.

A sparse event channel needs nothing but its own events, which a reader finds by
binary search. A dense one does: a 24-hour, 100-unit sort is a raster of
millions of points at overview zoom, and no viewer draws that. Counting events
into bins and folding the counts by summation gives the same picture in a read
the size of the screen, and the sum fold is exact at every level.

Counts sit on the same 4x ladder as the continuous levels, on the same
onset-relative timeline, but their base period is the producer's to choose
rather than a sample rate. Two things fix it here. The label axis multiplies
everything, so the finest level is sized to a target byte budget given how many
labels the channel has, and the ladder then steps up until a level is smaller
than the screen is wide. That lands the zoom-matched level around a megabyte for
a hundred units over a day, which is the size the format is aiming at.

Counting is streamed rather than accumulated whole. A finest level of a few
million bins across a hundred labels is gigabytes if you hold it, and the events
arrive sorted, so the bins complete in order and can be written as they close.
"""

import math
from collections.abc import Iterable, Iterator

import numpy as np
import numpy.typing as npt

from timeseries_zarr.constants import DECIMATION_FACTOR, UINT32_BYTES
from timeseries_zarr.types import LevelPlan

type EventBlock = tuple[npt.NDArray[np.int64], npt.NDArray[np.uint16] | None]
"""One window of onset-relative event times and, when the channel has them, labels."""


def count_columns(n_labels: int) -> int:
    """Return the width of a count row: one column per label, or one in total."""
    return max(1, n_labels)


def count_shape(bins: int, n_labels: int) -> tuple[int, ...]:
    """Return the array shape of a counts member.

    Labelled channels are (bins, n_labels); an unlabelled one is (bins,), since
    a single column would be a label axis nobody asked for.
    """
    return (bins, n_labels) if n_labels else (bins,)


def base_period_us(
    span_us: int, n_labels: int, target_bytes: int, floor_us: float = 1.0
) -> float:
    """Return the microseconds one bin spans at level 1.

    Sized so the finest level fits target_bytes, because the label axis is what
    makes this expensive: a hundred labels is a hundred times the bins for the
    same span. The floor stops a very short recording from asking for
    sub-microsecond bins.

    The period is a whole number of microseconds because events are binned by
    integer division. A fractional period would be declared but not used, and
    readers would place each bin a little later than the last, adding up to
    seconds by the end of a long recording. Rounded up so the level stays
    within its byte budget.
    """
    budget_bins = max(
        1, target_bytes // (UINT32_BYTES * count_columns(n_labels))
    )
    return float(max(math.ceil(floor_us), -(-span_us // budget_bins)))


def level_bins(span_us: int, period_us: float) -> int:
    """Return the bins a span occupies at one period, keeping a partial last bin."""
    if span_us <= 0:
        return 0
    return int(-(-span_us // int(max(1, round(period_us)))))


def plan_count_levels(
    span_us: int,
    n_labels: int,
    max_levels: int,
    min_bins: int,
    target_bytes: int,
) -> list[LevelPlan]:
    """Return the count pyramid for an event channel, one LevelPlan per level.

    Level 1 is sized by base_period_us and each level above it spans 4x more
    time, while a level still holds at least min_bins. An empty list means the
    channel is short enough that its own events already answer every zoom.
    """
    if span_us <= 0:
        return []
    base = base_period_us(span_us, n_labels, target_bytes)
    bins = level_bins(span_us, base)
    if bins <= 0:
        return []

    plans = [
        LevelPlan(level=1, shape=count_shape(bins, n_labels), period_us=base)
    ]
    for level in range(2, max_levels + 1):
        if bins < min_bins:
            break
        # Each level holds what the fold produces from the one below, not what
        # the span implies. Deriving it from the span instead lets rounding put
        # the array one row short of its own fold, which drops events.
        bins = -(-bins // DECIMATION_FACTOR)
        plans.append(
            LevelPlan(
                level=level,
                shape=count_shape(bins, n_labels),
                period_us=base * DECIMATION_FACTOR ** (level - 1),
            )
        )
    return plans


def fold_counts_block(
    counts: npt.NDArray[np.uint32],
) -> npt.NDArray[np.uint32]:
    """Sum 4 rows of counts into one, keeping a partial trailing group.

    Summation is why the count fold is exact at every level: nothing is averaged
    and nothing is dropped, so a coarse bin holds the true number of events in
    its span.
    """
    block = DECIMATION_FACTOR
    rows = counts.shape[0]
    n_full, tail = rows // block, rows % block
    pieces = []
    if n_full:
        head = counts[: n_full * block].reshape(
            n_full, block, *counts.shape[1:]
        )
        pieces.append(head.sum(axis=1, dtype=np.uint32))
    if tail:
        rest = counts[n_full * block :].sum(axis=0, dtype=np.uint32)
        pieces.append(rest.reshape(1, *counts.shape[1:]))
    if not pieces:
        return np.empty((0, *counts.shape[1:]), dtype=np.uint32)
    return np.concatenate(pieces, axis=0)


def iter_count_blocks(
    event_blocks: Iterable[EventBlock],
    n_labels: int,
    period_us: float,
    bins: int,
    block_bins: int,
) -> Iterator[npt.NDArray[np.uint32]]:
    """Turn sorted event windows into contiguous blocks of level-1 counts.

    Events arrive in time order, so a bin is finished once an event lands past
    it and the block holding it can be written and dropped. Blocks are emitted
    for the whole axis, including stretches no event falls in, so the array is
    covered. Raises ValueError if block_bins is not positive.
    """
    if block_bins <= 0:
        raise ValueError("block_bins must be positive")

    columns = count_columns(n_labels)
    period = int(max(1, round(period_us)))
    buffer = np.zeros((block_bins, columns), dtype=np.uint32)
    base = 0

    def emitted(rows: int) -> npt.NDArray[np.uint32]:
        window = buffer[:rows]
        return window.reshape(rows) if n_labels == 0 else window

    for times, labels in event_blocks:
        index = times // period
        column = (
            np.zeros(index.shape[0], dtype=np.int64)
            if labels is None
            else labels.astype(np.int64)
        )
        while index.shape[0]:
            limit = base + block_bins
            taken = int(np.searchsorted(index, limit))
            if taken:
                flat = (index[:taken] - base) * columns + column[:taken]
                buffer += (
                    np.bincount(flat, minlength=block_bins * columns)
                    .reshape(block_bins, columns)
                    .astype(np.uint32)
                )
            if taken == index.shape[0]:
                break
            yield emitted(min(block_bins, bins - base)).copy()
            base += block_bins
            buffer[:] = 0
            index, column = index[taken:], column[taken:]

    while base < bins:
        yield emitted(min(block_bins, bins - base)).copy()
        base += block_bins
        buffer[:] = 0
