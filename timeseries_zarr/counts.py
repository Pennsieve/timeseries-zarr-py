"""Count levels for event channels: per-bin event counts on a shared ladder."""

from collections.abc import Iterable, Iterator
from typing import Final

import numpy as np
import numpy.typing as npt

from timeseries_zarr.constants import (
    CANVAS_PIXELS,
    COUNT_LADDER_BASE_US,
    DECIMATION_FACTOR,
    EVENT_LEVEL_THRESHOLD,
    MAX_LEVELS,
    RASTER_BUDGET,
)
from timeseries_zarr.types import LevelPlan

COUNT_BLOCK_RANK: Final = 2
"""Rank of a block of counts: one axis of bins, one of label columns."""

type EventBlock = tuple[npt.NDArray[np.int64], npt.NDArray[np.uint16] | None]
"""One window of event times in microseconds from the origin, with the labels.

The labels are None for a channel that has none."""


def level1_period_us(n_events: int, last_event_us: int) -> int:
    """Return the period in microseconds of an event channel's count level 1.

    The finest count level a reader needs is one canvas pixel wide at the
    widest window it still serves from the events themselves: a window holding
    RASTER_BUDGET events at the channel's mean rate, drawn across
    CANVAS_PIXELS pixels. The period is that width rounded down to the ladder
    COUNT_LADDER_BASE_US times a power of 4, so every event channel in a
    bundle shares one set of bin grids. It is never finer than
    COUNT_LADDER_BASE_US, and a width exactly on a rung of the ladder keeps
    that rung. The arithmetic is exact in integers. last_event_us is the time
    of the last event from the bundle origin, and an event at the origin alone
    gives the finest period. Raises ValueError if n_events is below 1 or
    last_event_us is negative.
    """
    if n_events < 1:
        raise ValueError(f"n_events must be at least 1, got {n_events}")
    if last_event_us < 0:
        raise ValueError(
            f"last_event_us must be at least 0, got {last_event_us}"
        )
    width_us = RASTER_BUDGET * last_event_us // (n_events * CANVAS_PIXELS)
    period = COUNT_LADDER_BASE_US
    while period * DECIMATION_FACTOR <= width_us:
        period *= DECIMATION_FACTOR
    return period


def count_bins(last_event_us: int, period_us: int) -> int:
    """Return the number of bins that cover an event channel's last event.

    The bin axis starts at the bundle origin, and bin i spans the times from
    i * period_us up to but not including (i + 1) * period_us. The result is
    the bin holding the last event plus one, so an event that sits exactly on
    a bin boundary is counted in the bin it opens and is never dropped. A
    channel whose only event is at the origin has 1 bin. Raises ValueError if
    period_us is below 1 or last_event_us is negative.
    """
    if period_us < 1:
        raise ValueError(f"period_us must be at least 1, got {period_us}")
    if last_event_us < 0:
        raise ValueError(
            f"last_event_us must be at least 0, got {last_event_us}"
        )
    return last_event_us // period_us + 1


def plan_count_levels(
    n_events: int,
    last_event_us: int,
    max_levels: int = MAX_LEVELS,
    min_bins: int = 1024,
) -> list[LevelPlan]:
    """Return the count levels of an event channel, one LevelPlan per level.

    A channel with at most EVENT_LEVEL_THRESHOLD events gets none, because a
    reader serves any window from the events themselves. Level 1 takes the
    period from level1_period_us and the bin count from count_bins. Each level
    above it spans 4x the time and holds a quarter of the bins, rounded up, and
    the ladder stops before a level that would hold fewer than min_bins bins or
    would pass max_levels. A channel whose level 1 already holds fewer than
    min_bins bins gets none. The plan is the same for every label column.
    Raises ValueError if n_events is negative or above the uint32 range, which
    a count bin can hold when equal times put every event in one bin, or if
    last_event_us is negative.
    """
    if n_events < 0:
        raise ValueError(f"n_events must be at least 0, got {n_events}")
    if n_events > np.iinfo(np.uint32).max:
        raise ValueError(
            f"n_events {n_events} exceeds the uint32 range of a count bin"
        )
    if last_event_us < 0:
        raise ValueError(
            f"last_event_us must be at least 0, got {last_event_us}"
        )
    if n_events <= EVENT_LEVEL_THRESHOLD:
        return []
    period = level1_period_us(n_events, last_event_us)
    bins = count_bins(last_event_us, period)
    plans: list[LevelPlan] = []
    for level in range(1, max_levels + 1):
        if bins < min_bins:
            break
        plans.append(LevelPlan(level=level, bins=bins, period_us=float(period)))
        bins = -(-bins // DECIMATION_FACTOR)
        period *= DECIMATION_FACTOR
    return plans


def _check_event_block(
    times: npt.NDArray[np.int64],
    labels: npt.NDArray[np.uint16] | None,
    columns: int,
    previous_bin: int,
    period_us: int,
    bins: int,
) -> npt.NDArray[np.int64]:
    """Return the bin of each event, raising ValueError if the block is bad."""
    if labels is not None and labels.shape != times.shape:
        raise ValueError(
            f"labels have shape {labels.shape} but times have {times.shape}"
        )
    index = times // period_us
    if index[0] < max(previous_bin, 0) or (index[1:] < index[:-1]).any():
        raise ValueError("event times must be non-decreasing and at least 0")
    if index[-1] >= bins:
        raise ValueError(
            f"an event falls in bin {int(index[-1])}, past the {bins} bins"
        )
    if labels is not None and int(labels.max()) >= columns:
        raise ValueError(
            f"a label {int(labels.max())} is outside the {columns} columns"
        )
    return index


def iter_count_blocks(
    event_blocks: Iterable[EventBlock],
    n_labels: int,
    period_us: int,
    bins: int,
    block_bins: int,
) -> Iterator[npt.NDArray[np.uint32]]:
    """Yield level 1 counts in blocks of block_bins rows, covering every bin.

    Event i with time t counts in bin t // period_us, in the column of its
    label. The counts have max(1, n_labels) columns, and a block without
    labels counts every event in column 0. The blocks are contiguous and
    cover the whole axis of bins rows, including stretches no event falls in:
    every block has block_bins rows except the last, which holds the
    remainder. An axis of no bins yields nothing. Times must be non-decreasing
    across all blocks, so a block is complete, and is written out, as soon as
    an event lands past it; only one block of counts is held at a time. A
    block of no events is skipped. Raises ValueError if block_bins or
    period_us is below 1, bins or n_labels is negative, a time is negative or
    decreases, an event falls past the last bin, a label is outside the
    columns, or labels and times differ in shape.
    """
    if block_bins < 1:
        raise ValueError(f"block_bins must be at least 1, got {block_bins}")
    if period_us < 1:
        raise ValueError(f"period_us must be at least 1, got {period_us}")
    if bins < 0:
        raise ValueError(f"bins must be at least 0, got {bins}")
    if n_labels < 0:
        raise ValueError(f"n_labels must be at least 0, got {n_labels}")
    columns = max(1, n_labels)
    buffer = np.zeros((block_bins, columns), dtype=np.uint32)
    base = 0
    previous_bin = 0
    for times, labels in event_blocks:
        if not times.shape[0]:
            continue
        index = _check_event_block(
            times, labels, columns, previous_bin, period_us, bins
        )
        previous_bin = int(index[-1])
        column = (
            np.zeros(index.shape[0], dtype=np.int64)
            if labels is None
            else labels.astype(np.int64)
        )
        while index.shape[0]:
            taken = int(np.searchsorted(index, base + block_bins))
            if taken:
                flat = (index[:taken] - base) * columns + column[:taken]
                buffer += (
                    np.bincount(flat, minlength=block_bins * columns)
                    .reshape(block_bins, columns)
                    .astype(np.uint32)
                )
            if taken == index.shape[0]:
                break
            yield buffer.copy()
            base += block_bins
            buffer[:] = 0
            index, column = index[taken:], column[taken:]
    while base < bins:
        yield buffer[: min(block_bins, bins - base)].copy()
        base += block_bins
        buffer[:] = 0


def fold_count_block(
    counts: npt.NDArray[np.uint32],
) -> npt.NDArray[np.uint32]:
    """Fold a block of counts into the next coarser level over blocks of 4 rows.

    Each output row is the column-wise sum of up to 4 input rows, so a coarse
    bin holds exactly the events of its span. A final partial group of 1 to 3
    rows becomes one row. An empty input yields zero rows. The sums stay in
    uint32, which holds them because plan_count_levels caps the event count at
    the uint32 range. Raises ValueError if counts is not shaped (rows,
    columns).
    """
    if len(counts.shape) != COUNT_BLOCK_RANK:
        raise ValueError(
            f"counts have shape (rows, columns), got {counts.shape}"
        )
    n_full, tail_len = divmod(counts.shape[0], DECIMATION_FACTOR)
    split = n_full * DECIMATION_FACTOR
    columns = counts.shape[1]
    out = np.empty((n_full + (1 if tail_len else 0), columns), dtype=np.uint32)
    if n_full:
        grouped = counts[:split].reshape(n_full, DECIMATION_FACTOR, columns)
        out[:n_full] = grouped.sum(axis=1, dtype=np.uint32)
    if tail_len:
        out[n_full] = counts[split:].sum(axis=0, dtype=np.uint32)
    return out
