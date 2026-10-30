"""Count levels for event channels: per-bin event counts on a shared ladder."""

from collections.abc import Iterable, Iterator
from typing import Final

import numpy as np
import numpy.typing as npt

from timeseries_zarr.attrs import check_time_us
from timeseries_zarr.constants import (
    CANVAS_PIXELS,
    COUNT_LADDER_BASE_US,
    DEFAULT_DECIMATION_FACTOR,
    EVENT_COUNT_LIMIT,
    EVENT_LEVEL_THRESHOLD,
    MIN_DECIMATION_FACTOR,
    RASTER_BUDGET,
    max_levels_for,
)
from timeseries_zarr.types import LevelPlan

COUNT_BLOCK_RANK: Final = 2
"""Rank of a block of counts: one axis of bins, one of label columns."""

type EventBlock = tuple[npt.NDArray[np.int64], npt.NDArray[np.uint16] | None]
"""One window of event times in microseconds from the reference time, with labels.

The labels are None for a channel that has none."""


def _check_factor(decimation_factor: int) -> None:
    """Raise ValueError unless decimation_factor is at least the minimum."""
    if decimation_factor < MIN_DECIMATION_FACTOR:
        raise ValueError(
            f"decimation_factor must be at least {MIN_DECIMATION_FACTOR}, "
            f"got {decimation_factor}"
        )


def level1_period_us(
    n_events: int,
    span_us: int,
    decimation_factor: int = DEFAULT_DECIMATION_FACTOR,
) -> int:
    """Return the period in microseconds of an event channel's count level 1.

    The finest count level a reader needs is one canvas pixel wide at the
    widest window it still serves from the events themselves: a window holding
    RASTER_BUDGET events at the channel's mean rate, drawn across
    CANVAS_PIXELS pixels. The period is that width rounded down to the ladder
    COUNT_LADDER_BASE_US times a power of decimation_factor, so every event
    channel in a bundle shares one set of bin grids. It is never finer than
    COUNT_LADDER_BASE_US, and a width exactly on a rung of the ladder keeps
    that rung. The arithmetic is exact in integers. Events all at one time
    give the finest period.

    Args:
        n_events: The number of events in the channel.
        span_us: The time from the first event to the last.
        decimation_factor: The ratio of the periods of consecutive levels, at
            least MIN_DECIMATION_FACTOR.

    Raises:
        ValueError: If n_events is below 1, span_us is negative, or
            decimation_factor is below MIN_DECIMATION_FACTOR.
    """
    _check_factor(decimation_factor)
    if n_events < 1:
        raise ValueError(f"n_events must be at least 1, got {n_events}")
    if span_us < 0:
        raise ValueError(f"span_us must be at least 0, got {span_us}")
    width_us = RASTER_BUDGET * span_us // (n_events * CANVAS_PIXELS)
    period = COUNT_LADDER_BASE_US
    while period * decimation_factor <= width_us:
        period *= decimation_factor
    return period


def bin_offset_us(first_event_us: int, period_us: int) -> int:
    """Return the time of bin 0 of a count level: the first event's bin start.

    It is the largest multiple of period_us at or before first_event_us, and
    may be negative.

    Args:
        first_event_us: The time of the first event.
        period_us: The period of the level.

    Raises:
        ValueError: If period_us is below 1.
    """
    if period_us < 1:
        raise ValueError(f"period_us must be at least 1, got {period_us}")
    return (first_event_us // period_us) * period_us


def count_bins(first_event_us: int, last_event_us: int, period_us: int) -> int:
    """Return the number of bins from the first event's bin through the last's.

    Bin i spans the times from bin_offset_us(first_event_us, period_us) +
    i * period_us up to but not including the next multiple of period_us, so
    an event that sits exactly on a bin boundary is counted in the bin it
    opens and is never dropped. A channel whose events all share one time
    has 1 bin.

    Args:
        first_event_us: The time of the first event.
        last_event_us: The time of the last event.
        period_us: The period of the level.

    Raises:
        ValueError: If period_us is below 1 or last_event_us precedes
            first_event_us.
    """
    if period_us < 1:
        raise ValueError(f"period_us must be at least 1, got {period_us}")
    if last_event_us < first_event_us:
        raise ValueError(
            f"last_event_us {last_event_us} precedes first_event_us "
            f"{first_event_us}"
        )
    return last_event_us // period_us - first_event_us // period_us + 1


def plan_count_levels(
    n_events: int,
    first_event_us: int,
    last_event_us: int,
    max_levels: int | None = None,
    min_bins: int = 1024,
    *,
    decimation_factor: int = DEFAULT_DECIMATION_FACTOR,
) -> list[LevelPlan]:
    """Return the count levels of an event channel, one LevelPlan per level.

    A channel with at most EVENT_LEVEL_THRESHOLD events gets none, because a
    reader serves any window from the events themselves. Level 1 takes the
    period from level1_period_us. Each level above it has a period
    decimation_factor times that of the level below, and
    every level takes its offset_us from bin_offset_us and its bins from
    count_bins, so its bins cover every event from the first to the last. The
    ladder stops before a level that would hold fewer than min_bins bins or
    would pass max_levels. A channel whose level 1 already holds fewer than
    min_bins bins gets none. The plan is the same for every label column.

    Args:
        n_events: The number of events in the channel.
        first_event_us: The time of the first event.
        last_event_us: The time of the last event.
        max_levels: The most levels to plan. None means the most that
            max_levels_for allows at decimation_factor.
        min_bins: The fewest bins a level may hold.
        decimation_factor: The ratio of the periods of consecutive levels, at
            least MIN_DECIMATION_FACTOR.

    Raises:
        ValueError: If n_events is negative or above the uint32 range, which
            a count bin can hold when equal times put every event in one bin,
            if last_event_us precedes first_event_us, if decimation_factor
            is below MIN_DECIMATION_FACTOR, or if the offset or the end of the
            last bin of a level lies outside (-2^53, 2^53) microseconds.
    """
    _check_factor(decimation_factor)
    if max_levels is None:
        max_levels = max_levels_for(decimation_factor)
    if n_events < 0:
        raise ValueError(f"n_events must be at least 0, got {n_events}")
    if n_events >= EVENT_COUNT_LIMIT:
        raise ValueError(
            f"n_events {n_events} exceeds the uint32 range of a count bin"
        )
    if last_event_us < first_event_us:
        raise ValueError(
            f"last_event_us {last_event_us} precedes first_event_us "
            f"{first_event_us}"
        )
    if n_events <= EVENT_LEVEL_THRESHOLD:
        return []
    period = level1_period_us(
        n_events, last_event_us - first_event_us, decimation_factor
    )
    plans: list[LevelPlan] = []
    for level in range(1, max_levels + 1):
        bins = count_bins(first_event_us, last_event_us, period)
        if bins < min_bins:
            break
        offset_us = bin_offset_us(first_event_us, period)
        check_time_us(offset_us, f"offset_us of count level {level}")
        check_time_us(
            offset_us + bins * period,
            f"the end of the last bin of count level {level}",
        )
        plans.append(
            LevelPlan(
                level=level, bins=bins, period_us=period, offset_us=offset_us
            )
        )
        period *= decimation_factor
    return plans


def _check_event_block(
    times: npt.NDArray[np.int64],
    labels: npt.NDArray[np.uint16] | None,
    columns: int,
    previous_bin: int,
    period_us: int,
    first_bin: int,
    bins: int,
) -> npt.NDArray[np.int64]:
    """Return the bin of each event.

    Raises:
        ValueError: If the block is out of order or falls outside the bins or
            columns.
    """
    if labels is not None and labels.shape != times.shape:
        raise ValueError(
            f"labels have shape {labels.shape} but times have {times.shape}"
        )
    index = times // period_us - first_bin
    if index[0] < 0:
        raise ValueError(
            f"an event at {int(times[0])} precedes bin 0 at "
            f"{first_bin * period_us}"
        )
    if index[0] < previous_bin or (index[1:] < index[:-1]).any():
        raise ValueError("event times must be non-decreasing")
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
    offset_us: int,
    bins: int,
    block_bins: int,
) -> Iterator[npt.NDArray[np.uint32]]:
    """Yield level 1 counts in blocks of block_bins rows, covering every bin.

    Event i with time t counts in bin t // period_us - offset_us // period_us,
    in the column of its label. The counts have max(1, n_labels) columns, and
    a block without labels counts every event in column 0. The blocks are
    contiguous and
    cover the whole axis of bins rows, including stretches no event falls in:
    every block has block_bins rows except the last, which holds the
    remainder. An axis of no bins yields nothing. Times must be non-decreasing
    across all blocks, so a block is complete, and is written out, as soon as
    an event lands past it; only one block of counts is held at a time. A
    block of no events is skipped.

    Args:
        event_blocks: Windows of event times and labels, in time order.
        n_labels: The number of label columns. 0 counts every event in one
            column.
        period_us: The period of the level.
        offset_us: The time of bin 0, a multiple of period_us.
        bins: The number of bins on the axis.
        block_bins: The number of rows in each yielded block.

    Raises:
        ValueError: If block_bins or period_us is below 1, offset_us is not a
            multiple of period_us, bins or n_labels is negative, an event
            precedes bin 0, a time decreases, an event falls past the last
            bin, a label is outside the columns, or labels and times differ
            in shape.
    """
    if block_bins < 1:
        raise ValueError(f"block_bins must be at least 1, got {block_bins}")
    if period_us < 1:
        raise ValueError(f"period_us must be at least 1, got {period_us}")
    if offset_us % period_us:
        raise ValueError(
            f"offset_us must be a multiple of period_us, got {offset_us} "
            f"and {period_us}"
        )
    if bins < 0:
        raise ValueError(f"bins must be at least 0, got {bins}")
    if n_labels < 0:
        raise ValueError(f"n_labels must be at least 0, got {n_labels}")
    columns = max(1, n_labels)
    buffer = np.zeros((block_bins, columns), dtype=np.uint32)
    base = 0
    previous_bin = 0
    first_bin = offset_us // period_us
    for times, labels in event_blocks:
        if not times.shape[0]:
            continue
        index = _check_event_block(
            times, labels, columns, previous_bin, period_us, first_bin, bins
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
    counts: npt.NDArray[np.uint32], block: int
) -> npt.NDArray[np.uint32]:
    """Fold a block of counts into the next coarser level in groups of rows.

    Each output row is the column-wise sum of up to block input rows, so a
    coarse bin holds exactly the events of its span. A final partial group of
    fewer than block rows becomes one row. An empty input yields zero rows.
    The sums stay in uint32, which holds them because plan_count_levels caps
    the event count at the uint32 range.

    Args:
        counts: A block of counts shaped (rows, columns).
        block: The number of rows that fold into one, at least 1.

    Raises:
        ValueError: If counts is not shaped (rows, columns) or block is not
            positive.
    """
    if len(counts.shape) != COUNT_BLOCK_RANK:
        raise ValueError(
            f"counts have shape (rows, columns), got {counts.shape}"
        )
    if block < 1:
        raise ValueError(f"block must be at least 1, got {block}")
    n_full, tail_len = divmod(counts.shape[0], block)
    split = n_full * block
    columns = counts.shape[1]
    out = np.empty((n_full + (1 if tail_len else 0), columns), dtype=np.uint32)
    if n_full:
        grouped = counts[:split].reshape(n_full, block, columns)
        out[:n_full] = grouped.sum(axis=1, dtype=np.uint32)
    if tail_len:
        out[n_full] = counts[split:].sum(axis=0, dtype=np.uint32)
    return out
