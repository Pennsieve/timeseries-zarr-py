"""Count levels for event channels: per-bin event counts on a shared ladder."""

import numpy as np

from timeseries_zarr.constants import (
    CANVAS_PIXELS,
    COUNT_LADDER_BASE_US,
    DECIMATION_FACTOR,
    EVENT_LEVEL_THRESHOLD,
    MAX_LEVELS,
    RASTER_BUDGET,
)
from timeseries_zarr.types import LevelPlan


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
