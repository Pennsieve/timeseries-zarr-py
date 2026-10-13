"""Count levels for event channels: per-bin event counts on a shared ladder."""

from timeseries_zarr.constants import (
    CANVAS_PIXELS,
    COUNT_LADDER_BASE_US,
    DECIMATION_FACTOR,
    RASTER_BUDGET,
)


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
