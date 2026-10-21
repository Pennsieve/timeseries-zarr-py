"""Pyramid level planning: level counts, shapes, and time resolutions.

Raw samples are not a level. Levels are numbered from 1, and level k folds
4**k raw samples into each bin.
"""

from timeseries_zarr.constants import (
    DECIMATION_FACTOR,
    MAX_LEVELS,
    MICROSECONDS_PER_SECOND,
)
from timeseries_zarr.types import LevelPlan


def _require_level(level: int) -> None:
    """Raise ValueError if level is below 1."""
    if level < 1:
        raise ValueError(
            f"levels are numbered from 1 and raw is not a level, got {level}"
        )


def _require_rate(rate_hz: float) -> None:
    """Raise ValueError if rate_hz is not positive."""
    if rate_hz <= 0:
        raise ValueError(f"rate_hz must be positive, got {rate_hz!r}")


def level_count(
    num_samples: int, max_levels: int = MAX_LEVELS, min_bins: int = 1024
) -> int:
    """Return the number of levels for a channel of num_samples samples.

    Raw is not a level. Level k is included only while it would still hold at
    least min_bins complete bins, and the total is capped at max_levels. A
    partial trailing bin does not count toward min_bins, so this gate is
    stricter than level_bins, which keeps the partial bin. A channel too short
    for level 1 gets 0, and so does an empty one.

    Args:
        num_samples: Raw samples in the channel.
        max_levels: Most levels to return.
        min_bins: Fewest complete bins a level must hold.

    Returns:
        The number of levels.
    """
    count = 0
    for level in range(1, max_levels + 1):
        if num_samples // DECIMATION_FACTOR**level < min_bins:
            break
        count += 1
    return count


def level_bins(num_samples: int, level: int) -> int:
    """Return the number of bins at a level for num_samples raw samples.

    Level k has one bin per disjoint block of 4**k raw samples, and a partial
    final block becomes one bin.

    Args:
        num_samples: Raw samples in the channel.
        level: Level number, from 1.

    Returns:
        The number of bins.

    Raises:
        ValueError: level is below 1.
    """
    _require_level(level)
    samples_per_bin: int = DECIMATION_FACTOR**level
    return -(-num_samples // samples_per_bin)


def sample_period_us(rate_hz: float) -> float:
    """Return the spacing between raw samples in microseconds.

    Args:
        rate_hz: Sample rate in hertz.

    Raises:
        ValueError: rate_hz is not positive.
    """
    _require_rate(rate_hz)
    return MICROSECONDS_PER_SECOND / rate_hz


def level_period_us(rate_hz: float, level: int) -> float:
    """Return the microseconds that one bin spans at the given level.

    Each level spans 4x more time than the one below, and level 1 spans 4 raw
    samples.

    Args:
        rate_hz: Sample rate in hertz.
        level: Level number, from 1.

    Raises:
        ValueError: rate_hz is not positive or level is below 1.
    """
    _require_rate(rate_hz)
    _require_level(level)
    samples_per_bin: int = DECIMATION_FACTOR**level
    return samples_per_bin * MICROSECONDS_PER_SECOND / rate_hz


def raw_shape(num_samples: int) -> tuple[int, ...]:
    """Return the raw array's shape: one axis, one entry per sample."""
    return (num_samples,)


def plan_levels(
    num_samples: int,
    rate_hz: float,
    max_levels: int = MAX_LEVELS,
    min_bins: int = 1024,
) -> list[LevelPlan]:
    """Return the pyramid plan for a channel: one LevelPlan per level.

    Args:
        num_samples: Raw samples in the channel.
        rate_hz: Sample rate in hertz.
        max_levels: Most levels to plan.
        min_bins: Fewest complete bins a level must hold.

    Returns:
        The plans numbered from 1. An empty list means the channel carries raw
        samples and no levels.

    Raises:
        ValueError: rate_hz is not positive, even when no level fits.
    """
    _require_rate(rate_hz)
    return [
        LevelPlan(
            level=k,
            bins=level_bins(num_samples, k),
            period_us=level_period_us(rate_hz, k),
        )
        for k in range(1, level_count(num_samples, max_levels, min_bins) + 1)
    ]
