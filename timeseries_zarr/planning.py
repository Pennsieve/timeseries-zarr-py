"""Pyramid level planning: level counts, shapes, and time resolutions.

Raw samples are not a level. Levels are numbered from 1, and level k folds
decimation_factor**k raw samples into each bin.
"""

from timeseries_zarr.constants import (
    DEFAULT_DECIMATION_FACTOR,
    MICROSECONDS_PER_SECOND,
    MIN_DECIMATION_FACTOR,
    max_levels_for,
)
from timeseries_zarr.types import LevelPlan


def _require_samples_per_bin(samples_per_bin: int) -> None:
    """Raise ValueError if samples_per_bin is below 2."""
    if samples_per_bin < MIN_DECIMATION_FACTOR:
        raise ValueError(
            f"samples_per_bin must be {MIN_DECIMATION_FACTOR} or more, "
            f"got {samples_per_bin}"
        )


def _require_rate(rate_hz: float) -> None:
    """Raise ValueError if rate_hz is not positive."""
    if rate_hz <= 0:
        raise ValueError(f"rate_hz must be positive, got {rate_hz!r}")


def level_count(
    num_samples: int,
    max_levels: int | None = None,
    min_bins: int = 1024,
    *,
    decimation_factor: int = DEFAULT_DECIMATION_FACTOR,
) -> int:
    """Return the number of levels for a channel of num_samples samples.

    Raw is not a level. Level k is included only while it would still hold at
    least min_bins complete bins, and the total is capped at max_levels. A
    partial trailing bin does not count toward min_bins, so this gate is
    stricter than level_bins, which keeps the partial bin. A channel too short
    for level 1 gets 0, and so does an empty one.

    Args:
        num_samples: Raw samples in the channel.
        max_levels: Most levels to return. None means the most that
            max_levels_for allows at decimation_factor.
        min_bins: Fewest complete bins a level must hold.
        decimation_factor: The ratio of the bin widths of consecutive levels.

    Returns:
        The number of levels.

    Raises:
        ValueError: decimation_factor is below MIN_DECIMATION_FACTOR.
    """
    cap = max_levels_for(decimation_factor)
    if max_levels is None:
        max_levels = cap
    count = 0
    for level in range(1, max_levels + 1):
        if num_samples // decimation_factor**level < min_bins:
            break
        count += 1
    return count


def level_bins(num_samples: int, samples_per_bin: int) -> int:
    """Return the number of bins of a level for num_samples raw samples.

    A level has one bin per disjoint block of samples_per_bin raw samples, and
    a partial final block becomes one bin.

    Args:
        num_samples: Raw samples in the channel.
        samples_per_bin: Raw samples in one bin of the level.

    Returns:
        The number of bins.

    Raises:
        ValueError: samples_per_bin is below 2.
    """
    _require_samples_per_bin(samples_per_bin)
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


def level_period_us(rate_hz: float, samples_per_bin: int) -> float:
    """Return the microseconds that one bin of a level spans.

    Args:
        rate_hz: Sample rate in hertz.
        samples_per_bin: Raw samples in one bin of the level.

    Raises:
        ValueError: rate_hz is not positive or samples_per_bin is below 2.
    """
    _require_rate(rate_hz)
    _require_samples_per_bin(samples_per_bin)
    return samples_per_bin * MICROSECONDS_PER_SECOND / rate_hz


def raw_shape(num_samples: int) -> tuple[int, ...]:
    """Return the raw array's shape: one axis, one entry per sample."""
    return (num_samples,)


def plan_levels(
    num_samples: int,
    rate_hz: float,
    max_levels: int | None = None,
    min_bins: int = 1024,
    *,
    decimation_factor: int = DEFAULT_DECIMATION_FACTOR,
) -> list[LevelPlan]:
    """Return the pyramid plan for a channel: one LevelPlan per level.

    Args:
        num_samples: Raw samples in the channel.
        rate_hz: Sample rate in hertz.
        max_levels: Most levels to plan. None means the most that
            max_levels_for allows at decimation_factor.
        min_bins: Fewest complete bins a level must hold.
        decimation_factor: The ratio of the bin widths of consecutive levels.
            Level k folds decimation_factor**k raw samples into each bin.

    Returns:
        The plans numbered from 1. An empty list means the channel carries raw
        samples and no levels.

    Raises:
        ValueError: rate_hz is not positive, even when no level fits, or
            decimation_factor is below MIN_DECIMATION_FACTOR.
    """
    _require_rate(rate_hz)
    count = level_count(
        num_samples, max_levels, min_bins, decimation_factor=decimation_factor
    )
    return [
        LevelPlan(
            level=k,
            bins=level_bins(num_samples, decimation_factor**k),
            period_us=level_period_us(rate_hz, decimation_factor**k),
        )
        for k in range(1, count + 1)
    ]
