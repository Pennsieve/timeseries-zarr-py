"""Pyramid level planning: level counts, shapes, and time resolutions."""

from timeseries_zarr.constants import (
    DECIMATION_FACTOR,
    ENVELOPE_PAIR_SIZE,
    MICROSECONDS_PER_SECOND,
)
from timeseries_zarr.types import LevelPlan


def level_count(
    num_samples: int, max_levels: int = 8, min_bins: int = 1024
) -> int:
    """Return the number of pyramid levels for a channel of num_samples samples.

    Level 0 (raw) always counts, even when num_samples is 0. Each coarser level is
    included only while it would still hold at least min_bins complete bins, and
    the total is capped at max_levels. A partial trailing bin does not count
    toward min_bins, so this gate is one stricter than level_num_bins, which keeps
    the partial bin.
    """
    count = 1
    for level in range(1, max_levels):
        if num_samples // DECIMATION_FACTOR**level < min_bins:
            break
        count += 1
    return count


def level_num_bins(num_samples: int, level: int) -> int:
    """Return the number of bins at a pyramid level for num_samples samples.

    Level 0 has one bin per raw sample. Each coarser level has one bin per
    disjoint block of 4 bins from the level below, and a partial final block
    becomes one bin.
    """
    samples_per_bin: int = DECIMATION_FACTOR**level
    return -(-num_samples // samples_per_bin)


def level0_period_us(sample_rate_hz: float) -> float:
    """Return level-0 raw sample spacing in microseconds."""
    return MICROSECONDS_PER_SECOND / sample_rate_hz


def level_period_us(level0_period_us: float, level: int) -> float:
    """Return the microseconds that one bin spans at the given pyramid level.

    Each level spans 4x more time than the one below. Level 0 returns
    level0_period_us unchanged.
    """
    return float(level0_period_us * DECIMATION_FACTOR**level)


def level_shape(num_samples: int, level: int) -> tuple[int, ...]:
    """Return the array shape for a pyramid level.

    Level 0 is rank-1, a single axis of raw samples. Each coarser level is
    rank-2: one entry per bin, then a trailing axis of length 2 holding the
    (min, max) pair.
    """
    return (
        (num_samples,)
        if level == 0
        else (level_num_bins(num_samples, level), ENVELOPE_PAIR_SIZE)
    )


def plan_levels(
    num_samples: int,
    level0_period_us: float,
    max_levels: int = 8,
    min_bins: int = 1024,
) -> list[LevelPlan]:
    """Return the pyramid plan for a channel: one LevelPlan per level."""
    return [
        LevelPlan(
            level=k,
            bins=level_num_bins(num_samples, k),
            period_us=level_period_us(level0_period_us, k),
        )
        for k in range(level_count(num_samples, max_levels, min_bins))
    ]
