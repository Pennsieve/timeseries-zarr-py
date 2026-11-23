"""Format constants for the pyramid bundle."""

import operator
from collections.abc import Sequence
from itertools import accumulate
from typing import Final, SupportsIndex

import numpy as np

FORMAT_VERSION: Final = "1.0"
"""TS-Zarr specification version the writer produces (root tsz.version)."""

MAJOR_VERSION: Final = 1
"""Major version a bundle's tsz.version must carry for the reader to open it."""

TSZ_KEY: Final = "tsz"
"""Attribute key that holds the TS-Zarr fields of every node."""

CHANNEL_KINDS: Final = ("continuous", "event")
"""Values of tsz.kind that mark a channel group."""

ZARR_METADATA_FILE: Final = "zarr.json"
"""Name of the metadata file every Zarr v3 node holds, the root included."""

RAW_KEY: Final = "raw"
"""Key of the full-resolution sample array within a channel group."""

RUNS_KEY: Final = "runs"
"""Key of the table of runs within a continuous channel group."""

MINMAX_KEY: Final = "minmax"
"""Key of the (bins, 2) minimum and maximum member of a level group."""

MEAN_KEY: Final = "mean"
"""Key of the per-bin mean member of a level group."""

VAR_KEY: Final = "var"
"""Key of the optional per-bin variance member of a level group."""

COUNT_KEY: Final = "count"
"""Key of the per-bin count array of a level group or a count level group.

A continuous level counts the finite samples of each bin. A count level counts
the events of each bin per label.
"""

EVENTS_KEY: Final = "events"
"""Key of the event time array within an event channel group."""

DURATIONS_KEY: Final = "durations"
"""Key of the per-event duration array within an event channel group."""

LABELS_KEY: Final = "labels"
"""Key of the per-event label array within an event channel group."""

VALUES_KEY: Final = "values"
"""Key of the per-event measurement array within an event channel group."""

TEXT_KEY: Final = "text"
"""Key of the per-event text array within an event channel group."""

CHANNELS_KEY: Final = "channels"
"""Key of the per-event channel bitmask array within an event channel group."""

WAVEFORMS_KEY: Final = "waveforms"
"""Key of the per-event snippet array within an event channel group."""

WAVEFORM_MEAN_KEY: Final = "waveform_mean"
"""Key of the per-label mean snippet array within an event channel group."""

TIME_DIMENSION: Final = "time"
"""Dimension name of a sample axis or a bin axis."""

RUN_DIMENSION: Final = "run"
"""Dimension name of the per-run axis."""

EVENT_DIMENSION: Final = "event"
"""Dimension name of the per-event axis."""

LABEL_DIMENSION: Final = "label"
"""Dimension name of the label axis of a per-label array."""

DEFAULT_DECIMATION_FACTOR: Final = 4
"""Default ratio of the bin widths of consecutive levels."""

SHARD_INDEX_CACHE_BYTES: Final = 64 * 2**20
"""Bytes of shard indexes that the reader keeps in memory.

The index of a shard with 512 chunk slots takes 8196 bytes.
"""

MIN_DECIMATION_FACTOR: Final = 2
"""Smallest ratio of the bin widths of consecutive levels."""

LEVEL_PERIOD_TOLERANCE: Final = 1e-12
"""Relative tolerance of a level's tsz.period_us against the spec value."""

TIME_LIMIT_US: Final = 2**53
"""Exclusive bound on the magnitude of every time and duration in microseconds."""

EVENT_COUNT_LIMIT: Final = 2**32
"""Exclusive bound on the number of events in a channel."""

SAMPLES_PER_BIN_LIMIT: Final = 2**32
"""Exclusive bound on the width of a continuous level in samples."""

ENVELOPE_PAIR_SIZE: Final = 2
"""Length of the trailing (min, max) axis on every coarser pyramid level."""

RUNS_COLUMNS: Final = 3
"""Length of the trailing axis of the runs array: first sample, count, time."""

MAX_LABELS: Final = 2**16
"""Most distinct labels an event channel can hold (the uint16 label range)."""

FLOAT32_BYTES: Final = 4
"""Byte width of one float32 sample, for shard-size computation."""

INT64_BYTES: Final = 8
"""Byte width of one int64 event timestamp, for shard-size computation."""

UINT8_BYTES: Final = 1
"""Byte width of one uint8 cluster id, for shard-size computation."""

UINT16_BYTES: Final = 2
"""Byte width of one uint16 label, for shard-size computation."""

UINT32_BYTES: Final = 4
"""Byte width of one uint32 count, for shard-size computation."""

SHARD_INDEX_ENTRY_BYTES: Final = 16
"""Bytes of the shard index for one inner chunk: an (offset, length) pair of uint64."""

STATISTIC_ROW_BYTES: Final = 40
"""Bytes of one pending level row: five float64 statistics."""

EVENT_LEVEL_THRESHOLD: Final = 64_000
"""Events above which an event channel carries count levels."""

RASTER_BUDGET: Final = 10_000
"""Events a reader reads raw for one window, which sets count level 1."""

CANVAS_PIXELS: Final = 2000
"""Pixels across a reader's canvas, which sets count level 1."""

COUNT_LADDER_BASE_US: Final = 1000
"""Finest count level period in microseconds.

Level 1 lies on base * decimation_factor^j.
"""

MICROSECONDS_PER_SECOND: Final = 1_000_000.0
"""Microseconds in one second, for converting a sample rate to a period."""

INNER_CHUNK_SAMPLES: Final = 2**13
"""Target inner Zarr chunk length in samples (8192)."""

TARGET_SHARD_BYTES: Final = 16 * 2**20
"""Target outer shard size in bytes (~16 MiB) for sharded pyramid arrays."""

ENCODE_GROUP_BYTES: Final = 2**20
"""Smallest run of chunk bytes (1 MiB) that one thread-pool task encodes."""

WINDOW_BATCH_ROWS: Final = 2**18
"""Rows of a channel (256 Ki) that the writer gathers from windows for one job."""

WINDOW_BATCH_BYTES: Final = 2**26
"""Bytes of windows (64 MiB) at which the writer stops gathering for one job."""

CHANNEL_APPEND_BYTES: Final = 2**20
"""Most encoded bytes (1 MiB) that the writer of one channel holds unwritten."""

EVENTS_SEARCH_LIMIT_BYTES: Final = 2**20
"""Largest events array in bytes (1 MiB) that a window search reads whole."""

UNIT_TO_UV: Final = {
    "V": 1e6,
    "mV": 1e3,
    "uV": 1.0,
    "nV": 1e-3,
    "volts": 1e6,
    "millivolts": 1e3,
    "microvolts": 1.0,
    "nanovolts": 1e-3,
}
"""Conversion factors from a voltage unit to microvolts.

The symbols V, mV, uV, and nV match exactly, so "MV" and "Mv" are not voltage
units. The spelled-out names are lowercase and match without regard to case.
"""


_NOT_FACTOR_TYPES: Final = (
    bool,
    np.bool_,
    str,
    bytes,
    bytearray,
    memoryview,
    np.ndarray,
)
"""Types that never hold a decimation factor, whatever they contain."""


def integer_like(value: object) -> int | None:
    """Return value as a plain int, or None when it is not integer-like.

    An integer-like value is anything that operator.index accepts, such as an
    int or a numpy integer scalar. A bool, a numpy bool, a float, a string,
    bytes, a bytearray, a memoryview, and an array are not integer-like.

    Args:
        value: The value to convert.

    Returns:
        The value as an int, or None.
    """
    if isinstance(value, _NOT_FACTOR_TYPES):
        return None
    try:
        return operator.index(value)  # type: ignore[arg-type]
    except TypeError:
        return None


def normalize_factors(
    value: SupportsIndex | Sequence[SupportsIndex], name: str
) -> tuple[int, ...]:
    """Return a decimation factor or ladder as a tuple of ratios.

    An integer-like value is a ladder of one ratio. A sequence is copied to a
    tuple. Each ratio in the result is a plain int.

    Args:
        value: One ratio, or the ratios of consecutive levels, each at least
            MIN_DECIMATION_FACTOR. A ratio is an int or another integer-like
            value, such as a numpy integer.
        name: The field name that the error messages report.

    Returns:
        The ratios, at least one.

    Raises:
        TypeError: value is not an integer-like value or a sequence of them. A
            bool, a float, a string, bytes, a bytearray, a memoryview, an
            array, and a nested sequence are not integer-like.
        ValueError: value is an empty sequence, or a ratio is below
            MIN_DECIMATION_FACTOR.
    """
    single = integer_like(value)
    if single is not None:
        if single < MIN_DECIMATION_FACTOR:
            raise ValueError(
                f"{name} must be {MIN_DECIMATION_FACTOR} or more, got {single}"
            )
        return (single,)
    if isinstance(value, _NOT_FACTOR_TYPES) or not isinstance(value, Sequence):
        raise TypeError(
            f"{name} must be an int or a sequence of ints, "
            f"got {type(value).__name__} {value!r}"
        )
    given = tuple(value)
    if not given:
        raise ValueError(f"{name} must hold at least one ratio, got {value!r}")
    ratios: list[int] = []
    for ratio in given:
        converted = integer_like(ratio)
        if converted is None:
            raise TypeError(
                f"{name} must hold ints, got {type(ratio).__name__} "
                f"{ratio!r} in {given!r}"
            )
        ratios.append(converted)
    if min(ratios) < MIN_DECIMATION_FACTOR:
        raise ValueError(
            f"{name} must hold ratios of {MIN_DECIMATION_FACTOR} or more, "
            f"got {tuple(ratios)!r}"
        )
    return tuple(ratios)


def level_factors(
    factors: SupportsIndex | Sequence[SupportsIndex], levels: int
) -> tuple[int, ...]:
    """Return the ratios of the first levels of a ladder.

    The last ratio of the ladder repeats for every level beyond its length.
    Element k is the ratio of level k + 1 to level k, and element 0 is the
    ratio of level 1 to raw.

    Args:
        factors: One ratio, or the ratios of consecutive levels.
        levels: The number of ratios to return, 0 or more.

    Returns:
        Exactly levels ratios.

    Raises:
        TypeError: factors is not integer-like or a sequence of them.
        ValueError: A ratio is below MIN_DECIMATION_FACTOR, factors is empty,
            or levels is negative.
    """
    if levels < 0:
        raise ValueError(f"levels must be 0 or more, got {levels}")
    ratios = normalize_factors(factors, "decimation_factor")
    padded = ratios + (ratios[-1],) * (levels - len(ratios))
    return padded[:levels]


def level_widths(
    factors: SupportsIndex | Sequence[SupportsIndex], levels: int
) -> tuple[int, ...]:
    """Return the samples per bin of the first levels of a ladder.

    Args:
        factors: One ratio, or the ratios of consecutive levels.
        levels: The number of levels, 0 or more.

    Returns:
        The width in raw samples of levels 1 through levels.

    Raises:
        TypeError: factors is not integer-like or a sequence of them.
        ValueError: A ratio is below MIN_DECIMATION_FACTOR, factors is empty,
            or levels is negative.
    """
    return tuple(accumulate(level_factors(factors, levels), operator.mul))


def max_levels_for(factors: SupportsIndex | Sequence[SupportsIndex]) -> int:
    """Return the most levels a channel can hold at a decimation ladder.

    Raw is not a level. The bin width of level k is the product of the first k
    ratios of the ladder, and a bin count is stored as uint32, so the width of
    the last level is at most 2**32 - 1.

    Args:
        factors: One ratio, or the ratios of consecutive levels, each at least
            MIN_DECIMATION_FACTOR. The last ratio repeats for further levels.

    Returns:
        The largest level number whose bin width fits.

    Raises:
        TypeError: factors is not integer-like or a sequence of them.
        ValueError: A ratio is below MIN_DECIMATION_FACTOR or factors is
            empty.
    """
    ratios = normalize_factors(factors, "decimation_factor")
    limit = 2**32 - 1
    levels = 0
    width = ratios[0]
    while width <= limit:
        levels += 1
        width *= ratios[min(levels, len(ratios) - 1)]
    return levels
