"""Format constants for the pyramid bundle."""

from typing import Final

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

EVENT_DIMENSION: Final = "event"
"""Dimension name of the per-event axis."""

LABEL_DIMENSION: Final = "label"
"""Dimension name of the label axis of a per-label array."""

DEFAULT_DECIMATION_FACTOR: Final = 4
"""Default ratio of the bin widths of consecutive levels."""

MIN_DECIMATION_FACTOR: Final = 2
"""Smallest ratio of the bin widths of consecutive levels."""

LEVEL_PERIOD_TOLERANCE: Final = 1e-9
"""Relative tolerance of a level's tsz.period_us against the spec value."""

ENVELOPE_PAIR_SIZE: Final = 2
"""Length of the trailing (min, max) axis on every coarser pyramid level."""

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
    "volts": 1e6,
    "v": 1e6,
    "millivolts": 1e3,
    "mv": 1e3,
    "microvolts": 1.0,
    "uv": 1.0,
}
"""Conversion factors from a volts-family unit name to microvolts."""


def max_levels_for(decimation_factor: int) -> int:
    """Return the most levels a channel can hold at a decimation factor.

    Raw is not a level. The bin width of level k is decimation_factor to the
    power k, and a bin count is stored as uint32, so the width of the last
    level is at most 2**32 - 1.

    Args:
        decimation_factor: The ratio of the bin widths of consecutive levels,
            at least MIN_DECIMATION_FACTOR.

    Returns:
        The largest level number whose bin width fits.

    Raises:
        ValueError: decimation_factor is below MIN_DECIMATION_FACTOR.
    """
    if decimation_factor < MIN_DECIMATION_FACTOR:
        raise ValueError(
            f"decimation_factor must be {MIN_DECIMATION_FACTOR} or more, "
            f"got {decimation_factor}"
        )
    limit = 2**32 - 1
    levels = 0
    width = decimation_factor
    while width <= limit:
        levels += 1
        width *= decimation_factor
    return levels
