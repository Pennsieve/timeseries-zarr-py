"""Format constants for the pyramid bundle."""

from typing import Final

FORMAT_VERSION: Final = "1.0"
"""Version of the TS-Zarr specification the writer produces, the root tsz.version."""

DECIMATION_FACTOR: Final = 4
"""Samples folded into one bin per pyramid level (4x coarser each level)."""

MAX_LEVELS: Final = 15
"""Most folded levels a channel can hold; raw is not a level.

A level-k bin spans 4^k samples and its count is stored as uint32, which a
level-16 bin would exceed.
"""

ENVELOPE_PAIR_SIZE: Final = 2
"""Length of the trailing (min, max) axis on every coarser pyramid level."""

STATISTIC_COLUMNS: Final = 5
"""Columns of an in-flight float64 statistics block.

In order: minimum, maximum, mean, variance, count of finite samples.
"""

MIN_COLUMN: Final = 0
"""Column of a statistics block holding a bin's minimum."""

MAX_COLUMN: Final = 1
"""Column of a statistics block holding a bin's maximum."""

MEAN_COLUMN: Final = 2
"""Column of a statistics block holding a bin's mean."""

VAR_COLUMN: Final = 3
"""Column of a statistics block holding a bin's population variance."""

COUNT_COLUMN: Final = 4
"""Column of a statistics block holding a bin's count of finite samples."""

MAX_UNIT_CLUSTERS: Final = 256
"""Most distinct clusters a unit channel can hold (the uint8 cluster-id range)."""

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

EVENT_LEVEL_THRESHOLD: Final = 64_000
"""Events above which an event channel carries count levels."""

RASTER_BUDGET: Final = 10_000
"""Events a reader reads raw for one window, which sets count level 1."""

CANVAS_PIXELS: Final = 2000
"""Pixels across a reader's canvas, which sets count level 1."""

COUNT_LADDER_BASE_US: Final = 1000
"""Finest count level period in microseconds; level 1 lies on base * 4^j."""

MICROSECONDS_PER_SECOND: Final = 1_000_000.0
"""Microseconds in one second, for converting a sample rate to a period."""

INNER_CHUNK_SAMPLES: Final = 2**13
"""Target inner Zarr chunk length in samples (8192)."""

TARGET_SHARD_BYTES: Final = 16 * 2**20
"""Target outer shard size in bytes (~16 MiB) for sharded pyramid arrays."""

UNIT_TO_UV: Final = {
    "volts": 1e6,
    "v": 1e6,
    "millivolts": 1e3,
    "mv": 1e3,
    "microvolts": 1.0,
    "uv": 1.0,
}
"""Conversion factors from a volts-family unit name to microvolts."""
