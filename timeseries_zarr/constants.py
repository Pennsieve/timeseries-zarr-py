"""Format constants for the pyramid bundle."""

from typing import Final

DECIMATION_FACTOR: Final = 4
"""Samples folded into one bin per pyramid level (4x coarser each level)."""

MAX_LEVELS: Final = 7
"""Most level groups a channel can hold: levels 1 through 7, a 16384x range.

Raw samples are not a level, so this counts only the folded ones.
"""

ENVELOPE_PAIR_SIZE: Final = 2
"""Length of the trailing (min, max) axis of a level's env member."""

STAT_COLUMNS: Final = 5
"""Columns of an in-flight stat block: min, max, mean, count, valid."""

MIN_COL: Final = 0
"""Stat-block column holding a bin's minimum."""

MAX_COL: Final = 1
"""Stat-block column holding a bin's maximum."""

MEAN_COL: Final = 2
"""Stat-block column holding a bin's mean."""

COUNT_COL: Final = 3
"""Stat-block column holding the raw samples behind a bin.

Time support, not a count of finite samples: a bin full of NaN still spans
its slots. Never written to disk, because a level read back can rebuild it
from its own number. The valid column is the one that counts finite samples.
"""

VALID_COL: Final = 4
"""Stat-block column holding the finite samples behind a bin.

Equal to COUNT_COL only where nothing is missing. It is what separates one
bad sample in a full bin from an amplifier stop, and it is data rather than
arithmetic, so unlike the count it is written and read back.
"""

MAX_VALID_COUNT: Final = 2**16 - 1
"""Largest per-bin valid count the u2 member can hold.

A level-7 bin spans 16384 raw samples and fits. Level 8 would span 65536 and
would not, so MAX_LEVELS is what keeps this member in range.
"""

MAX_LABEL_VALUES: Final = 2**16
"""Most distinct labels an event channel can hold (the u2 range).

A modern sorter on a high-density probe passes 256 units without trying, which
is what the old u1 column allowed.
"""

FLOAT32_BYTES: Final = 4
"""Byte width of one float32 sample, for shard-size computation."""

INT64_BYTES: Final = 8
"""Byte width of one int64 event timestamp, for shard-size computation."""

UINT16_BYTES: Final = 2
"""Byte width of one uint16 label, for shard-size computation."""

UINT32_BYTES: Final = 4
"""Byte width of one uint32 bin count, for shard-size computation."""

EVENT_LEVEL_THRESHOLD: Final = 64 * 2**10
"""Events above which a channel gets a count pyramid rather than raster reads.

Below it a reader binary-searches the events themselves and needs nothing more.
"""

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
