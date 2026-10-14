"""Shared value types for the writer."""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from timeseries_zarr.constants import (
    INNER_CHUNK_SAMPLES,
    MAX_LEVELS,
    TARGET_SHARD_BYTES,
)

type ChannelKind = Literal["continuous", "event"]

type VoltagePolicy = Literal["convert", "require_uv"]
"""How the writer treats the unit of a continuous channel.

convert scales a recognized volts-family unit to microvolts. require_uv
raises unless such a unit is exactly "uV".
"""

type EventColumn = Literal[
    "durations", "labels", "text", "channels", "waveform_mean"
]
"""Optional per-event column an event channel source can provide."""


@dataclass(frozen=True, slots=True)
class LevelPlan:
    """Resolved size and time resolution of one pyramid level of a channel.

    bins is the number of entries along the level's time axis. period_us is
    the microseconds that one entry spans, widening 4x with each coarser
    level.
    """

    level: int
    bins: int
    period_us: float

    @property
    def name(self) -> str:
        """Zarr key for this level: its decimal level number."""
        return str(self.level)


@dataclass(frozen=True, slots=True)
class LevelMembers[T]:
    """Handles to the member arrays of one pyramid level.

    minmax is the (bins, 2) envelope, mean and count are (bins,), and var is
    None on a level written without it. T is whatever array handle the caller
    works with.
    """

    minmax: T
    mean: T
    var: T | None
    count: T


@dataclass(frozen=True, slots=True)
class ResolvedUnit:
    """Scale factor and stored unit string of one continuous channel.

    factor multiplies the source samples. unit is the string the channel
    group declares.
    """

    factor: float
    unit: str


@dataclass(frozen=True, slots=True)
class RecordingMeta:
    """Identity and wall-clock start of a recording, for recording.json.

    start is the session's wall-clock start and must carry a UTC offset. The
    three objects are producer-defined and may be empty.
    """

    start: datetime
    subject: dict[str, object]
    session: dict[str, object]
    source: dict[str, object]


@dataclass(frozen=True, slots=True)
class ChunkShard:
    """Inner-chunk and outer-shard shapes for one Zarr array.

    chunk_shape is the inner (compressed) chunk; shard_shape is the outer shard
    that groups whole chunks under the ZEP2 sharding codec.
    """

    chunk_shape: tuple[int, ...]
    shard_shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class WriteOpts:
    """Tunable settings shared by the channel writers.

    max_levels and min_bins bound the pyramid; inner_len and
    target_shard_bytes size the inner chunk and the outer shard. write_var
    adds the var member to every level. voltage_policy sets how a continuous
    channel with a volts-family unit reaches microvolts.
    """

    zstd_level: int = 5
    max_levels: int = MAX_LEVELS
    min_bins: int = 1024
    inner_len: int = INNER_CHUNK_SAMPLES
    target_shard_bytes: int = TARGET_SHARD_BYTES
    write_var: bool = False
    voltage_policy: VoltagePolicy = "convert"
