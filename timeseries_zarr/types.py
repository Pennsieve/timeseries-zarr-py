"""Shared value types for the writer."""

from dataclasses import dataclass
from typing import Literal

from timeseries_zarr.constants import (
    EVENT_LEVEL_THRESHOLD,
    INNER_CHUNK_SAMPLES,
    MAX_LEVELS,
    TARGET_SHARD_BYTES,
)

type ChannelKind = Literal["continuous", "event"]
"""What a channel holds. Spikes and annotations are both event channels;
which columns are present, not the kind, decides what a view can draw."""


@dataclass(frozen=True, slots=True)
class RecordingMeta:
    """Recording-level metadata for the bundle's meta/ group.

    Everything identifying a bundle carries lives in these three objects and
    nowhere else, so that deleting one directory de-identifies it. session
    gains start_us when the bundle is written; the onset is the earliest
    start across every channel, which only the bundle knows.
    """

    subject: dict[str, object]
    session: dict[str, object]
    source: dict[str, object]


@dataclass(frozen=True, slots=True)
class LevelPlan:
    """Resolved shape and time resolution of one pyramid level of a channel.

    Levels are numbered from 1 and hold (min, max) envelopes; raw samples are
    not a level. period_us is the microseconds one bin spans, widening 4x with
    each coarser level. shape describes the level's env member, the only one
    written today; a level group may hold more.
    """

    level: int
    shape: tuple[int, ...]
    period_us: float


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
    target_shard_bytes size the inner chunk and the outer shard.
    event_level_threshold is the event count above which a channel earns a
    count pyramid at all.
    """

    zstd_level: int = 5
    max_levels: int = MAX_LEVELS
    min_bins: int = 1024
    inner_len: int = INNER_CHUNK_SAMPLES
    target_shard_bytes: int = TARGET_SHARD_BYTES
    event_level_threshold: int = EVENT_LEVEL_THRESHOLD
