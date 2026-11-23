"""Shared value types for the writer."""

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from timeseries_zarr.constants import (
    DEFAULT_DECIMATION_FACTOR,
    INNER_CHUNK_SAMPLES,
    TARGET_SHARD_BYTES,
    max_levels_for,
    normalize_factors,
)


def _available_cpus() -> int:
    """Return the CPUs that this process may run on, at least 1.

    The CPU affinity bounds the count where the platform reports it. A
    container limited to a few CPUs on a large host reports those CPUs.
    """
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        return len(affinity(0)) or 1
    return os.cpu_count() or 1


type ChannelKind = Literal["continuous", "event"]

type VoltagePolicy = Literal["convert", "require_uv"]
"""How the writer treats the unit of a continuous channel.

"convert" scales a recognized volts-family unit to microvolts. "require_uv"
raises unless such a unit is exactly "uV".
"""

type EventColumn = Literal[
    "durations", "labels", "text", "channels", "waveform_mean"
]
"""Optional per-event column an event channel source can provide."""


@dataclass(frozen=True, slots=True)
class LevelPlan:
    """Resolved size and time resolution of one pyramid level of a channel.

    Attributes:
        level: The level number, counted from 1.
        bins: The number of entries along the level's time axis.
        period_us: The microseconds that one entry spans. It is a whole
            multiple of the period of the level below, at least 2 times it.
            A count level holds an int.
        offset_us: The time of entry 0 from the reference time. It is 0 for a
            level of a continuous channel, whose entries count from the
            channel's own offset, and a multiple of period_us for a count
            level.
    """

    level: int
    bins: int
    period_us: float
    offset_us: int = 0

    @property
    def name(self) -> str:
        """Zarr key for this level: its decimal level number."""
        return str(self.level)


@dataclass(frozen=True, slots=True)
class LevelMembers[T]:
    """Handles to the member arrays of one pyramid level.

    The type parameter T is whatever array handle the caller works with.

    Attributes:
        minmax: The (bins, 2) envelope.
        mean: The (bins,) mean.
        var: The (bins,) variance. None on a level written without it.
        count: The (bins,) count of finite samples.
    """

    minmax: T
    mean: T
    var: T | None
    count: T


@dataclass(frozen=True, slots=True)
class ResolvedUnit:
    """Scale factor and stored unit string of one continuous channel.

    Attributes:
        factor: The multiplier applied to the source samples.
        unit: The unit string the channel group declares.
    """

    factor: float
    unit: str


@dataclass(frozen=True, slots=True)
class RecordingMeta:
    """Identity and datetimes of a recording, for recording.json.

    Each time must carry a UTC offset.

    Attributes:
        session_start_time: The datetime acquisition began. The writer stores
            it as the start_time of the session object. It places no data.
        reference_time: The instant every time in the bundle counts from.
            When given, the writer uses it as it is, so a rewrite keeps a
            bundle's reference time and a channel that begins before it gets
            a negative offset. When None, the writer places it at the
            earlier of the session start and the first sample.
        subject: Producer-defined subject object.
        session: Producer-defined session object.
        source: Producer-defined source object.
        wall_clock: Whether the source times are microseconds since the Unix
            epoch. When False, the times count from an unknown instant, the
            session start does not place the reference time,
            recording.json holds no reference_time, and reference_time must
            be None.
    """

    session_start_time: datetime | None = None
    reference_time: datetime | None = None
    subject: dict[str, object] = field(default_factory=dict)
    session: dict[str, object] = field(default_factory=dict)
    source: dict[str, object] = field(default_factory=dict)
    wall_clock: bool = True


@dataclass(frozen=True, slots=True)
class ChunkShard:
    """Inner-chunk and outer-shard shapes for one Zarr array.

    Attributes:
        chunk_shape: The inner (compressed) chunk.
        shard_shape: The outer shard that groups whole chunks under the ZEP2
            sharding codec.
    """

    chunk_shape: tuple[int, ...]
    shard_shape: tuple[int, ...]


def _require_int_factor(name: str, value: int) -> None:
    """Raise TypeError unless value is a plain int, and ValueError below 2."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{name} must be an int, got {type(value).__name__} {value!r}; "
            "set continuous_decimation_factor for a ladder of ratios"
        )
    normalize_factors(value, name)


@dataclass(frozen=True, slots=True)
class WriteOpts:
    """Tunable settings shared by the channel writers.

    Attributes:
        zstd_level: The Zstandard compression level.
        decimation_factor: The base ratio of the bin widths of consecutive
            levels, an int of at least MIN_DECIMATION_FACTOR. It sets the
            ratio of every level of a continuous channel and of an event
            channel unless an override below applies.
        continuous_decimation_factor: Overrides decimation_factor for
            continuous channels. An int sets one ratio. A sequence is a
            ladder: element 0 is the number of samples in a bin of level 1,
            element k is the ratio of level k + 1 to level k, and the last
            element repeats for further levels. Each ratio is at least
            MIN_DECIMATION_FACTOR. None uses decimation_factor. A sequence is
            stored as a tuple.
        event_decimation_factor: Overrides decimation_factor for the count
            levels of event channels, at least MIN_DECIMATION_FACTOR. None
            uses decimation_factor.
        max_levels: The most levels to write, 0 through the larger of the
            level limits of continuous and event channels at their factors. A
            value of 0 writes the raw samples and event arrays with no
            levels. None writes as many levels as level_cap and
            event_level_cap allow. A channel kind with a lower limit stops at
            that limit.
        min_bins: The fewest bins a level may hold, at least 1.
        inner_len: The inner chunk length in rows, at least 1.
        target_shard_bytes: The target size of an outer shard in bytes, at
            least 1.
        write_var: Whether to add the var member to every level. Requires
            max_levels of 1 or more.
        voltage_policy: How a continuous channel with a volts-family unit
            reaches microvolts. The policy also applies to the unit of an event
            channel.
        max_workers: The threads that fold and encode the channels of a
            window, at least 1. None uses the CPUs available to the process.
            The bytes written do not depend on it.

    Raises:
        TypeError: If a decimation factor has the wrong type. A sequence is
            valid only in continuous_decimation_factor.
        ValueError: If a setting lies outside its range, as stated above.
    """

    zstd_level: int = 5
    decimation_factor: int = DEFAULT_DECIMATION_FACTOR
    continuous_decimation_factor: int | Sequence[int] | None = None
    event_decimation_factor: int | None = None
    max_levels: int | None = None
    min_bins: int = 1024
    inner_len: int = INNER_CHUNK_SAMPLES
    target_shard_bytes: int = TARGET_SHARD_BYTES
    write_var: bool = False
    voltage_policy: VoltagePolicy = "convert"
    max_workers: int | None = None

    @property
    def worker_count(self) -> int:
        """The threads to use: max_workers, or the available CPUs when None."""
        return self.max_workers or _available_cpus()

    @property
    def continuous_factors(self) -> tuple[int, ...]:
        """The decimation ladder of a continuous channel."""
        override = self.continuous_decimation_factor
        if override is None:
            return (self.decimation_factor,)
        if isinstance(override, int):
            return (override,)
        return tuple(override)

    @property
    def event_factor(self) -> int:
        """The decimation factor of the count levels of an event channel."""
        if self.event_decimation_factor is None:
            return self.decimation_factor
        return self.event_decimation_factor

    @property
    def level_cap(self) -> int:
        """The most levels of a continuous channel."""
        cap = max_levels_for(self.continuous_factors)
        if self.max_levels is None:
            return cap
        return min(self.max_levels, cap)

    @property
    def event_level_cap(self) -> int:
        """The most count levels of an event channel."""
        cap = max_levels_for(self.event_factor)
        if self.max_levels is None:
            return cap
        return min(self.max_levels, cap)

    def __post_init__(self) -> None:
        """Raise on a setting outside its type or range.

        Raises:
            TypeError: A decimation factor has the wrong type.
            ValueError: A setting lies outside its range.
        """
        _require_int_factor("decimation_factor", self.decimation_factor)
        if self.event_decimation_factor is not None:
            _require_int_factor(
                "event_decimation_factor", self.event_decimation_factor
            )
        override = self.continuous_decimation_factor
        if override is not None:
            ladder = normalize_factors(override, "continuous_decimation_factor")
            if not isinstance(override, int):
                object.__setattr__(self, "continuous_decimation_factor", ladder)
        if self.max_workers is not None and self.max_workers < 1:
            raise ValueError(
                f"max_workers must be 1 or more, got {self.max_workers}"
            )
        cap = max(
            max_levels_for(self.continuous_factors),
            max_levels_for(self.event_factor),
        )
        if self.max_levels is not None and not 0 <= self.max_levels <= cap:
            if (
                self.continuous_decimation_factor is None
                and self.event_decimation_factor is None
            ):
                where = f"decimation_factor {self.decimation_factor}"
            else:
                where = (
                    f"continuous factors {self.continuous_factors} and "
                    f"event factor {self.event_factor}"
                )
            raise ValueError(
                f"max_levels must be 0 through {cap} at {where}, "
                f"got {self.max_levels}"
            )
        for name, value in (
            ("min_bins", self.min_bins),
            ("inner_len", self.inner_len),
            ("target_shard_bytes", self.target_shard_bytes),
        ):
            if value < 1:
                raise ValueError(f"{name} must be 1 or more, got {value}")
        if self.write_var and self.level_cap == 0:
            raise ValueError(
                "write_var adds var to every level and max_levels 0 writes no "
                "level; set max_levels to 1 or more, or write_var to False"
            )


@dataclass(frozen=True, slots=True)
class Finding:
    """One requirement of docs/spec.md that a bundle violates.

    Attributes:
        path: The node or file the violation lies in, relative to the bundle
            root, such as "0/1/mean" or "recording.json". The empty string
            names the root group.
        section: The section of docs/spec.md that states the violated
            requirement, such as "4.2".
        message: What is wrong and what is valid, on one line.
    """

    path: str
    section: str
    message: str
