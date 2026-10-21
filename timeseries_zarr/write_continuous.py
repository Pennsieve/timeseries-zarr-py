"""Compose streaming and zarr I/O to write one continuous channel.

A channel is the raw samples plus a ladder of level groups above them. Raw is
written from the source, and every level is folded from the samples below it.
A level group holds one array per statistic over a shared bin axis. An empty
input creates the arrays and writes nothing.

write_continuous_channel composes three steps. create_channel_group makes the
channel group with its attributes, write_raw makes and fills the raw array,
and write_levels folds the levels from the stored raw array.
write_continuous_channels writes every channel of a multi-channel source. It
fills the raw arrays of all channels in one sweep over the source and then
folds the levels of each channel.
"""

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt

from timeseries_zarr.attrs import continuous_channel_attrs, level_group_attrs
from timeseries_zarr.constants import (
    COUNT_KEY,
    DECIMATION_FACTOR,
    ENVELOPE_PAIR_SIZE,
    MEAN_KEY,
    MINMAX_KEY,
    RAW_KEY,
    TIME_DIMENSION,
    VAR_KEY,
)
from timeseries_zarr.fold import (
    fold_raw_block,
    fold_statistics_block,
    split_members,
)
from timeseries_zarr.planning import plan_levels, raw_shape
from timeseries_zarr.protocols import (
    ContinuousChannelMetadata,
    ContinuousChannelSource,
    MultiChannelSource,
    source_provenance,
)
from timeseries_zarr.sizing import level_sizing, member_sizing, raw_sizing
from timeseries_zarr.streaming import (
    BlockReadableArray,
    Regrouper,
    _rebuffer_and_fold,
    iter_array_blocks,
    iter_member_blocks,
    iter_raw_blocks,
    regroup,
    scale,
    scale_block,
    tap,
)
from timeseries_zarr.types import (
    ChunkShard,
    LevelMembers,
    LevelPlan,
    WriteOpts,
)
from timeseries_zarr.units import resolve_unit
from timeseries_zarr.zarr_io import (
    ZarrArray,
    ZarrGroup,
    create_array,
    create_group_with_attrs,
    write_region,
)


def _write_blocks(
    array: ZarrArray, blocks: Iterable[npt.NDArray[np.float32]]
) -> None:
    """Write each block to array at its running axis-0 offset.

    Blocks arrive shard-sized, so each write covers a whole shard. A write
    narrower than a shard makes the sharding codec read that shard back,
    re-encode every inner chunk, and rewrite it.
    """
    start = 0
    for block in blocks:
        write_region(array, start, block)
        start += block.shape[0]


def create_channel_group(
    parent: ZarrGroup,
    key: int,
    channel: ContinuousChannelMetadata,
    *,
    reference_us: int,
    unit: str,
) -> ZarrGroup:
    """Create the channel group named str(key) with the continuous attributes.

    The group stores unit as given, whatever unit the channel declares. Its
    offset_us is the distance in microseconds from reference_us, the
    reference time, to the first sample, negative for a channel that begins
    before it. The group carries the channel's provenance when the channel has
    one. Raises ValueError if the rate is not positive or the provenance is
    not a JSON object of finite values.
    """
    return create_group_with_attrs(
        parent,
        str(key),
        continuous_channel_attrs(
            channel.name,
            channel.id,
            channel.rate_hz(),
            channel.start_us() - reference_us,
            unit,
            source_provenance(channel),
        ),
    )


def create_raw(
    group: ZarrGroup,
    num_samples: int,
    sizing: ChunkShard,
    zstd_level: int,
) -> ZarrArray:
    """Create the raw array under group, with every sample unwritten.

    The array is float32, named "raw", and shaped (num_samples,). Its fill
    value is NaN, its axis is named "time", and it has no attributes. The
    channel group carries the sample rate. A num_samples of 0 gives a
    zero-length array.
    """
    return create_array(
        group,
        RAW_KEY,
        raw_shape(num_samples),
        np.float32,
        sizing.chunk_shape,
        sizing.shard_shape,
        {},
        zstd_level,
        dimension_names=(TIME_DIMENSION,),
    )


def write_raw(
    group: ZarrGroup,
    source: ContinuousChannelSource,
    sizing: ChunkShard,
    zstd_level: int,
    factor: float = 1.0,
) -> ZarrArray:
    """Create the raw array under group and stream the source's samples into it.

    The array is the one create_raw makes. The samples are multiplied by
    factor, so a non-finite sample stays a gap and a DC offset stays in. They
    are read in shard-sized blocks, so every write covers a whole shard. An
    empty source creates a zero-length array and writes nothing.
    """
    array = create_raw(group, source.num_samples(), sizing, zstd_level)
    _write_blocks(
        array, scale(iter_raw_blocks(source, sizing.shard_shape[0]), factor)
    )
    return array


def create_level(
    parent: ZarrGroup,
    plan: LevelPlan,
    sizing: ChunkShard,
    zstd_level: int,
    write_var: bool,
) -> LevelMembers[ZarrArray]:
    """Create the level group named by plan.level with its member arrays.

    The group carries period_us. It holds minmax shaped (bins, 2), and mean
    and count shaped (bins,), plus var when write_var is true. The float
    members are float32 and count is uint32. sizing is the sizing of minmax,
    and the other members take its row counts, so one block of rows fills one
    shard of every member. Every member names its bin axis "time". Raises
    ValueError if plan.level is below 1.
    """
    if plan.level < 1:
        raise ValueError("levels are numbered from 1; raw is not a level")
    group = create_group_with_attrs(
        parent, plan.name, level_group_attrs(plan.period_us)
    )
    member_rows = member_sizing(sizing)

    def member(
        key: str,
        dtype: type[np.generic],
        shape: tuple[int, ...],
        array_sizing: ChunkShard,
        dimension_names: tuple[str | None, ...],
    ) -> ZarrArray:
        return create_array(
            group,
            key,
            shape,
            dtype,
            array_sizing.chunk_shape,
            array_sizing.shard_shape,
            {},
            zstd_level,
            dimension_names=dimension_names,
        )

    bins_shape = (plan.bins,)
    time_only = (TIME_DIMENSION,)
    return LevelMembers(
        minmax=member(
            MINMAX_KEY,
            np.float32,
            (plan.bins, ENVELOPE_PAIR_SIZE),
            sizing,
            (TIME_DIMENSION, None),
        ),
        mean=member(MEAN_KEY, np.float32, bins_shape, member_rows, time_only),
        var=(
            member(VAR_KEY, np.float32, bins_shape, member_rows, time_only)
            if write_var
            else None
        ),
        count=member(COUNT_KEY, np.uint32, bins_shape, member_rows, time_only),
    )


def _statistics_writer(
    members: LevelMembers[ZarrArray],
) -> Callable[[npt.NDArray[np.float64]], None]:
    """Return a function that writes successive statistics blocks to members."""
    start = 0

    def write(block: npt.NDArray[np.float64]) -> None:
        nonlocal start
        minmax, mean, var, count = split_members(block, members.var is not None)
        write_region(members.minmax, start, minmax)
        write_region(members.mean, start, mean)
        if members.var is not None and var is not None:
            write_region(members.var, start, var)
        write_region(members.count, start, count)
        start += block.shape[0]

    return write


def write_statistics_blocks(
    members: LevelMembers[ZarrArray],
    blocks: Iterable[npt.NDArray[np.float64]],
) -> None:
    """Write a stream of statistics blocks into a level's member arrays.

    Each block is split into the members and written at a running row offset,
    so the blocks must arrive in level order and cover the level from row 0.
    Blocks of shard_shape[0] rows make every write cover a whole shard of
    every member, and only the last block may be shorter. A level without a
    var member drops the variance column. An empty stream writes nothing.
    """
    write = _statistics_writer(members)
    for block in blocks:
        write(block)


def _write_levels_by_read_back(
    group: ZarrGroup,
    raw: ZarrArray,
    plans: list[LevelPlan],
    opts: WriteOpts,
) -> None:
    """Write each level by reading the one below it back from disk.

    Level 1 folds the stored raw samples and each later level folds the stored
    float32 members of the level below. One shard of the level below is read per
    fold step, so memory stays at one shard.
    """
    previous: LevelMembers[ZarrArray] | None = None
    for plan in plans:
        sizing = level_sizing(
            plan.bins, opts.inner_len, opts.target_shard_bytes
        )
        members = create_level(group, plan, sizing, opts.zstd_level, False)
        read_rows = DECIMATION_FACTOR * sizing.shard_shape[0]
        folded: Iterator[npt.NDArray[np.float64]]
        if previous is None:
            folded = _rebuffer_and_fold(
                iter_array_blocks(cast("BlockReadableArray", raw), read_rows),
                fold_raw_block,
            )
        else:
            folded = _rebuffer_and_fold(
                iter_member_blocks(
                    cast("LevelMembers[BlockReadableArray]", previous),
                    read_rows,
                ),
                fold_statistics_block,
            )
        write_statistics_blocks(members, regroup(folded, sizing.shard_shape[0]))
        previous = members


def _write_levels_in_one_pass(
    group: ZarrGroup,
    raw: ZarrArray,
    plans: list[LevelPlan],
    opts: WriteOpts,
) -> None:
    """Write every level in one pass over the stored raw samples.

    Level 1 folds the raw samples. Each later level folds the float64
    statistics of the level below, never its stored float32 members, so the
    variance keeps its full precision. The levels form one chain of generators
    that is drained once, and a shard of statistics rows per level is in
    flight.
    """
    first = level_sizing(plans[0].bins, opts.inner_len, opts.target_shard_bytes)
    stream: Iterator[npt.NDArray[np.float64]] = _rebuffer_and_fold(
        iter_array_blocks(
            cast("BlockReadableArray", raw),
            DECIMATION_FACTOR * first.shard_shape[0],
        ),
        fold_raw_block,
    )
    for index, plan in enumerate(plans):
        sizing = level_sizing(
            plan.bins, opts.inner_len, opts.target_shard_bytes
        )
        members = create_level(group, plan, sizing, opts.zstd_level, True)
        stream = tap(
            regroup(stream, sizing.shard_shape[0]), _statistics_writer(members)
        )
        if index + 1 < len(plans):
            stream = _rebuffer_and_fold(stream, fold_statistics_block)
    for _ in stream:
        pass


def write_levels(
    group: ZarrGroup,
    raw: ZarrArray,
    rate_hz: float,
    opts: WriteOpts,
) -> None:
    """Write the levels of a channel by folding its stored raw samples.

    The levels are planned from the length of raw and from rate_hz, and every
    array is sized and compressed per opts. Raw too short for level 1 gets no
    level.

    With opts.write_var false, each level is folded from the stored members of
    the level below. With it true, every level is folded in one pass from the
    raw samples through float64 statistics. The variance then stays exact on a
    channel with a large DC offset.
    """
    plans = plan_levels(raw.shape[0], rate_hz, opts.max_levels, opts.min_bins)
    if not plans:
        return
    if opts.write_var:
        _write_levels_in_one_pass(group, raw, plans, opts)
    else:
        _write_levels_by_read_back(group, raw, plans, opts)


def write_continuous_channel(
    parent: ZarrGroup,
    key: int,
    source: ContinuousChannelSource,
    *,
    reference_us: int,
    opts: WriteOpts,
) -> None:
    """Write one continuous channel as the subgroup named str(key).

    A voltage unit resolves under opts.voltage_policy before the group is
    created, and the raw samples are scaled to match. A channel too short for
    level 1 gets raw and nothing else. Raises ValueError if the source's rate
    is not positive, if its provenance is not a JSON object of finite values,
    or, under voltage_policy require_uv, if a voltage unit is not "uV".
    """
    resolved = resolve_unit(source.unit, opts.voltage_policy)
    group = create_channel_group(
        parent, key, source, reference_us=reference_us, unit=resolved.unit
    )
    raw = write_raw(
        group,
        source,
        raw_sizing(
            source.num_samples(), opts.inner_len, opts.target_shard_bytes
        ),
        opts.zstd_level,
        resolved.factor,
    )
    write_levels(group, raw, source.rate_hz(), opts)


@dataclass(slots=True)
class _RawSink:
    """Collect one channel's blocks of a sweep and write them to its raw array."""

    key: int
    channel: ContinuousChannelMetadata
    group: ZarrGroup
    raw: ZarrArray
    factor: float
    regrouper: Regrouper[np.float32]
    num_samples: int
    received: int = 0
    written: int = 0

    def _label(self) -> str:
        return f"channel {self.key} ({self.channel.id!r})"

    def _write(self, block: npt.NDArray[np.float32]) -> None:
        write_region(self.raw, self.written, block)
        self.written += block.shape[0]

    def push(self, block: npt.ArrayLike) -> None:
        """Write each whole shard that the samples of block complete.

        The samples are cast to float32 and multiplied by factor. Samples
        that do not fill a shard wait for the next push. Raises ValueError,
        before it writes any sample of block, if block is not rank 1 or if the
        channel has received more samples than it declares.
        """
        samples = np.asarray(block, dtype=np.float32)
        if samples.ndim != 1:
            raise ValueError(
                f"{self._label()} gives a block of rank {samples.ndim}; "
                "each block must be a rank-1 array"
            )
        self.received += samples.shape[0]
        if self.received > self.num_samples:
            raise ValueError(
                f"{self._label()} yields at least {self.received} samples "
                f"and declares {self.num_samples}; its blocks must total "
                f"{self.num_samples}"
            )
        for shard in self.regrouper.push(scale_block(samples, self.factor)):
            self._write(shard)

    def finish(self) -> None:
        """Write the samples that wait for a shard to fill.

        Raises ValueError if the channel received fewer samples than it
        declares.
        """
        if self.received != self.num_samples:
            raise ValueError(
                f"{self._label()} yields {self.received} samples and "
                f"declares {self.num_samples}; its blocks must total "
                f"{self.num_samples}"
            )
        tail = self.regrouper.flush()
        if tail is not None:
            self._write(tail)


def write_continuous_channels(
    parent: ZarrGroup,
    first_key: int,
    source: MultiChannelSource,
    *,
    reference_us: int,
    opts: WriteOpts,
) -> None:
    """Write the channels of source as the subgroups named from str(first_key).

    Each channel group is created as write_continuous_channel creates it. One pass over the
    windows of source fills the raw array of every channel, a shard at a time.
    The levels of each channel are then folded from its stored raw samples.
    Memory holds one shard of raw per channel and the current window. A source
    without channels writes nothing, and iter_windows is not called.

    Raises ValueError for a rate that is not positive, for provenance that is
    not a JSON object of finite values, and under voltage_policy require_uv
    for a voltage unit that is not "uV". Raises ValueError also for a window
    that does not hold one block per channel, for a block that is not rank 1,
    and for a channel whose blocks do not total its num_samples. Each of these
    errors comes before the first level is folded. A failure leaves the
    groups written so far in place.
    """
    channels = list(source.channels())
    if not channels:
        return
    sinks: list[_RawSink] = []
    for key, channel in enumerate(channels, start=first_key):
        resolved = resolve_unit(channel.unit, opts.voltage_policy)
        group = create_channel_group(
            parent, key, channel, reference_us=reference_us, unit=resolved.unit
        )
        num_samples = channel.num_samples()
        sizing = raw_sizing(
            num_samples, opts.inner_len, opts.target_shard_bytes
        )
        raw = create_raw(group, num_samples, sizing, opts.zstd_level)
        sinks.append(
            _RawSink(
                key,
                channel,
                group,
                raw,
                resolved.factor,
                Regrouper(sizing.shard_shape[0]),
                num_samples,
            )
        )
    for index, window in enumerate(source.iter_windows()):
        if len(window) != len(sinks):
            raise ValueError(
                f"window {index} holds {len(window)} blocks and the source "
                f"has {len(sinks)} channels; a window must hold one block "
                "for each channel"
            )
        for sink, block in zip(sinks, window, strict=False):
            sink.push(block)
    for sink in sinks:
        sink.finish()
    for sink in sinks:
        write_levels(sink.group, sink.raw, sink.channel.rate_hz(), opts)
