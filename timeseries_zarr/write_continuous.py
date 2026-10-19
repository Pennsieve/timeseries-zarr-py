"""Compose streaming and zarr I/O to write one continuous channel.

A channel is the raw samples plus a ladder of level groups above them. Raw is
written from the source, and every level is folded from the samples below it.
A level group holds one array per statistic over a shared bin axis. An empty
input creates the arrays and writes nothing.
"""

from collections.abc import Callable, Iterable, Iterator
from typing import cast

import numpy as np
import numpy.typing as npt

from timeseries_zarr.attrs import continuous_channel_attrs, level_group_attrs
from timeseries_zarr.constants import (
    COUNT_KEY,
    DECIMATION_FACTOR,
    ENVELOPE_PAIR_SIZE,
    FLOAT32_BYTES,
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
    ContinuousChannelSource,
    source_provenance,
)
from timeseries_zarr.sizing import chunk_and_shard, level_sizing, member_sizing
from timeseries_zarr.streaming import (
    BlockReadableArray,
    _rebuffer_and_fold,
    iter_array_blocks,
    iter_member_blocks,
    iter_raw_blocks,
    regroup,
    scale,
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


def write_raw(
    group: ZarrGroup,
    source: ContinuousChannelSource,
    sizing: ChunkShard,
    zstd_level: int,
    factor: float = 1.0,
) -> ZarrArray:
    """Create the raw array under group and stream the source's samples into it.

    A rank-1 float32 array named "raw" of shape (num_samples,), with the
    dimension name "time" and no attributes: the sample period is the
    channel's rate_hz, which the channel group carries. The samples are
    multiplied by factor, so a non-finite sample stays a gap and a DC offset
    stays in. They are read in shard-sized blocks, so every write covers
    a whole shard. An empty source creates a zero-length array and writes
    nothing.
    """
    array = create_array(
        group,
        RAW_KEY,
        raw_shape(source.num_samples()),
        np.float32,
        sizing.chunk_shape,
        sizing.shard_shape,
        {},
        zstd_level,
        dimension_names=(TIME_DIMENSION,),
    )
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


def write_continuous_channel(
    parent: ZarrGroup,
    key: int,
    source: ContinuousChannelSource,
    *,
    reference_us: int,
    opts: WriteOpts,
) -> None:
    """Write one continuous channel as the subgroup named str(key).

    The channel group carries the continuous attributes, with offset_us the
    distance in microseconds from reference_us, the reference time, to the
    first sample, negative for a channel that begins before it, and the
    source's provenance when it has one. The raw samples are written first.
    The levels are planned from the sample count and rate, and every array is
    sized and compressed per opts. A channel too short for level 1 gets raw
    and nothing else.

    With opts.write_var false, each level is folded from the stored members of
    the level below. With it true, every level is folded in one pass from the
    raw samples through float64 statistics, which keeps the variance exact on a
    channel with a large DC offset. Raises ValueError if the source's
    provenance is not a JSON object of finite values, or, under
    voltage_policy require_uv, if a voltage unit is not "uV".
    """
    rate_hz = source.rate_hz()
    resolved = resolve_unit(source.unit, opts.voltage_policy)
    group = create_group_with_attrs(
        parent,
        str(key),
        continuous_channel_attrs(
            source.name,
            source.id,
            rate_hz,
            source.start_us() - reference_us,
            resolved.unit,
            source_provenance(source),
        ),
    )
    raw = write_raw(
        group,
        source,
        chunk_and_shard(
            raw_shape(source.num_samples()),
            FLOAT32_BYTES,
            opts.inner_len,
            opts.target_shard_bytes,
        ),
        opts.zstd_level,
        resolved.factor,
    )
    plans = plan_levels(
        source.num_samples(), rate_hz, opts.max_levels, opts.min_bins
    )
    if not plans:
        return
    if opts.write_var:
        _write_levels_in_one_pass(group, raw, plans, opts)
    else:
        _write_levels_by_read_back(group, raw, plans, opts)
