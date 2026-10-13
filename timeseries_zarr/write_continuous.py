"""Compose streaming and zarr I/O to write one continuous channel's levels.

Each level writer shapes its array from the plan, takes its chunk and shard
shapes from sizing, then streams blocks in. An empty input creates the array
and writes nothing.
"""

from collections.abc import Iterable
from typing import Final, cast

import numpy as np
import numpy.typing as npt

from timeseries_zarr.attrs import (
    channel_group_attrs,
    level_array_attrs,
    level_group_attrs,
)
from timeseries_zarr.constants import (
    DECIMATION_FACTOR,
    ENVELOPE_PAIR_SIZE,
    FLOAT32_BYTES,
)
from timeseries_zarr.fold import fold_block, split_members
from timeseries_zarr.planning import plan_levels, raw_shape, sample_period_us
from timeseries_zarr.protocols import ContinuousChannelSource
from timeseries_zarr.sizing import chunk_and_shard, member_sizing
from timeseries_zarr.streaming import (
    BlockReadableArray,
    _rebuffer_and_fold,
    iter_array_blocks,
    iter_raw_blocks,
)
from timeseries_zarr.types import ChunkShard, LevelMembers, LevelPlan, WriteOpts
from timeseries_zarr.zarr_io import (
    ZarrArray,
    ZarrGroup,
    create_array,
    create_group_with_attrs,
    write_region,
)


def _array_shape(plan: LevelPlan) -> tuple[int, ...]:
    """Return the array shape of a level: rank 1 for raw, (bins, 2) above it."""
    return (plan.bins,) if plan.level == 0 else (plan.bins, ENVELOPE_PAIR_SIZE)


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


RAW_KEY: Final = "raw"
"""Key of the full-resolution sample array within a channel group."""

MINMAX_KEY: Final = "minmax"
"""Key of the (bins, 2) minimum and maximum member of a level group."""

MEAN_KEY: Final = "mean"
"""Key of the per-bin mean member of a level group."""

VAR_KEY: Final = "var"
"""Key of the optional per-bin variance member of a level group."""

COUNT_KEY: Final = "count"
"""Key of the per-bin count of finite samples member of a level group."""

TIME_DIMENSION: Final = "time"
"""Dimension name of a sample axis or a bin axis."""


def write_raw(
    group: ZarrGroup,
    source: ContinuousChannelSource,
    sizing: ChunkShard,
    zstd_level: int,
) -> ZarrArray:
    """Create the raw array under group and stream the source's samples into it.

    A rank-1 float32 array named "raw" of shape (num_samples,), with the
    dimension name "time" and no attributes: the sample period is the
    channel's rate_hz, which the channel group carries. The samples are stored
    as the source gives them, so a non-finite sample stays a gap and a DC
    offset stays in. They are read in shard-sized blocks, so every write covers
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
    _write_blocks(array, iter_raw_blocks(source, sizing.shard_shape[0]))
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
    start = 0
    for block in blocks:
        minmax, mean, var, count = split_members(block, members.var is not None)
        write_region(members.minmax, start, minmax)
        write_region(members.mean, start, mean)
        if members.var is not None and var is not None:
            write_region(members.var, start, var)
        write_region(members.count, start, count)
        start += block.shape[0]


def write_level0(
    group: ZarrGroup,
    source: ContinuousChannelSource,
    plan: LevelPlan,
    sizing: ChunkShard,
    zstd_level: int,
) -> ZarrArray:
    """Create the level-0 raw array under group and stream the source into it.

    The array is named "0", holds float32, and carries the level period_us as
    its sole attribute. The source's raw samples are read in shard-sized
    blocks. Raises ValueError if plan does not describe level 0 (raw).
    """
    if plan.level != 0:
        raise ValueError("plan level must be raw")

    array = create_array(
        group,
        str(plan.level),
        _array_shape(plan),
        np.float32,
        sizing.chunk_shape,
        sizing.shard_shape,
        level_array_attrs(plan.period_us),
        zstd_level,
    )
    _write_blocks(array, iter_raw_blocks(source, sizing.shard_shape[0]))
    return array


def write_level_from_previous(
    group: ZarrGroup,
    prev: ZarrArray,
    plan: LevelPlan,
    sizing: ChunkShard,
    zstd_level: int,
) -> ZarrArray:
    """Create the level named plan.level by folding the previous level into it.

    The array is named str(plan.level), holds float32, and carries the level
    period_us as its sole attribute. prev is read in axis-0 blocks of
    DECIMATION_FACTOR shards and folded across block boundaries by fold_block,
    so each folded block fills one shard of the new array. Raises ValueError if
    plan describes level 0.
    """
    if plan.level == 0:
        raise ValueError("level must not be raw")

    array = create_array(
        group,
        str(plan.level),
        _array_shape(plan),
        np.float32,
        sizing.chunk_shape,
        sizing.shard_shape,
        level_array_attrs(plan.period_us),
        zstd_level,
    )

    _write_blocks(
        array,
        _rebuffer_and_fold(
            iter_array_blocks(
                cast("BlockReadableArray", prev),
                DECIMATION_FACTOR * sizing.shard_shape[0],
            ),
            fold_block,
        ),
    )
    return array


def write_continuous_channel(
    parent: ZarrGroup,
    index: int,
    source: ContinuousChannelSource,
    *,
    opts: WriteOpts,
) -> None:
    """Write one continuous channel as the subgroup named str(index).

    Creates the channel group under parent carrying its continuous-kind
    attributes, then writes level 0 from the source and folds each coarser
    level from the one written below it. The pyramid is planned from the
    source's sample count and rate; every level array is sized and compressed
    per opts.
    """
    attributes = channel_group_attrs(
        source.id,
        source.rate_hz(),
        source.start_us(),
        "continuous",
        source.name,
        source.unit,
    )
    group = create_group_with_attrs(parent, str(index), attributes)
    previous = None
    plans = [
        LevelPlan(
            level=0,
            bins=source.num_samples(),
            period_us=sample_period_us(source.rate_hz()),
        ),
        *plan_levels(
            source.num_samples(),
            source.rate_hz(),
            opts.max_levels,
            opts.min_bins,
        ),
    ]
    for plan in plans:
        sizing = chunk_and_shard(
            level_shape=_array_shape(plan),
            dtype_size=FLOAT32_BYTES,
            inner_len=opts.inner_len,
            target_shard_bytes=opts.target_shard_bytes,
        )
        if previous is None:
            previous = write_level0(
                group=group,
                source=source,
                plan=plan,
                sizing=sizing,
                zstd_level=opts.zstd_level,
            )
        else:
            previous = write_level_from_previous(
                group=group,
                prev=previous,
                plan=plan,
                sizing=sizing,
                zstd_level=opts.zstd_level,
            )
