"""Compose streaming and zarr I/O to write one continuous channel's levels.

Each level writer shapes its array from the plan, takes its chunk and shard
shapes from sizing, then streams blocks in. An empty input creates the array
and writes nothing.
"""

from collections.abc import Iterable
from typing import cast

import numpy as np
import numpy.typing as npt

from timeseries_zarr.attrs import channel_group_attrs, level_array_attrs
from timeseries_zarr.constants import (
    DECIMATION_FACTOR,
    ENVELOPE_PAIR_SIZE,
    FLOAT32_BYTES,
)
from timeseries_zarr.fold import fold_block
from timeseries_zarr.planning import level0_period_us, plan_levels
from timeseries_zarr.protocols import ContinuousChannelSource
from timeseries_zarr.sizing import chunk_and_shard
from timeseries_zarr.streaming import (
    BlockReadableArray,
    _rebuffer_and_fold,
    iter_array_blocks,
    iter_raw_blocks,
)
from timeseries_zarr.types import ChunkShard, LevelPlan, WriteOpts
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
    for plan in plan_levels(
        source.num_samples(),
        level0_period_us(source.rate_hz()),
        opts.max_levels,
        opts.min_bins,
    ):
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
