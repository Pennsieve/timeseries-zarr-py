"""Compose streaming and zarr I/O to write one continuous channel.

A channel is the raw samples plus a ladder of level groups above them. Raw is
written first and from the source; each level is folded from the one below,
level 1 from raw. A level group holds one array per statistic over a shared bin
axis: env, mean and, on a channel with gaps, valid.

The fold carries every statistic in one stream of stat blocks and the write
splits that stream into its member arrays, so raw is read once no matter how
many statistics a level holds. An empty input creates the arrays and writes
nothing.
"""

from collections.abc import Iterable, Iterator
from typing import cast

import numpy as np
import numpy.typing as npt

from timeseries_zarr.attrs import channel_group_attrs, level_group_attrs
from timeseries_zarr.constants import (
    DECIMATION_FACTOR,
    FLOAT32_BYTES,
    MAX_COL,
    MEAN_COL,
    MIN_COL,
    VALID_COL,
)
from timeseries_zarr.fold import fold_block
from timeseries_zarr.planning import plan_levels, raw_shape, sample_period_us
from timeseries_zarr.protocols import ContinuousChannelSource
from timeseries_zarr.sizing import chunk_and_shard
from timeseries_zarr.streaming import (
    BlockReadableArray,
    _rebuffer_and_fold,
    iter_level_stat_blocks,
    iter_offset_removed_blocks,
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

RAW_KEY = "raw"
"""Key of the full-resolution sample array within a channel group."""

ENV_KEY = "env"
"""Key of the (min, max) envelope member within a level group."""

MEAN_KEY = "mean"
"""Key of the per-bin mean member within a level group."""

VALID_KEY = "valid"
"""Key of the per-bin finite-sample count within a level group."""


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


def _write_stat_blocks(
    env: ZarrArray,
    mean: ZarrArray,
    valid: ZarrArray | None,
    blocks: Iterable[npt.NDArray[np.float64]],
) -> None:
    """Split a stream of stat blocks across a level's member arrays.

    Every member shares the block's row offset, which is why they are sized
    from one row geometry: a mean row is half an env row and a valid row half
    of that, so sizing them independently would put the arrays on different
    shard boundaries and only one could be written a whole shard at a time.
    """
    start = 0
    for block in blocks:
        rows = block.shape[0]
        write_region(
            env, start, block[:, MIN_COL : MAX_COL + 1].astype(np.float32)
        )
        write_region(mean, start, block[:, MEAN_COL].astype(np.float32))
        if valid is not None:
            write_region(valid, start, block[:, VALID_COL].astype(np.uint16))
        start += rows


def write_raw(
    group: ZarrGroup,
    source: ContinuousChannelSource,
    sizing: ChunkShard,
    zstd_level: int,
) -> tuple[ZarrArray, bool]:
    """Create the raw array under group and stream the source's samples into it.

    Rank-1 float32 under the key "raw", with no attributes: the sample period is
    the channel's rate_hz and restating it here would be a second place to keep
    it right. The samples keep their DC offset; only the statistics above them
    have it removed.

    Returns the array and whether any sample was non-finite. The spec writes
    valid only on channels that have gaps, and raw is the one pass that sees
    every sample.
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
    )
    has_gaps = False

    def _noting_gaps(
        blocks: Iterable[npt.NDArray[np.float32]],
    ) -> Iterator[npt.NDArray[np.float32]]:
        nonlocal has_gaps
        for block in blocks:
            has_gaps = has_gaps or not np.isfinite(block).all()
            yield block

    _write_blocks(
        array, _noting_gaps(iter_raw_blocks(source, sizing.shard_shape[0]))
    )
    return array, has_gaps


def write_level(
    parent: ZarrGroup,
    blocks: Iterable[npt.NDArray[np.float64]],
    plan: LevelPlan,
    sizing: ChunkShard,
    zstd_level: int,
    *,
    write_valid: bool = True,
) -> tuple[ZarrArray, ZarrArray, ZarrArray | None]:
    """Create the level group named plan.level and stream stat blocks into it.

    The group carries period_us and holds one array per statistic over a shared
    bin axis. Returns (env, mean, valid), which the next level folds from; valid
    is None when write_valid is false. Raises ValueError if plan describes a
    level below 1.
    """
    if plan.level < 1:
        raise ValueError("levels are numbered from 1; raw is not a level")

    group = create_group_with_attrs(
        parent, str(plan.level), level_group_attrs(plan.period_us)
    )
    env = create_array(
        group,
        ENV_KEY,
        plan.shape,
        np.float32,
        sizing.chunk_shape,
        sizing.shard_shape,
        {},
        zstd_level,
    )
    mean = create_array(
        group,
        MEAN_KEY,
        (plan.shape[0],),
        np.float32,
        (sizing.chunk_shape[0],),
        (sizing.shard_shape[0],),
        {},
        zstd_level,
    )
    valid = (
        create_array(
            group,
            VALID_KEY,
            (plan.shape[0],),
            np.uint16,
            (sizing.chunk_shape[0],),
            (sizing.shard_shape[0],),
            {},
            zstd_level,
        )
        if write_valid
        else None
    )
    _write_stat_blocks(env, mean, valid, blocks)
    return env, mean, valid


def write_continuous_channel(
    parent: ZarrGroup,
    index: int,
    source: ContinuousChannelSource,
    *,
    onset_us: int,
    opts: WriteOpts,
) -> None:
    """Write one continuous channel as the subgroup named str(index).

    1. Create the channel group with its continuous-kind attributes.
    2. Write the raw samples from the source.
    3. Fold level 1 from raw, with the channel's DC offset removed, and each
       level after it from the one written below. valid is written only if
       raw held a non-finite sample.

    The pyramid is planned from the source's sample count and rate; every array
    is sized and compressed per opts. A channel too short to fill one level gets
    raw and nothing else.

    onset_us is the bundle's onset in wall-clock microseconds. The channel
    records its distance from it, not its own wall-clock start: no absolute
    time may appear outside meta/.
    """
    offset_uv = source.offset_uv()
    attributes = channel_group_attrs(
        source.id,
        source.rate_hz(),
        source.start_us() - onset_us,
        "continuous",
        source.name,
        source.unit,
        offset_uv,
    )
    group = create_group_with_attrs(parent, str(index), attributes)

    def _sizing(shape: tuple[int, ...]) -> ChunkShard:
        return chunk_and_shard(
            level_shape=shape,
            dtype_size=FLOAT32_BYTES,
            inner_len=opts.inner_len,
            target_shard_bytes=opts.target_shard_bytes,
        )

    num_samples = source.num_samples()
    raw, has_gaps = write_raw(
        group=group,
        source=source,
        sizing=_sizing(raw_shape(num_samples)),
        zstd_level=opts.zstd_level,
    )

    previous: tuple[ZarrArray, ZarrArray, ZarrArray | None, int] | None = None
    for plan in plan_levels(
        num_samples,
        sample_period_us(source.rate_hz()),
        opts.max_levels,
        opts.min_bins,
    ):
        sizing = _sizing(plan.shape)
        # Read the level below 4 shards at a time, so one folded block fills
        # exactly one shard of the level being written.
        read_len = DECIMATION_FACTOR * sizing.shard_shape[0]
        if previous is None:
            below: Iterator[npt.NDArray[np.float64]] = (
                iter_offset_removed_blocks(
                    cast("BlockReadableArray", raw), offset_uv, read_len
                )
            )
        else:
            prev_env, prev_mean, prev_valid, prev_level = previous
            below = iter_level_stat_blocks(
                cast("BlockReadableArray", prev_env),
                cast("BlockReadableArray", prev_mean),
                cast("BlockReadableArray | None", prev_valid),
                num_samples,
                prev_level,
                read_len,
            )
        env, mean, valid = write_level(
            parent=group,
            blocks=_rebuffer_and_fold(below, fold_block),
            plan=plan,
            sizing=sizing,
            zstd_level=opts.zstd_level,
            write_valid=has_gaps,
        )
        previous = (env, mean, valid, plan.level)
