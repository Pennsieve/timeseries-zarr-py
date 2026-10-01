"""Compose streaming and zarr I/O to write one unit (spike) event channel.

Each array writer takes its chunk and shard shapes from sizing and streams the
source's rows in. An empty source (no events) creates the array and writes
nothing.
"""

from collections.abc import Callable, Iterable, Iterator
from typing import cast

import numpy as np
import numpy.typing as npt

from timeseries_zarr.attrs import (
    channel_group_attrs,
    level_group_attrs,
    waveform_array_attrs,
)
from timeseries_zarr.constants import (
    DECIMATION_FACTOR,
    FLOAT32_BYTES,
    INT64_BYTES,
    UINT16_BYTES,
    UINT32_BYTES,
)
from timeseries_zarr.counts import (
    fold_counts_block,
    iter_count_blocks,
    plan_count_levels,
)
from timeseries_zarr.planning import sample_period_us
from timeseries_zarr.protocols import UnitChannelSource
from timeseries_zarr.sizing import chunk_and_shard
from timeseries_zarr.streaming import (
    BlockReadableArray,
    _rebuffer_and_fold,
    iter_array_blocks,
)
from timeseries_zarr.types import ChunkShard, LevelPlan, WriteOpts
from timeseries_zarr.zarr_io import (
    ZarrArray,
    ZarrGroup,
    create_array,
    create_group_with_attrs,
    write_region,
)


def _write_source_blocks[T: np.generic](
    array: ZarrArray,
    n: int,
    block_len: int,
    reader: Callable[[int, int], npt.NDArray[T]],
    on_block: Callable[[npt.NDArray[T]], None] | None = None,
) -> None:
    """Stream a source's rows into array in block_len-sized axis-0 blocks.

    Each [begin, begin + block_len) window is read through reader and written
    at the running axis-0 offset. on_block, when given, inspects a block before
    its write. Callers pass the shard length: a write narrower than a shard
    makes the sharding codec read that shard back, re-encode every inner chunk,
    and rewrite it.
    """
    start = 0
    for begin in range(0, n, block_len):
        block = reader(begin, min(begin + block_len, n))
        if on_block is not None:
            on_block(block)
        write_region(array, start, block)
        start += block.shape[0]


def write_events_array(
    group: ZarrGroup,
    source: UnitChannelSource,
    onset_us: int,
    sizing: ChunkShard,
    zstd_level: int,
) -> ZarrArray:
    """Create the events array under group and stream the source's timestamps in.

    A rank-1 int64 array named "events", with no custom attributes. The
    source reports wall-clock microseconds and the array stores the distance
    from onset_us instead, because no absolute time may appear outside
    meta/. Raises ValueError if the timestamps are not non-decreasing,
    including across a block boundary.
    """
    array = create_array(
        group=group,
        name="events",
        shape=(source.num_events(),),
        dtype=np.int64,
        chunk_shape=sizing.chunk_shape,
        shard_shape=sizing.shard_shape,
        attrs={},
        zstd_level=zstd_level,
    )

    def _read_relative(start: int, stop: int) -> npt.NDArray[np.int64]:
        return source.read_events(start, stop) - onset_us

    prev_last: np.int64 | None = None

    def _check_ascending(block: npt.NDArray[np.int64]) -> None:
        nonlocal prev_last
        descends_inside = bool((block[1:] < block[:-1]).any())
        descends_at_seam = prev_last is not None and block[0] < prev_last
        if descends_inside or descends_at_seam:
            raise ValueError("event timestamps must be non-decreasing")
        prev_last = block[-1]

    _write_source_blocks(
        array,
        source.num_events(),
        sizing.shard_shape[0],
        _read_relative,
        _check_ascending,
    )
    return array


def write_labels_array(
    group: ZarrGroup,
    source: UnitChannelSource,
    sizing: ChunkShard,
    zstd_level: int,
) -> ZarrArray:
    """Create the labels array under group and stream the source's labels in.

    A rank-1 uint16 array named "labels", with no custom attributes. Row k
    categorizes the event at events[k]: a cluster id for a sorted spike.
    """
    array = create_array(
        group=group,
        name="labels",
        shape=(source.num_events(),),
        dtype=np.uint16,
        chunk_shape=sizing.chunk_shape,
        shard_shape=sizing.shard_shape,
        attrs={},
        zstd_level=zstd_level,
    )

    _write_source_blocks(
        array,
        source.num_events(),
        sizing.shard_shape[0],
        source.read_labels,
    )
    return array


def write_waveforms_array(
    group: ZarrGroup,
    source: UnitChannelSource,
    period_us: float,
    sizing: ChunkShard,
    zstd_level: int,
) -> ZarrArray:
    """Create the waveforms array under group and stream the source's waveforms in.

    A rank-2 float32 array named "waveforms" of shape (n_events,
    points_per_event), carrying period_us as its sole attribute: the
    microseconds between adjacent waveform samples. Row k is the waveform for
    the event at events[k].
    """
    array = create_array(
        group=group,
        name="waveforms",
        shape=(source.num_events(), source.points_per_event()),
        dtype=np.float32,
        chunk_shape=sizing.chunk_shape,
        shard_shape=sizing.shard_shape,
        attrs=waveform_array_attrs(period_us),
        zstd_level=zstd_level,
    )

    _write_source_blocks(
        array,
        source.num_events(),
        sizing.shard_shape[0],
        source.read_waveforms,
    )
    return array


def write_unit_channel(
    parent: ZarrGroup,
    index: int,
    source: UnitChannelSource,
    *,
    onset_us: int,
    opts: WriteOpts,
) -> None:
    """Write one unit channel as the subgroup named str(index).

    Creates the channel group under parent carrying its unit-kind attributes,
    then writes the events, labels, and waveforms arrays from the source, each
    sized and compressed per opts. The waveform period comes from the source's
    sample rate. num_events sizes all three arrays, so the source must keep
    events, labels, and waveforms the same length.

    onset_us is the bundle's onset in wall-clock microseconds; the channel
    offset and every event timestamp are stored as distances from it.
    """
    attributes = channel_group_attrs(
        source.id,
        source.rate_hz(),
        source.start_us() - onset_us,
        "event",
        source.name,
        source.unit,
    )
    group = create_group_with_attrs(parent, str(index), attributes)

    n = source.num_events()

    def _sizing(level_shape: tuple[int, ...], dtype_size: int) -> ChunkShard:
        return chunk_and_shard(
            level_shape=level_shape,
            dtype_size=dtype_size,
            inner_len=opts.inner_len,
            target_shard_bytes=opts.target_shard_bytes,
        )

    write_events_array(
        group,
        source,
        onset_us,
        _sizing((n,), INT64_BYTES),
        opts.zstd_level,
    )
    write_labels_array(
        group,
        source,
        _sizing((n,), UINT16_BYTES),
        opts.zstd_level,
    )
    write_waveforms_array(
        group,
        source,
        sample_period_us(source.rate_hz()),
        _sizing((n, source.points_per_event()), FLOAT32_BYTES),
        opts.zstd_level,
    )
    last_us = int(source.read_events(n - 1, n)[0]) - onset_us if n else 0
    write_count_levels(group, source, onset_us, last_us + 1, opts)


COUNTS_KEY = "counts"
"""Key of the per-bin event-count member within a level group."""


def _iter_event_blocks(
    source: UnitChannelSource, onset_us: int, block_events: int
) -> Iterator[tuple[npt.NDArray[np.int64], npt.NDArray[np.uint16] | None]]:
    """Yield the source's events and labels together, onset-relative, in windows."""
    n = source.num_events()
    labelled = source.num_labels() > 0
    for begin in range(0, n, block_events):
        stop = min(begin + block_events, n)
        times = source.read_events(begin, stop) - onset_us
        yield times, (source.read_labels(begin, stop) if labelled else None)


def write_count_level(
    parent: ZarrGroup,
    blocks: Iterable[npt.NDArray[np.uint32]],
    plan: LevelPlan,
    sizing: ChunkShard,
    zstd_level: int,
) -> ZarrArray:
    """Create the level group named plan.level and stream its counts member in.

    The label axis is never chunked: a reader asking for one window wants every
    label in it, and splitting them would turn one read into many.
    """
    group = create_group_with_attrs(
        parent, str(plan.level), level_group_attrs(plan.period_us)
    )
    array = create_array(
        group=group,
        name=COUNTS_KEY,
        shape=plan.shape,
        dtype=np.uint32,
        chunk_shape=sizing.chunk_shape,
        shard_shape=sizing.shard_shape,
        attrs={},
        zstd_level=zstd_level,
    )
    start = 0
    for block in blocks:
        write_region(array, start, block)
        start += block.shape[0]
    return array


def write_count_levels(
    group: ZarrGroup,
    source: UnitChannelSource,
    onset_us: int,
    span_us: int,
    opts: WriteOpts,
) -> None:
    """Write the channel's count pyramid, if its event count earns one.

    A channel a reader can raster directly gets nothing: below the threshold the
    events answer every zoom on their own, and a pyramid would be weight with no
    reader. Level 1 is counted from the events in one streamed pass and each
    level above it is the 4x sum fold of the one below, read back from disk the
    way a continuous level is.
    """
    n = source.num_events()
    if n < opts.event_level_threshold:
        return

    n_labels = source.num_labels()
    plans = plan_count_levels(
        span_us,
        n_labels,
        opts.max_levels,
        opts.min_bins,
        opts.target_shard_bytes,
    )
    if not plans:
        return

    def _sizing(shape: tuple[int, ...]) -> ChunkShard:
        rows = chunk_and_shard(
            level_shape=(shape[0],),
            dtype_size=UINT32_BYTES * max(1, n_labels),
            inner_len=opts.inner_len,
            target_shard_bytes=opts.target_shard_bytes,
        )
        trailing = shape[1:]
        return ChunkShard(
            chunk_shape=(rows.chunk_shape[0], *trailing),
            shard_shape=(rows.shard_shape[0], *trailing),
        )

    first, *rest = plans
    sizing = _sizing(first.shape)
    previous = write_count_level(
        group,
        iter_count_blocks(
            _iter_event_blocks(source, onset_us, opts.inner_len),
            n_labels,
            first.period_us,
            first.shape[0],
            sizing.shard_shape[0],
        ),
        first,
        sizing,
        opts.zstd_level,
    )
    for plan in rest:
        sizing = _sizing(plan.shape)
        previous = write_count_level(
            group,
            _rebuffer_and_fold(
                iter_array_blocks(
                    cast("BlockReadableArray", previous),
                    DECIMATION_FACTOR * sizing.shard_shape[0],
                ),
                fold_counts_block,
            ),
            plan,
            sizing,
            opts.zstd_level,
        )
