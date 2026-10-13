"""Compose streaming and zarr I/O to write one unit (spike) channel's arrays.

Each array writer takes its chunk and shard shapes from sizing and streams the
source's rows in. An empty source (no events) creates the array and writes
nothing.
"""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from timeseries_zarr.attrs import channel_group_attrs, waveform_array_attrs
from timeseries_zarr.constants import FLOAT32_BYTES, INT64_BYTES, UINT8_BYTES
from timeseries_zarr.planning import sample_period_us
from timeseries_zarr.protocols import UnitChannelSource
from timeseries_zarr.sizing import chunk_and_shard
from timeseries_zarr.types import ChunkShard, WriteOpts
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
    sizing: ChunkShard,
    zstd_level: int,
) -> ZarrArray:
    """Create the events array under group and stream the source's timestamps in.

    A rank-1 int64 array named "events" of absolute-microsecond timestamps,
    with no custom attributes. Raises ValueError if the timestamps are not
    non-decreasing, including across a block boundary.
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
        source.read_events,
        _check_ascending,
    )
    return array


def write_units_array(
    group: ZarrGroup,
    source: UnitChannelSource,
    sizing: ChunkShard,
    zstd_level: int,
) -> ZarrArray:
    """Create the units array under group and stream the source's cluster ids in.

    A rank-1 uint8 array named "units", with no custom attributes. Row k
    classifies the event at events[k].
    """
    array = create_array(
        group=group,
        name="units",
        shape=(source.num_events(),),
        dtype=np.uint8,
        chunk_shape=sizing.chunk_shape,
        shard_shape=sizing.shard_shape,
        attrs={},
        zstd_level=zstd_level,
    )

    _write_source_blocks(
        array,
        source.num_events(),
        sizing.shard_shape[0],
        source.read_units,
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
    opts: WriteOpts,
) -> None:
    """Write one unit channel as the subgroup named str(index).

    Creates the channel group under parent carrying its unit-kind attributes,
    then writes the events, units, and waveforms arrays from the source, each
    sized and compressed per opts. The waveform period comes from the source's
    sample rate. num_events sizes all three arrays, so the source must keep
    events, units, and waveforms the same length.
    """
    attributes = channel_group_attrs(
        source.id,
        source.rate_hz(),
        source.start_us(),
        "unit",
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
        _sizing((n,), INT64_BYTES),
        opts.zstd_level,
    )
    write_units_array(
        group,
        source,
        _sizing((n,), UINT8_BYTES),
        opts.zstd_level,
    )
    write_waveforms_array(
        group,
        source,
        sample_period_us(source.rate_hz()),
        _sizing((n, source.points_per_event()), FLOAT32_BYTES),
        opts.zstd_level,
    )
