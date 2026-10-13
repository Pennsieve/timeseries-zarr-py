"""Compose streaming and zarr I/O to write one event channel's arrays.

Each array writer takes its chunk and shard shapes from sizing and streams the
source's rows in. An empty source, with no events, creates the array and
writes nothing.
"""

from collections.abc import Callable
from typing import Final

import numpy as np
import numpy.typing as npt

from timeseries_zarr.protocols import EventChannelSource
from timeseries_zarr.types import ChunkShard
from timeseries_zarr.zarr_io import (
    ZarrArray,
    ZarrGroup,
    create_array,
    write_region,
)

EVENTS_KEY: Final = "events"
"""Key of the event time array within an event channel group."""

EVENT_DIMENSION: Final = "event"
"""Dimension name of the per-event axis."""


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
    source: EventChannelSource,
    origin_us: int,
    sizing: ChunkShard,
    zstd_level: int,
) -> ZarrArray:
    """Create the events array under group and stream the event times into it.

    A rank-1 int64 array named "events" with the dimension name "event" and no
    attributes. Each time is the source's wall-clock microseconds minus
    origin_us, the bundle origin, so the stored times count from the origin.
    Equal times are allowed. Raises ValueError if a time falls before the
    origin or if the times decrease, within a block or across a block
    boundary.
    """
    array = create_array(
        group,
        EVENTS_KEY,
        (source.num_events(),),
        np.int64,
        sizing.chunk_shape,
        sizing.shard_shape,
        {},
        zstd_level,
        dimension_names=(EVENT_DIMENSION,),
    )
    previous_last: int | None = None

    def read(start: int, stop: int) -> npt.NDArray[np.int64]:
        return source.read_events(start, stop) - origin_us

    def check(block: npt.NDArray[np.int64]) -> None:
        nonlocal previous_last
        if block[0] < 0:
            raise ValueError(
                f"an event precedes the bundle origin by {-int(block[0])} "
                "microseconds"
            )
        decreases_inside = bool((block[1:] < block[:-1]).any())
        decreases_at_seam = (
            previous_last is not None and block[0] < previous_last
        )
        if decreases_inside or decreases_at_seam:
            raise ValueError("event times must be non-decreasing")
        previous_last = int(block[-1])

    _write_source_blocks(
        array, source.num_events(), sizing.shard_shape[0], read, check
    )
    return array
