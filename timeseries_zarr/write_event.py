"""Compose streaming and zarr I/O to write one event channel's arrays.

Each array writer takes its chunk and shard shapes from sizing and streams the
source's rows in. An empty source, with no events, creates the array and
writes nothing.
"""

from collections.abc import Callable, Mapping
from typing import Final

import numpy as np
import numpy.typing as npt

from timeseries_zarr.columns import channels_rows, channels_width, text_rows
from timeseries_zarr.constants import MAX_LABELS
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

DURATIONS_KEY: Final = "durations"
"""Key of the per-event duration array within an event channel group."""

LABELS_KEY: Final = "labels"
"""Key of the per-event label array within an event channel group."""

TEXT_KEY: Final = "text"
"""Key of the per-event text array within an event channel group."""

CHANNELS_KEY: Final = "channels"
"""Key of the per-event channel bitmask array within an event channel group."""

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


def write_durations_array(
    group: ZarrGroup,
    source: EventChannelSource,
    sizing: ChunkShard,
    zstd_level: int,
) -> ZarrArray:
    """Create the durations array under group and stream the durations into it.

    A rank-1 int64 array named "durations" of microseconds, with the dimension
    name "event" and no attributes. A duration of 0 is a point event. Raises
    ValueError if the source reports no max_duration_us, a duration is
    negative, or a duration exceeds max_duration_us, which the channel declares
    as a bound on every duration.
    """
    bound = source.max_duration_us()
    if bound is None:
        raise ValueError("a source with durations must report max_duration_us")
    array = create_array(
        group,
        DURATIONS_KEY,
        (source.num_events(),),
        np.int64,
        sizing.chunk_shape,
        sizing.shard_shape,
        {},
        zstd_level,
        dimension_names=(EVENT_DIMENSION,),
    )

    def check(block: npt.NDArray[np.int64]) -> None:
        if block.min() < 0:
            raise ValueError(f"a duration is negative: {int(block.min())}")
        if block.max() > bound:
            raise ValueError(
                f"a duration {int(block.max())} exceeds max_duration_us {bound}"
            )

    _write_source_blocks(
        array,
        source.num_events(),
        sizing.shard_shape[0],
        source.read_durations,
        check,
    )
    return array


def write_labels_array(
    group: ZarrGroup,
    source: EventChannelSource,
    sizing: ChunkShard,
    zstd_level: int,
) -> ZarrArray:
    """Create the labels array under group and stream the event labels into it.

    A rank-1 uint16 array named "labels", with the dimension name "event" and
    no attributes. Each label indexes the channel's label categories, so it
    must be below source.num_labels(). Raises ValueError if num_labels() is
    above MAX_LABELS, which a uint16 cannot index, or if a label is at or above
    num_labels().
    """
    categories = source.num_labels()
    if categories > MAX_LABELS:
        raise ValueError(
            f"{categories} label categories exceed the uint16 limit {MAX_LABELS}"
        )
    array = create_array(
        group,
        LABELS_KEY,
        (source.num_events(),),
        np.uint16,
        sizing.chunk_shape,
        sizing.shard_shape,
        {},
        zstd_level,
        dimension_names=(EVENT_DIMENSION,),
    )

    def check(block: npt.NDArray[np.uint16]) -> None:
        largest = int(block.max())
        if largest >= categories:
            raise ValueError(
                f"a label {largest} is outside the {categories} label categories"
            )

    _write_source_blocks(
        array,
        source.num_events(),
        sizing.shard_shape[0],
        source.read_labels,
        check,
    )
    return array


def write_text_array(
    group: ZarrGroup,
    source: EventChannelSource,
    sizing: ChunkShard,
    zstd_level: int,
) -> ZarrArray:
    """Create the text array under group and stream the event texts into it.

    A uint8 array named "text" shaped (num_events, W), with W the source's
    text_width(), and the dimension names "event" and None. Row i holds the
    UTF-8 bytes of event i's text followed by zero bytes. The array is stored
    without a compression codec, so a shard object is the rows as plain text
    followed by the shard index, which a text tool can read and redact.
    sizing is the sizing of the (num_events, W) shape, with the width axis
    whole. Raises ValueError if text_width() is below 1, a text is longer than
    the width or holds a NUL character, or the source returns a different
    number of texts than requested.
    """
    width = source.text_width()
    if width < 1:
        raise ValueError(f"text_width must be at least 1, got {width}")
    array = create_array(
        group,
        TEXT_KEY,
        (source.num_events(), width),
        np.uint8,
        sizing.chunk_shape,
        sizing.shard_shape,
        {},
        zstd_level,
        dimension_names=(EVENT_DIMENSION, None),
        compress=False,
    )

    def read(start: int, stop: int) -> npt.NDArray[np.uint8]:
        texts = source.read_text(start, stop)
        if len(texts) != stop - start:
            raise ValueError(
                f"read_text returned {len(texts)} texts for {stop - start} events"
            )
        return text_rows(texts, width)

    _write_source_blocks(
        array, source.num_events(), sizing.shard_shape[0], read
    )
    return array


def write_channels_array(
    group: ZarrGroup,
    source: EventChannelSource,
    channel_keys: Mapping[str, int],
    max_key: int,
    sizing: ChunkShard,
    zstd_level: int,
) -> ZarrArray:
    """Create the channels array under group and stream the event bitmasks in.

    A uint8 array named "channels" shaped (num_events, M), with M the byte
    width of a bitmask for keys up to max_key, and the dimension names "event"
    and None. Row i has the bit of each channel key that event i applies to,
    and a row of zeros means the whole recording. The source names channels by
    id and channel_keys maps each id to the channel key it has in the bundle.
    sizing is the sizing of the (num_events, M) shape, with the width axis
    whole. Raises ValueError if max_key is negative, an id is not in
    channel_keys, a key is outside 0 to max_key, or the source returns a
    different number of id lists than requested.
    """
    width = channels_width(max_key)
    array = create_array(
        group,
        CHANNELS_KEY,
        (source.num_events(), width),
        np.uint8,
        sizing.chunk_shape,
        sizing.shard_shape,
        {},
        zstd_level,
        dimension_names=(EVENT_DIMENSION, None),
    )

    def read(start: int, stop: int) -> npt.NDArray[np.uint8]:
        id_lists = source.read_channel_ids(start, stop)
        if len(id_lists) != stop - start:
            raise ValueError(
                f"read_channel_ids returned {len(id_lists)} lists for "
                f"{stop - start} events"
            )
        key_lists = []
        for ids in id_lists:
            for channel_id in ids:
                if channel_id not in channel_keys:
                    raise ValueError(f"unknown channel id {channel_id!r}")
            key_lists.append([channel_keys[channel_id] for channel_id in ids])
        return channels_rows(key_lists, max_key)

    _write_source_blocks(
        array, source.num_events(), sizing.shard_shape[0], read
    )
    return array
