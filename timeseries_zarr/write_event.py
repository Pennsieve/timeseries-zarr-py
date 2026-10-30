"""Compose streaming and zarr I/O to write one event channel's arrays.

Each array writer takes its chunk and shard shapes from sizing and streams the
source's rows in. An empty source, with no events, creates the array and
writes nothing.
"""

from collections.abc import Callable, Iterable, Iterator, Mapping
from functools import partial
from itertools import chain
from typing import Final, cast

import numpy as np
import numpy.typing as npt

from timeseries_zarr.attrs import (
    check_time_us,
    event_channel_attrs,
    level_group_attrs,
    waveform_array_attrs,
)
from timeseries_zarr.columns import channels_rows, channels_width, text_rows
from timeseries_zarr.constants import (
    CHANNELS_KEY,
    COUNT_KEY,
    DURATIONS_KEY,
    EVENT_COUNT_LIMIT,
    EVENT_DIMENSION,
    EVENTS_KEY,
    FLOAT32_BYTES,
    INT64_BYTES,
    LABEL_DIMENSION,
    LABELS_KEY,
    MAX_LABELS,
    TEXT_KEY,
    TIME_DIMENSION,
    UINT8_BYTES,
    UINT16_BYTES,
    UINT32_BYTES,
    WAVEFORM_MEAN_KEY,
)
from timeseries_zarr.counts import (
    EventBlock,
    fold_count_block,
    iter_count_blocks,
    plan_count_levels,
)
from timeseries_zarr.protocols import EventChannelSource, source_provenance
from timeseries_zarr.sizing import chunk_and_shard
from timeseries_zarr.streaming import (
    BlockReadableArray,
    _rebuffer_and_fold,
    iter_array_blocks,
    regroup,
)
from timeseries_zarr.types import ChunkShard, LevelPlan, WriteOpts
from timeseries_zarr.zarr_io import (
    ZarrArray,
    ZarrGroup,
    create_array,
    create_group_with_attrs,
    write_region,
)

WAVEFORM_MEAN_RANK: Final = 2
"""Rank of the waveform_mean array: one axis of labels, one of samples."""


def _write_source_blocks[T: np.generic](
    array: ZarrArray,
    n: int,
    block_len: int,
    reader: Callable[[int, int], npt.NDArray[T]],
    on_block: Callable[[npt.NDArray[T]], None] | None = None,
) -> None:
    """Stream a source's rows into array in block_len-sized axis-0 blocks.

    Each [begin, begin + block_len) window is read through reader and written
    at the running axis-0 offset. Callers pass the shard length: a write
    narrower than a shard makes the sharding codec read that shard back,
    re-encode every inner chunk, and rewrite it.

    Args:
        array: Array to write into.
        n: Total rows to stream.
        block_len: Rows per block.
        reader: Returns the rows from begin up to end.
        on_block: Called with each block before its write.
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
    reference_us: int,
    sizing: ChunkShard,
    zstd_level: int,
) -> ZarrArray:
    """Create the events array under group and stream the event times into it.

    A rank-1 int64 array named "events" with the dimension name "event" and no
    attributes. The stored times count from the reference time, and an event
    before it is negative. Equal times are allowed.

    Args:
        group: Channel group to create the array in.
        source: Source of the event times.
        reference_us: Reference time in microseconds since the Unix epoch.
            The source times are in the same unit.
        sizing: Chunk and shard shapes of the array.
        zstd_level: Zstandard compression level.

    Returns:
        The created array.

    Raises:
        ValueError: If the times decrease, within a block or across a block
            boundary, or a stored time lies outside (-2^53, 2^53).
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
        return source.read_events(start, stop) - reference_us

    def check(block: npt.NDArray[np.int64]) -> None:
        nonlocal previous_last
        decreases_inside = bool((block[1:] < block[:-1]).any())
        decreases_at_seam = (
            previous_last is not None and block[0] < previous_last
        )
        if decreases_inside or decreases_at_seam:
            raise ValueError("event times must be non-decreasing")
        check_time_us(int(block.min()), "an event time")
        check_time_us(int(block.max()), "an event time")
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
    *,
    reference_us: int = 0,
) -> ZarrArray:
    """Create the durations array under group and stream the durations into it.

    A rank-1 int64 array named "durations" of microseconds, with the dimension
    name "event" and no attributes. A duration of 0 is a point event.

    Args:
        group: Channel group to create the array in.
        source: Source of the durations.
        sizing: Chunk and shard shapes of the array.
        zstd_level: Zstandard compression level.
        reference_us: Reference time in microseconds since the Unix epoch,
            which the end of each interval counts from.

    Returns:
        The created array.

    Raises:
        ValueError: If the source reports no max_duration_us, a duration is
            negative, a duration exceeds max_duration_us, which the channel
            declares as a bound on every duration, or the end of an interval
            lies outside (-2^53, 2^53).
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

    def read(start: int, stop: int) -> npt.NDArray[np.int64]:
        block = source.read_durations(start, stop)
        if block.min() < 0:
            raise ValueError(f"a duration is negative: {int(block.min())}")
        if block.max() > bound:
            raise ValueError(
                f"a duration {int(block.max())} exceeds max_duration_us {bound}"
            )
        ends = source.read_events(start, stop) - reference_us + block
        check_time_us(int(ends.max()), "the end of an interval")
        return block

    _write_source_blocks(
        array, source.num_events(), sizing.shard_shape[0], read
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
    must be below source.num_labels().

    Args:
        group: Channel group to create the array in.
        source: Source of the labels.
        sizing: Chunk and shard shapes of the array.
        zstd_level: Zstandard compression level.

    Returns:
        The created array.

    Raises:
        ValueError: If num_labels() is above MAX_LABELS, which a uint16 cannot
            index, or if a label is at or above num_labels().
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

    Args:
        group: Channel group to create the array in.
        source: Source of the texts.
        sizing: Chunk and shard shapes of the (num_events, W) shape, with the
            width axis whole.
        zstd_level: Zstandard compression level.

    Returns:
        The created array.

    Raises:
        ValueError: If text_width() is below 1, a text is longer than the
            width or holds a NUL character, or the source returns a different
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
    id.

    Args:
        group: Channel group to create the array in.
        source: Source of the channel ids of each event.
        channel_keys: Channel key in the bundle, by channel id.
        max_key: Largest channel key in the bundle.
        sizing: Chunk and shard shapes of the (num_events, M) shape, with the
            width axis whole.
        zstd_level: Zstandard compression level.

    Returns:
        The created array.

    Raises:
        ValueError: If max_key is negative, an id is not in channel_keys, a
            key is outside 0 to max_key, or the source returns a different
            number of id lists than requested.
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


def write_waveform_mean_array(
    group: ZarrGroup,
    source: EventChannelSource,
    zstd_level: int,
) -> ZarrArray:
    """Create the waveform_mean array under group and write the source's means.

    A float32 array named "waveform_mean" shaped (labels, samples), with the
    dimension names "label" and None, holding the mean snippet of each label
    category. It carries period_us, the microseconds between snippet samples,
    and no event_sample. A label that has no events keeps whatever the source
    gives, NaN by the format's convention. The array has one row per label, so
    it is sized with the default chunk and shard targets and written whole.

    Args:
        group: Channel group to create the array in.
        source: Source of the means and the sample period.
        zstd_level: Zstandard compression level.

    Returns:
        The created array.

    Raises:
        ValueError: If the means are not shaped (labels, samples), have no
            samples, or have a row count other than source.num_labels(), or if
            the period is not finite and positive.
    """
    means = np.asarray(source.waveform_mean(), dtype=np.float32)
    if len(means.shape) != WAVEFORM_MEAN_RANK:
        raise ValueError(
            f"waveform_mean has shape (labels, samples), got {means.shape}"
        )
    labels, samples = means.shape
    if samples < 1:
        raise ValueError("waveform_mean needs at least 1 sample per label")
    if labels != source.num_labels():
        raise ValueError(
            f"waveform_mean has {labels} rows but the channel has "
            f"{source.num_labels()} label categories"
        )
    attrs = waveform_array_attrs(source.waveform_period_us(), samples=samples)
    sizing = chunk_and_shard(means.shape, FLOAT32_BYTES)
    array = create_array(
        group,
        WAVEFORM_MEAN_KEY,
        means.shape,
        np.float32,
        sizing.chunk_shape,
        sizing.shard_shape,
        attrs,
        zstd_level,
        dimension_names=(LABEL_DIMENSION, None),
    )
    write_region(array, 0, means)
    return array


def _row_sizing(
    rows: int, columns: int, itemsize: int, opts: WriteOpts
) -> ChunkShard:
    """Return the chunk and shard shapes of a (rows, columns) array.

    The inner chunk spans at most opts.inner_len rows, and fewer when the row
    is wide, so that one chunk stays within opts.target_shard_bytes and a
    block of rows never outgrows a shard. The column axis is never chunked.

    Args:
        rows: Length of axis 0.
        columns: Length of axis 1.
        itemsize: Bytes per element.
        opts: Chunk and shard targets.
    """
    fitting = max(1, opts.target_shard_bytes // (columns * itemsize))
    return chunk_and_shard(
        (rows, columns),
        itemsize,
        min(opts.inner_len, fitting),
        opts.target_shard_bytes,
    )


def _event_blocks(
    source: EventChannelSource,
    reference_us: int,
    block_len: int,
    with_labels: bool,
) -> Iterator[EventBlock]:
    """Yield the source's event times from the reference time, with labels.

    Args:
        source: Source of the events.
        reference_us: Reference time in microseconds since the Unix epoch.
        block_len: Events per block.
        with_labels: Whether each block carries labels. Otherwise they are
            None.

    Yields:
        A block of times and its labels, or None for the labels.
    """
    total = source.num_events()
    for start in range(0, total, block_len):
        stop = min(start + block_len, total)
        times = source.read_events(start, stop) - reference_us
        labels = source.read_labels(start, stop) if with_labels else None
        yield times, labels


def _write_count_blocks(
    array: ZarrArray, blocks: Iterable[npt.NDArray[np.uint32]]
) -> None:
    """Write each block of counts to array at its running row offset.

    Args:
        array: Array to write into.
        blocks: Blocks of rows, in order.
    """
    start = 0
    for block in blocks:
        write_region(array, start, block)
        start += block.shape[0]


def write_count_levels(
    group: ZarrGroup,
    source: EventChannelSource,
    reference_us: int,
    opts: WriteOpts,
) -> None:
    """Write the count levels of a dense event channel under group.

    Each level is a group named by its number, carrying period_us and
    offset_us, with one uint32 array "count" shaped (bins, L). L is the label
    count, or 1 without labels, and the bin axis is named "time". Level 1 is
    counted from the source's event times and labels, streamed in blocks of
    events; every level above it sums decimation_factor rows of the stored
    level below, one shard at a time, after the rows that precede its bin 0 are
    padded with zeros so the two grids align. Channels with too few events, or
    whose level 1 would hold fewer than opts.min_bins bins, get no levels and
    nothing is created. Every write covers a whole shard.

    Args:
        group: Channel group to create the levels in.
        source: Source of the event times and labels.
        reference_us: Reference time in microseconds since the Unix epoch.
        opts: Level limits, chunk and shard targets, and compression level.

    Raises:
        ValueError: If the event times decrease or if a label is outside the
            label categories.
    """
    total = source.num_events()
    if total == 0:
        return
    first_event_us = int(source.read_events(0, 1)[0]) - reference_us
    last_event_us = int(source.read_events(total - 1, total)[0]) - reference_us
    factor = opts.decimation_factor
    plans = plan_count_levels(
        total,
        first_event_us,
        last_event_us,
        opts.level_cap,
        opts.min_bins,
        decimation_factor=factor,
    )
    if not plans:
        return
    with_labels = "labels" in source.columns()
    n_labels = source.num_labels() if with_labels else 0
    columns = max(1, n_labels)
    block_len = chunk_and_shard(
        (total,), INT64_BYTES, opts.inner_len, opts.target_shard_bytes
    ).shard_shape[0]

    previous: tuple[ZarrArray, LevelPlan] | None = None
    for plan in plans:
        sizing = _row_sizing(plan.bins, columns, UINT32_BYTES, opts)
        level = create_group_with_attrs(
            group, plan.name, level_group_attrs(plan.period_us, plan.offset_us)
        )
        array = create_array(
            level,
            COUNT_KEY,
            (plan.bins, columns),
            np.uint32,
            sizing.chunk_shape,
            sizing.shard_shape,
            {},
            opts.zstd_level,
            dimension_names=(TIME_DIMENSION, None),
        )
        shard_rows = sizing.shard_shape[0]
        counts: Iterator[npt.NDArray[np.uint32]]
        if previous is None:
            counts = iter_count_blocks(
                _event_blocks(source, reference_us, block_len, with_labels),
                n_labels,
                int(plan.period_us),
                plan.offset_us,
                plan.bins,
                shard_rows,
            )
        else:
            below, below_plan = previous
            # Bin 0 of this level may open before bin 0 of the level below.
            shift = (below_plan.offset_us - plan.offset_us) // int(
                below_plan.period_us
            )
            blocks: Iterable[npt.NDArray[np.uint32]] = iter_array_blocks(
                cast("BlockReadableArray", below),
                factor * shard_rows,
            )
            if shift:
                blocks = chain(
                    [np.zeros((shift, columns), dtype=np.uint32)], blocks
                )
            counts = regroup(
                _rebuffer_and_fold(
                    blocks,
                    partial(fold_count_block, block=factor),
                    group=factor,
                ),
                shard_rows,
            )
        _write_count_blocks(array, counts)
        previous = (array, plan)


def check_event_channel(
    source: EventChannelSource, reference_us: int, opts: WriteOpts
) -> None:
    """Raise ValueError if a limit of the format rules out writing source.

    The checks read at most the first and last event. They run before the
    writer creates any node. A check on every event or duration runs while the
    array is written.

    Args:
        source: Source of the event channel.
        reference_us: Reference time in microseconds since the Unix epoch.
        opts: Level limits of the count levels the writer would plan.

    Raises:
        ValueError: If the channel holds 2^32 events or more, the first or
            last event time, max_duration_us, or a count level offset or end
            lies outside (-2^53, 2^53) microseconds.
    """
    total = source.num_events()
    if total >= EVENT_COUNT_LIMIT:
        raise ValueError(
            f"an event channel must hold fewer than 2^32 events, got {total}"
        )
    if "durations" in source.columns():
        bound = source.max_duration_us()
        if bound is not None:
            check_time_us(bound, "max_duration_us")
    if total == 0:
        return
    first_event_us = int(source.read_events(0, 1)[0]) - reference_us
    last_event_us = int(source.read_events(total - 1, total)[0]) - reference_us
    check_time_us(first_event_us, "an event time")
    check_time_us(last_event_us, "an event time")
    plan_count_levels(
        total,
        first_event_us,
        last_event_us,
        opts.level_cap,
        opts.min_bins,
        decimation_factor=opts.decimation_factor,
    )


def write_event_channel(
    parent: ZarrGroup,
    key: int,
    source: EventChannelSource,
    *,
    reference_us: int,
    channel_keys: Mapping[str, int],
    max_key: int,
    opts: WriteOpts,
) -> None:
    """Write one event channel as the subgroup named str(key).

    The channel group carries the event attributes: kind, name, id, the unit,
    the source's provenance, and, when the source provides the column,
    label_names and the duration bound max_duration_us. The events array
    comes first, then each optional column the source names in columns(),
    then the count levels of a dense channel.

    Args:
        parent: Group to create the channel group in.
        key: Channel key, the name of the channel group.
        source: Source of the event channel.
        reference_us: Reference time in microseconds since the Unix epoch.
            Times are stored from it.
        channel_keys: Channel key in the bundle, by channel id.
        max_key: Largest channel key in the bundle, which sizes the channels
            bitmask.
        opts: Chunk and shard targets and compression level of every array.

    Raises:
        ValueError: If the source offers waveform_mean without labels, its
            provenance is not a JSON object of finite values, or
            check_event_channel rejects it, and whatever the column writers
            raise for data that breaks the format.
    """
    check_event_channel(source, reference_us, opts)
    columns = source.columns()
    if "waveform_mean" in columns and "labels" not in columns:
        raise ValueError("waveform_mean requires a labels column")
    group = create_group_with_attrs(
        parent,
        str(key),
        event_channel_attrs(
            source.name,
            source.id,
            source.unit,
            source.label_names() if "labels" in columns else None,
            source.max_duration_us() if "durations" in columns else None,
            source_provenance(source),
        ),
    )
    total = source.num_events()

    def sizing(itemsize: int) -> ChunkShard:
        return chunk_and_shard(
            (total,), itemsize, opts.inner_len, opts.target_shard_bytes
        )

    write_events_array(
        group, source, reference_us, sizing(INT64_BYTES), opts.zstd_level
    )
    if "durations" in columns:
        write_durations_array(
            group,
            source,
            sizing(INT64_BYTES),
            opts.zstd_level,
            reference_us=reference_us,
        )
    if "labels" in columns:
        write_labels_array(group, source, sizing(UINT16_BYTES), opts.zstd_level)
    if "text" in columns:
        write_text_array(
            group,
            source,
            _row_sizing(total, source.text_width(), UINT8_BYTES, opts),
            opts.zstd_level,
        )
    if "channels" in columns:
        write_channels_array(
            group,
            source,
            channel_keys,
            max_key,
            _row_sizing(total, channels_width(max_key), UINT8_BYTES, opts),
            opts.zstd_level,
        )
    if "waveform_mean" in columns:
        write_waveform_mean_array(group, source, opts.zstd_level)
    write_count_levels(group, source, reference_us, opts)
