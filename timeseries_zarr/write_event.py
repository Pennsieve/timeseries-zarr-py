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
    label_count,
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
from timeseries_zarr.protocols import (
    EventChannelSource,
    source_provenance,
    source_waveform_event_sample,
)
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

_LABEL_BLOCK_ROWS: Final = 2**20
"""Rows of labels the writer reads at a time to find the labels in use."""


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
    must be below source.num_labels(). Without label_names, num_labels() must
    be one more than the largest label.

    Args:
        group: Channel group to create the array in.
        source: Source of the labels.
        sizing: Chunk and shard shapes of the array.
        zstd_level: Zstandard compression level.

    Returns:
        The created array.

    Raises:
        ValueError: If num_labels() is above MAX_LABELS, which a uint16 cannot
            index, a label is at or above num_labels(), or the source has no
            label_names and num_labels() is not one more than the largest
            label.
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

    largest_label = -1

    def check(block: npt.NDArray[np.uint16]) -> None:
        nonlocal largest_label
        largest = int(block.max())
        if largest >= categories:
            raise ValueError(
                f"a label {largest} is outside the {categories} label categories"
            )
        largest_label = max(largest_label, largest)

    _write_source_blocks(
        array,
        source.num_events(),
        sizing.shard_shape[0],
        source.read_labels,
        check,
    )
    names = source.label_names()
    if names is None and source.num_events():
        expected = label_count(None, largest_label)
        if categories != expected:
            raise ValueError(
                f"num_labels() is {categories}, but the largest label "
                f"{largest_label} gives {expected} label categories"
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
    UTF-8 bytes of event i's text followed by zero bytes.

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
    channel_indices: Mapping[str, int],
    max_index: int,
    sizing: ChunkShard,
    zstd_level: int,
) -> ZarrArray:
    """Create the channels array under group and stream the event bitmasks in.

    A uint8 array named "channels" shaped (num_events, M), with M the byte
    width of a bitmask for indices up to max_index, and the dimension names "event"
    and None. Row i has the bit of each channel index that event i applies to,
    and a row of zeros names no channel. The source names channels by id.

    Args:
        group: Channel group to create the array in.
        source: Source of the channel ids of each event.
        channel_indices: Channel index in the bundle, by channel id.
        max_index: Largest channel index in the bundle.
        sizing: Chunk and shard shapes of the (num_events, M) shape, with the
            width axis whole.
        zstd_level: Zstandard compression level.

    Returns:
        The created array.

    Raises:
        ValueError: If max_index is negative, an id is not in channel_indices, a
            index is outside 0 to max_index, or the source returns a different
            number of id lists than requested.
    """
    width = channels_width(max_index)
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
        index_lists = []
        for ids in id_lists:
            for channel_id in ids:
                if channel_id not in channel_indices:
                    raise ValueError(f"unknown channel id {channel_id!r}")
            index_lists.append(
                [channel_indices[channel_id] for channel_id in ids]
            )
        return channels_rows(index_lists, max_index)

    _write_source_blocks(
        array, source.num_events(), sizing.shard_shape[0], read
    )
    return array


def _labels_in_use(
    source: EventChannelSource, categories: int
) -> npt.NDArray[np.bool_]:
    """Return whether each label category has an event, read from the source.

    Args:
        source: Source of the labels.
        categories: The number of label categories.

    Raises:
        ValueError: If a label is at or above categories.
    """
    in_use = np.zeros(categories, dtype=np.bool_)
    total = source.num_events()
    for start in range(0, total, _LABEL_BLOCK_ROWS):
        block = source.read_labels(start, min(start + _LABEL_BLOCK_ROWS, total))
        tally = np.bincount(block, minlength=categories)
        if tally.size > categories:
            raise ValueError(
                f"a label {tally.size - 1} is outside the {categories} "
                "label categories"
            )
        in_use |= tally > 0
    return in_use


def write_waveform_mean_array(
    group: ZarrGroup,
    source: EventChannelSource,
    zstd_level: int,
) -> ZarrArray:
    """Create the waveform_mean array under group and write the source's means.

    A float32 array named "waveform_mean" shaped (labels, samples), with the
    dimension names "label" and None, holding the mean snippet of each label
    category. It carries period_us, the microseconds between snippet samples,
    and event_sample, the snippet sample at the event time, when the source
    defines waveform_event_sample as WaveformEventSampleSource declares it and
    the index is above 0. A label that has no events has a row of NaN. The
    array has one row per label, so it is sized with the default chunk and
    shard targets and written whole.

    Args:
        group: Channel group to create the array in.
        source: Source of the means, the sample period, and the labels.
        zstd_level: Zstandard compression level.

    Returns:
        The created array.

    Raises:
        ValueError: If the means are not shaped (labels, samples), have no
            samples, or have a row count other than source.num_labels(), if
            a label that has no events has a mean that is not NaN, or if the
            period is not finite and positive or the event sample is outside
            the samples.
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
    in_use = _labels_in_use(source, labels)
    unused = ~in_use & ~np.isnan(means).all(axis=1)
    if unused.any():
        raise ValueError(
            f"label {int(np.flatnonzero(unused)[0])} has no events, so its "
            "waveform_mean row must be NaN"
        )
    event_sample = source_waveform_event_sample(source)
    attrs = waveform_array_attrs(
        source.waveform_period_us(), event_sample or None, samples=samples
    )
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


def _check_label_categories(source: EventChannelSource, total: int) -> None:
    """Raise ValueError if num_labels() contradicts what the metadata fixes.

    The metadata fixes the number of label categories when the source has
    label_names, and when it has no events. Otherwise the labels fix it, and
    write_labels_array checks it while it writes.

    Raises:
        ValueError: If label_names is empty or the number of categories
            differs from the number the metadata fixes.
    """
    names = source.label_names()
    if names is not None and not names:
        raise ValueError("label_names must name at least one category")
    if names is None and total:
        return
    expected = label_count(None if names is None else len(names), None)
    categories = source.num_labels()
    if categories != expected:
        basis = (
            "label_names" if names is not None else "a channel with no events"
        )
        raise ValueError(
            f"num_labels() is {categories}, but {basis} gives {expected} "
            "label categories"
        )


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
            lies outside (-2^53, 2^53) microseconds, label_names is empty, or
            num_labels() disagrees with label_names or with a channel that
            has no events.
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
    if "labels" in source.columns():
        _check_label_categories(source, total)
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
    index: int,
    source: EventChannelSource,
    *,
    reference_us: int,
    channel_indices: Mapping[str, int],
    max_index: int,
    opts: WriteOpts,
) -> None:
    """Write one event channel as the subgroup named str(index).

    The channel group carries the event attributes: kind, name, id, the unit,
    the source's provenance, and, when the source provides the column,
    label_names and the duration bound max_duration_us. The events array
    comes first, then each optional column the source names in columns(),
    then the count levels of a dense channel.

    Args:
        parent: Group to create the channel group in.
        index: Channel index, the name of the channel group.
        source: Source of the event channel.
        reference_us: Reference time in microseconds since the Unix epoch.
            Times are stored from it.
        channel_indices: Channel index in the bundle, by channel id.
        max_index: Largest channel index in the bundle, which sizes the channels
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
        str(index),
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
            channel_indices,
            max_index,
            _row_sizing(total, channels_width(max_index), UINT8_BYTES, opts),
            opts.zstd_level,
        )
    if "waveform_mean" in columns:
        write_waveform_mean_array(group, source, opts.zstd_level)
    write_count_levels(group, source, reference_us, opts)
