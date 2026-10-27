"""Write continuous channels in one pass.

A channel is the raw samples plus a ladder of level groups above them. A
level group holds one array per statistic over a shared bin axis. An empty
input creates the arrays and writes nothing.

ChannelWriter writes one channel in one pass. Each block of samples goes to
the raw array and through a LevelFolder, and every chunk that the folder
emits goes to its member array. Nothing is read back from the store.
write_continuous_channel writes one source, and write_continuous_channels
writes every channel of a multi-channel source in one sweep over its windows.
"""

from collections.abc import Sequence
from concurrent.futures import Executor, Future, ThreadPoolExecutor, wait
from contextlib import nullcontext
from typing import cast

import numpy as np
import numpy.typing as npt

from timeseries_zarr.attrs import continuous_channel_attrs, level_group_attrs
from timeseries_zarr.constants import (
    COUNT_KEY,
    ENVELOPE_PAIR_SIZE,
    MEAN_KEY,
    MINMAX_KEY,
    RAW_KEY,
    TIME_DIMENSION,
    VAR_KEY,
)
from timeseries_zarr.fold import LevelChunk, LevelFolder
from timeseries_zarr.planning import plan_levels, raw_shape
from timeseries_zarr.protocols import (
    ContinuousChannelMetadata,
    ContinuousChannelSource,
    MultiChannelSource,
    source_provenance,
)
from timeseries_zarr.shards import ArrayWriter, ShardLayout
from timeseries_zarr.sizing import level_sizing, member_sizing, raw_sizing
from timeseries_zarr.streaming import iter_raw_blocks, scale_block
from timeseries_zarr.types import (
    ChunkShard,
    LevelMembers,
    LevelPlan,
    ResolvedUnit,
    WriteOpts,
)
from timeseries_zarr.units import resolve_unit
from timeseries_zarr.zarr_io import (
    NodeMetadata,
    ZarrArray,
    ZarrGroup,
    array_metadata,
    create_nodes,
    group_metadata,
    shard_layout,
)


def _channel_group_metadata(
    channel: ContinuousChannelMetadata,
    *,
    reference_us: int,
    unit: str,
) -> NodeMetadata:
    """Return the metadata of the channel group, as create_channel_group stores it."""
    return group_metadata(
        continuous_channel_attrs(
            channel.name,
            channel.id,
            channel.rate_hz(),
            channel.start_us() - reference_us,
            unit,
            source_provenance(channel),
        )
    )


def create_channel_group(
    parent: ZarrGroup,
    key: int,
    channel: ContinuousChannelMetadata,
    *,
    reference_us: int,
    unit: str,
) -> ZarrGroup:
    """Create the channel group named str(key) with the continuous attributes.

    The group stores unit as given, whatever unit the channel declares. Its
    offset_us is the distance in microseconds from reference_us to the first
    sample, negative for a channel that begins before it. The group carries
    the channel's provenance when the channel has one.

    Args:
        parent: The group that receives the channel group.
        key: The channel number that names the group.
        channel: The channel metadata.
        reference_us: The reference time in microseconds.
        unit: The unit string to store.

    Returns:
        The new channel group.

    Raises:
        ValueError: If the rate is not positive or the provenance is not a
            JSON object of finite values.
    """
    name = str(key)
    metadata = _channel_group_metadata(
        channel, reference_us=reference_us, unit=unit
    )
    return cast("ZarrGroup", create_nodes(parent, {name: metadata})[name])


def _raw_metadata(
    num_samples: int, sizing: ChunkShard, zstd_level: int
) -> NodeMetadata:
    """Return the metadata of the raw array, as create_raw stores it."""
    return array_metadata(
        raw_shape(num_samples),
        np.float32,
        sizing.chunk_shape,
        sizing.shard_shape,
        zstd_level,
        {},
        dimension_names=(TIME_DIMENSION,),
    )


def create_raw(
    group: ZarrGroup,
    num_samples: int,
    sizing: ChunkShard,
    zstd_level: int,
) -> ZarrArray:
    """Create the raw array under group, with every sample unwritten.

    The array is float32, named "raw", and shaped (num_samples,). Its fill
    value is NaN, its axis is named "time", and it has no attributes. The
    channel group carries the sample rate. A num_samples of 0 gives a
    zero-length array.

    Args:
        group: The channel group.
        num_samples: The length of the array.
        sizing: The chunk and shard shapes.
        zstd_level: The Zstandard compression level.

    Returns:
        The new array.
    """
    metadata = _raw_metadata(num_samples, sizing, zstd_level)
    return cast("ZarrArray", create_nodes(group, {RAW_KEY: metadata})[RAW_KEY])


def _level_metadata(
    plan: LevelPlan,
    sizing: ChunkShard,
    zstd_level: int,
    write_var: bool,
    prefix: str = "",
) -> dict[str, NodeMetadata]:
    """Return the metadata of a level group and its members, as create_level stores them.

    Each key is the path of the node below the parent group, with prefix
    before the level name.
    """
    member_rows = member_sizing(sizing)
    base = f"{prefix}{plan.name}"
    bins_shape = (plan.bins,)
    time_only = (TIME_DIMENSION,)

    def member(
        key: str,
        dtype: type[np.generic],
        shape: tuple[int, ...],
        array_sizing: ChunkShard,
        dimension_names: tuple[str | None, ...],
    ) -> NodeMetadata:
        return array_metadata(
            shape,
            dtype,
            array_sizing.chunk_shape,
            array_sizing.shard_shape,
            zstd_level,
            {},
            dimension_names=dimension_names,
        )

    nodes: dict[str, NodeMetadata] = {
        base: group_metadata(level_group_attrs(plan.period_us)),
        f"{base}/{MINMAX_KEY}": member(
            MINMAX_KEY,
            np.float32,
            (plan.bins, ENVELOPE_PAIR_SIZE),
            sizing,
            (TIME_DIMENSION, None),
        ),
        f"{base}/{MEAN_KEY}": member(
            MEAN_KEY, np.float32, bins_shape, member_rows, time_only
        ),
        f"{base}/{COUNT_KEY}": member(
            COUNT_KEY, np.uint32, bins_shape, member_rows, time_only
        ),
    }
    if write_var:
        nodes[f"{base}/{VAR_KEY}"] = member(
            VAR_KEY, np.float32, bins_shape, member_rows, time_only
        )
    return nodes


def _level_members(
    handles: dict[str, ZarrGroup | ZarrArray],
    plan: LevelPlan,
    write_var: bool,
    prefix: str = "",
) -> LevelMembers[ZarrArray]:
    """Return the member handles of a level that _level_metadata described."""
    base = f"{prefix}{plan.name}"

    def member(key: str) -> ZarrArray:
        return cast("ZarrArray", handles[f"{base}/{key}"])

    return LevelMembers(
        minmax=member(MINMAX_KEY),
        mean=member(MEAN_KEY),
        var=member(VAR_KEY) if write_var else None,
        count=member(COUNT_KEY),
    )


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
    and the other members take its row counts. Every member names its bin
    axis "time".

    Args:
        parent: The channel group.
        plan: The size and period of the level.
        sizing: The chunk and shard shapes of minmax.
        zstd_level: The Zstandard compression level.
        write_var: Whether to create the var member.

    Returns:
        The handles of the new member arrays.

    Raises:
        ValueError: If plan.level is below 1.
    """
    if plan.level < 1:
        raise ValueError("levels are numbered from 1; raw is not a level")
    handles = create_nodes(
        parent, _level_metadata(plan, sizing, zstd_level, write_var)
    )
    return _level_members(handles, plan, write_var)


class _LevelWriters:
    """The member writers of one level."""

    def __init__(
        self, layouts: LevelMembers[ShardLayout], executor: Executor | None
    ) -> None:
        """Start a writer for each member that layouts holds."""
        self._minmax = ArrayWriter(layouts.minmax, executor)
        self._mean = ArrayWriter(layouts.mean, executor)
        self._var = (
            None if layouts.var is None else ArrayWriter(layouts.var, executor)
        )
        self._count = ArrayWriter(layouts.count, executor)

    def push(self, chunk: LevelChunk) -> None:
        """Write the rows of chunk to each member."""
        self._minmax.push(chunk.minmax)
        self._mean.push(chunk.mean)
        if self._var is not None and chunk.var is not None:
            self._var.push(chunk.var)
        self._count.push(chunk.count)

    def finish(self) -> None:
        """Close the last shard of each member."""
        self._minmax.finish()
        self._mean.finish()
        if self._var is not None:
            self._var.finish()
        self._count.finish()


class ChannelWriter:
    """Writes the raw array and every level of one channel as samples arrive.

    Each push scales the block to the stored unit, appends it to raw, folds it
    into every level, and appends each chunk the folder emits to its member
    array. The writer holds one partial chunk of each array and the pending
    rows of each level. Its memory does not grow with the length of the
    channel. Every shard is written once, as it fills. The bytes written do not
    depend on the executor.
    """

    def __init__(
        self,
        key: int,
        channel: ContinuousChannelMetadata,
        raw: ShardLayout,
        levels: Sequence[LevelMembers[ShardLayout]],
        factor: float,
        inner_len: int,
        executor: Executor | None = None,
    ) -> None:
        """Start a channel.

        Args:
            key: The channel number.
            channel: The channel metadata.
            raw: The layout of the raw array.
            levels: The layout of each level's members, in level order.
            factor: The multiplier applied to every sample.
            inner_len: The rows of one chunk of a level.
            executor: Encodes the chunks of a large push in groups. A worker
                of the executor may call push while another worker is idle.
                Chunks encode in the calling thread when None.
        """
        self._key = key
        self._channel = channel
        self._factor = factor
        self._num_samples = channel.num_samples()
        self._received = 0
        self._raw = ArrayWriter(raw, executor)
        self._folder = (
            LevelFolder(
                len(levels),
                inner_len,
                bool(levels) and levels[0].var is not None,
            )
            if levels
            else None
        )
        self._levels = [_LevelWriters(layouts, executor) for layouts in levels]

    @property
    def label(self) -> str:
        """The channel key and id, as an error message names the channel."""
        return f"channel {self._key} ({self._channel.id!r})"

    def push(self, block: npt.ArrayLike) -> None:
        """Write the samples of block and fold them into every level.

        The samples are cast to float32 and multiplied by the factor.

        Args:
            block: The next samples of the channel.

        Raises:
            ValueError: If block is not rank 1 or if the channel has received
                more samples than it declares. Nothing from block is written
                first.
        """
        samples = np.asarray(block, dtype=np.float32)
        if samples.ndim != 1:
            raise ValueError(
                f"{self.label} gives a block of rank {samples.ndim}; "
                "each block must be a rank-1 array"
            )
        self._received += samples.shape[0]
        if self._received > self._num_samples:
            raise ValueError(
                f"{self.label} yields at least {self._received} samples "
                f"and declares {self._num_samples}; its blocks must total "
                f"{self._num_samples}"
            )
        if not samples.shape[0]:
            return
        scaled = np.ascontiguousarray(scale_block(samples, self._factor))
        self._raw.push(scaled)
        if self._folder is not None:
            self._route(self._folder.push(scaled))

    def finish(self) -> None:
        """Fold the tails and close the last shard of every array.

        Raises:
            ValueError: If the channel received fewer samples than it
                declares.
        """
        if self._received != self._num_samples:
            raise ValueError(
                f"{self.label} yields {self._received} samples and "
                f"declares {self._num_samples}; its blocks must total "
                f"{self._num_samples}"
            )
        self._raw.finish()
        if self._folder is not None:
            self._route(self._folder.finish())
        for writers in self._levels:
            writers.finish()

    def _route(self, chunks: Sequence[LevelChunk]) -> None:
        """Write each chunk to the members of its level."""
        for chunk in chunks:
            self._levels[chunk.level - 1].push(chunk)


def _add_channel_nodes(
    nodes: dict[str, NodeMetadata],
    key: int,
    channel: ContinuousChannelMetadata,
    resolved: ResolvedUnit,
    reference_us: int,
    opts: WriteOpts,
) -> list[LevelPlan]:
    """Add the nodes of one channel to nodes and return its level plans.

    The nodes are the channel group, its raw array, and every level group
    with its members, keyed by their path below the parent of the channel.
    """
    num_samples = channel.num_samples()
    plans = plan_levels(
        num_samples, channel.rate_hz(), opts.max_levels, opts.min_bins
    )
    nodes[str(key)] = _channel_group_metadata(
        channel, reference_us=reference_us, unit=resolved.unit
    )
    nodes[f"{key}/{RAW_KEY}"] = _raw_metadata(
        num_samples,
        raw_sizing(num_samples, opts.inner_len, opts.target_shard_bytes),
        opts.zstd_level,
    )
    for plan in plans:
        nodes.update(
            _level_metadata(
                plan,
                level_sizing(
                    plan.bins, opts.inner_len, opts.target_shard_bytes
                ),
                opts.zstd_level,
                opts.write_var,
                prefix=f"{key}/",
            )
        )
    return plans


def _channel_writer(
    handles: dict[str, ZarrGroup | ZarrArray],
    key: int,
    channel: ContinuousChannelMetadata,
    plans: Sequence[LevelPlan],
    factor: float,
    opts: WriteOpts,
    executor: Executor | None,
) -> ChannelWriter:
    """Return the writer of a channel whose nodes handles holds."""
    levels = []
    for plan in plans:
        members = _level_members(handles, plan, opts.write_var, f"{key}/")
        levels.append(
            LevelMembers(
                minmax=shard_layout(members.minmax),
                mean=shard_layout(members.mean),
                var=None if members.var is None else shard_layout(members.var),
                count=shard_layout(members.count),
            )
        )
    return ChannelWriter(
        key,
        channel,
        shard_layout(cast("ZarrArray", handles[f"{key}/{RAW_KEY}"])),
        levels,
        factor,
        opts.inner_len,
        executor,
    )


def _pool(workers: int) -> ThreadPoolExecutor | nullcontext[None]:
    """Return a context that yields a pool of workers threads, or None for one."""
    if workers == 1:
        return nullcontext(None)
    return ThreadPoolExecutor(workers, thread_name_prefix="timeseries-zarr")


def _wait_for(jobs: Sequence[tuple[ChannelWriter, Future[None]]]) -> None:
    """Wait for every job, then raise the error of the first that failed.

    The error gains a note with the label of the job's channel.
    """
    wait([future for _, future in jobs])
    for writer, future in jobs:
        error = future.exception()
        if error is not None:
            error.add_note(f"while writing {writer.label}")
            raise error


def write_continuous_channel(
    parent: ZarrGroup,
    key: int,
    source: ContinuousChannelSource,
    *,
    reference_us: int,
    opts: WriteOpts,
) -> None:
    """Write one continuous channel as the subgroup named str(key).

    A voltage unit resolves under opts.voltage_policy before the group is
    created, and the raw samples are scaled to match. The source is read in
    blocks of one raw shard, and the chunks of each block encode in groups on
    opts.max_workers threads. A channel too short for level 1 gets raw and
    nothing else.

    Args:
        parent: The group that receives the channel group.
        key: The channel number that names the group.
        source: The source of the channel.
        reference_us: The reference time in microseconds.
        opts: The write settings.

    Raises:
        ValueError: If the source's rate is not positive, if its provenance
            is not a JSON object of finite values, if a read returns fewer
            samples than the source declares, or, under voltage_policy
            require_uv, if a voltage unit is not "uV".
    """
    resolved = resolve_unit(source.unit, opts.voltage_policy)
    nodes: dict[str, NodeMetadata] = {}
    plans = _add_channel_nodes(nodes, key, source, resolved, reference_us, opts)
    handles = create_nodes(parent, nodes)
    block_samples = raw_sizing(
        source.num_samples(), opts.inner_len, opts.target_shard_bytes
    ).shard_shape[0]
    workers = opts.worker_count
    with _pool(workers) as pool:
        writer = _channel_writer(
            handles, key, source, plans, resolved.factor, opts, pool
        )
        for block in iter_raw_blocks(source, block_samples):
            writer.push(block)
        writer.finish()


def write_continuous_channels(
    parent: ZarrGroup,
    first_key: int,
    source: MultiChannelSource,
    *,
    reference_us: int,
    opts: WriteOpts,
) -> None:
    """Write the channels of source as the subgroups named from str(first_key).

    The channel groups, the raw arrays, and the level groups and members of
    every channel are created in one call, with the attributes and shapes that
    write_continuous_channel gives them. One pass over the windows of source
    writes the raw array and every level of every channel. Each window
    dispatches one job per channel to opts.max_workers threads, and the next
    window is read while the jobs run. A source must not overwrite the arrays
    of a window it has yielded. When there are fewer channels than threads,
    each job also encodes its chunks in groups on the idle threads. Memory
    holds one partial chunk per array, the pending rows of each level, and two
    windows. A source without channels writes nothing, and iter_windows is not
    called.

    Args:
        parent: The group that receives the channel groups.
        first_key: The channel number of the first channel.
        source: The multi-channel source.
        reference_us: The reference time in microseconds.
        opts: The write settings.

    Raises:
        ValueError: If a rate is not positive, if provenance is not a JSON
            object of finite values, if a voltage unit is not "uV" under
            voltage_policy require_uv, if a window does not hold one block
            per channel, if a block is not rank 1, or if a channel's blocks
            do not total its num_samples. A unit or provenance error comes
            before any node is created. A window or block error leaves the
            created nodes and the shards written so far in place. Any error
            from a channel's job gains a note with the channel label, and
            the first channel in channel order wins when several fail in one
            window.
    """
    channels = list(source.channels())
    if not channels:
        return
    nodes: dict[str, NodeMetadata] = {}
    plans_by_key: dict[int, list[LevelPlan]] = {}
    factors: dict[int, float] = {}
    for key, channel in enumerate(channels, start=first_key):
        resolved = resolve_unit(channel.unit, opts.voltage_policy)
        plans_by_key[key] = _add_channel_nodes(
            nodes, key, channel, resolved, reference_us, opts
        )
        factors[key] = resolved.factor
    handles = create_nodes(parent, nodes)
    workers = opts.worker_count
    pool = ThreadPoolExecutor(workers, thread_name_prefix="timeseries-zarr")
    try:
        # A job that waits on the pool for chunk groups needs an idle worker.
        encoder = pool if len(channels) < workers else None
        writers = [
            _channel_writer(
                handles,
                key,
                channel,
                plans_by_key[key],
                factors[key],
                opts,
                encoder,
            )
            for key, channel in enumerate(channels, start=first_key)
        ]
        jobs: list[tuple[ChannelWriter, Future[None]]] = []
        for index, window in enumerate(source.iter_windows()):
            _wait_for(jobs)
            if len(window) != len(writers):
                raise ValueError(
                    f"window {index} holds {len(window)} blocks and the "
                    f"source has {len(writers)} channels; a window must hold "
                    "one block for each channel"
                )
            jobs = [
                (writer, pool.submit(writer.push, block))
                for writer, block in zip(writers, window, strict=True)
            ]
        _wait_for(jobs)
        _wait_for([(writer, pool.submit(writer.finish)) for writer in writers])
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
