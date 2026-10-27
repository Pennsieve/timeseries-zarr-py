"""Chunk and shard shape computation for Zarr pyramid arrays."""

from math import ceil, prod

from timeseries_zarr.constants import (
    CHANNEL_APPEND_BYTES,
    ENVELOPE_PAIR_SIZE,
    FLOAT32_BYTES,
    INNER_CHUNK_SAMPLES,
    SHARD_INDEX_ENTRY_BYTES,
    STATISTIC_ROW_BYTES,
    TARGET_SHARD_BYTES,
)
from timeseries_zarr.planning import raw_shape
from timeseries_zarr.types import ChunkShard, WriteOpts

_LONG_ARRAY_ROWS = 2**62
"""A length that fills every shard, so a sizing has its full shard width."""


def chunk_shape_for_level(
    level_shape: tuple[int, ...], inner_len: int = INNER_CHUNK_SAMPLES
) -> tuple[int, ...]:
    """Return the inner Zarr chunk shape for a pyramid level.

    The chunk spans up to inner_len bins along the time axis (axis 0), never
    more than the level's own length and never fewer than one bin, so a
    zero-length level still gets a shape Zarr accepts. The trailing length-2
    (min, max) axis of an envelope level is never chunked.

    Args:
        level_shape: Shape of the level array.
        inner_len: Target chunk length in bins along axis 0.
    """
    n, *rest = level_shape
    return (max(1, min(n, inner_len)), *rest)


def shard_shape_for_level(
    chunk_shape: tuple[int, ...],
    level_shape: tuple[int, ...],
    dtype_size: int,
    target_shard_bytes: int = TARGET_SHARD_BYTES,
) -> tuple[int, ...]:
    """Return the outer Zarr shard shape for a pyramid level.

    The shard spans as many whole inner chunks along the time axis (axis 0) as
    fit in target_shard_bytes, at least one and at most the number of chunks
    the level holds. A zero-length level takes one chunk. The shard is an
    integer multiple of chunk_shape along axis 0, and the trailing length-2
    (min, max) axis stays size 2.

    Args:
        chunk_shape: Inner chunk shape of the level.
        level_shape: Shape of the level array.
        dtype_size: Bytes per element.
        target_shard_bytes: Target shard size in bytes.
    """
    chunk_bytes = prod(chunk_shape) * dtype_size
    chunks_per_shard = max(1, target_shard_bytes // chunk_bytes)
    chunks_in_level = ceil(level_shape[0] / chunk_shape[0])
    shard_chunks = max(1, min(chunks_per_shard, chunks_in_level))
    return (shard_chunks * chunk_shape[0], *chunk_shape[1:])


def chunk_and_shard(
    level_shape: tuple[int, ...],
    dtype_size: int,
    inner_len: int = INNER_CHUNK_SAMPLES,
    target_shard_bytes: int = TARGET_SHARD_BYTES,
) -> ChunkShard:
    """Return the inner-chunk and outer-shard shapes for a level."""
    chunk_shape = chunk_shape_for_level(level_shape, inner_len)
    shard_shape = shard_shape_for_level(
        chunk_shape, level_shape, dtype_size, target_shard_bytes
    )

    return ChunkShard(chunk_shape=chunk_shape, shard_shape=shard_shape)


def raw_sizing(
    num_samples: int,
    inner_len: int = INNER_CHUNK_SAMPLES,
    target_shard_bytes: int = TARGET_SHARD_BYTES,
) -> ChunkShard:
    """Return the chunk and shard shapes of a channel's raw array.

    raw has shape (num_samples,) and holds float32. A channel with no samples
    takes one-row chunks and shards.
    """
    return chunk_and_shard(
        raw_shape(num_samples), FLOAT32_BYTES, inner_len, target_shard_bytes
    )


def level_sizing(
    bins: int,
    inner_len: int = INNER_CHUNK_SAMPLES,
    target_shard_bytes: int = TARGET_SHARD_BYTES,
) -> ChunkShard:
    """Return the chunk and shard shapes of a level's minmax array.

    minmax has shape (bins, 2) and is the widest member of a level, so its row
    counts size every member: the others take them through member_sizing, and
    one shard of folded rows then fills one shard of each. The trailing pair
    axis is never chunked. A level with no bins takes one-row chunks and
    shards.
    """
    return chunk_and_shard(
        (bins, ENVELOPE_PAIR_SIZE),
        FLOAT32_BYTES,
        inner_len,
        target_shard_bytes,
    )


def member_sizing(sizing: ChunkShard) -> ChunkShard:
    """Return the sizing of a one-dimensional level member.

    Keeps the first axis of sizing and drops the rest, so mean, var, and
    count share the row counts of minmax. A one-dimensional sizing comes back
    equal.
    """
    return ChunkShard(
        chunk_shape=sizing.chunk_shape[:1], shard_shape=sizing.shard_shape[:1]
    )


def _index_bytes(sizing: ChunkShard) -> int:
    """Return the bytes of the index of one open shard of an array."""
    chunks = sizing.shard_shape[0] // sizing.chunk_shape[0]
    return chunks * SHARD_INDEX_ENTRY_BYTES


def channel_writer_bytes(opts: WriteOpts) -> int:
    """Return the bytes of memory that the writer of one channel holds.

    The estimate covers the state that lives as long as the channel: the
    partial chunk of raw with its shard index, for each of opts.max_levels
    levels the pending statistics of one chunk with the shard index of
    minmax, and CHANNEL_APPEND_BYTES of encoded chunks that wait to be
    written. It does not depend on the length of the channel.
    The window and the output of the chunks in flight are not part of it.

    Args:
        opts: The write settings.

    Returns:
        The estimate in bytes.
    """
    raw = raw_sizing(_LONG_ARRAY_ROWS, opts.inner_len, opts.target_shard_bytes)
    level = level_sizing(
        _LONG_ARRAY_ROWS, opts.inner_len, opts.target_shard_bytes
    )
    per_level = opts.inner_len * STATISTIC_ROW_BYTES + _index_bytes(level)
    return (
        opts.inner_len * FLOAT32_BYTES
        + _index_bytes(raw)
        + opts.max_levels * per_level
        + CHANNEL_APPEND_BYTES
    )
