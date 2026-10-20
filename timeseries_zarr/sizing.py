"""Chunk and shard shape computation for Zarr pyramid arrays."""

from math import ceil, prod

from timeseries_zarr.constants import (
    ENVELOPE_PAIR_SIZE,
    FLOAT32_BYTES,
    INNER_CHUNK_SAMPLES,
    TARGET_SHARD_BYTES,
)
from timeseries_zarr.planning import raw_shape
from timeseries_zarr.types import ChunkShard


def chunk_shape_for_level(
    level_shape: tuple[int, ...], inner_len: int = INNER_CHUNK_SAMPLES
) -> tuple[int, ...]:
    """Return the inner Zarr chunk shape for a pyramid level.

    The chunk spans up to inner_len bins along the time axis (axis 0), never
    more than the level's own length and never fewer than one bin, so a
    zero-length level still gets a shape Zarr accepts. The trailing length-2
    (min, max) axis of an envelope level is never chunked.
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
    the level holds. A zero-length level takes one chunk. It is always an
    integer multiple of chunk_shape along axis 0; the trailing length-2
    (min, max) axis stays size 2.
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
    """Return the inner-chunk and outer-shard shapes for a level as a ChunkShard."""
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
