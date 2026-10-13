"""Bounded-memory streaming generators that drive the folds over a source."""

from collections.abc import Callable, Iterable, Iterator
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from timeseries_zarr.constants import DECIMATION_FACTOR
from timeseries_zarr.fold import assemble_statistics
from timeseries_zarr.protocols import ContinuousChannelSource
from timeseries_zarr.types import LevelMembers


class BlockReadableArray(Protocol):
    """Minimal structural view of an on-disk array iterated along axis 0.

    Any object with a shape and axis-0 slicing satisfies this: a zarr Array, a
    numpy array.
    """

    @property
    def shape(self) -> tuple[int, ...]:
        """Array shape; axis 0 is the iterated sample/bin axis."""
        ...

    def __getitem__(self, item: slice) -> npt.NDArray[Any]:
        """Return the rows in the given axis-0 slice."""
        ...


def _rebuffer_and_fold[TIn: np.generic, TOut: np.generic](
    blocks: Iterable[npt.NDArray[TIn]],
    fold_fn: Callable[[npt.NDArray[TIn]], npt.NDArray[TOut]],
    group: int = DECIMATION_FACTOR,
) -> Iterator[npt.NDArray[TOut]]:
    """Fold a stream of blocks into the next coarser level.

    Blocks are sliced and joined along axis 0 only, whatever their rank and
    dtype. The concatenation of the yielded arrays equals fold_fn applied to
    the whole concatenated input, computed in bounded memory: at most group-1
    rows are carried across a block boundary.
    """
    carry: npt.NDArray[TIn] | None = None
    for block in blocks:
        # An exhausted carry still concatenates to the block itself, so skip the
        # copy: with shard-aligned inputs that is every iteration but the last.
        buffer = (
            block
            if carry is None or not carry.shape[0]
            else np.concatenate([carry, block], axis=0)
        )
        n_full = buffer.shape[0] // group
        split = n_full * group
        if n_full:
            yield fold_fn(buffer[:split])
        carry = buffer[split:]
    if carry is not None and carry.shape[0]:
        yield fold_fn(carry)


def iter_raw_blocks(
    source: ContinuousChannelSource,
    block_samples: int,
) -> Iterator[npt.NDArray[np.float32]]:
    """Yield successive read_samples windows covering the source's raw samples.

    Walks [0, source.num_samples()) in block_samples-sized windows; the final
    window may be shorter. Yields nothing when the source has no samples.
    Raises ValueError if block_samples is not positive.
    """
    if block_samples <= 0:
        raise ValueError("block_samples must be positive")
    n = source.num_samples()
    for start in range(0, n, block_samples):
        yield source.read_samples(start, min(start + block_samples, n))


def iter_array_blocks(
    array: BlockReadableArray, block_len: int
) -> Iterator[npt.NDArray[Any]]:
    """Yield successive axis-0 windows of an on-disk array.

    Walks [0, array.shape[0]) in block_len-sized windows; the final window may
    be shorter (keep-tail). Works for rank-1 raw and rank-2 (min, max) arrays
    alike. Yields nothing when the array is empty. Raises ValueError if
    block_len is not positive.
    """
    if block_len <= 0:
        raise ValueError("block_len must be positive")
    n = array.shape[0]
    for start in range(0, n, block_len):
        stop = min(n, start + block_len)
        yield array[start:stop]


def regroup[T: np.generic](
    blocks: Iterable[npt.NDArray[T]], rows: int
) -> Iterator[npt.NDArray[T]]:
    """Re-cut a stream of blocks into blocks of exactly rows rows.

    Blocks of any size and rank join across boundaries along axis 0, so the
    concatenation of the output equals the concatenation of the input. Every
    yielded block has rows rows except the last, which holds the remainder and
    is never empty. A block with no rows is skipped, and an empty stream
    yields nothing. Fewer than rows rows carry from one input block to the
    next. Raises ValueError if rows is not positive.
    """
    if rows <= 0:
        raise ValueError("rows must be positive")
    pending: list[npt.NDArray[T]] = []
    pending_rows = 0
    for block in blocks:
        if not block.shape[0]:
            continue
        pending.append(block)
        pending_rows += block.shape[0]
        if pending_rows < rows:
            continue
        merged = (
            pending[0] if len(pending) == 1 else np.concatenate(pending, axis=0)
        )
        whole = merged.shape[0] // rows * rows
        # Copy the carry before yielding, so it neither keeps the merged block
        # alive nor changes if the caller reuses an input array between pulls.
        remainder = merged[whole:].copy()
        for start in range(0, whole, rows):
            yield merged[start : start + rows]
        pending = [remainder] if remainder.shape[0] else []
        pending_rows = remainder.shape[0]
    if pending_rows:
        yield (
            pending[0] if len(pending) == 1 else np.concatenate(pending, axis=0)
        )


def tap[T: np.generic](
    blocks: Iterable[npt.NDArray[T]],
    sink: Callable[[npt.NDArray[T]], None],
) -> Iterator[npt.NDArray[T]]:
    """Yield each block unchanged after passing it to sink.

    sink runs once per block, in order, when the consumer asks for the block
    and before the block is yielded. An exception from sink reaches the
    consumer. An empty stream never calls sink.
    """
    for block in blocks:
        sink(block)
        yield block


def iter_member_blocks(
    members: LevelMembers[BlockReadableArray], block_len: int
) -> Iterator[npt.NDArray[np.float64]]:
    """Yield statistics blocks read back from a level's member arrays.

    Walks the level in block_len-row windows, the last possibly shorter, and
    assembles each window from minmax, mean, count, and var when present with
    assemble_statistics. A level with no rows yields nothing. Raises
    ValueError if block_len is not positive or if the members do not all have
    the same number of rows.
    """
    if block_len <= 0:
        raise ValueError("block_len must be positive")
    rows = members.minmax.shape[0]
    others = (
        ("mean", members.mean),
        ("var", members.var),
        ("count", members.count),
    )
    for name, member in others:
        if member is not None and member.shape[0] != rows:
            raise ValueError(
                f"{name} has {member.shape[0]} rows but minmax has {rows}"
            )
    for start in range(0, rows, block_len):
        stop = min(start + block_len, rows)
        yield assemble_statistics(
            members.minmax[start:stop],
            members.mean[start:stop],
            None if members.var is None else members.var[start:stop],
            members.count[start:stop],
        )
