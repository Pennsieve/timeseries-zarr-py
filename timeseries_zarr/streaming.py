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

    A zarr Array and a numpy array satisfy this protocol.
    """

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the array. Axis 0 is the sample or bin axis."""
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

    Args:
        blocks: The blocks to fold, in order.
        fold_fn: The fold applied to each run of whole groups.
        group: The number of rows that fold into one.

    Yields:
        The fold of each run of whole groups, then the fold of the tail.
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
    """Yield successive read_samples windows covering the raw samples.

    Walks [0, source.num_samples()) in block_samples-sized windows; the final
    window may be shorter. Yields nothing when the source has no samples.

    Args:
        source: The source to read.
        block_samples: The number of samples in each window.

    Raises:
        ValueError: If block_samples is not positive.
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
    alike. Yields nothing when the array is empty.

    Args:
        array: The array to read.
        block_len: The number of rows in each window.

    Raises:
        ValueError: If block_len is not positive.
    """
    if block_len <= 0:
        raise ValueError("block_len must be positive")
    n = array.shape[0]
    for start in range(0, n, block_len):
        stop = min(n, start + block_len)
        yield array[start:stop]


class Regrouper[T: np.generic]:
    """Cut blocks of any length into blocks with a fixed number of rows.

    Blocks join along axis 0 whatever their rank, and rows keep their order. A
    push returns the complete blocks that the new rows finish. Fewer rows than
    one block wait for a later push, and flush returns them at the end of the
    stream.
    """

    def __init__(self, rows: int) -> None:
        """Create a regrouper that returns blocks of the given number of rows.

        Args:
            rows: The number of rows in each block.

        Raises:
            ValueError: If rows is not positive.
        """
        if rows <= 0:
            raise ValueError("rows must be positive")
        self._rows = rows
        self._carry: npt.NDArray[T] | None = None
        self._filled = 0

    def push(self, block: npt.NDArray[T]) -> list[npt.NDArray[T]]:
        """Return the whole blocks that the pushed rows complete, in order.

        A block with no rows adds nothing. A block cut from the pushed rows
        alone is a view of the pushed array. Rows that wait for a later push
        are copied.

        Args:
            block: The rows to add. Trailing dimensions must match earlier
                pushes.
        """
        rows = self._rows
        count = block.shape[0]
        complete: list[npt.NDArray[T]] = []
        position = 0
        carry = self._carry
        if carry is not None:
            position = min(count, rows - self._filled)
            carry[self._filled : self._filled + position] = block[:position]
            self._filled += position
            if self._filled == rows:
                complete.append(carry)
                self._carry = None
                self._filled = 0
        while count - position >= rows:
            complete.append(block[position : position + rows])
            position += rows
        if position < count:
            self._carry = np.empty((rows, *block.shape[1:]), dtype=block.dtype)
            self._filled = count - position
            self._carry[: self._filled] = block[position:]
        return complete

    def flush(self) -> npt.NDArray[T] | None:
        """Return the rows left over, or None when no rows wait.

        The returned block is never empty. The regrouper holds no rows
        afterward.
        """
        if self._carry is None:
            return None
        tail = self._carry[: self._filled]
        self._carry = None
        self._filled = 0
        return tail


def regroup[T: np.generic](
    blocks: Iterable[npt.NDArray[T]], rows: int
) -> Iterator[npt.NDArray[T]]:
    """Re-cut a stream of blocks into blocks of exactly rows rows.

    Blocks of any size and rank join across boundaries along axis 0, so the
    concatenation of the output equals the concatenation of the input. Every
    yielded block has rows rows except the last, which holds the remainder and
    is never empty. A block with no rows is skipped, and an empty stream
    yields nothing. Fewer than rows rows carry from one input block to the
    next.

    Args:
        blocks: The blocks to re-cut.
        rows: The number of rows in each yielded block.

    Yields:
        Blocks of rows rows, then a final shorter block when rows remain.

    Raises:
        ValueError: If rows is not positive.
    """
    regrouper: Regrouper[T] = Regrouper(rows)
    for block in blocks:
        yield from regrouper.push(block)
    tail = regrouper.flush()
    if tail is not None:
        yield tail


def tap[T: np.generic](
    blocks: Iterable[npt.NDArray[T]],
    sink: Callable[[npt.NDArray[T]], None],
) -> Iterator[npt.NDArray[T]]:
    """Yield each block unchanged after passing it to sink.

    sink runs once per block, in order, when the consumer asks for the block
    and before the block is yielded. An exception from sink reaches the
    consumer. An empty stream never calls sink.

    Args:
        blocks: The blocks to pass through.
        sink: The function that receives each block.
    """
    for block in blocks:
        sink(block)
        yield block


def scale_block(
    block: npt.NDArray[np.float32], factor: float
) -> npt.NDArray[np.float32]:
    """Return block multiplied by factor, in float32.

    A factor of 1.0 returns block itself. NaN and infinity stay non-finite.

    Args:
        block: The samples to scale.
        factor: The multiplier.
    """
    if factor == 1.0:
        return block
    return block * np.float32(factor)


def scale(
    blocks: Iterable[npt.NDArray[np.float32]], factor: float
) -> Iterator[npt.NDArray[np.float32]]:
    """Yield each block multiplied by factor, in float32.

    A factor of 1.0 yields the input blocks themselves. NaN and infinity stay
    non-finite.

    Args:
        blocks: The blocks to scale.
        factor: The multiplier.
    """
    for block in blocks:
        yield scale_block(block, factor)


def iter_member_blocks(
    members: LevelMembers[BlockReadableArray], block_len: int
) -> Iterator[npt.NDArray[np.float64]]:
    """Yield statistics blocks read back from a level's member arrays.

    Walks the level in block_len-row windows, the last possibly shorter, and
    assembles each window from minmax, mean, count, and var when present with
    assemble_statistics. A level with no rows yields nothing.

    Args:
        members: The member arrays of the level.
        block_len: The number of rows in each window.

    Raises:
        ValueError: If block_len is not positive or if the members do not all
            have the same number of rows.
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
