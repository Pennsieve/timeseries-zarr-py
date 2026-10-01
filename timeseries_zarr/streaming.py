"""Bounded-memory streaming generators that drive the folds over a source."""

from collections.abc import Callable, Iterable, Iterator
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from timeseries_zarr.constants import (
    COUNT_COL,
    DECIMATION_FACTOR,
    MAX_COL,
    MEAN_COL,
    MIN_COL,
    STAT_COLUMNS,
    VALID_COL,
)
from timeseries_zarr.fold import fold_raw_block
from timeseries_zarr.planning import bin_counts
from timeseries_zarr.protocols import ContinuousChannelSource


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

    Blocks are raw runs, stat rows or count rows; only their axis-0 length
    matters here. The concatenation of the yielded arrays equals fold_fn
    applied to the whole concatenated input,
    computed in bounded memory: at most group-1 rows are carried across a block
    boundary.
    """
    carry: npt.NDArray[Any] | None = None
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


def iter_raw_to_level1(
    source: ContinuousChannelSource, block_samples: int
) -> Iterator[npt.NDArray[np.float64]]:
    """Yield level-1 stat rows folded from the source's raw samples.

    One row per 4 raw samples, keep-tail. The concatenated output equals
    folding the whole series at once, whatever block_samples is. Raises
    ValueError if block_samples is not positive.
    """
    if block_samples <= 0:
        raise ValueError("block_samples must be positive")
    yield from _rebuffer_and_fold(
        iter_raw_blocks(source, block_samples), fold_raw_block
    )


def iter_array_blocks(
    array: BlockReadableArray, block_len: int
) -> Iterator[npt.NDArray[Any]]:
    """Yield successive axis-0 windows of an on-disk array.

    Walks [0, array.shape[0]) in block_len-sized windows; the final window may
    be shorter (keep-tail). The dtype and trailing shape are whatever the
    array holds. Yields nothing when the array is empty. Raises ValueError if
    block_len is not positive.
    """
    if block_len <= 0:
        raise ValueError("block_len must be positive")
    n = array.shape[0]
    for start in range(0, n, block_len):
        stop = min(n, start + block_len)
        yield array[start:stop]


def iter_offset_removed_blocks(
    array: BlockReadableArray, offset_uv: float, block_len: int
) -> Iterator[npt.NDArray[np.float64]]:
    """Yield an on-disk raw array in float64 blocks with its DC offset removed.

    The subtraction is what the offset_uv attribute means, and it happens in
    float64 before anything is summed. Doing it in float32 would spend the
    precision the attribute exists to protect. Raises ValueError if block_len is
    not positive.
    """
    if block_len <= 0:
        raise ValueError("block_len must be positive")
    for block in iter_array_blocks(array, block_len):
        yield block.astype(np.float64) - offset_uv


def iter_level_stat_blocks(
    env: BlockReadableArray,
    mean: BlockReadableArray,
    valid: BlockReadableArray | None,
    num_samples: int,
    level: int,
    block_len: int,
) -> Iterator[npt.NDArray[np.float64]]:
    """Yield a level already on disk as the stat blocks that fold the next one.

    Reassembles what the write narrowed. env, mean and valid come back from
    their arrays; only the count is rebuilt, because a bin's time support is
    fixed by its level while how much of it was finite is not. valid is None on
    a channel with no gaps, where every bin is full and valid equals the count.
    Raises ValueError if block_len is not positive.
    """
    if block_len <= 0:
        raise ValueError("block_len must be positive")
    n = env.shape[0]
    for start in range(0, n, block_len):
        stop = min(n, start + block_len)
        out = np.empty((stop - start, STAT_COLUMNS), dtype=np.float64)
        out[:, MIN_COL : MAX_COL + 1] = env[start:stop]
        out[:, MEAN_COL] = mean[start:stop]
        out[:, COUNT_COL] = bin_counts(num_samples, level, start, stop)
        out[:, VALID_COL] = (
            out[:, COUNT_COL] if valid is None else valid[start:stop]
        )
        yield out
