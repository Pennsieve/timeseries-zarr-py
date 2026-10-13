"""Bounded-memory streaming generators that drive the folds over a source."""

from collections.abc import Callable, Iterable, Iterator
from typing import Protocol

import numpy as np
import numpy.typing as npt

from timeseries_zarr.constants import DECIMATION_FACTOR
from timeseries_zarr.fold import fold_raw_block
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

    def __getitem__(self, item: slice) -> npt.NDArray[np.float32]:
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


def iter_level0_to_level1(
    source: ContinuousChannelSource, block_samples: int
) -> Iterator[npt.NDArray[np.float64]]:
    """Yield level-1 statistics blocks folded from the source's raw samples.

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
) -> Iterator[npt.NDArray[np.float32]]:
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
