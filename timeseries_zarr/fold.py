"""Pyramid folding: reduce one level to the next over disjoint blocks of 4.

A folded level travels as one float64 array of five columns -- min, max, mean,
count and valid. One array means the streaming machinery carries every
statistic in a single pass and across a block boundary without knowing what
the columns mean.

Count and valid are different numbers and both are needed. Count is time
support, the raw slots a bin spans, and it exists because a trailing partial
bin holds fewer than 4 samples and a plain mean of means would over-weight it;
it never reaches disk, because a level read back rebuilds it from its own
number. Valid is how many of those slots held a finite sample, which is data
and cannot be rebuilt, so it is written.

The arithmetic is float64 throughout and narrows to float32 only at the write.
A mean is a sum, and summing 16384 samples of a signal riding on a large DC
offset is exactly where float32 loses the part you wanted.

Every fold uses plain min, max and mean, so a NaN propagates into the bin
holding it. A NaN-aware mean would average different time supports on different
channels, which breaks the montage exactness the mean member exists for.
"""

from typing import Any

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


def _block_split(
    length: int, block: int = DECIMATION_FACTOR
) -> tuple[int, int]:
    """Return (n_full, tail_len) for splitting length items into blocks.

    n_full is the count of complete blocks of size block; tail_len is the
    leftover count (0..block-1), kept as a final partial block with no padding.
    Raises ValueError if block is not positive.
    """
    if block <= 0:
        raise ValueError("block must be positive")
    return (length // block, length % block)


def _empty_stats(rows: int) -> npt.NDArray[np.float64]:
    """Return an uninitialized stat block of the given row count."""
    return np.empty((rows, STAT_COLUMNS), dtype=np.float64)


def fold_raw_block(
    raw: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.float64]:
    """Fold raw samples into stat rows over disjoint blocks of 4.

    Takes rank-1 samples and returns one row per bin: min, max, mean, the
    number of samples behind it, and how many of those were finite. A final
    partial block of 1-3 samples becomes one row carrying its true counts.
    """
    block = DECIMATION_FACTOR
    n_full, tail_len = _block_split(raw.shape[0], block)
    split = n_full * block
    out = _empty_stats(n_full + (1 if tail_len else 0))
    values = raw.astype(np.float64, copy=False)
    if n_full:
        full = values[:split].reshape(n_full, block)
        out[:n_full, MIN_COL] = full.min(axis=1)
        out[:n_full, MAX_COL] = full.max(axis=1)
        out[:n_full, MEAN_COL] = full.mean(axis=1)
        out[:n_full, COUNT_COL] = block
        out[:n_full, VALID_COL] = np.isfinite(full).sum(axis=1)
    if tail_len:
        tail = values[split:]
        out[n_full, MIN_COL] = tail.min()
        out[n_full, MAX_COL] = tail.max()
        out[n_full, MEAN_COL] = tail.mean()
        out[n_full, COUNT_COL] = tail_len
        out[n_full, VALID_COL] = np.isfinite(tail).sum()
    return out


def fold_stat_block(
    stats: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Fold stat rows into coarser stat rows over disjoint blocks of 4.

    Each output row takes the smallest of the 4 mins, the largest of the 4
    maxes, the count-weighted mean of the 4 means, and the sums of the 4 counts
    and the 4 valid counts. Weighting by count is what keeps the mean exact
    where the block below ends in a partial bin. A final partial block of 1-3
    rows becomes one row.
    """
    block = DECIMATION_FACTOR
    n_full, tail_len = _block_split(stats.shape[0], block)
    split = n_full * block
    out = _empty_stats(n_full + (1 if tail_len else 0))
    if n_full:
        full = stats[:split].reshape(n_full, block, STAT_COLUMNS)
        _reduce_into(out[:n_full], full)
    if tail_len:
        tail = stats[split:].reshape(1, tail_len, STAT_COLUMNS)
        _reduce_into(out[n_full : n_full + 1], tail)
    return out


def _reduce_into(
    out: npt.NDArray[np.float64], groups: npt.NDArray[np.float64]
) -> None:
    """Reduce (rows, members, 4) groups of stat rows into one row each."""
    counts = groups[:, :, COUNT_COL]
    total = counts.sum(axis=1)
    out[:, MIN_COL] = groups[:, :, MIN_COL].min(axis=1)
    out[:, MAX_COL] = groups[:, :, MAX_COL].max(axis=1)
    out[:, MEAN_COL] = (groups[:, :, MEAN_COL] * counts).sum(axis=1) / total
    out[:, COUNT_COL] = total
    out[:, VALID_COL] = groups[:, :, VALID_COL].sum(axis=1)


def fold_block(
    arr: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.float64]:
    """Fold one pyramid level to the next coarser one, dispatching on rank.

    Rank-1 is raw samples; rank-2 is stat rows, which must carry all four
    columns. Raises ValueError for any other shape.
    """
    match arr.ndim:
        case 1:
            return fold_raw_block(arr)
        case 2 if arr.shape[1] == STAT_COLUMNS:
            return fold_stat_block(arr.astype(np.float64, copy=False))
        case _:
            raise ValueError(
                f"cannot fold an array of shape {arr.shape}: expected rank 1 "
                f"raw samples or rank 2 stat rows of {STAT_COLUMNS} columns"
            )
