"""Pyramid folding: reduce a level to the next coarser one in blocks of 4.

A folded level travels as a float64 stat block of STATISTIC_COLUMNS columns: min,
max, mean, population variance, and the count of finite samples. Every
statistic uses the plain reduction, so a NaN propagates into the bin holding
it; only the count stays finite.
"""

import numpy as np
import numpy.typing as npt

from timeseries_zarr.constants import (
    COUNT_COLUMN,
    DECIMATION_FACTOR,
    ENVELOPE_PAIR_SIZE,
    MAX_COLUMN,
    MEAN_COLUMN,
    MIN_COLUMN,
    STATISTIC_COLUMNS,
    VAR_COLUMN,
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


def _reduce_into(
    out: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    finite: npt.NDArray[np.bool_],
) -> None:
    """Fill out's rows with the statistics of each row of values."""
    out[:, MIN_COLUMN] = values.min(axis=1)
    out[:, MAX_COLUMN] = values.max(axis=1)
    out[:, MEAN_COLUMN] = values.mean(axis=1)
    out[:, VAR_COLUMN] = values.var(axis=1)
    out[:, COUNT_COLUMN] = finite.sum(axis=1)


def fold_raw_block(raw: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
    """Fold raw samples into per-bin statistics over disjoint blocks of 4.

    Returns a float64 stat block with one row per block: the minimum,
    maximum, mean, and population variance of the block's samples, then the
    number of finite samples. A non-finite sample (NaN or infinity) counts as
    NaN, so a block holding one has NaN in its four statistics while the
    count still reports the finite samples; a block of nothing but non-finite
    samples has count 0. A final partial block of 1 to 3 samples becomes one
    row. An empty input yields zero rows. The input is never modified.
    """
    block = DECIMATION_FACTOR
    n_full, tail_len = _block_split(raw.shape[0], block)
    split = n_full * block
    values = raw.astype(np.float64)
    finite = np.isfinite(values)
    values[~finite] = np.nan
    out = np.empty(
        (n_full + (1 if tail_len else 0), STATISTIC_COLUMNS), dtype=np.float64
    )
    if n_full:
        _reduce_into(
            out[:n_full],
            values[:split].reshape(n_full, block),
            finite[:split].reshape(n_full, block),
        )
    if tail_len:
        _reduce_into(
            out[n_full:], values[split:][None, :], finite[split:][None, :]
        )
    return out


def _combine_into(
    out: npt.NDArray[np.float64], children: npt.NDArray[np.float64]
) -> None:
    """Fill out's rows by combining each row's children, shaped (rows, k, 5)."""
    counts = children[:, :, COUNT_COLUMN]
    means = children[:, :, MEAN_COLUMN]
    n = counts.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        parent_mean = (counts * means).sum(axis=1) / n
        spread = means - parent_mean[:, None]
        parent_var = (counts * (children[:, :, VAR_COLUMN] + spread**2)).sum(
            axis=1
        ) / n
    out[:, MIN_COLUMN] = children[:, :, MIN_COLUMN].min(axis=1)
    out[:, MAX_COLUMN] = children[:, :, MAX_COLUMN].max(axis=1)
    out[:, MEAN_COLUMN] = parent_mean
    out[:, VAR_COLUMN] = parent_var
    out[:, COUNT_COLUMN] = n


def fold_stat_block(stats: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Fold a stat block into the next coarser level over blocks of 4 rows.

    Each output row combines up to 4 child rows: the minimum of the minimums,
    the maximum of the maximums, the count-weighted mean of the means, the
    count-weighted mean of each child's variance plus the squared distance of
    its mean from the parent mean, and the sum of the counts. In exact
    arithmetic the result equals the statistics of the raw samples the
    children cover. A NaN in any child statistic makes the parent's four
    statistics NaN, and so do children whose counts sum to 0; the count is
    the plain sum in every case. A final partial block of 1 to 3 rows becomes
    one row. An empty input yields zero rows. Raises ValueError if stats is
    not rank 2 with STATISTIC_COLUMNS columns.
    """
    if stats.shape[1:] != (STATISTIC_COLUMNS,):
        raise ValueError(
            f"a stat block has shape (rows, {STATISTIC_COLUMNS}), got {stats.shape}"
        )
    block = DECIMATION_FACTOR
    n_full, tail_len = _block_split(stats.shape[0], block)
    split = n_full * block
    out = np.empty(
        (n_full + (1 if tail_len else 0), STATISTIC_COLUMNS), dtype=np.float64
    )
    if n_full:
        _combine_into(
            out[:n_full],
            stats[:split].reshape(n_full, block, STATISTIC_COLUMNS),
        )
    if tail_len:
        _combine_into(out[n_full:], stats[split:][None, :, :])
    return out


def fold_pair_block(pairs: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Fold (min, max) pairs into coarser pairs over disjoint blocks of 4.

    Each output row takes the smallest of the 4 mins (column 0) and the largest
    of the 4 maxes (column 1). A final partial block of 1-3 rows becomes one
    row.
    """
    block = DECIMATION_FACTOR
    n_full, tail_len = _block_split(pairs.shape[0], block)
    split = n_full * block
    out = np.empty(
        (n_full + (1 if tail_len else 0), ENVELOPE_PAIR_SIZE), dtype=np.float32
    )
    if n_full:
        full = pairs[:split].reshape(n_full, block, ENVELOPE_PAIR_SIZE)
        out[:n_full, 0] = full[:, :, 0].min(axis=1)
        out[:n_full, 1] = full[:, :, 1].max(axis=1)
    if tail_len:
        tail = pairs[split:]
        out[n_full, 0] = tail[:, 0].min()
        out[n_full, 1] = tail[:, 1].max()
    return out


def fold_block(arr: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Fold one pyramid level to the next coarser one, dispatching on rank.

    Rank-1 is raw samples, folded by fold_raw_block and reduced to its
    float32 (min, max) columns; rank-2 is (min, max) pairs, folded by
    fold_pair_block. Raises ValueError for any other rank.
    """
    match arr.ndim:
        case 1:
            return fold_raw_block(arr)[:, [MIN_COLUMN, MAX_COLUMN]].astype(
                np.float32
            )
        case 2:
            return fold_pair_block(arr)
        case _:
            raise ValueError(f"unsupported array rank: {arr.ndim}")
