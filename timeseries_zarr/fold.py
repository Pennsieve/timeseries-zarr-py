"""Pyramid folding: reduce a level to the next coarser one in blocks of 4.

A folded level travels as a float64 statistics block of STATISTIC_COLUMNS
columns: minimum, maximum, mean, population variance, and the count of finite
samples. Every
statistic uses the plain reduction, so a NaN propagates into the bin holding
it; only the count stays finite.
"""

from typing import Any

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


def _require_statistics_block(statistics: npt.NDArray[np.float64]) -> None:
    """Raise ValueError unless statistics has shape (rows, STATISTIC_COLUMNS)."""
    if statistics.shape[1:] != (STATISTIC_COLUMNS,):
        raise ValueError(
            f"a statistics block has shape (rows, {STATISTIC_COLUMNS}), "
            f"got {statistics.shape}"
        )


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

    Returns a float64 statistics block with one row per block: the minimum,
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
    total_count = counts.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        parent_mean = (counts * means).sum(axis=1) / total_count
        spread = means - parent_mean[:, None]
        parent_variance = (
            counts * (children[:, :, VAR_COLUMN] + spread**2)
        ).sum(axis=1) / total_count
    out[:, MIN_COLUMN] = children[:, :, MIN_COLUMN].min(axis=1)
    out[:, MAX_COLUMN] = children[:, :, MAX_COLUMN].max(axis=1)
    out[:, MEAN_COLUMN] = parent_mean
    out[:, VAR_COLUMN] = parent_variance
    out[:, COUNT_COLUMN] = total_count


def fold_statistics_block(
    statistics: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Fold a statistics block into the next coarser level over blocks of 4 rows.

    Each output row combines up to 4 child rows: the minimum of the minimums,
    the maximum of the maximums, the count-weighted mean of the means, the
    count-weighted mean of each child's variance plus the squared distance of
    its mean from the parent mean, and the sum of the counts. In exact
    arithmetic the result equals the statistics of the raw samples the
    children cover. A NaN in any child statistic makes the parent's four
    statistics NaN, and so do children whose counts sum to 0; the count is
    the plain sum in every case. A final partial block of 1 to 3 rows becomes
    one row. An empty input yields zero rows. Raises ValueError if statistics is
    not rank 2 with STATISTIC_COLUMNS columns.
    """
    _require_statistics_block(statistics)
    block = DECIMATION_FACTOR
    n_full, tail_len = _block_split(statistics.shape[0], block)
    split = n_full * block
    out = np.empty(
        (n_full + (1 if tail_len else 0), STATISTIC_COLUMNS), dtype=np.float64
    )
    if n_full:
        _combine_into(
            out[:n_full],
            statistics[:split].reshape(n_full, block, STATISTIC_COLUMNS),
        )
    if tail_len:
        _combine_into(out[n_full:], statistics[split:][None, :, :])
    return out


def split_members(
    statistics: npt.NDArray[np.float64], with_var: bool
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32] | None,
    npt.NDArray[np.uint32],
]:
    """Split a statistics block into the member arrays of one level.

    Returns (minmax, mean, var, count) in their on-disk dtypes. minmax is
    float32 of shape (rows, 2) with the minimum in column 0 and the maximum in
    column 1. mean and var are float32 of shape (rows,), and var is None when
    with_var is false. count is uint32 of shape (rows,) and exact, because a
    level within MAX_LEVELS never counts 2**32 samples. A NaN stays NaN, and a
    value beyond the float32 range becomes infinity without a warning. An
    empty block yields zero rows in every member. Raises ValueError if statistics
    is not shaped (rows, STATISTIC_COLUMNS).
    """
    _require_statistics_block(statistics)
    with np.errstate(over="ignore"):
        minmax = statistics[:, [MIN_COLUMN, MAX_COLUMN]].astype(np.float32)
        mean = statistics[:, MEAN_COLUMN].astype(np.float32)
        var = statistics[:, VAR_COLUMN].astype(np.float32) if with_var else None
    count = statistics[:, COUNT_COLUMN].astype(np.uint32)
    return minmax, mean, var, count


def assemble_statistics(
    minmax: npt.NDArray[Any],
    mean: npt.NDArray[Any],
    var: npt.NDArray[Any] | None,
    count: npt.NDArray[Any],
) -> npt.NDArray[np.float64]:
    """Assemble the member arrays of one level into a float64 statistics block.

    The inverse of split_members, up to the float32 rounding the stored
    statistics already carry. minmax is shaped (rows, 2) with the minimum in
    column 0 and the maximum in column 1; mean, var, and count are shaped
    (rows,). A var of None leaves the variance column NaN. An empty level
    yields zero rows. Raises ValueError if minmax is not (rows, 2) or if mean,
    var, or count does not have the same number of rows.
    """
    if minmax.shape[1:] != (ENVELOPE_PAIR_SIZE,):
        raise ValueError(
            f"minmax has shape (rows, {ENVELOPE_PAIR_SIZE}), got {minmax.shape}"
        )
    rows = minmax.shape[0]
    for name, member in (("mean", mean), ("var", var), ("count", count)):
        if member is not None and member.shape != (rows,):
            raise ValueError(
                f"{name} has shape ({rows},) to match minmax, "
                f"got {member.shape}"
            )
    out = np.empty((rows, STATISTIC_COLUMNS), dtype=np.float64)
    out[:, MIN_COLUMN] = minmax[:, 0]
    out[:, MAX_COLUMN] = minmax[:, 1]
    out[:, MEAN_COLUMN] = mean
    out[:, VAR_COLUMN] = np.nan if var is None else var
    out[:, COUNT_COLUMN] = count
    return out
