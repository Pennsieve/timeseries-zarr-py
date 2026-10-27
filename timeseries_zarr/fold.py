"""Pyramid folding: reduce a level to the next coarser one in blocks of 4.

A folded level travels as a float64 statistics block of STATISTIC_COLUMNS
columns: minimum, maximum, mean, population variance, and the count of finite
samples. Every statistic uses the plain reduction, so a NaN propagates into
the bin holding it; only the count stays finite.

Statistics holds the same five statistics as one float64 array each.
fold_raw and fold_statistics work on it with column views, which keeps every
numpy call a full pass over the block instead of a loop of 4 elements per row.
"""

from collections.abc import Sequence
from dataclasses import dataclass
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

    Args:
        length: Number of items.
        block: Block size.

    Returns:
        The count of complete blocks, and the leftover count (0 to block - 1)
        that forms a final partial block with no padding.

    Raises:
        ValueError: block is not positive.
    """
    if block <= 0:
        raise ValueError("block must be positive")
    return (length // block, length % block)


def _require_statistics_block(statistics: npt.NDArray[np.float64]) -> None:
    """Raise ValueError unless statistics is (rows, STATISTIC_COLUMNS)."""
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

    A non-finite sample (NaN or infinity) counts as NaN, so a block holding
    one has NaN in its four statistics while the count still reports the
    finite samples. A block of only non-finite samples has count 0. A final
    partial block of 1 to 3 samples becomes one row. An empty input yields
    zero rows. The input is not modified.

    Args:
        raw: Raw samples of one channel.

    Returns:
        A float64 statistics block with one row per block: the minimum,
        maximum, mean, and population variance of the block's samples, then
        the number of finite samples.
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
    """Fold a statistics block into the next coarser level in blocks of 4 rows.

    Each output row combines up to 4 child rows: the minimum of the minimums,
    the maximum of the maximums, the count-weighted mean of the means, the
    count-weighted mean of each child's variance plus the squared distance of
    its mean from the parent mean, and the sum of the counts. In exact
    arithmetic the result equals the statistics of the raw samples the
    children cover. A NaN in any child statistic makes the parent's four
    statistics NaN, and so do children whose counts sum to 0; the count is
    the plain sum in every case. A final partial block of 1 to 3 rows becomes
    one row. An empty input yields zero rows.

    Args:
        statistics: Statistics block of the finer level.

    Returns:
        The statistics block of the coarser level.

    Raises:
        ValueError: statistics is not rank 2 with STATISTIC_COLUMNS columns.
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

    A NaN stays NaN, and a value beyond the float32 range becomes infinity
    without a warning. An empty block yields zero rows in every member.

    Args:
        statistics: Statistics block of one level.
        with_var: Whether to return the variance member.

    Returns:
        The tuple (minmax, mean, var, count) in on-disk dtypes. minmax is
        float32 of shape (rows, 2) with the minimum in column 0 and the
        maximum in column 1. mean and var are float32 of shape (rows,), and
        var is None when with_var is false. count is uint32 of shape (rows,)
        and exact, because a level within MAX_LEVELS never counts 2**32
        samples.

    Raises:
        ValueError: statistics is not shaped (rows, STATISTIC_COLUMNS).
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
    statistics already carry. An empty level yields zero rows.

    Args:
        minmax: Array of shape (rows, 2) with the minimum in column 0 and the
            maximum in column 1.
        mean: Array of shape (rows,).
        var: Array of shape (rows,). None leaves the variance column NaN.
        count: Array of shape (rows,).

    Returns:
        The float64 statistics block.

    Raises:
        ValueError: minmax is not (rows, 2), or mean, var, or count has a
            different number of rows.
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


@dataclass(frozen=True, eq=False)
class Statistics:
    """The statistics of consecutive bins, one float64 array per statistic.

    Attributes:
        minimum: Minimum of each bin. NaN when the bin holds a non-finite
            sample.
        maximum: Maximum of each bin. NaN when the bin holds a non-finite
            sample.
        mean: Mean of each bin. NaN when the bin holds a non-finite sample.
        variance: Population variance of each bin. NaN when the bin holds a
            non-finite sample.
        count: Number of finite samples in each bin.
    """

    minimum: npt.NDArray[np.float64]
    maximum: npt.NDArray[np.float64]
    mean: npt.NDArray[np.float64]
    variance: npt.NDArray[np.float64]
    count: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        """Raise ValueError unless every member is a float64 array of rows."""
        members = {
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "variance": self.variance,
            "count": self.count,
        }
        for name, member in members.items():
            if member.ndim != 1 or member.dtype != np.float64:
                raise ValueError(
                    f"{name} is a float64 array of shape (rows,), "
                    f"got {member.dtype} with shape {member.shape}"
                )
            if member.shape != self.minimum.shape:
                raise ValueError(
                    f"{name} has {member.shape[0]} rows to match minimum, "
                    f"got {self.minimum.shape[0]}"
                )

    @property
    def rows(self) -> int:
        """The number of bins."""
        return int(self.minimum.shape[0])

    def members(
        self, with_var: bool
    ) -> tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.float32] | None,
        npt.NDArray[np.uint32],
    ]:
        """Convert the statistics to the member arrays of one level.

        A NaN stays NaN, and a value beyond the float32 range becomes infinity
        without a warning. An empty block yields zero rows in every member.

        Args:
            with_var: Whether to return the variance member.

        Returns:
            The tuple (minmax, mean, var, count) in on-disk dtypes. minmax is
            float32 of shape (rows, 2) with the minimum in column 0 and the
            maximum in column 1. mean and var are float32 of shape (rows,),
            and var is None when with_var is false. count is uint32 of shape
            (rows,) and exact, because a level within MAX_LEVELS never counts
            2**32 samples.
        """
        minmax = np.empty((self.rows, ENVELOPE_PAIR_SIZE), dtype=np.float32)
        with np.errstate(over="ignore"):
            minmax[:, 0] = self.minimum
            minmax[:, 1] = self.maximum
            mean = self.mean.astype(np.float32)
            var = self.variance.astype(np.float32) if with_var else None
        return minmax, mean, var, self.count.astype(np.uint32)


def _reduce_columns(columns: Sequence[npt.NDArray[Any]]) -> Statistics:
    """Return the statistics of each row of the (rows, len(columns)) samples.

    Args:
        columns: One equal-length view per sample position in the bin.
    """
    width = len(columns)
    rows = columns[0].shape[0]
    with np.errstate(invalid="ignore", over="ignore"):
        minimum = (
            np.minimum(columns[0], columns[1])
            if width > 1
            else columns[0].copy()
        )
        maximum = (
            np.maximum(columns[0], columns[1])
            if width > 1
            else columns[0].copy()
        )
        for column in columns[2:]:
            np.minimum(minimum, column, out=minimum)
            np.maximum(maximum, column, out=maximum)
        mean = (
            np.add(columns[0], columns[1], dtype=np.float64)
            if (width > 1)
            else columns[0].astype(np.float64)
        )
        for column in columns[2:]:
            np.add(mean, column, out=mean)
        mean /= width
        variance = np.zeros(rows, dtype=np.float64)
        deviation = np.empty(rows, dtype=np.float64)
        for column in columns:
            np.subtract(column, mean, out=deviation)
            np.multiply(deviation, deviation, out=deviation)
            variance += deviation
        variance /= width
    count = np.full(rows, float(width), dtype=np.float64)
    minimum64 = minimum.astype(np.float64)
    maximum64 = maximum.astype(np.float64)
    bad = np.flatnonzero(~(np.isfinite(minimum) & np.isfinite(maximum)))
    if bad.size:
        finite = np.zeros(bad.size, dtype=np.float64)
        for column in columns:
            finite += np.isfinite(column[bad])
        count[bad] = finite
        minimum64[bad] = np.nan
        maximum64[bad] = np.nan
        mean[bad] = np.nan
        variance[bad] = np.nan
    return Statistics(minimum64, maximum64, mean, variance, count)


def _join(parts: list[Statistics]) -> Statistics:
    """Concatenate the statistics of consecutive bins."""
    if len(parts) == 1:
        return parts[0]
    return Statistics(
        np.concatenate([part.minimum for part in parts]),
        np.concatenate([part.maximum for part in parts]),
        np.concatenate([part.mean for part in parts]),
        np.concatenate([part.variance for part in parts]),
        np.concatenate([part.count for part in parts]),
    )


def fold_raw(raw: npt.NDArray[np.float32]) -> Statistics:
    """Fold raw samples into per-bin statistics over disjoint blocks of 4.

    A non-finite sample (NaN or infinity) counts as NaN, so a block holding
    one has NaN in its four statistics while the count still reports the
    finite samples. A block of only non-finite samples has count 0. A final
    partial block of 1 to 3 samples becomes one row. An empty input yields
    zero rows. The input is not modified.

    Args:
        raw: Raw samples of one channel.

    Returns:
        The statistics with one row per block.

    Raises:
        ValueError: raw is not rank 1.
    """
    if raw.ndim != 1:
        raise ValueError(f"raw samples are rank 1, got shape {raw.shape}")
    block = DECIMATION_FACTOR
    n_full, tail_len = _block_split(raw.shape[0], block)
    split = n_full * block
    parts = []
    if n_full or not tail_len:
        grid = raw[:split].reshape(n_full, block)
        parts.append(_reduce_columns([grid[:, j] for j in range(block)]))
    if tail_len:
        tail = raw[split:]
        parts.append(
            _reduce_columns([tail[j : j + 1] for j in range(tail_len)])
        )
    return _join(parts)


def _combine_columns(children: Sequence[Statistics]) -> Statistics:
    """Combine the statistics of each row's children, one Statistics each.

    Args:
        children: For each child position, the statistics of that child in
            every row.
    """
    first = children[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        minimum = first.minimum.copy()
        maximum = first.maximum.copy()
        total = first.count.copy()
        weighted = first.count * first.mean
        for child in children[1:]:
            np.minimum(minimum, child.minimum, out=minimum)
            np.maximum(maximum, child.maximum, out=maximum)
            total += child.count
            weighted += child.count * child.mean
        weighted /= total
        parent_mean = weighted
        variance = np.zeros(first.rows, dtype=np.float64)
        spread = np.empty(first.rows, dtype=np.float64)
        for child in children:
            np.subtract(child.mean, parent_mean, out=spread)
            np.multiply(spread, spread, out=spread)
            spread += child.variance
            spread *= child.count
            variance += spread
        variance /= total
    return Statistics(minimum, maximum, parent_mean, variance, total)


def _children(
    statistics: Statistics, start: int, rows: int, width: int
) -> list[Statistics]:
    """Return the width child positions of rows consecutive parents."""
    stop = start + rows * width

    def column(
        array: npt.NDArray[np.float64], j: int
    ) -> npt.NDArray[np.float64]:
        return array[start:stop].reshape(rows, width)[:, j]

    return [
        Statistics(
            column(statistics.minimum, j),
            column(statistics.maximum, j),
            column(statistics.mean, j),
            column(statistics.variance, j),
            column(statistics.count, j),
        )
        for j in range(width)
    ]


def fold_statistics(statistics: Statistics) -> Statistics:
    """Fold statistics into the next coarser level in blocks of 4 rows.

    Each output row combines up to 4 child rows: the minimum of the minimums,
    the maximum of the maximums, the count-weighted mean of the means, the
    count-weighted mean of each child's variance plus the squared distance of
    its mean from the parent mean, and the sum of the counts. In exact
    arithmetic the result equals the statistics of the raw samples the
    children cover. A NaN in any child statistic makes the parent's four
    statistics NaN, and so do children whose counts sum to 0; the count is
    the plain sum in every case. A final partial block of 1 to 3 rows becomes
    one row. An empty input yields zero rows.

    Args:
        statistics: Statistics of the finer level.

    Returns:
        The statistics of the coarser level.
    """
    block = DECIMATION_FACTOR
    n_full, tail_len = _block_split(statistics.rows, block)
    parts = []
    if n_full or not tail_len:
        parts.append(_combine_columns(_children(statistics, 0, n_full, block)))
    if tail_len:
        parts.append(
            _combine_columns(_children(statistics, n_full * block, 1, tail_len))
        )
    return _join(parts)


@dataclass(frozen=True, eq=False)
class LevelChunk:
    """One inner chunk of every member of one level, in on-disk dtypes.

    Attributes:
        level: The level number, counted from 1.
        minmax: float32 array of shape (rows, 2) with the minimum in column 0
            and the maximum in column 1.
        mean: float32 array of shape (rows,).
        var: float32 array of shape (rows,), or None when the folder does not
            keep the variance.
        count: uint32 array of shape (rows,).
    """

    level: int
    minmax: npt.NDArray[np.float32]
    mean: npt.NDArray[np.float32]
    var: npt.NDArray[np.float32] | None
    count: npt.NDArray[np.uint32]

    @property
    def rows(self) -> int:
        """The number of bins in the chunk."""
        return int(self.mean.shape[0])


def _slice(statistics: Statistics, start: int, stop: int) -> Statistics:
    """Return the rows start to stop of statistics as views."""
    return Statistics(
        statistics.minimum[start:stop],
        statistics.maximum[start:stop],
        statistics.mean[start:stop],
        statistics.variance[start:stop],
        statistics.count[start:stop],
    )


def _copy(statistics: Statistics) -> Statistics:
    """Return statistics with its own memory."""
    return Statistics(
        statistics.minimum.copy(),
        statistics.maximum.copy(),
        statistics.mean.copy(),
        statistics.variance.copy(),
        statistics.count.copy(),
    )


_STATISTIC_FIELDS = 5


class _Level:
    """The pending rows and the unfolded input of one level of a LevelFolder."""

    def __init__(self, inner_len: int) -> None:
        """Allocate a pending buffer of inner_len rows."""
        self.buffer = np.empty((_STATISTIC_FIELDS, inner_len), dtype=np.float64)
        self.filled = 0
        self.carry_raw: npt.NDArray[np.float32] | None = None
        self.carry_statistics: Statistics | None = None

    def pending(self, start: int, stop: int) -> Statistics:
        """Return the pending rows start to stop as views."""
        view = self.buffer[:, start:stop]
        return Statistics(view[0], view[1], view[2], view[3], view[4])


class LevelFolder:
    """Fold every level of one channel as its raw samples arrive.

    Level 1 folds the raw samples. Each later level folds the float64
    statistics of the level below, never its stored float32 members, so the
    variance keeps its full precision. A level holds at most inner_len pending
    rows plus the 1 to 3 unfolded input rows of an unfinished block of 4, so
    the state does not grow with the length of the channel.

    A level emits a chunk of exactly inner_len rows each time its pending rows
    fill. finish emits the last chunk of each level, which holds the
    remaining rows and may be shorter. The chunks of one level concatenate to
    the fold of the whole channel.
    """

    def __init__(self, levels: int, inner_len: int, with_var: bool) -> None:
        """Create a folder.

        Args:
            levels: The number of levels to fold, at least 1.
            inner_len: The rows of one emitted chunk, at least 1.
            with_var: Whether the emitted chunks carry the variance.

        Raises:
            ValueError: levels or inner_len is less than 1.
        """
        if levels < 1:
            raise ValueError(f"levels must be at least 1, got {levels}")
        if inner_len < 1:
            raise ValueError(f"inner_len must be at least 1, got {inner_len}")
        self._inner_len = inner_len
        self._with_var = with_var
        self._levels = [_Level(inner_len) for _ in range(levels)]
        self._finished = False
        self._emitted: list[LevelChunk] = []

    @property
    def pending_rows(self) -> list[int]:
        """The rows each level holds that no chunk has emitted yet."""
        return [level.filled for level in self._levels]

    def push(self, block: npt.NDArray[np.float32]) -> list[LevelChunk]:
        """Fold raw samples into every level.

        Args:
            block: The next raw samples of the channel, rank 1, any length.

        Returns:
            The chunks that filled, in the order they filled.

        Raises:
            ValueError: block is not rank 1.
            RuntimeError: finish was called.
        """
        if self._finished:
            raise RuntimeError("push after finish")
        if block.ndim != 1:
            raise ValueError(f"raw samples are rank 1, got shape {block.shape}")
        self._feed_raw(block)
        return self._drain()

    def finish(self) -> list[LevelChunk]:
        """Fold the unfolded tails and emit the last chunk of every level.

        Returns:
            The final chunks, finest level first.

        Raises:
            RuntimeError: finish was called before.
        """
        if self._finished:
            raise RuntimeError("finish called twice")
        self._finished = True
        for index, level in enumerate(self._levels):
            if index == 0 and level.carry_raw is not None:
                tail = fold_raw(level.carry_raw)
                level.carry_raw = None
                self._append(0, tail)
            elif index > 0 and level.carry_statistics is not None:
                tail = fold_statistics(level.carry_statistics)
                level.carry_statistics = None
                self._append(index, tail)
            if level.filled:
                self._emit(index, level.pending(0, level.filled))
                level.filled = 0
        return self._drain()

    def _drain(self) -> list[LevelChunk]:
        """Return and clear the emitted chunks."""
        emitted, self._emitted = self._emitted, []
        return emitted

    def _feed_raw(self, block: npt.NDArray[np.float32]) -> None:
        """Fold block into level 1 behind the carried samples."""
        level = self._levels[0]
        start = 0
        if level.carry_raw is not None:
            take = min(
                DECIMATION_FACTOR - level.carry_raw.shape[0], block.shape[0]
            )
            joined = np.concatenate([level.carry_raw, block[:take]])
            start = take
            if joined.shape[0] < DECIMATION_FACTOR:
                level.carry_raw = joined
                return
            level.carry_raw = None
            self._append(0, fold_raw(joined))
        whole = (
            (block.shape[0] - start) // DECIMATION_FACTOR * DECIMATION_FACTOR
        )
        if whole:
            self._append(0, fold_raw(block[start : start + whole]))
        if start + whole < block.shape[0]:
            level.carry_raw = block[start + whole :].copy()

    def _feed_statistics(self, index: int, rows: Statistics) -> None:
        """Fold the rows of level index to the level above it."""
        level = self._levels[index + 1]
        start = 0
        carry = level.carry_statistics
        if carry is not None:
            take = min(DECIMATION_FACTOR - carry.rows, rows.rows)
            joined = _join([carry, _slice(rows, 0, take)])
            start = take
            if joined.rows < DECIMATION_FACTOR:
                level.carry_statistics = _copy(joined)
                return
            level.carry_statistics = None
            self._append(index + 1, fold_statistics(joined))
        whole = (rows.rows - start) // DECIMATION_FACTOR * DECIMATION_FACTOR
        if whole:
            self._append(
                index + 1, fold_statistics(_slice(rows, start, start + whole))
            )
        if start + whole < rows.rows:
            level.carry_statistics = _copy(
                _slice(rows, start + whole, rows.rows)
            )

    def _append(self, index: int, rows: Statistics) -> None:
        """Add rows to the pending rows of level index, emitting full chunks."""
        level = self._levels[index]
        start = 0
        while start < rows.rows:
            if level.filled == 0 and rows.rows - start >= self._inner_len:
                stop = start + self._inner_len
                self._emit(index, _slice(rows, start, stop))
            else:
                stop = min(rows.rows, start + self._inner_len - level.filled)
                level.buffer[:, level.filled : level.filled + stop - start] = [
                    rows.minimum[start:stop],
                    rows.maximum[start:stop],
                    rows.mean[start:stop],
                    rows.variance[start:stop],
                    rows.count[start:stop],
                ]
                level.filled += stop - start
                if level.filled == self._inner_len:
                    self._emit(index, level.pending(0, level.filled))
                    level.filled = 0
            start = stop

    def _emit(self, index: int, chunk: Statistics) -> None:
        """Record chunk as an emitted chunk and fold it into the next level."""
        minmax, mean, var, count = chunk.members(self._with_var)
        self._emitted.append(LevelChunk(index + 1, minmax, mean, var, count))
        if index + 1 < len(self._levels):
            self._feed_statistics(index, chunk)
