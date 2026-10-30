"""Pyramid folding: reduce a level to the next coarser one in fixed blocks.

A folded level travels as Statistics: the minimum, maximum, mean, population
variance, and count of the finite samples of each bin, as one float64 array
each. A NaN or infinite sample does not count and does not enter any
statistic. A bin with no finite sample has count 0 and NaN in the other four
statistics.

fold_raw and fold_statistics work on column views, which keeps every numpy
call a full pass over the block instead of a loop over the elements of a row.
LevelFolder holds the state of one channel's pyramid.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from timeseries_zarr.constants import ENVELOPE_PAIR_SIZE, MIN_DECIMATION_FACTOR


def _block_split(length: int, block: int) -> tuple[int, int]:
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


@dataclass(frozen=True, eq=False)
class Statistics:
    """The statistics of consecutive bins, one float64 array per statistic.

    Attributes:
        minimum: Minimum of the finite samples of each bin. NaN when the count
            is 0.
        maximum: Maximum of the finite samples of each bin. NaN when the count
            is 0.
        mean: Mean of the finite samples of each bin. NaN when the count is 0.
        variance: Population variance of the finite samples of each bin. NaN
            when the count is 0.
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
            (rows,) and exact, because a level within max_levels_for never
            counts 2**32 samples.
        """
        minmax = np.empty((self.rows, ENVELOPE_PAIR_SIZE), dtype=np.float32)
        with np.errstate(over="ignore"):
            minmax[:, 0] = self.minimum
            minmax[:, 1] = self.maximum
            mean = self.mean.astype(np.float32)
            var = self.variance.astype(np.float32) if with_var else None
        return minmax, mean, var, self.count.astype(np.uint32)


def _reduce_finite(columns: Sequence[npt.NDArray[Any]]) -> Statistics:
    """Return the statistics of the finite samples of each row.

    Args:
        columns: One equal-length view per sample position in the bin.
    """
    rows = columns[0].shape[0]
    count = np.zeros(rows, dtype=np.float64)
    total = np.zeros(rows, dtype=np.float64)
    minimum = np.full(rows, np.inf, dtype=np.float64)
    maximum = np.full(rows, -np.inf, dtype=np.float64)
    masks = [np.isfinite(column) for column in columns]
    values = [column.astype(np.float64) for column in columns]
    for mask, value in zip(masks, values, strict=True):
        count += mask
        total += np.where(mask, value, 0.0)
        np.minimum(minimum, np.where(mask, value, np.inf), out=minimum)
        np.maximum(maximum, np.where(mask, value, -np.inf), out=maximum)
    empty = count == 0
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
        variance = np.zeros(rows, dtype=np.float64)
        for mask, value in zip(masks, values, strict=True):
            deviation = np.where(mask, value - mean, 0.0)
            variance += deviation * deviation
        variance /= count
    minimum[empty] = np.nan
    maximum[empty] = np.nan
    return Statistics(minimum, maximum, mean, variance, count)


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
        patch = _reduce_finite([column[bad] for column in columns])
        minimum64[bad] = patch.minimum
        maximum64[bad] = patch.maximum
        mean[bad] = patch.mean
        variance[bad] = patch.variance
        count[bad] = patch.count
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


def fold_raw(raw: npt.NDArray[np.float32], block: int) -> Statistics:
    """Fold raw samples into per-bin statistics over disjoint blocks.

    A non-finite sample (NaN or infinity) enters no statistic and does not
    count. A block with no finite sample has count 0 and NaN in the other four
    statistics. A final partial block of fewer than block samples becomes one
    row. An empty input yields zero rows. The input is not modified.

    Args:
        raw: Raw samples of one channel.
        block: The number of samples that fold into one row, at least 1.

    Returns:
        The statistics with one row per block.

    Raises:
        ValueError: raw is not rank 1, or block is not positive.
    """
    if raw.ndim != 1:
        raise ValueError(f"raw samples are rank 1, got shape {raw.shape}")
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

    A child with count 0 is absent from every statistic.

    Args:
        children: For each child position, the statistics of that child in
            every row.
    """
    rows = children[0].rows
    present = [child.count > 0 for child in children]
    minimum = np.full(rows, np.inf, dtype=np.float64)
    maximum = np.full(rows, -np.inf, dtype=np.float64)
    total = np.zeros(rows, dtype=np.float64)
    weighted = np.zeros(rows, dtype=np.float64)
    for child, here in zip(children, present, strict=True):
        np.minimum(minimum, np.where(here, child.minimum, np.inf), out=minimum)
        np.maximum(maximum, np.where(here, child.maximum, -np.inf), out=maximum)
        total += child.count
        weighted += child.count * np.where(here, child.mean, 0.0)
    empty = total == 0
    with np.errstate(invalid="ignore", divide="ignore"):
        parent_mean = weighted / total
        variance = np.zeros(rows, dtype=np.float64)
        for child, here in zip(children, present, strict=True):
            spread = child.mean - parent_mean
            spread *= spread
            spread += child.variance
            spread *= child.count
            variance += np.where(here, spread, 0.0)
        variance /= total
    minimum[empty] = np.nan
    maximum[empty] = np.nan
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


def fold_statistics(statistics: Statistics, block: int) -> Statistics:
    """Fold statistics into the next coarser level in blocks of rows.

    Each output row combines up to block child rows with count above 0: the
    minimum of the minimums, the maximum of the maximums, the count-weighted
    mean of the means, the count-weighted mean of each child's variance plus
    the squared distance of its mean from the parent mean, and the sum of the
    counts. In exact arithmetic the result equals the statistics of the finite
    raw samples the children cover. A child with count 0 contributes nothing.
    A row whose children all have count 0 has count 0 and NaN in the other
    four statistics. A final partial block of fewer than block rows becomes
    one row. An empty input yields zero rows.

    Args:
        statistics: Statistics of the finer level.
        block: The number of rows that fold into one, at least 1.

    Returns:
        The statistics of the coarser level.

    Raises:
        ValueError: block is not positive.
    """
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
    rows plus the fewer than decimation_factor unfolded input rows of an
    unfinished block, so the state does not grow with the length of the
    channel.

    A level emits a chunk of exactly inner_len rows each time its pending rows
    fill. finish emits the last chunk of each level, which holds the
    remaining rows and may be shorter. The chunks of one level concatenate to
    the fold of the whole channel.
    """

    def __init__(
        self,
        levels: int,
        inner_len: int,
        with_var: bool,
        decimation_factor: int,
    ) -> None:
        """Create a folder.

        Args:
            levels: The number of levels to fold, at least 1.
            inner_len: The rows of one emitted chunk, at least 1.
            with_var: Whether the emitted chunks carry the variance.
            decimation_factor: The number of rows of a level that fold into
                one row of the next, at least 2.

        Raises:
            ValueError: levels or inner_len is less than 1, or
                decimation_factor is less than 2.
        """
        if levels < 1:
            raise ValueError(f"levels must be at least 1, got {levels}")
        if inner_len < 1:
            raise ValueError(f"inner_len must be at least 1, got {inner_len}")
        if decimation_factor < MIN_DECIMATION_FACTOR:
            raise ValueError(
                "decimation_factor must be at least "
                f"{MIN_DECIMATION_FACTOR}, got {decimation_factor}"
            )
        self._inner_len = inner_len
        self._decimation_factor = decimation_factor
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
                tail = fold_raw(level.carry_raw, self._decimation_factor)
                level.carry_raw = None
                self._append(0, tail)
            elif index > 0 and level.carry_statistics is not None:
                tail = fold_statistics(
                    level.carry_statistics, self._decimation_factor
                )
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
        factor = self._decimation_factor
        level = self._levels[0]
        start = 0
        if level.carry_raw is not None:
            take = min(factor - level.carry_raw.shape[0], block.shape[0])
            joined = np.concatenate([level.carry_raw, block[:take]])
            start = take
            if joined.shape[0] < factor:
                level.carry_raw = joined
                return
            level.carry_raw = None
            self._append(0, fold_raw(joined, factor))
        whole = (block.shape[0] - start) // factor * factor
        if whole:
            self._append(0, fold_raw(block[start : start + whole], factor))
        if start + whole < block.shape[0]:
            level.carry_raw = block[start + whole :].copy()

    def _feed_statistics(self, index: int, rows: Statistics) -> None:
        """Fold the rows of level index to the level above it."""
        factor = self._decimation_factor
        level = self._levels[index + 1]
        start = 0
        carry = level.carry_statistics
        if carry is not None:
            take = min(factor - carry.rows, rows.rows)
            joined = _join([carry, _slice(rows, 0, take)])
            start = take
            if joined.rows < factor:
                level.carry_statistics = _copy(joined)
                return
            level.carry_statistics = None
            self._append(index + 1, fold_statistics(joined, factor))
        whole = (rows.rows - start) // factor * factor
        if whole:
            self._append(
                index + 1,
                fold_statistics(_slice(rows, start, start + whole), factor),
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
