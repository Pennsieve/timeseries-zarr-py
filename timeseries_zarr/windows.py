"""Window arithmetic over the time grids of a bundle.

A grid holds equal-width entries. The raw samples of a channel form one grid,
and so do the bins of each level and of each count level. Entry i of a grid
starts at offset_us + i * period microseconds from the reference time. Index
arithmetic is exact in Fraction, and a computed time axis is float64. The
module also checks the integer arguments of a read and parses the decimal
names of channel groups and level groups. It checks a label range and a
column name argument, and builds the error for an array of the wrong layout.
A channel with a runs table also maps windows and indices to the exact times
that the table gives. The table is an int64 array of shape (runs, 3) that holds
the first sample, the number of samples, and the time of the first sample of
each run.
"""

import math
import operator
from collections.abc import Iterable, Mapping
from fractions import Fraction
from typing import Any, Final, SupportsIndex, cast

import numpy as np
import numpy.typing as npt

from timeseries_zarr.constants import MICROSECONDS_PER_SECOND
from timeseries_zarr.zarr_io import ZarrArray

_HALF: Final = 0.5


def bin_period(rate_hz: float, samples_per_bin: int) -> Fraction:
    """Return the exact microseconds that one entry of a grid spans.

    Args:
        rate_hz: The sample rate in hertz.
        samples_per_bin: The samples in one entry. It is 1 for raw samples
            and the samples per bin of a level.

    Raises:
        ValueError: If rate_hz is not positive and finite or samples_per_bin
            is below 1.
    """
    if not (math.isfinite(rate_hz) and rate_hz > 0):
        raise ValueError(
            f"rate_hz must be positive and finite, got {rate_hz!r}"
        )
    if samples_per_bin < 1:
        raise ValueError(
            f"samples_per_bin must be at least 1, got {samples_per_bin!r}"
        )
    return (
        samples_per_bin * Fraction(MICROSECONDS_PER_SECOND) / Fraction(rate_hz)
    )


def nearest_integer(value: float) -> int:
    """Return the integer nearest value, the larger one when two are as near.

    Args:
        value: A finite number.
    """
    floor = math.floor(value)
    return floor + 1 if value - floor >= _HALF else floor


def duration_us(count: int, rate_hz: float) -> int:
    """Return the microseconds that count samples at rate_hz span, rounded up.

    A window of this length from the first sample holds every sample.

    Args:
        count: The number of samples.
        rate_hz: The sample rate in hertz.

    Raises:
        ValueError: If rate_hz is not positive and finite.
    """
    return math.ceil(count * bin_period(rate_hz, 1))


def integer_argument(
    value: object, name: str, expectation: str = "an integer"
) -> int:
    """Return value as an int.

    Args:
        value: The argument to check.
        name: The name of the argument, which begins the error message.
        expectation: The phrase that follows name in the error message.

    Raises:
        TypeError: If value is not an integer.
    """
    try:
        return operator.index(cast("SupportsIndex", value))
    except TypeError:
        raise TypeError(
            f"{name} must be {expectation}, got {value!r}"
        ) from None


def _window_bound(value: SupportsIndex | None, name: str) -> int | None:
    """Return value as an int, or None for an open bound.

    Raises:
        TypeError: If value is not an integer.
    """
    if value is None:
        return None
    return integer_argument(value, name, "an integer count of microseconds")


def window_bounds(
    start_us: SupportsIndex | None, end_us: SupportsIndex | None
) -> tuple[int | None, int | None]:
    """Return the bounds of a window as ints, None for an open side.

    Args:
        start_us: The start of the window, or None for an open side.
        end_us: The end of the window, or None for an open side.

    Raises:
        TypeError: If a bound is not an integer.
        ValueError: If end_us is before start_us.
    """
    start = _window_bound(start_us, "start_us")
    end = _window_bound(end_us, "end_us")
    if start is not None and end is not None and end < start:
        raise ValueError(
            f"end_us must be at or after start_us, got start_us {start} "
            f"and end_us {end}"
        )
    return start, end


def index_range(
    start_us: int | None,
    end_us: int | None,
    offset_us: int,
    period: Fraction,
    length: int,
) -> tuple[int, int]:
    """Return the half-open index range of the grid entries in a time window.

    Every entry that overlaps the window is in the range, and an entry that
    starts at end_us is not. Equal bounds and a window outside the grid give
    an empty range.

    Args:
        start_us: The start of the window in microseconds from the reference
            time. None leaves that side open.
        end_us: The end of the window, exclusive. None leaves that side open.
        offset_us: The time of entry 0 of the grid.
        period: The microseconds one entry spans.
        length: The number of entries in the grid.

    Raises:
        TypeError: If a bound is not an integer.
        ValueError: If end_us is before start_us.
    """
    start, end = window_bounds(start_us, end_us)
    first = 0
    if start is not None:
        first = min(max(math.floor((start - offset_us) / period), 0), length)
        if end == start:
            return first, first
    stop = length
    if end is not None:
        stop = min(max(math.ceil((end - offset_us) / period), 0), length)
    return first, max(stop, first)


def _run_of(table: npt.NDArray[np.int64], index: int) -> int:
    """Return the row of the last run that starts at or before index.

    The result is -1 when index is before the first run.
    """
    return int(np.searchsorted(table[:, 0], index, side="right")) - 1


def _run_bounds(
    table: npt.NDArray[np.int64], row: int, length: int
) -> tuple[int, int]:
    """Return the end of run row and the first sample of the next run.

    The end of the grid stands for the first sample of a run that does not
    exist.
    """
    end = int(table[row, 0] + table[row, 1])
    following = int(table[row + 1, 0]) if row + 1 < len(table) else length
    return end, following


def run_index_range(
    start_us: int | None,
    end_us: int | None,
    table: npt.NDArray[np.int64],
    offset_us: int,
    period: Fraction,
    length: int,
) -> tuple[int, int]:
    """Return the half-open index range of the samples in a time window.

    A sample inside a run starts at the exact time that the table gives. A
    sample between two runs starts at its grid time. Every sample that
    overlaps the window is in the range, and a sample that starts at end_us is
    not. Equal bounds and a window outside the grid give an empty range.

    Args:
        start_us: The start of the window in microseconds from the reference
            time. None leaves that side open.
        end_us: The end of the window, exclusive. None leaves that side open.
        table: The runs table of the channel.
        offset_us: The time of sample 0 of the grid.
        period: The microseconds one sample spans.
        length: The number of samples in the grid.

    Raises:
        TypeError: If a bound is not an integer.
        ValueError: If end_us is before start_us.
    """
    start, end = window_bounds(start_us, end_us)
    first = 0
    if start is not None:
        row = int(np.searchsorted(table[:, 2], start, side="right")) - 1
        if row >= 0:
            index = int(table[row, 0]) + math.floor(
                (start - int(table[row, 2])) / period
            )
            run_end, following = _run_bounds(table, row, length)
            if index >= run_end:
                grid = math.floor((start - offset_us) / period)
                index = min(max(grid, run_end), following)
            first = min(index, length)
        if end == start:
            return first, first
    stop = length
    if end is not None:
        row = int(np.searchsorted(table[:, 2], end, side="left")) - 1
        stop = 0
        if row >= 0:
            index = int(table[row, 0]) + math.ceil(
                (end - int(table[row, 2])) / period
            )
            run_end, following = _run_bounds(table, row, length)
            if index >= run_end:
                grid = math.ceil((end - offset_us) / period)
                index = min(max(grid, run_end), following)
            stop = min(index, length)
    return first, max(stop, first)


def entry_time(
    index: int,
    offset_us: int,
    period: Fraction,
    table: npt.NDArray[np.int64] | None = None,
) -> Fraction:
    """Return the exact start time of grid entry index in microseconds.

    Args:
        index: The grid index.
        offset_us: The time of entry 0 of the grid.
        period: The microseconds one entry spans.
        table: The runs table of the channel, or None for a grid without
            runs. An entry inside a run starts at the time that the table
            gives, and any other entry starts at its grid time.
    """
    if table is not None:
        row = _run_of(table, index)
        if row >= 0 and index < table[row, 0] + table[row, 1]:
            return int(table[row, 2]) + (index - int(table[row, 0])) * period
    return offset_us + index * period


def run_time_axis(
    first: int,
    count: int,
    offset_us: int,
    period_us: float,
    table: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    """Return the start time in microseconds of count samples from first.

    A sample inside a run starts at the time that the table gives, and any
    other sample starts at its grid time.

    Args:
        first: The index of the first sample.
        count: The number of samples.
        offset_us: The time of sample 0 of the grid.
        period_us: The microseconds one sample spans.
        table: The runs table of the channel.
    """
    indices = np.arange(first, first + count, dtype=np.int64)
    rows = np.maximum(
        np.searchsorted(table[:, 0], indices, side="right") - 1, 0
    )
    firsts = table[rows, 0]
    inside = indices < firsts + table[rows, 1]
    exact = table[rows, 2] + (indices - firsts).astype(np.float64) * period_us
    grid = offset_us + indices.astype(np.float64) * period_us
    return np.where(inside, exact, grid)


def time_axis(
    first: int, count: int, offset_us: int, period_us: float
) -> npt.NDArray[np.float64]:
    """Return the start time in microseconds of count entries from index first.

    Each time derives from its own index and carries no rounding error
    accumulated along the axis.

    Args:
        first: The index of the first entry.
        count: The number of entries.
        offset_us: The time of entry 0 of the grid.
        period_us: The microseconds one entry spans.
    """
    indices = np.arange(first, first + count, dtype=np.float64)
    return offset_us + indices * period_us


def canonical_number(name: str) -> int | None:
    """Return the number that a node name spells, or None.

    A name spells a number when it holds only ASCII decimal digits and has no
    leading zero. The name 0 spells 0, and 01 spells nothing.
    """
    if name.isdecimal() and name == str(int(name)):
        return int(name)
    return None


def check_level_numbers(
    where: str, noun: str, names: Iterable[str]
) -> list[int]:
    """Return the level numbers that names spell, in ascending order.

    A level number is a number of at least 1 that canonical_number reads
    from a name. The result leaves out every other name.

    Args:
        where: The location, which begins the error message.
        noun: The name of a level in the error message.
        names: The node names to read.

    Raises:
        ValueError: Unless the numbers run from 1 with no gap.
    """
    numbers = sorted(
        number
        for number in map(canonical_number, names)
        if number is not None and number >= 1
    )
    if numbers != list(range(1, len(numbers) + 1)):
        listed = ", ".join(str(number) for number in numbers)
        raise ValueError(
            f"{where}: {noun} must be numbered from 1 with no gap, got {listed}"
        )
    return numbers


def check_level_arrays(
    where: str, noun: str, member: str, names: Iterable[str]
) -> None:
    """Raise ValueError if one of the array names spells a level number.

    A level is a group that holds the array member.

    Args:
        where: The location, which begins the error message.
        noun: The name of a level in the error message.
        member: The name of the array a level holds.
        names: The paths of the arrays below a channel group.

    Raises:
        ValueError: If a name spells a level number.
    """
    for name in names:
        number = canonical_number(name)
        if number is not None and number >= 1:
            raise ValueError(
                f"{where}: {noun} {number} is an array; a {noun} must be a "
                f"group that holds {member}"
            )


def array_error(
    where: str, name: str, expectation: str, array: ZarrArray | None
) -> ValueError:
    """Return the error for a missing array or one of the wrong layout.

    Args:
        where: The location, which begins the message.
        name: The name of the array, which follows where.
        expectation: The phrase that names a valid array.
        array: The array found, or None when it is missing.
    """
    if array is None:
        return ValueError(
            f"{where}: {name} is missing; it must be {expectation}"
        )
    return ValueError(
        f"{where}: {name} must be {expectation}, got shape {array.shape} "
        f"and dtype {array.dtype}"
    )


def check_label_range(
    where: str,
    labels: npt.NDArray[Any],
    limit: int,
    limit_phrase: str | None = None,
) -> None:
    """Raise ValueError unless every label is at least 0 and below limit.

    Args:
        where: The location, which begins the error message.
        labels: The labels to check.
        limit: The exclusive upper bound.
        limit_phrase: Text that follows limit in the error message.

    Raises:
        ValueError: If a label is below 0 or at least limit.
    """
    if not labels.size:
        return
    smallest = int(labels.min())
    largest = int(labels.max())
    if smallest >= 0 and largest < limit:
        return
    value = smallest if smallest < 0 else largest
    bound = f"{limit}" if limit_phrase is None else f"{limit}, {limit_phrase}"
    raise ValueError(
        f"{where}: labels must be at least 0 and below {bound}, got {value}"
    )


def check_not_string(
    names: object, argument: str, noun: str = "column names"
) -> None:
    """Raise TypeError if names is a string.

    Args:
        names: The value of argument, which takes a collection of names.
        argument: The name of the argument in the error message.
        noun: The kind of name in the error message.

    Raises:
        TypeError: If names is a string.
    """
    if isinstance(names, str):
        raise TypeError(
            f"{argument} must be a collection of {noun}, got the string "
            f"{names!r}"
        )


def select_level(counts: Mapping[int, int], bins: int) -> int:
    """Return the coarsest level whose window holds at least bins entries.

    Args:
        counts: A level number mapped to the entries a window holds at that
            level.
        bins: The entries a window must hold.

    Returns:
        The level number, or 0 when no level holds enough.

    Raises:
        ValueError: If bins is below 1.
    """
    if bins < 1:
        raise ValueError(f"bins must be at least 1, got {bins!r}")
    for level in sorted(counts, reverse=True):
        if counts[level] >= bins:
            return level
    return 0


def merge_spans(
    populated: npt.NDArray[np.bool_],
    first: int,
    offset_us: int,
    period: Fraction,
    low_us: int,
    high_us: int,
    gap_us: int,
    table: npt.NDArray[np.int64] | None = None,
) -> list[tuple[int, int]]:
    """Return the time spans that the populated entries of a grid cover.

    Each run of populated entries becomes one half-open span, widened outward
    to whole microseconds and clamped to [low_us, high_us). The result omits a
    span that clamps to nothing. A span that starts at most gap_us after the
    previous span ends extends the previous span.

    Args:
        populated: Marks of grid entries. Element i marks entry first + i.
        first: The grid index of populated[0].
        offset_us: The time of entry 0 of the grid.
        period: The microseconds one entry spans.
        low_us: The inclusive lower clamp.
        high_us: The exclusive upper clamp.
        gap_us: The widest gap that merges two spans.
        table: The runs table of the channel when the grid is its raw
            samples, so that a span starts and ends at exact sample times.
            None puts every entry at its grid time.
    """
    padded = np.concatenate(([False], populated, [False])).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    spans: list[tuple[int, int]] = []
    for run_start, run_stop in zip(
        edges[0::2].tolist(), edges[1::2].tolist(), strict=True
    ):
        start = max(
            low_us,
            math.floor(entry_time(first + run_start, offset_us, period, table)),
        )
        end = min(
            high_us,
            math.ceil(
                entry_time(first + run_stop - 1, offset_us, period, table)
                + period
            ),
        )
        if start >= end:
            continue
        if spans and start - spans[-1][1] <= gap_us:
            spans[-1] = (spans[-1][0], max(spans[-1][1], end))
        else:
            spans.append((start, end))
    return spans
