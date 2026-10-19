"""Window arithmetic over the time grids of a bundle.

A grid holds equal-width entries. The raw samples of a channel form one grid,
and so do the bins of each level and of each count level. Entry i of a grid
starts at offset_us + i * period microseconds from the bundle origin. Index
arithmetic is exact in Fraction, and a computed time axis is float64. The
module also checks the integer arguments of a read and parses the decimal
names of channel groups and level groups. It checks a label range and a
column name argument, and builds the error for an array of the wrong layout.
"""

import math
import operator
from collections.abc import Iterable, Mapping
from fractions import Fraction
from typing import Any, SupportsIndex, cast

import numpy as np
import numpy.typing as npt

from timeseries_zarr.constants import MICROSECONDS_PER_SECOND
from timeseries_zarr.zarr_io import ZarrArray


def bin_period(rate_hz: float, samples_per_bin: int) -> Fraction:
    """Return the exact microseconds that one entry of a grid spans.

    samples_per_bin is 1 for raw samples and 4^k for level k. Raises
    ValueError if rate_hz is not positive and finite or samples_per_bin is
    below 1.
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


def duration_us(count: int, rate_hz: float) -> int:
    """Return the microseconds that count samples at rate_hz span, rounded up.

    A window of this length from the first sample holds every sample. Raises
    ValueError if rate_hz is not positive and finite.
    """
    return math.ceil(count * bin_period(rate_hz, 1))


def integer_argument(
    value: object, name: str, expectation: str = "an integer"
) -> int:
    """Return value as an int.

    name and expectation begin the error message. Raises TypeError if value
    is not an integer.
    """
    try:
        return operator.index(cast("SupportsIndex", value))
    except TypeError:
        raise TypeError(
            f"{name} must be {expectation}, got {value!r}"
        ) from None


def _window_bound(value: SupportsIndex | None, name: str) -> int | None:
    """Return value as an int, or None for an open bound.

    Raises TypeError if value is not an integer.
    """
    if value is None:
        return None
    return integer_argument(value, name, "an integer count of microseconds")


def window_bounds(
    start_us: SupportsIndex | None, end_us: SupportsIndex | None
) -> tuple[int | None, int | None]:
    """Return the bounds of a window as ints, None for an open side.

    Raises TypeError if a bound is not an integer and ValueError if end_us
    is before start_us.
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

    The window is [start_us, end_us) in microseconds from the bundle origin,
    and a None bound leaves that side open. The grid holds length entries of
    period microseconds from offset_us. Every entry that overlaps the window
    is in the range, and an entry that starts at end_us is not. Equal bounds
    and a window outside the grid give an empty range. Raises TypeError if a
    bound is not an integer and ValueError if end_us is before start_us.
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


def time_axis(
    first: int, count: int, offset_us: int, period_us: float
) -> npt.NDArray[np.float64]:
    """Return the start time in microseconds of count entries from index first.

    Each time derives from its own index and carries no rounding error
    accumulated along the axis.
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
    from a name. The result leaves out every other name. where and noun
    begin the error message. Raises ValueError unless the numbers run from 1
    with no gap.
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

    names are the paths of the arrays below a channel group. A level is a
    group that holds the array member. where and noun begin the error
    message.
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

    where and name begin the message, and expectation is the phrase that
    names a valid array.
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

    where begins the error message, and limit_phrase follows limit in it.
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


def check_not_string(names: object, argument: str) -> None:
    """Raise TypeError if names is a string.

    names is the value of argument, which takes a collection of column names.
    """
    if isinstance(names, str):
        raise TypeError(
            f"{argument} must be a collection of column names, got the "
            f"string {names!r}"
        )


def select_level(counts: Mapping[int, int], bins: int) -> int:
    """Return the coarsest level whose window holds at least bins entries.

    counts maps a level number to the entries a window holds at that level.
    The result is 0 when no level holds enough. Raises ValueError if bins is
    below 1.
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
) -> list[tuple[int, int]]:
    """Return the time spans that the populated entries of a grid cover.

    populated[i] marks grid entry first + i. Each run of populated entries
    becomes one half-open span, widened outward to whole microseconds and
    clamped to [low_us, high_us). The result omits a span that clamps to
    nothing. A span that starts at most gap_us after the previous span ends
    extends the previous span.
    """
    padded = np.concatenate(([False], populated, [False])).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    spans: list[tuple[int, int]] = []
    for run_start, run_stop in zip(
        edges[0::2].tolist(), edges[1::2].tolist(), strict=True
    ):
        start = max(
            low_us, offset_us + math.floor((first + run_start) * period)
        )
        end = min(high_us, offset_us + math.ceil((first + run_stop) * period))
        if start >= end:
            continue
        if spans and start - spans[-1][1] <= gap_us:
            spans[-1] = (spans[-1][0], max(spans[-1][1], end))
        else:
            spans.append((start, end))
    return spans
