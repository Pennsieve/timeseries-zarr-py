"""Windows of several continuous channels stacked on one shared grid.

Every time argument is an integer count of microseconds from the reference
time. Channels stack when they share rate_hz and offset_us. Entry i of
every row then starts at the same time.
"""

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal, SupportsIndex, get_args

import numpy as np
import numpy.typing as npt

from timeseries_zarr.constants import (
    COUNT_KEY,
    MEAN_KEY,
    MICROSECONDS_PER_SECOND,
    MINMAX_KEY,
    VAR_KEY,
)
from timeseries_zarr.convert import (
    TIME_COLUMN,
    Columns,
    stack_to_dataframe,
    stack_to_xarray,
)
from timeseries_zarr.read_continuous import (
    ContinuousChannel,
    member_arrays,
    read_entries,
)
from timeseries_zarr.windows import integer_argument, time_axis, window_bounds

if TYPE_CHECKING:
    import pandas as pd
    import xarray as xr

type BinMember = Literal["min", "max", "mean", "var", "count"]
"""Statistic of a level that a stack reads from each bin.

min and max are the two columns of minmax.
"""

_MEMBERS: Final[tuple[BinMember, ...]] = get_args(BinMember.__value__)
_MEMBER_ARRAYS: Final[dict[BinMember, tuple[str, int | None]]] = {
    "min": (MINMAX_KEY, 0),
    "max": (MINMAX_KEY, 1),
    "mean": (MEAN_KEY, None),
    "var": (VAR_KEY, None),
    "count": (COUNT_KEY, None),
}
"""Array of a level and its column, or None for a rank-1 array, per member."""


@dataclass(frozen=True, slots=True, eq=False)
class Stack:
    """Windows of several continuous channels on one shared grid.

    Entry j of a row is sample or bin first + j, which starts at offset_us +
    (first + j) * period_us microseconds from the reference time. A row holds
    NaN past the end of its channel, or 0 when member is count.

    Attributes:
        keys: The channel keys, one for each row of values.
        names: The channel names, in the order of keys.
        units: The channel units, in the order of keys.
        level: The level of the entries. Level 0 holds raw samples.
        member: The statistic each entry holds. None at level 0.
        offset_us: The time of sample or bin 0, in microseconds from the
            reference time.
        first: The index of the first entry of each row.
        period_us: The duration of one entry in microseconds.
        values: One row for each channel, in the order of keys.
    """

    keys: tuple[int, ...]
    names: tuple[str, ...]
    units: tuple[str, ...]
    level: int
    member: BinMember | None
    offset_us: int
    first: int
    period_us: float
    values: npt.NDArray[Any]

    @property
    def time_us(self) -> npt.NDArray[np.float64]:
        """Start time of each entry in microseconds from the reference time."""
        return time_axis(self.first, len(self), self.offset_us, self.period_us)

    @property
    def time_s(self) -> npt.NDArray[np.float64]:
        """Start time of each entry in seconds from the reference time."""
        return self.time_us / MICROSECONDS_PER_SECOND

    def __len__(self) -> int:
        """Return the number of entries in each row."""
        return int(self.values.shape[1])

    def columns(self) -> Columns:
        """Return time_us and one column per channel.

        The name of each channel column is its channel key as a decimal string.
        """
        columns: Columns = {TIME_COLUMN: self.time_us}
        for key, row in zip(self.keys, self.values, strict=True):
            columns[str(key)] = row
        return columns

    def to_dataframe(self) -> "pd.DataFrame":
        """Return the stack as a pandas DataFrame indexed by time_us."""
        return stack_to_dataframe(self)

    def to_xarray(self) -> "xr.DataArray":
        """Return the stack as an xarray DataArray.

        The dimensions are channel and time_us.
        """
        return stack_to_xarray(self)


def _check_member(number: int, member: BinMember | None) -> None:
    """Raise ValueError unless member suits level number.

    Level 0 takes None, and every other level takes a name in _MEMBERS.

    Args:
        number: The level.
        member: The statistic to read.
    """
    if number == 0:
        if member is not None:
            raise ValueError(
                "level 0 holds raw samples and takes member None, "
                f"got {member!r}"
            )
    elif member not in _MEMBERS:
        listed = ", ".join(repr(name) for name in _MEMBERS)
        raise ValueError(
            f"member must be one of {listed} at level {number}, got {member!r}"
        )


def _check_rate_and_offset(
    channels: Sequence[ContinuousChannel], number: int
) -> None:
    """Raise ValueError if a channel differs from the first in either field."""
    shared = (
        "rate_hz and offset_us"
        if number == 0
        else f"rate_hz, offset_us, and level {number}"
    )
    reference = channels[0]
    for channel in channels:
        for field, value, expected in (
            ("rate_hz", channel.rate_hz, reference.rate_hz),
            ("offset_us", channel.offset_us, reference.offset_us),
        ):
            if value != expected:
                raise ValueError(
                    f"channel {channel.key} has {field} {value!r} and "
                    f"channel {reference.key} has {expected!r}; stack needs "
                    f"channels that share {shared}"
                )


def check_shared_grid(
    channels: Sequence[ContinuousChannel],
    level: int,
    member: BinMember | None,
) -> None:
    """Check that channels share one grid at level and hold member there.

    Level 0 is the grid of raw samples and takes member None. Every other
    level takes a member that BinMember names.

    Args:
        channels: The channels to stack.
        level: The level to read.
        member: The statistic to read at level.

    Raises:
        ValueError: If channels is empty, if a channel differs from the first
            in rate_hz or offset_us, if a channel lacks the level, if member
            does not suit the level, or if a level lacks member.
        TypeError: If level is not an integer.
    """
    if not channels:
        raise ValueError("stack needs at least one channel, got none")
    number = integer_argument(level, "level")
    _check_member(number, member)
    _check_rate_and_offset(channels, number)
    # Only level 0 takes None, and every channel holds raw samples.
    if member is None:
        return
    for channel in channels:
        chosen = channel.levels.get(number)
        if chosen is None:
            raise ValueError(
                f"channel {channel.key} has no level {number}; stack needs "
                "a level that every channel holds"
            )
        name, _ = _MEMBER_ARRAYS[member]
        if name not in member_arrays(chosen):
            raise ValueError(
                f"level {number} of channel {channel.key} has no {member}; "
                "stack needs a member that every channel holds at level "
                f"{number}"
            )


def _check_distinct(channels: Sequence[ContinuousChannel]) -> None:
    """Raise ValueError if two channels share a key."""
    seen: set[int] = set()
    for channel in channels:
        if channel.key in seen:
            raise ValueError(
                f"stack needs distinct channels, got channel {channel.key} "
                "more than once"
            )
        seen.add(channel.key)


def _check_max_workers(max_workers: SupportsIndex) -> int:
    """Return max_workers as an int.

    Raises:
        TypeError: If max_workers is not an integer.
        ValueError: If max_workers is below 1.
    """
    count = integer_argument(max_workers, "max_workers")
    if count < 1:
        raise ValueError(f"max_workers must be at least 1, got {count}")
    return count


def stack_channels(
    channels: Sequence[ContinuousChannel],
    start_us: int | None,
    end_us: int | None,
    *,
    level: int = 0,
    member: BinMember = "mean",
    max_workers: SupportsIndex = 1,
) -> Stack:
    """Return the window of each channel as one row of a Stack.

    The rows span every entry that overlaps the window on any channel. A row
    holds NaN past the end of its channel, or 0 when member is count.
    The result does not depend on max_workers.

    Args:
        channels: The channels to stack, in row order.
        start_us: The start of the window in microseconds from the reference
            time. None leaves that side open.
        end_us: The end of the window, exclusive. None leaves that side open.
        level: The level to read. Level 0 reads the raw samples and ignores
            member.
        member: The statistic to read from the bins of a level of at least 1.
        max_workers: The number of threads that read the channels. 1 reads
            them in order on the calling thread.

    Raises:
        ValueError: If two channels share a key, if check_shared_grid rejects
            the channels, if end_us is before start_us, or if max_workers is
            below 1.
        TypeError: If a bound, level, or max_workers is not an integer.
    """
    _check_distinct(channels)
    number = integer_argument(level, "level")
    chosen = None if number == 0 else member
    check_shared_grid(channels, number, chosen)
    count = _check_max_workers(max_workers)
    start, end = window_bounds(start_us, end_us)
    name, column = _MEMBER_ARRAYS[member]

    def read_one(
        channel: ContinuousChannel,
    ) -> tuple[int, float, npt.NDArray[Any]]:
        first, period_us, rows = read_entries(channel, start, end, number, name)
        if number != 0 and column is not None:
            rows = rows[:, column]
        return first, period_us, rows

    workers = min(count, len(channels))
    if workers == 1:
        results = [read_one(channel) for channel in channels]
    else:
        with ThreadPoolExecutor(workers) as executor:
            results = list(executor.map(read_one, channels))
    rows = [row for _, _, row in results]
    # The channels share one grid and differ only in length. A channel that
    # ends before the window reads nothing, and every other read starts at
    # the shared first.
    first = max(entry_first for entry_first, _, _ in results)
    stop = max(entry_first + row.shape[0] for entry_first, _, row in results)
    dtype = np.result_type(*(row.dtype for row in rows))
    fill = 0 if chosen == COUNT_KEY else np.nan
    values = np.full((len(rows), stop - first), fill, dtype=dtype)
    for index, row in enumerate(rows):
        values[index, : row.shape[0]] = row
    return Stack(
        keys=tuple(channel.key for channel in channels),
        names=tuple(channel.name for channel in channels),
        units=tuple(channel.unit for channel in channels),
        level=number,
        member=chosen,
        offset_us=channels[0].offset_us,
        first=first,
        period_us=results[0][1],
        values=values,
    )
