"""Windows of several continuous channels stacked on one shared grid.

Every time argument is an integer count of microseconds from the reference
time. Channels stack when they share rate_hz and offset_us. Entry i of
every row then starts at the same time.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal, get_args

import numpy as np
import numpy.typing as npt

from timeseries_zarr.constants import COUNT_KEY, MICROSECONDS_PER_SECOND
from timeseries_zarr.convert import (
    TIME_COLUMN,
    Columns,
    stack_to_dataframe,
    stack_to_xarray,
)
from timeseries_zarr.read_continuous import Bins, ContinuousChannel, Samples
from timeseries_zarr.windows import integer_argument, time_axis

if TYPE_CHECKING:
    import pandas as pd
    import xarray as xr

type BinMember = Literal["min", "max", "mean", "var", "count"]
"""Statistic of a level that a stack reads from each bin.

min and max are the two columns of minmax.
"""

_MEMBERS: Final[tuple[BinMember, ...]] = get_args(BinMember.__value__)


@dataclass(frozen=True, slots=True, eq=False)
class Stack:
    """Windows of several continuous channels on one shared grid.

    values has one row per channel in the order of keys. Entry j of a row
    is sample or bin first + j, which starts at offset_us + (first + j) *
    period_us microseconds from the reference time. names and units hold the
    name and unit of each channel in the order of keys. Level 0 holds raw
    samples, and member is then None. A row holds NaN past the end of its
    channel, or 0 when member is count.
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
    level takes a member that BinMember names. Raises ValueError if channels
    is empty, if a channel differs from the first in rate_hz or offset_us,
    or if a channel lacks the level. Also raises ValueError if member does
    not suit the level or a level lacks member. Raises TypeError if level is
    not an integer.
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
        held = {
            "min": chosen.minmax,
            "max": chosen.minmax,
            "mean": chosen.mean,
            "var": chosen.var,
            "count": chosen.count,
        }
        if held[member] is None:
            raise ValueError(
                f"level {number} of channel {channel.key} has no {member}; "
                "stack needs a member that every channel holds at level "
                f"{number}"
            )


def _entries(result: Samples | Bins, member: BinMember) -> npt.NDArray[Any]:
    """Return the samples of a raw read, or the member column of bins."""
    if isinstance(result, Samples):
        return result.values
    # Each name in _MEMBERS is a field of Bins, and check_shared_grid has
    # confirmed that the level holds member.
    entries: npt.NDArray[Any] = getattr(result, member)
    return entries


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


def stack_channels(
    channels: Sequence[ContinuousChannel],
    start_us: int | None,
    end_us: int | None,
    *,
    level: int = 0,
    member: BinMember = "mean",
) -> Stack:
    """Return the window of each channel as one row of a Stack.

    The window is [start_us, end_us), and a None bound leaves that side
    open. Level 0 reads the raw samples and ignores member. A level of at
    least 1 reads member from the bins of that level. The rows span every
    entry that overlaps the window on any channel. A row holds NaN past the
    end of its channel, or 0 when member is count.

    Raises ValueError if two channels share a key, if check_shared_grid
    rejects the channels, or if end_us is before start_us. Raises TypeError
    if a bound or level is not an integer.
    """
    _check_distinct(channels)
    number = integer_argument(level, "level")
    chosen = None if number == 0 else member
    check_shared_grid(channels, number, chosen)
    results = [
        channel.read(start_us, end_us, level=number) for channel in channels
    ]
    rows = [_entries(result, member) for result in results]
    # The channels share one grid and differ only in length. A channel that
    # ends before the window reads nothing, and every other read starts at
    # the shared first.
    first = max(result.first for result in results)
    stop = max(result.first + len(result) for result in results)
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
        period_us=results[0].period_us,
        values=values,
    )
