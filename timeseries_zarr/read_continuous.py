"""Records and window reads for the continuous channels of a bundle.

Every time argument is an integer count of microseconds from the reference
time. A window is half-open, and a read widens it outward to whole samples
or whole bins.
"""

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal, SupportsIndex, overload

import numpy as np
import numpy.typing as npt

from timeseries_zarr.attrs import parse_continuous_attrs, parse_level_attrs
from timeseries_zarr.constants import (
    COUNT_KEY,
    DEFAULT_DECIMATION_FACTOR,
    ENVELOPE_PAIR_SIZE,
    LEVEL_PERIOD_TOLERANCE,
    MEAN_KEY,
    MICROSECONDS_PER_SECOND,
    MINMAX_KEY,
    RAW_KEY,
    TSZ_KEY,
    VAR_KEY,
)
from timeseries_zarr.convert import (
    TIME_COLUMN,
    VALUES_COLUMN,
    Columns,
    bins_to_dataframe,
    bins_to_xarray,
    samples_to_series,
    samples_to_xarray,
)
from timeseries_zarr.planning import (
    level_bins,
    level_period_us,
    sample_period_us,
)
from timeseries_zarr.windows import (
    array_error,
    bin_period,
    check_level_arrays,
    check_level_numbers,
    check_not_string,
    duration_us,
    index_range,
    integer_argument,
    merge_spans,
    select_level,
    time_axis,
    window_bounds,
)
from timeseries_zarr.zarr_io import ZarrArray, ZarrGroup, read_rows

if TYPE_CHECKING:
    import pandas as pd
    import xarray as xr

_RAW_EXPECTATION = "a rank-1 floating-point array"
_LEVEL_MEMBERS: Final = (MINMAX_KEY, MEAN_KEY, VAR_KEY, COUNT_KEY)


@dataclass(frozen=True, slots=True, eq=False)
class Samples:
    """Raw samples of a continuous channel in a window.

    Sample first + i is values[i], at offset_us + (first + i) * period_us
    microseconds from the reference time. A NaN value is a gap.

    Attributes:
        offset_us: The time of sample 0 of the channel in microseconds.
        first: The index of the first sample in the window.
        period_us: The spacing of the samples in microseconds.
        values: The sample values.
    """

    offset_us: int
    first: int
    period_us: float
    values: npt.NDArray[np.floating[Any]]

    @property
    def time_us(self) -> npt.NDArray[np.float64]:
        """Time of each sample in microseconds from the reference time."""
        return time_axis(self.first, len(self), self.offset_us, self.period_us)

    @property
    def time_s(self) -> npt.NDArray[np.float64]:
        """Time of each sample in seconds from the reference time."""
        return self.time_us / MICROSECONDS_PER_SECOND

    def __len__(self) -> int:
        """Return the number of samples."""
        return int(self.values.shape[0])

    def columns(self) -> Columns:
        """Return time_us and values keyed by those names."""
        return {TIME_COLUMN: self.time_us, VALUES_COLUMN: self.values}

    def to_series(self, *, name: str | None = None) -> "pd.Series":
        """Return the samples as a pandas Series indexed by time_us.

        Args:
            name: The name of the series.
        """
        return samples_to_series(self, name=name)

    def to_xarray(
        self, *, name: str | None = None, unit: str | None = None
    ) -> "xr.DataArray":
        """Return the samples as an xarray DataArray along time_us.

        Args:
            name: The name of the array.
            unit: The unit to record on the array.
        """
        return samples_to_xarray(self, name=name, unit=unit)


@dataclass(frozen=True, slots=True, eq=False)
class Bins:
    """Statistics of the bins of one level in a window.

    Entry i describes bin first + i, which starts at offset_us + (first + i)
    * period_us microseconds from the reference time. A bin with a non-finite
    sample has NaN statistics, and count holds the number of its finite
    samples.

    Attributes:
        level: The level number.
        offset_us: The time of sample 0 of the channel in microseconds.
        first: The index of the first bin in the window.
        period_us: The bin width in microseconds.
        min: The minimum of each bin, the first column of minmax.
        max: The maximum of each bin, the second column of minmax.
        mean: The mean of each bin. None when the level does not hold it or
            the read left it out.
        var: The variance of each bin. None when the level does not hold it
            or the read left it out.
        count: The count of finite samples in each bin. None when the level
            does not hold it or the read left it out.
    """

    level: int
    offset_us: int
    first: int
    period_us: float
    min: npt.NDArray[np.floating[Any]]
    max: npt.NDArray[np.floating[Any]]
    mean: npt.NDArray[np.floating[Any]] | None
    var: npt.NDArray[np.floating[Any]] | None
    count: npt.NDArray[np.integer[Any]] | None

    @property
    def time_us(self) -> npt.NDArray[np.float64]:
        """Start time of each bin in microseconds from the reference time."""
        return time_axis(self.first, len(self), self.offset_us, self.period_us)

    @property
    def time_s(self) -> npt.NDArray[np.float64]:
        """Start time of each bin in seconds from the reference time."""
        return self.time_us / MICROSECONDS_PER_SECOND

    def __len__(self) -> int:
        """Return the number of bins."""
        return int(self.min.shape[0])

    def columns(self) -> Columns:
        """Return time_us, min, and max, then each other statistic held.

        mean, var, and count follow in that order when the bins hold them.
        """
        columns: Columns = {
            TIME_COLUMN: self.time_us,
            "min": self.min,
            "max": self.max,
        }
        optional = {
            MEAN_KEY: self.mean,
            VAR_KEY: self.var,
            COUNT_KEY: self.count,
        }
        for name, values in optional.items():
            if values is not None:
                columns[name] = values
        return columns

    def to_dataframe(self) -> "pd.DataFrame":
        """Return the bins as a pandas DataFrame indexed by time_us."""
        return bins_to_dataframe(self)

    def to_xarray(self) -> "xr.Dataset":
        """Return the bins as an xarray Dataset along time_us."""
        return bins_to_xarray(self)


@dataclass(frozen=True, slots=True, eq=False)
class Level:
    """One level of a continuous channel.

    Attributes:
        number: The level number, counted from 1.
        period_us: The bin width in microseconds that the level group states.
        samples_per_bin: The raw samples that each bin folds. The last bin
            may fold fewer.
        minmax: The zarr array of shape (bins, 2).
        mean: A rank-1 zarr array, or None when the level does not hold it.
        var: A rank-1 zarr array, or None when the level does not hold it.
        count: A rank-1 zarr array, or None when the level does not hold it.
    """

    number: int
    period_us: float
    samples_per_bin: int
    minmax: ZarrArray
    mean: ZarrArray | None
    var: ZarrArray | None
    count: ZarrArray | None

    @property
    def bins(self) -> int:
        """Number of bins in the level."""
        return self.minmax.shape[0]


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class ContinuousChannel:
    """A continuous channel of a bundle.

    Attributes:
        key: The name of the channel group.
        name: The channel name.
        id: The channel id. None when the group omits it.
        unit: The unit of the samples.
        rate_hz: The sample rate in hertz.
        offset_us: The time of sample 0 in microseconds from the reference
            time.
        provenance: The provenance object. None when the group omits it.
        raw: The zarr array of samples.
        levels: Maps each level number to its Level, in level order.
    """

    key: int
    name: str
    id: str | None
    unit: str
    rate_hz: float
    offset_us: int
    provenance: Mapping[str, object] | None
    raw: ZarrArray
    levels: Mapping[int, Level]

    @property
    def kind(self) -> Literal["continuous"]:
        """Channel kind, which is continuous."""
        return "continuous"

    @property
    def num_samples(self) -> int:
        """Number of raw samples."""
        return self.raw.shape[0]

    @property
    def period_us(self) -> float:
        """Spacing of the raw samples in microseconds."""
        return sample_period_us(self.rate_hz)

    @property
    def start_us(self) -> int:
        """Time of sample 0 in microseconds from the reference time."""
        return self.offset_us

    @property
    def end_us(self) -> int:
        """End of the last sample in microseconds from the reference time.

        The end rounds up to a whole microsecond. A window from start_us to
        end_us holds every sample. A channel with no samples has end_us equal
        to start_us.
        """
        return self.offset_us + duration_us(self.num_samples, self.rate_hz)

    def __repr__(self) -> str:
        """Return the key, name, sample rate, sample count, and level count."""
        return (
            f"ContinuousChannel(key={self.key}, name={self.name!r}, "
            f"rate_hz={self.rate_hz!r}, num_samples={self.num_samples}, "
            f"levels={len(self.levels)})"
        )

    @overload
    def read(
        self,
        start_us: SupportsIndex | None = None,
        end_us: SupportsIndex | None = None,
        *,
        level: None = None,
        members: Iterable[str] | None = None,
    ) -> Samples: ...

    @overload
    def read(
        self,
        start_us: SupportsIndex | None = None,
        end_us: SupportsIndex | None = None,
        *,
        level: SupportsIndex | None = None,
        members: Iterable[str] | None = None,
    ) -> Samples | Bins: ...

    def read(
        self,
        start_us: SupportsIndex | None = None,
        end_us: SupportsIndex | None = None,
        *,
        level: SupportsIndex | None = None,
        members: Iterable[str] | None = None,
    ) -> Samples | Bins:
        """Return the raw samples or the bins of a level in a window.

        The result holds every sample or bin that overlaps the window.

        Args:
            start_us: The inclusive start of the window in microseconds from
                the reference time. None leaves the start open.
            end_us: The exclusive end of the window. None leaves the end
                open.
            level: The level to read. 0 and None read the raw samples.
            members: The members of the level to read from minmax, mean,
                var, and count. None reads every member the level holds. The
                read always holds min and max.

        Returns:
            Samples for level 0 or None, otherwise Bins.

        Raises:
            TypeError: If a bound or level is not an integer, or if members
                is a string.
            ValueError: If end_us is before start_us, the channel has no such
                level, or members comes with a raw read or names a member
                the level lacks.
        """
        number = self._level_number(level)
        start, end = window_bounds(start_us, end_us)
        selected = self._selected_members(number, members)
        first, stop = self._index_range(start, end, number)
        if number == 0:
            return Samples(
                offset_us=self.offset_us,
                first=first,
                period_us=self.period_us,
                values=read_rows(self.raw, first, stop),
            )
        chosen = self.levels[number]
        minmax = read_rows(chosen.minmax, first, stop)
        return Bins(
            level=number,
            offset_us=self.offset_us,
            first=first,
            period_us=level_period_us(self.rate_hz, chosen.samples_per_bin),
            min=minmax[:, 0],
            max=minmax[:, 1],
            mean=_read_member(chosen.mean, MEAN_KEY, selected, first, stop),
            var=_read_member(chosen.var, VAR_KEY, selected, first, stop),
            count=_read_member(chosen.count, COUNT_KEY, selected, first, stop),
        )

    def level_for(
        self,
        start_us: SupportsIndex | None = None,
        end_us: SupportsIndex | None = None,
        *,
        bins: SupportsIndex,
    ) -> int:
        """Return the coarsest level at which a window holds bins bins.

        Each count is the length of the read at that level.

        Args:
            start_us: The inclusive start of the window in microseconds from
                the reference time. None leaves the start open.
            end_us: The exclusive end of the window. None leaves the end
                open.
            bins: The number of entries wanted, at least 1.

        Returns:
            The level number, or 0 for the raw samples when no level holds
            enough.

        Raises:
            TypeError: If a bound or bins is not an integer.
            ValueError: If end_us is before start_us or bins is below 1.
        """
        wanted = integer_argument(bins, "bins")
        start, end = window_bounds(start_us, end_us)
        counts: dict[int, int] = {}
        for number in self.levels:
            first, stop = self._index_range(start, end, number)
            counts[number] = stop - first
        return select_level(counts, wanted)

    def spans(
        self,
        start_us: SupportsIndex | None = None,
        end_us: SupportsIndex | None = None,
        *,
        level: SupportsIndex | None = None,
        gap_us: SupportsIndex = 0,
    ) -> list[tuple[int, int]]:
        """Return the half-open time spans in a window that hold data.

        A bin holds data when its count is above 0, or on a level without
        count when a minmax value is finite. A sample holds data when it is
        finite. Each span runs from the floor of its first bin start to the
        ceiling of its last bin end, in microseconds. It stays within the
        window and within the start_us and end_us of the channel.

        Args:
            start_us: The inclusive start of the window in microseconds from
                the reference time. None leaves the start open.
            end_us: The exclusive end of the window. None leaves the end
                open.
            level: The level to read. None reads the coarsest level, or the
                raw samples when the channel has no levels. 0 reads the raw
                samples.
            gap_us: A span that starts at most this many microseconds after
                the previous span ends joins it.

        Returns:
            The (start, end) pairs in time order.

        Raises:
            TypeError: If a bound, level, or gap_us is not an integer.
            ValueError: If end_us is before start_us, gap_us is below 0, or
                the channel has no such level.
        """
        gap = _gap_us(gap_us)
        if level is None:
            number = max(self.levels, default=0)
        else:
            number = self._level_number(level)
        period, length = self._grid(number)
        start, end = window_bounds(start_us, end_us)
        first, stop = index_range(start, end, self.offset_us, period, length)
        low_us = self.start_us if start is None else max(start, self.start_us)
        high_us = self.end_us if end is None else min(end, self.end_us)
        return merge_spans(
            self._populated(number, first, stop),
            first,
            self.offset_us,
            period,
            low_us,
            high_us,
            gap,
        )

    def _level_number(self, level: SupportsIndex | None) -> int:
        """Return the number of the level to read, 0 for the raw samples.

        Raises:
            TypeError: If level is not an integer or None.
            ValueError: If the channel has no such level.
        """
        if level is None:
            return 0
        number = integer_argument(level, "level", "an integer or None")
        if number != 0 and number not in self.levels:
            raise ValueError(self._level_choices())
        return number

    def _selected_members(
        self, number: int, members: Iterable[str] | None
    ) -> set[str]:
        """Return the members of level number that a read returns.

        The result always holds minmax, and it is empty for a raw read.
        Raises TypeError if members is a string. Raises ValueError if
        members comes with a raw read or names a member the level lacks.
        """
        if number == 0:
            if members is not None:
                raise ValueError(
                    "level 0 holds raw samples and takes members None, "
                    f"got {members!r}"
                )
            return set()
        held = tuple(member_arrays(self.levels[number]))
        if members is None:
            return set(held)
        check_not_string(members, "members", "member names")
        selected = {MINMAX_KEY}
        for name in members:
            if name not in held:
                raise ValueError(
                    f"channel {self.key} level {number} has members {held!r}; "
                    f"members must name only these, got {name!r}"
                )
            selected.add(name)
        return selected

    def _level_choices(self) -> str:
        """Return the error message for a level the channel does not hold."""
        where = f"channel {self.key}"
        raw = "0 or None for raw samples"
        count = len(self.levels)
        if count == 0:
            return f"{where} has no levels; level must be {raw}"
        if count == 1:
            return f"{where} has level 1; level must be 1, or {raw}"
        return (
            f"{where} has levels 1 through {count}; level must be one of "
            f"them, or {raw}"
        )

    def _grid(self, number: int) -> tuple[Fraction, int]:
        """Return the exact entry width and the length of a level's grid.

        Level 0 is the grid of raw samples.
        """
        if number == 0:
            return bin_period(self.rate_hz, 1), self.num_samples
        chosen = self.levels[number]
        return bin_period(self.rate_hz, chosen.samples_per_bin), chosen.bins

    def _index_range(
        self, start_us: int | None, end_us: int | None, number: int
    ) -> tuple[int, int]:
        """Return the index range of a window on a level's grid."""
        period, length = self._grid(number)
        return index_range(start_us, end_us, self.offset_us, period, length)

    def _populated(
        self, number: int, first: int, stop: int
    ) -> npt.NDArray[np.bool_]:
        """Return which entries from first to stop hold data on a level."""
        if number == 0:
            return np.isfinite(read_rows(self.raw, first, stop))
        chosen = self.levels[number]
        if chosen.count is not None:
            return read_rows(chosen.count, first, stop) > 0
        envelope = read_rows(chosen.minmax, first, stop)
        return np.isfinite(envelope).any(axis=1)


def _read_member(
    array: ZarrArray | None,
    name: str,
    selected: set[str],
    first: int,
    stop: int,
) -> npt.NDArray[Any] | None:
    """Return rows first through stop - 1 of member name.

    The result is None when the level lacks the member or selected does not
    hold name.
    """
    if array is None or name not in selected:
        return None
    return read_rows(array, first, stop)


def member_arrays(level: Level) -> dict[str, ZarrArray]:
    """Return the members a level holds, keyed by name in spec order."""
    present = dict(
        zip(
            _LEVEL_MEMBERS,
            (level.minmax, level.mean, level.var, level.count),
            strict=True,
        )
    )
    return {name: array for name, array in present.items() if array is not None}


def read_entries(
    channel: ContinuousChannel,
    start_us: int | None,
    end_us: int | None,
    number: int,
    name: str,
) -> tuple[int, float, npt.NDArray[Any]]:
    """Return first, period_us, and the rows of one array in a window.

    number 0 reads the raw samples and ignores name. Otherwise name is the
    member of level number, and minmax returns rows of shape (n, 2). Raises
    ValueError if the channel lacks the level or the level lacks name.
    """
    if number != 0 and number not in channel.levels:
        raise ValueError(channel._level_choices())
    first, stop = channel._index_range(start_us, end_us, number)
    if number == 0:
        return first, channel.period_us, read_rows(channel.raw, first, stop)
    held = member_arrays(channel.levels[number])
    array = held.get(name)
    if array is None:
        raise ValueError(
            f"channel {channel.key} level {number} has members "
            f"{tuple(held)!r}; name must be one of these, got {name!r}"
        )
    period_us = level_period_us(
        channel.rate_hz, channel.levels[number].samples_per_bin
    )
    return first, period_us, read_rows(array, first, stop)


def _gap_us(gap_us: SupportsIndex) -> int:
    """Return gap_us as an int.

    Raises:
        TypeError: If gap_us is not an integer.
        ValueError: If gap_us is below 0.
    """
    gap = integer_argument(gap_us, "gap_us", "an integer count of microseconds")
    if gap < 0:
        raise ValueError(f"gap_us must be at least 0, got {gap}")
    return gap


def _parse_raw(where: str, arrays: Mapping[str, ZarrArray]) -> ZarrArray:
    """Return the raw array of a channel.

    Raises:
        ValueError: Unless raw is a rank-1 floating-point array.
    """
    raw = arrays.get(RAW_KEY)
    if (
        raw is None
        or raw.ndim != 1
        or not np.issubdtype(raw.dtype, np.floating)
    ):
        raise array_error(where, RAW_KEY, _RAW_EXPECTATION, raw)
    return raw


def _parse_member(
    where: str,
    arrays: Mapping[str, ZarrArray],
    number: int,
    name: str,
    shape: tuple[int, ...],
    *,
    integer: bool = False,
) -> ZarrArray | None:
    """Return member name of level number, or None when the level lacks it.

    Raises:
        ValueError: Unless the member has the given shape and an integer
            dtype when integer is true, or a floating-point dtype otherwise.
    """
    array = arrays.get(f"{number}/{name}")
    if array is None:
        return None
    kind = np.integer if integer else np.floating
    if array.shape != shape or not np.issubdtype(array.dtype, kind):
        raise array_error(
            where, name, _member_expectation(shape, integer=integer), array
        )
    return array


def _member_expectation(shape: tuple[int, ...], *, integer: bool) -> str:
    """Return the phrase that names the shape and dtype kind of a member."""
    kind = "an integer" if integer else "a floating-point"
    return f"{kind} array of shape {shape}"


def _parse_level(
    key: int,
    number: int,
    group: ZarrGroup,
    arrays: Mapping[str, ZarrArray],
    rate_hz: float,
    num_samples: int,
) -> Level:
    """Return one level of channel key.

    Raises:
        ValueError: If period_us does not match rate_hz, if minmax is
            missing, or if a member has the wrong shape or dtype kind.
    """
    where = f"channel {key} level {number}"
    period_us = parse_level_attrs(group.attrs, where)
    samples_per_bin = DEFAULT_DECIMATION_FACTOR**number
    expected = level_period_us(rate_hz, samples_per_bin)
    if not math.isclose(period_us, expected, rel_tol=LEVEL_PERIOD_TOLERANCE):
        raise ValueError(
            f"{where}: {TSZ_KEY}.period_us must be {expected!r} to match "
            f"rate_hz {rate_hz!r}, got {period_us!r}"
        )
    bins = level_bins(num_samples, samples_per_bin)
    envelope = (bins, ENVELOPE_PAIR_SIZE)
    column = (bins,)
    minmax = _parse_member(where, arrays, number, MINMAX_KEY, envelope)
    if minmax is None:
        raise array_error(
            where,
            MINMAX_KEY,
            _member_expectation(envelope, integer=False),
            None,
        )
    return Level(
        number=number,
        period_us=period_us,
        samples_per_bin=samples_per_bin,
        minmax=minmax,
        mean=_parse_member(where, arrays, number, MEAN_KEY, column),
        var=_parse_member(where, arrays, number, VAR_KEY, column),
        count=_parse_member(
            where, arrays, number, COUNT_KEY, column, integer=True
        ),
    )


def _level_numbers(
    where: str,
    groups: Mapping[str, ZarrGroup],
    arrays: Mapping[str, ZarrArray],
) -> list[int]:
    """Return the level numbers of a channel in level order.

    Raises:
        ValueError: If a child array has a level name or if the levels are
            not numbered from 1 without a gap.
    """
    check_level_arrays(where, "level", MINMAX_KEY, arrays)
    return check_level_numbers(where, "levels", groups)


def parse_continuous_channel(
    key: int,
    attributes: Mapping[str, object],
    groups: Mapping[str, ZarrGroup],
    arrays: Mapping[str, ZarrArray],
) -> ContinuousChannel:
    """Return the continuous channel that channel group key describes.

    The parser ignores unknown nodes and unknown tsz keys.

    Args:
        key: The name of the channel group.
        attributes: The attributes of the channel group.
        groups: The descendant groups, keyed by the path below the group,
            such as 1.
        arrays: The descendant arrays, keyed by the path below the group,
            such as 1/minmax.

    Raises:
        ValueError: If a required key or array is missing, an array has the
            wrong shape or dtype kind, the level numbers have a gap, a level
            is stored as an array, or a level's period_us does not match
            rate_hz. The message names the channel and the level at fault.
    """
    where = f"channel {key}"
    fields = parse_continuous_attrs(attributes, where)
    numbers = _level_numbers(where, groups, arrays)
    raw = _parse_raw(where, arrays)
    levels = {
        number: _parse_level(
            key,
            number,
            groups[str(number)],
            arrays,
            fields.rate_hz,
            raw.shape[0],
        )
        for number in numbers
    }
    return ContinuousChannel(
        key=key,
        name=fields.name,
        id=fields.id,
        unit=fields.unit,
        rate_hz=fields.rate_hz,
        offset_us=fields.offset_us,
        provenance=fields.provenance,
        raw=raw,
        levels=MappingProxyType(levels),
    )
