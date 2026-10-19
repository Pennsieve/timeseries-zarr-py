"""Records and window reads for the continuous channels of a bundle.

Every time argument is an integer count of microseconds from the bundle
origin. A window is half-open, and a read widens it outward to whole samples
or whole bins.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, SupportsIndex, overload

import numpy as np
import numpy.typing as npt

from timeseries_zarr.attrs import parse_continuous_attrs, parse_level_attrs
from timeseries_zarr.constants import (
    COUNT_KEY,
    DECIMATION_FACTOR,
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


@dataclass(frozen=True, slots=True, eq=False)
class Samples:
    """Raw samples of a continuous channel in a window.

    values[i] is sample first + i, at offset_us + (first + i) * period_us
    microseconds from the bundle origin. A NaN value is a gap.
    """

    offset_us: int
    first: int
    period_us: float
    values: npt.NDArray[np.floating[Any]]

    @property
    def time_us(self) -> npt.NDArray[np.float64]:
        """Time of each sample in microseconds from the bundle origin."""
        return time_axis(self.first, len(self), self.offset_us, self.period_us)

    @property
    def time_s(self) -> npt.NDArray[np.float64]:
        """Time of each sample in seconds from the bundle origin."""
        return self.time_us / MICROSECONDS_PER_SECOND

    def __len__(self) -> int:
        """Return the number of samples."""
        return int(self.values.shape[0])

    def columns(self) -> Columns:
        """Return time_us and values keyed by those names."""
        return {TIME_COLUMN: self.time_us, VALUES_COLUMN: self.values}

    def to_series(self, *, name: str | None = None) -> "pd.Series":
        """Return the samples as a pandas Series indexed by time_us."""
        return samples_to_series(self, name=name)

    def to_xarray(
        self, *, name: str | None = None, unit: str | None = None
    ) -> "xr.DataArray":
        """Return the samples as an xarray DataArray along time_us."""
        return samples_to_xarray(self, name=name, unit=unit)


@dataclass(frozen=True, slots=True, eq=False)
class Bins:
    """Statistics of the bins of one level in a window.

    Entry i describes bin first + i, which starts at offset_us + (first + i)
    * period_us microseconds from the bundle origin. min and max are the two
    columns of minmax. mean, var, and count are None when the level does not
    hold them. A bin with a non-finite sample has NaN statistics, and count
    holds the number of its finite samples.
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
        """Start time of each bin in microseconds from the bundle origin."""
        return time_axis(self.first, len(self), self.offset_us, self.period_us)

    @property
    def time_s(self) -> npt.NDArray[np.float64]:
        """Start time of each bin in seconds from the bundle origin."""
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

    period_us is the bin width in microseconds that the level group states.
    Each bin folds samples_per_bin raw samples, and the last bin may fold
    fewer. minmax is a zarr array of shape (bins, 2). mean, var, and count
    are rank-1 zarr arrays, or None when the level does not hold them.
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

    key is the name of the channel group. rate_hz is the sample rate in
    hertz. offset_us is the time of sample 0 in microseconds from the bundle
    origin. id and provenance are None when the group omits them. raw is the
    zarr array of samples. levels maps each level number to its Level in
    level order.
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
        """Time of sample 0 in microseconds from the bundle origin."""
        return self.offset_us

    @property
    def end_us(self) -> int:
        """End of the last sample in microseconds from the origin, rounded up.

        A window from start_us to end_us holds every sample. A channel with no
        samples has end_us equal to start_us.
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
    ) -> Samples: ...

    @overload
    def read(
        self,
        start_us: SupportsIndex | None = None,
        end_us: SupportsIndex | None = None,
        *,
        level: SupportsIndex | None = None,
    ) -> Samples | Bins: ...

    def read(
        self,
        start_us: SupportsIndex | None = None,
        end_us: SupportsIndex | None = None,
        *,
        level: SupportsIndex | None = None,
    ) -> Samples | Bins:
        """Return the raw samples or the bins of a level in a window.

        The window is [start_us, end_us), and a None bound leaves that side
        open. Level 0 and None read the raw samples. The result holds every
        sample or bin that overlaps the window. Raises TypeError if a bound
        or level is not an integer. Raises ValueError if end_us is before
        start_us or the channel has no such level.
        """
        number = self._level_number(level)
        start, end = window_bounds(start_us, end_us)
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
            period_us=level_period_us(self.rate_hz, number),
            min=minmax[:, 0],
            max=minmax[:, 1],
            mean=_read_member(chosen.mean, first, stop),
            var=_read_member(chosen.var, first, stop),
            count=_read_member(chosen.count, first, stop),
        )

    def level_for(
        self,
        start_us: SupportsIndex | None = None,
        end_us: SupportsIndex | None = None,
        *,
        bins: SupportsIndex,
    ) -> int:
        """Return the coarsest level at which a window holds bins bins.

        The window is [start_us, end_us), and each count is the length of
        the read at that level. The result is 0, for the raw samples, when
        no level holds enough. Raises TypeError if a bound or bins is not an
        integer. Raises ValueError if end_us is before start_us or bins is
        below 1.
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

        level None reads the coarsest level, or the raw samples when the
        channel has no levels. Level 0 reads the raw samples. A bin holds
        data when its count is above 0, or on a level without count when a
        minmax value is finite. A sample holds data when it is finite. Each
        span runs from the floor of its first bin start to the ceiling of
        its last bin end, in microseconds. It stays within the window and
        within the start_us and end_us of the channel. A span that starts at
        most gap_us after the previous span ends joins it.

        Raises TypeError if a bound, level, or gap_us is not an integer.
        Raises ValueError if end_us is before start_us, gap_us is below 0,
        or the channel has no such level.
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

        Raises TypeError if level is not an integer or None. Raises
        ValueError if the channel has no such level.
        """
        if level is None:
            return 0
        number = integer_argument(level, "level", "an integer or None")
        if number != 0 and number not in self.levels:
            raise ValueError(self._level_choices())
        return number

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
    array: ZarrArray | None, first: int, stop: int
) -> npt.NDArray[Any] | None:
    """Return rows first through stop - 1 of a member, or None without one."""
    return None if array is None else read_rows(array, first, stop)


def _gap_us(gap_us: SupportsIndex) -> int:
    """Return gap_us as an int.

    Raises TypeError if gap_us is not an integer and ValueError if it is
    below 0.
    """
    gap = integer_argument(gap_us, "gap_us", "an integer count of microseconds")
    if gap < 0:
        raise ValueError(f"gap_us must be at least 0, got {gap}")
    return gap


def _parse_raw(where: str, arrays: Mapping[str, ZarrArray]) -> ZarrArray:
    """Return the raw array of a channel.

    Raises ValueError unless raw is a rank-1 floating-point array.
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

    Raises ValueError unless the member has the given shape and an integer
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

    Raises ValueError if period_us does not match rate_hz, if minmax is
    missing, or if a member has the wrong shape or dtype kind.
    """
    where = f"channel {key} level {number}"
    period_us = parse_level_attrs(group.attrs, where)
    expected = level_period_us(rate_hz, number)
    if not math.isclose(period_us, expected, rel_tol=LEVEL_PERIOD_TOLERANCE):
        raise ValueError(
            f"{where}: {TSZ_KEY}.period_us must be {expected!r} to match "
            f"rate_hz {rate_hz!r}, got {period_us!r}"
        )
    bins = level_bins(num_samples, number)
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
        samples_per_bin=DECIMATION_FACTOR**number,
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

    Raises ValueError if a child array has a level name or if the levels
    are not numbered from 1 without a gap.
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

    attributes are the attributes of the channel group. groups and arrays
    hold its descendants keyed by the path below the group, such as 1 and
    1/minmax. The parser ignores unknown nodes and unknown tsz keys.

    Raises ValueError naming the channel, and the level at fault. A missing
    required key or array raises, as does an array of the wrong shape or
    dtype kind. So does a gap in the level numbers, a level stored as an
    array, or a level whose period_us does not match rate_hz.
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
