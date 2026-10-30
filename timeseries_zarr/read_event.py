"""Records and window reads for the event channels of a bundle.

Every time argument is an integer count of microseconds from the reference
time, and a window is half-open.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, SupportsIndex

import numpy as np
import numpy.typing as npt

from timeseries_zarr.attrs import (
    label_count,
    parse_count_level_attrs,
    parse_event_attrs,
    parse_waveform_attrs,
)
from timeseries_zarr.columns import (
    channels_width,
    check_channels_rows,
    decode_channels_rows,
    decode_event_text,
)
from timeseries_zarr.constants import (
    CHANNELS_KEY,
    COUNT_KEY,
    DURATIONS_KEY,
    EVENT_DIMENSION,
    EVENTS_KEY,
    EVENTS_SEARCH_LIMIT_BYTES,
    LABEL_DIMENSION,
    LABELS_KEY,
    MICROSECONDS_PER_SECOND,
    MIN_DECIMATION_FACTOR,
    TEXT_KEY,
    TSZ_KEY,
    VALUES_KEY,
    WAVEFORM_MEAN_KEY,
    WAVEFORMS_KEY,
)
from timeseries_zarr.convert import (
    TIME_COLUMN,
    Columns,
    counts_to_dataframe,
    counts_to_xarray,
    events_to_dataframe,
    waveform_to_dataframe,
    waveform_to_xarray,
)
from timeseries_zarr.windows import (
    array_error,
    check_label_range,
    check_level_arrays,
    check_level_numbers,
    check_not_string,
    index_range,
    integer_argument,
    select_level,
    time_axis,
    window_bounds,
)
from timeseries_zarr.zarr_io import ZarrArray, ZarrGroup, read_int, read_rows

if TYPE_CHECKING:
    import pandas as pd
    import xarray as xr


@dataclass(frozen=True, slots=True)
class _Layout:
    """Rank and dtype kind that an array of an event channel must have."""

    rank: int
    kind: npt.DTypeLike
    kind_name: str


_INTEGER_COLUMN = _Layout(1, np.integer, "integer")
_FLOATING_COLUMN = _Layout(1, np.floating, "floating-point")
_BYTE_ROWS = _Layout(2, np.uint8, "uint8")
_FLOATING_ROWS = _Layout(2, np.floating, "floating-point")
_INTEGER_ROWS = _Layout(2, np.integer, "integer")

_COLUMN_LAYOUTS: Mapping[str, _Layout] = MappingProxyType(
    {
        DURATIONS_KEY: _INTEGER_COLUMN,
        LABELS_KEY: _INTEGER_COLUMN,
        VALUES_KEY: _FLOATING_COLUMN,
        TEXT_KEY: _BYTE_ROWS,
        CHANNELS_KEY: _BYTE_ROWS,
    }
)
"""Layout of each per-event column but waveforms, in spec order."""


def _snippet_time_us(
    event_sample: int, samples: int, period_us: float
) -> npt.NDArray[np.float64]:
    """Return the time of each snippet sample relative to the event time."""
    return time_axis(-event_sample, samples, 0, period_us)


def _object_column(
    items: Sequence[object] | None,
) -> npt.NDArray[np.object_] | None:
    """Return a rank-1 object array of items, or None when items is None."""
    if items is None:
        return None
    column = np.empty(len(items), dtype=np.object_)
    # np.array would read equal-length index lists as a second axis.
    for index, item in enumerate(items):
        column[index] = item
    return column


@dataclass(frozen=True, slots=True, eq=False)
class Waveform:
    """Snippet rows read from a waveforms or waveform_mean array.

    Attributes:
        values: One row per snippet and one column per snippet sample.
        period_us: The snippet sample period in microseconds.
        event_sample: The column that holds the sample recorded at the event
            time.
        rows: What a row stands for, event for waveforms and label for
            waveform_mean.
    """

    values: npt.NDArray[np.floating[Any]]
    period_us: float
    event_sample: int
    rows: Literal["event", "label"]

    @property
    def samples(self) -> int:
        """Number of samples in each snippet."""
        return int(self.values.shape[1])

    @property
    def time_us(self) -> npt.NDArray[np.float64]:
        """Time of each snippet sample in microseconds from the event time."""
        return _snippet_time_us(self.event_sample, self.samples, self.period_us)

    @property
    def time_s(self) -> npt.NDArray[np.float64]:
        """Time of each snippet sample in seconds from the event time."""
        return self.time_us / MICROSECONDS_PER_SECOND

    def __len__(self) -> int:
        """Return the number of samples in each snippet."""
        return self.samples

    def columns(self) -> Columns:
        """Return time_us and one column per snippet row.

        The name of each row column is its row index as a decimal string.
        """
        columns: Columns = {TIME_COLUMN: self.time_us}
        for row, values in enumerate(self.values):
            columns[str(row)] = values
        return columns

    def to_dataframe(self) -> "pd.DataFrame":
        """Return the snippets as a pandas DataFrame indexed by time_us."""
        return waveform_to_dataframe(self)

    def to_xarray(self) -> "xr.DataArray":
        """Return the snippets as an xarray DataArray.

        The first dimension takes its name from rows, and the second is
        time_us.
        """
        return waveform_to_xarray(self)


@dataclass(frozen=True, slots=True, eq=False)
class Snippets:
    """A waveforms or waveform_mean array of an event channel.

    Attributes:
        array: A rank-2 zarr array with one row per snippet and one column
            per snippet sample.
        period_us: The snippet sample period in microseconds.
        event_sample: The column that holds the sample recorded at the event
            time.
        rows: What a row stands for, event for waveforms and label for
            waveform_mean.
    """

    array: ZarrArray
    period_us: float
    event_sample: int
    rows: Literal["event", "label"]

    @property
    def samples(self) -> int:
        """Number of samples in each snippet."""
        return self.array.shape[1]

    @property
    def time_us(self) -> npt.NDArray[np.float64]:
        """Time of each snippet sample in microseconds from the event time."""
        return _snippet_time_us(self.event_sample, self.samples, self.period_us)

    def read(self) -> Waveform:
        """Return every row of the array."""
        return Waveform(
            values=read_rows(self.array, 0, self.array.shape[0]),
            period_us=self.period_us,
            event_sample=self.event_sample,
            rows=self.rows,
        )


@dataclass(frozen=True, slots=True, eq=False)
class CountLevel:
    """One count level of an event channel.

    Bin i starts at offset_us + i * period_us.

    Attributes:
        number: The level number, from 1.
        period_us: The bin width in microseconds.
        offset_us: The time of bin 0 in microseconds from the reference time.
            A multiple of period_us, and may be negative.
        count: A rank-2 zarr array whose entry (i, j) is the number of events
            of label j that start in bin i.
    """

    number: int
    period_us: int
    offset_us: int
    count: ZarrArray

    @property
    def bins(self) -> int:
        """Number of bins in the level."""
        return self.count.shape[0]

    @property
    def num_labels(self) -> int:
        """Number of label columns in count."""
        return self.count.shape[1]


@dataclass(frozen=True, slots=True, eq=False)
class Counts:
    """Event counts of one count level in a window.

    Bin first + i starts at offset_us + (first + i) * period_us microseconds
    from the reference time.

    Attributes:
        level: The count level number.
        first: The index of the first bin in the window.
        period_us: The bin width in microseconds.
        offset_us: The time of bin 0 in microseconds from the reference time.
        count: Entry (i, j) is the number of events of label j that start in
            bin first + i.
        label_names: The name of each label column, or None when the channel
            has no label_names.
    """

    level: int
    first: int
    period_us: float
    offset_us: int
    count: npt.NDArray[np.integer[Any]]
    label_names: tuple[str, ...] | None

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
        return int(self.count.shape[0])

    def columns(self) -> Columns:
        """Return time_us and one column per label.

        label_names names the label columns. Without label_names, the name of
        each label column is its label index as a decimal string.

        Raises:
            ValueError: Two label_names are equal or one is time_us.
        """
        names = (
            self.label_names
            if self.label_names is not None
            else tuple(str(label) for label in range(self.count.shape[1]))
        )
        if len(set(names)) != len(names) or TIME_COLUMN in names:
            raise ValueError(
                f"label_names must be distinct and must not include "
                f"{TIME_COLUMN!r} to name count columns, got {names!r}"
            )
        columns: Columns = {TIME_COLUMN: self.time_us}
        for name, values in zip(names, self.count.T, strict=True):
            columns[name] = values
        return columns

    def to_dataframe(self) -> "pd.DataFrame":
        """Return the counts as a pandas DataFrame indexed by time_us."""
        return counts_to_dataframe(self)

    def to_xarray(self) -> "xr.DataArray":
        """Return the counts as an xarray DataArray.

        The dimensions are time_us and label.
        """
        return counts_to_xarray(self)


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class EventTable:
    """The events of an event channel in a window, one row per event.

    A column is None when the channel lacks it or the read left it out.

    Attributes:
        time_us: The time of each event in microseconds from the reference
            time, in non-decreasing order.
        duration_us: The duration of each event in microseconds.
        label: The label of each event.
        value: The value of each event, in the unit of the channel.
        text: The decoded text of each event.
        channel_indices: The channel indices each event applies to. An empty list
            means the event applies to no channel.
        waveform: The snippet of each event.
        label_names: The name of each label value by index, or None when the
            channel has no label_names.
        whole_recording: True when the channel has no channels column.
            applies_to then returns True for every event.
    """

    time_us: npt.NDArray[np.integer[Any]]
    duration_us: npt.NDArray[np.integer[Any]] | None
    label: npt.NDArray[np.integer[Any]] | None
    value: npt.NDArray[np.floating[Any]] | None
    text: list[str] | None
    channel_indices: list[list[int]] | None
    waveform: Waveform | None
    label_names: tuple[str, ...] | None
    whole_recording: bool = False

    @property
    def label_name(self) -> list[str] | None:
        """Name of the label of each event.

        None when the table has no label column or the channel has no
        label_names.
        """
        if self.label is None or self.label_names is None:
            return None
        names = self.label_names
        return [names[label] for label in self.label.tolist()]

    @property
    def time_s(self) -> npt.NDArray[np.float64]:
        """Time of each event in seconds from the reference time."""
        return np.asarray(
            self.time_us / MICROSECONDS_PER_SECOND, dtype=np.float64
        )

    def __len__(self) -> int:
        """Return the number of events."""
        return int(self.time_us.shape[0])

    def __repr__(self) -> str:
        """Return the event count, the column names, and whole_recording."""
        return (
            f"EventTable(len={len(self)}, columns={self._column_names()!r}, "
            f"whole_recording={self.whole_recording})"
        )

    def _column_names(self) -> tuple[str, ...]:
        """Return the names of the columns that columns returns, in order."""
        held = {
            "duration_us": self.duration_us is not None,
            "label": self.label is not None,
            "label_name": (
                self.label is not None and self.label_names is not None
            ),
            "value": self.value is not None,
            "text": self.text is not None,
            "channel_indices": self.channel_indices is not None,
        }
        return (
            TIME_COLUMN,
            *(name for name, present in held.items() if present),
        )

    def columns(self) -> Columns:
        """Return time_us and each other per-event column the table holds.

        The order is time_us, duration_us, label, label_name, value, text,
        and channel_indices. label_name, text, and channel_indices are object
        arrays that hold a str or a list of indices per event. waveform and
        whole_recording are not columns.
        """
        present = {
            "duration_us": self.duration_us,
            "label": self.label,
            "label_name": _object_column(self.label_name),
            "value": self.value,
            "text": _object_column(self.text),
            "channel_indices": _object_column(self.channel_indices),
        }
        columns: Columns = {TIME_COLUMN: self.time_us}
        for name, values in present.items():
            if values is not None:
                columns[name] = values
        return columns

    def to_dataframe(self) -> "pd.DataFrame":
        """Return the events as a pandas DataFrame, one row per event."""
        return events_to_dataframe(self)

    def applies_to(self, index: int) -> npt.NDArray[np.bool_]:
        """Return a mask of the events that apply to channel index.

        An event applies to index when its channel_indices names index. Every event
        applies when whole_recording is true.

        Args:
            index: A channel index.

        Raises:
            ValueError: channel_indices is None and whole_recording is false.
        """
        if self.channel_indices is None:
            if self.whole_recording:
                return np.ones(len(self), dtype=np.bool_)
            raise ValueError(
                "the table has no channel_indices; read with 'channels' among "
                "columns to find the events that apply to a channel"
            )
        return np.array(
            [index in indices for indices in self.channel_indices],
            dtype=np.bool_,
        )


type _Times = npt.NDArray[np.integer[Any]]


def _search_times(events: ZarrArray) -> _Times | None:
    """Return events read whole, or None when it is too large for that.

    It reads the whole array when the array holds at most
    EVENTS_SEARCH_LIMIT_BYTES bytes.
    """
    if events.nbytes <= EVENTS_SEARCH_LIMIT_BYTES:
        return read_rows(events, 0, events.shape[0])
    return None


def _first_at_or_after(
    events: ZarrArray, times: _Times | None, value: int, low: int = 0
) -> int:
    """Return the index of the first event from index low at or after value.

    times holds events read whole, or is None to search events in the
    store. The result is the number of events when every event from low on
    is before value.
    """
    # numpy compares a bound outside the dtype range in float64, which
    # rounds the events near it.
    limits = np.iinfo(events.dtype)
    if value > limits.max:
        return events.shape[0]
    value = max(value, int(limits.min))
    if times is not None:
        return max(low, int(np.searchsorted(times, value, side="left")))
    high = events.shape[0]
    # A probe fetches its whole inner chunk. Inside one chunk, a single
    # read costs less than further probes.
    while high - low > events.chunks[0]:
        middle = (low + high) // 2
        if read_int(events, middle) < value:
            low = middle + 1
        else:
            high = middle
    rows = read_rows(events, low, high)
    return low + int(np.searchsorted(rows, value, side="left"))


def _window_times(
    events: ZarrArray, times: _Times | None, first: int, stop: int
) -> _Times:
    """Return the times of events first through stop - 1."""
    if times is None:
        return read_rows(events, first, stop)
    return times[first:stop]


def _kept(
    rows: npt.NDArray[Any], keep: npt.NDArray[np.bool_] | None
) -> npt.NDArray[Any]:
    """Return the rows that keep marks, or every row when keep is None."""
    return rows if keep is None else rows[keep]


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class EventChannel:
    """An event channel of a bundle.

    Attributes:
        index: The name of the channel group.
        name: The channel name.
        id: The channel id, or None when the group omits it.
        unit: The unit of values, or None when the group omits it.
        label_names: The name of each label, where label_names[j] names label
            j, or None when the group omits it.
        max_duration_us: A bound on every duration in microseconds, or None
            when the group omits it.
        provenance: The provenance object, or None when the group omits it.
        events: The zarr array of event times.
        durations: The zarr array of durations, or None when absent.
        labels: The zarr array of labels, or None when absent.
        values: The zarr array of values, or None when absent.
        text: The zarr array of event text, or None when absent.
        channels: The zarr array of channel indices, or None when absent.
        waveforms: The per-event snippets, or None when absent.
        waveform_mean: The per-label mean snippets, or None when absent.
        count_levels: Each count level by number, in level order.
        channel_count: The number of channels in the bundle, or None when
            unknown. A known count lets a read reject a channels bit that
            names a channel the bundle lacks.
    """

    index: int
    name: str
    id: str | None
    unit: str | None
    label_names: tuple[str, ...] | None
    max_duration_us: int | None
    provenance: Mapping[str, object] | None
    events: ZarrArray
    durations: ZarrArray | None
    labels: ZarrArray | None
    values: ZarrArray | None
    text: ZarrArray | None
    channels: ZarrArray | None
    waveforms: Snippets | None
    waveform_mean: Snippets | None
    count_levels: Mapping[int, CountLevel]
    channel_count: int | None = None

    @property
    def kind(self) -> Literal["event"]:
        """Channel kind, which is event."""
        return "event"

    @property
    def num_events(self) -> int:
        """Number of events."""
        return self.events.shape[0]

    def num_labels(self) -> int:
        """Return the number of label categories of the channel.

        The number is the length of label_names. Without label_names it is one
        more than the largest label, or 1 when the channel has no labels or no
        events. The last case reads the whole labels column.
        """
        names = self.label_names
        largest = None
        if names is None and self.labels is not None and self.num_events:
            largest = int(read_rows(self.labels, 0, self.num_events).max())
        return label_count(None if names is None else len(names), largest)

    @property
    def columns(self) -> tuple[str, ...]:
        """Names of the per-event columns the channel holds, in spec order."""
        return tuple(self._column_arrays())

    def __repr__(self) -> str:
        """Return the index, name, event count, and per-event columns."""
        return (
            f"EventChannel(index={self.index}, name={self.name!r}, "
            f"num_events={self.num_events}, columns={self.columns!r})"
        )

    def read(
        self,
        start_us: SupportsIndex | None = None,
        end_us: SupportsIndex | None = None,
        *,
        columns: Iterable[str] | None = None,
    ) -> EventTable:
        """Return the events in a window and their per-event columns.

        The window is [start_us, end_us), and a None bound leaves that side
        open. A point event, which has no duration or a duration of 0, is in
        the window when its time is at or after start_us and before end_us. An
        interval is in the window when it starts before end_us and ends after
        start_us. The table always holds the times.

        Args:
            start_us: Inclusive start of the window. None leaves it open.
            end_us: Exclusive end of the window. None leaves it open.
            columns: The per-event columns to read. None reads every column
                the channel holds.

        Raises:
            TypeError: A bound is not an integer or columns is a string.
            ValueError: end_us is before start_us, or columns names a column
                the channel lacks. Also raised when a label is below 0 or at
                least the length of label_names, or the text of an event in
                the window is not valid UTF-8.
        """
        start, end = window_bounds(start_us, end_us)
        selected = self._selected_columns(columns)
        events = self.events
        times = _search_times(events)
        first = 0
        lookback = self._lookback_us()
        if start is not None and lookback is not None:
            first = _first_at_or_after(events, times, start - lookback)
        stop = self.num_events
        if end is not None:
            stop = max(first, _first_at_or_after(events, times, end, first))
        window_times = _window_times(events, times, first, stop)
        arrays = self._column_arrays()
        rows: dict[str, npt.NDArray[Any]] = {}
        keep: npt.NDArray[np.bool_] | None = None
        if self.durations is not None and start is not None:
            rows[DURATIONS_KEY] = read_rows(self.durations, first, stop)
            # A narrow stored dtype would wrap the sum of time and duration.
            times64 = np.asarray(window_times, dtype=np.int64)
            durations64 = np.asarray(rows[DURATIONS_KEY], dtype=np.int64)
            ends = times64 + durations64
            # A positive duration whose sum falls below its time ends past
            # 2^63 - 1, so the interval reaches every int64 start.
            keep = (
                (ends > start)
                | ((durations64 == 0) & (times64 >= start))
                | ((durations64 > 0) & (ends < times64))
            )
        for name in selected:
            if name not in rows:
                rows[name] = read_rows(arrays[name], first, stop)
        kept = {name: _kept(rows[name], keep) for name in selected}
        indices = _kept(np.arange(first, stop), keep)
        return self._table(_kept(window_times, keep), kept, indices)

    def read_counts(
        self,
        start_us: SupportsIndex | None = None,
        end_us: SupportsIndex | None = None,
        *,
        level: SupportsIndex,
    ) -> Counts:
        """Return the bins of a count level that overlap a window.

        The window is [start_us, end_us). A window past the last bin gives no
        bins.

        Args:
            start_us: Inclusive start of the window. None leaves it open.
            end_us: Exclusive end of the window. None leaves it open.
            level: The count level number.

        Raises:
            TypeError: A bound or level is not an integer.
            ValueError: end_us is before start_us or the channel has no such
                count level.
        """
        number = integer_argument(level, "level")
        chosen = self.count_levels.get(number)
        if chosen is None:
            raise ValueError(self._count_level_choices(number))
        start, end = window_bounds(start_us, end_us)
        first, stop = index_range(
            start,
            end,
            chosen.offset_us,
            Fraction(chosen.period_us),
            chosen.bins,
        )
        return Counts(
            level=chosen.number,
            first=first,
            period_us=chosen.period_us,
            offset_us=chosen.offset_us,
            count=read_rows(chosen.count, first, stop),
            label_names=self.label_names,
        )

    def level_for(
        self,
        start_us: SupportsIndex | None = None,
        end_us: SupportsIndex | None = None,
        *,
        bins: SupportsIndex,
    ) -> int:
        """Return the coarsest count level that holds bins bins in a window.

        The window is [start_us, end_us). An open start_us stands for the
        first event, and an open end_us for 1 microsecond after the last
        event. Each count is the number of bins read_counts returns for the
        window with those bounds. The result is 0, for the events, when no
        count level holds enough. A channel without events or count levels
        gives 0.

        Args:
            start_us: Inclusive start of the window. None leaves it open.
            end_us: Exclusive end of the window. None leaves it open.
            bins: The number of bins the level must hold.

        Raises:
            TypeError: A bound or bins is not an integer.
            ValueError: end_us is before start_us or bins is below 1.
        """
        wanted = integer_argument(bins, "bins")
        start, end = window_bounds(start_us, end_us)
        if self.num_events == 0 or not self.count_levels:
            return select_level({}, wanted)
        if start is None:
            start = read_int(self.events, 0)
            if end is not None:
                start = min(start, end)
        if end is None:
            end = max(read_int(self.events, self.num_events - 1) + 1, start)
        counts: dict[int, int] = {}
        for number, chosen in self.count_levels.items():
            first, stop = index_range(
                start,
                end,
                chosen.offset_us,
                Fraction(chosen.period_us),
                chosen.bins,
            )
            counts[number] = stop - first
        return select_level(counts, wanted)

    def _column_arrays(self) -> dict[str, ZarrArray]:
        """Return the per-event columns the channel holds, in spec order."""
        present = {
            DURATIONS_KEY: self.durations,
            LABELS_KEY: self.labels,
            VALUES_KEY: self.values,
            TEXT_KEY: self.text,
            CHANNELS_KEY: self.channels,
            WAVEFORMS_KEY: (
                None if self.waveforms is None else self.waveforms.array
            ),
        }
        return {
            name: array for name, array in present.items() if array is not None
        }

    def _selected_columns(self, columns: Iterable[str] | None) -> list[str]:
        """Return the per-event columns a read returns.

        Raises:
            TypeError: columns is a string.
            ValueError: columns names a column the channel lacks.
        """
        present = self.columns
        if columns is None:
            return list(present)
        check_not_string(columns, "columns")
        selected = list(columns)
        for name in selected:
            if name not in present:
                raise ValueError(
                    f"channel {self.index} has columns {present!r}; columns "
                    f"must name only these, got {name!r}"
                )
        return selected

    def _lookback_us(self) -> int | None:
        """Return how far before start_us a window search starts.

        None means the search starts at the first event, for a channel with
        durations and no max_duration_us.
        """
        if self.durations is None:
            return 0
        return self.max_duration_us

    def _table(
        self,
        times: npt.NDArray[Any],
        rows: Mapping[str, npt.NDArray[Any]],
        indices: npt.NDArray[np.int64],
    ) -> EventTable:
        """Return the table of the kept times and per-event rows.

        Args:
            times: The kept event times.
            rows: The kept rows of each column by name.
            indices: The event index of each kept row.

        Raises:
            ValueError: A label is below 0 or at least the length of
                label_names, or the text of an event is not valid UTF-8.
        """
        label = rows.get(LABELS_KEY)
        names = self.label_names
        if label is not None and names is not None:
            check_label_range(
                f"channel {self.index}",
                label,
                len(names),
                "the length of label_names",
            )
        text = rows.get(TEXT_KEY)
        channels = rows.get(CHANNELS_KEY)
        if channels is not None and self.channel_count is not None:
            check_channels_rows(
                channels,
                self.channel_count,
                indices.tolist(),
                f"channel {self.index}",
            )
        snippets = rows.get(WAVEFORMS_KEY)
        waveform = None
        if snippets is not None and self.waveforms is not None:
            waveform = Waveform(
                values=snippets,
                period_us=self.waveforms.period_us,
                event_sample=self.waveforms.event_sample,
                rows=EVENT_DIMENSION,
            )
        return EventTable(
            time_us=times,
            duration_us=rows.get(DURATIONS_KEY),
            label=label,
            value=rows.get(VALUES_KEY),
            text=(
                None
                if text is None
                else decode_event_text(
                    text, indices.tolist(), f"channel {self.index}"
                )
            ),
            channel_indices=(
                None if channels is None else decode_channels_rows(channels)
            ),
            waveform=waveform,
            label_names=names,
            whole_recording=self.channels is None,
        )

    def _count_level_choices(self, level: int) -> str:
        """Return the error message for a count level the channel lacks."""
        where = f"channel {self.index}"
        count = len(self.count_levels)
        if level == 0:
            choices = ""
            if count == 1:
                choices = ", or pass count level 1"
            elif count > 1:
                choices = f", or pass a count level from 1 through {count}"
            return (
                f"{where}: level 0 stands for the events; read them with "
                f"read{choices}"
            )
        if count == 0:
            return (
                f"{where} has no count levels; read its events with read, "
                f"got level {level!r}"
            )
        if count == 1:
            return f"{where} has count level 1; level must be 1, got {level!r}"
        return (
            f"{where} has count levels 1 through {count}; level must be one "
            f"of them, got {level!r}"
        )


def _expectation(layout: _Layout, rows: int | None) -> str:
    """Return the phrase naming the rank, dtype kind, and rows of an array."""
    phrase = f"a rank-{layout.rank} {layout.kind_name} array"
    return phrase if rows is None else f"{phrase} with {rows} rows"


def _checked(
    array: ZarrArray | None,
    where: str,
    name: str,
    layout: _Layout,
    rows: int | None,
) -> ZarrArray | None:
    """Return array, or None when it is absent.

    Args:
        array: The array to check, or None.
        where: Names the channel in an error message.
        name: The array name.
        layout: The rank and dtype kind the array must have.
        rows: The required number of rows. None allows any number.

    Raises:
        ValueError: The array lacks the rank, dtype kind, or rows given.
    """
    if array is None:
        return None
    if (
        array.ndim != layout.rank
        or not np.issubdtype(array.dtype, layout.kind)
        or (rows is not None and array.shape[0] != rows)
    ):
        raise array_error(where, name, _expectation(layout, rows), array)
    return array


def _required(
    array: ZarrArray | None,
    where: str,
    name: str,
    layout: _Layout,
    rows: int | None,
) -> ZarrArray:
    """Return array after the checks of _checked.

    Raises:
        ValueError: The array is absent.
    """
    checked = _checked(array, where, name, layout, rows)
    if checked is None:
        raise array_error(where, name, _expectation(layout, rows), None)
    return checked


def _parse_snippets(
    where: str,
    arrays: Mapping[str, ZarrArray],
    name: str,
    rows: int | None,
) -> Snippets | None:
    """Return the snippet array name of a channel, or None without one.

    Raises:
        ValueError: The array is not a rank-2 floating-point array with rows
            rows that carries tsz.period_us, or tsz.event_sample is not below
            the number of columns.
    """
    array = _checked(arrays.get(name), where, name, _FLOATING_ROWS, rows)
    if array is None:
        return None
    timing = parse_waveform_attrs(array.attrs, f"{where} {name}")
    if timing.event_sample >= array.shape[1]:
        raise ValueError(
            f"{where} {name}: {TSZ_KEY}.event_sample must be below the "
            f"{array.shape[1]} samples of a snippet, got {timing.event_sample!r}"
        )
    return Snippets(
        array=array,
        period_us=timing.period_us,
        event_sample=timing.event_sample,
        rows=EVENT_DIMENSION if name == WAVEFORMS_KEY else LABEL_DIMENSION,
    )


@dataclass(frozen=True, slots=True)
class _KnownLabels:
    """The number of label categories that the metadata of a channel fixes.

    Attributes:
        count: The number of label categories.
        reason: The phrase that names where the number comes from.
    """

    count: int
    reason: str


def _known_labels(
    label_names: tuple[str, ...] | None,
    labels: ZarrArray | None,
    num_events: int,
) -> _KnownLabels | None:
    """Return the number of label categories, or None when it needs the data.

    The number follows from the metadata when the channel has label_names,
    and when it has no labels or no events. Otherwise it is one more than the
    largest label, which only a read of the labels column finds.
    """
    if label_names is not None:
        return _KnownLabels(len(label_names), "one per label name")
    if labels is None or num_events == 0:
        return _KnownLabels(
            label_count(None, None),
            "the one category of a channel without labels or events",
        )
    return None


def _parse_count_levels(
    where: str,
    groups: Mapping[str, ZarrGroup],
    arrays: Mapping[str, ZarrArray],
    labels: _KnownLabels | None,
) -> Mapping[int, CountLevel]:
    """Return the count levels of a channel keyed by number in level order.

    Raises:
        ValueError: A count level is an array, the level numbers have a gap,
            or a level lacks period_us or count. Also raised when offset_us
            is not a multiple of period_us, count is not a rank-2 integer
            array, the levels disagree on the number of labels, or a
            period_us is not an integer multiple, at least 2, of the one
            before it.
    """
    check_level_arrays(where, "count level", COUNT_KEY, arrays)
    numbers = check_level_numbers(where, "count levels", groups)
    levels: dict[int, CountLevel] = {}
    for number in numbers:
        level_where = f"{where} count level {number}"
        attrs = parse_count_level_attrs(groups[str(number)].attrs, level_where)
        count = _required(
            arrays.get(f"{number}/{COUNT_KEY}"),
            level_where,
            COUNT_KEY,
            _INTEGER_ROWS,
            None,
        )
        previous = levels.get(number - 1)
        if previous is not None:
            _check_period(level_where, attrs.period_us, previous)
        _check_label_columns(level_where, count, levels.get(1), labels)
        levels[number] = CountLevel(
            number=number,
            period_us=attrs.period_us,
            offset_us=attrs.offset_us,
            count=count,
        )
    return MappingProxyType(levels)


def _check_period(where: str, period_us: int, previous: CountLevel) -> None:
    """Check that period_us follows the count level previous.

    The test is exact integer arithmetic.

    Raises:
        ValueError: period_us is not an integer multiple of the period_us of
            previous, or the multiple is below 2.
    """
    if (
        period_us % previous.period_us
        or period_us < MIN_DECIMATION_FACTOR * previous.period_us
    ):
        raise ValueError(
            f"{where}: {TSZ_KEY}.period_us must be an integer multiple of "
            f"{previous.period_us!r}, the period_us of count level "
            f"{previous.number}, at least {MIN_DECIMATION_FACTOR} times it, "
            f"got {period_us!r}"
        )


def _check_label_columns(
    where: str,
    count: ZarrArray,
    first_level: CountLevel | None,
    labels: _KnownLabels | None,
) -> None:
    """Check that count has one column per label.

    The number of labels is the number the metadata fixes. When only the data
    fixes it, count must have the columns of count level 1.

    Raises:
        ValueError: The number of columns differs from the number of labels.
    """
    columns = count.shape[1]
    if labels is not None and columns != labels.count:
        raise ValueError(
            f"{where}: {COUNT_KEY} must have {labels.count} columns, "
            f"{labels.reason}, got shape {count.shape}"
        )
    if first_level is not None and columns != first_level.num_labels:
        raise ValueError(
            f"{where}: {COUNT_KEY} must have the {first_level.num_labels} "
            f"columns of count level 1, got shape {count.shape}"
        )


def _check_waveform_rows(
    where: str,
    mean: Snippets,
    labels: _KnownLabels | None,
    first_level: CountLevel | None,
) -> None:
    """Check that waveform_mean has one row per label.

    The number of labels is the number the metadata fixes. When only the data
    fixes it, waveform_mean must have the label columns of count level 1.

    Raises:
        ValueError: The number of rows differs from the number of labels.
    """
    rows = mean.array.shape[0]
    if labels is not None and rows != labels.count:
        raise ValueError(
            f"{where} {WAVEFORM_MEAN_KEY}: must have {labels.count} rows, "
            f"{labels.reason}, got shape {mean.array.shape}"
        )
    if (
        labels is None
        and first_level is not None
        and rows != first_level.num_labels
    ):
        raise ValueError(
            f"{where} {WAVEFORM_MEAN_KEY}: must have the "
            f"{first_level.num_labels} label columns of count level 1 as "
            f"rows, got shape {mean.array.shape}"
        )


def _check_waveform_agreement(
    where: str, mean: Snippets, waveforms: Snippets
) -> None:
    """Check that waveform_mean and waveforms describe the same trace.

    Raises:
        ValueError: The arrays differ in the number of samples, period_us, or
            event_sample.
    """
    for field, expected, actual in (
        ("samples", waveforms.array.shape[1], mean.array.shape[1]),
        ("period_us", waveforms.period_us, mean.period_us),
        ("event_sample", waveforms.event_sample, mean.event_sample),
    ):
        if actual != expected:
            raise ValueError(
                f"{where} {WAVEFORM_MEAN_KEY}: {field} must match "
                f"{WAVEFORMS_KEY}, which has {expected!r}, got {actual!r}"
            )


def parse_event_channel(
    index: int,
    attributes: Mapping[str, object],
    groups: Mapping[str, ZarrGroup],
    arrays: Mapping[str, ZarrArray],
    channel_count: int | None = None,
) -> EventChannel:
    """Return the event channel that channel group index describes.

    The parser ignores unknown nodes and unknown tsz keys.

    Args:
        index: The channel index.
        attributes: The attributes of the channel group.
        groups: The descendant groups by path below the channel group, such
            as 1.
        arrays: The descendant arrays by path below the channel group, such
            as 1/count.
        channel_count: The number of channels in the bundle. None skips the
            checks on the channels column.

    Raises:
        ValueError: A required key or array is missing, or an array has the
            wrong rank, dtype kind, or number of rows. Also raised when count
            level numbers have a gap, count levels disagree on the number of
            labels, a count level is stored as an array, or a period_us is
            not an integer multiple, at least 2, of the one before it. Also
            raised when count or waveform_mean disagrees with the number of
            label categories, waveform_mean disagrees with waveforms, or
            channels has other than ceil(channel_count / 8) columns. The
            message names the channel, and the array or count level at fault.
    """
    where = f"channel {index}"
    fields = parse_event_attrs(attributes, where)
    events = _required(
        arrays.get(EVENTS_KEY), where, EVENTS_KEY, _INTEGER_COLUMN, None
    )
    count = events.shape[0]
    columns = {
        name: _checked(arrays.get(name), where, name, layout, count)
        for name, layout in _COLUMN_LAYOUTS.items()
    }
    channels = columns[CHANNELS_KEY]
    if channels is not None and channel_count is not None:
        width = channels_width(channel_count - 1)
        if channels.shape[1] != width:
            raise ValueError(
                f"{where}: {CHANNELS_KEY} must have {width} columns, one bit "
                f"per channel of the {channel_count} channels, got shape "
                f"{channels.shape}"
            )
    known = _known_labels(fields.label_names, columns[LABELS_KEY], count)
    waveforms = _parse_snippets(where, arrays, WAVEFORMS_KEY, count)
    waveform_mean = _parse_snippets(where, arrays, WAVEFORM_MEAN_KEY, None)
    count_levels = _parse_count_levels(where, groups, arrays, known)
    if waveform_mean is not None:
        if columns[LABELS_KEY] is not None:
            _check_waveform_rows(
                where, waveform_mean, known, count_levels.get(1)
            )
        if waveforms is not None:
            _check_waveform_agreement(where, waveform_mean, waveforms)
    return EventChannel(
        index=index,
        name=fields.name,
        id=fields.id,
        unit=fields.unit,
        label_names=fields.label_names,
        max_duration_us=fields.max_duration_us,
        provenance=fields.provenance,
        events=events,
        durations=columns[DURATIONS_KEY],
        labels=columns[LABELS_KEY],
        values=columns[VALUES_KEY],
        text=columns[TEXT_KEY],
        channels=channels,
        waveforms=waveforms,
        waveform_mean=waveform_mean,
        count_levels=count_levels,
        channel_count=channel_count,
    )
