"""Records and window reads for the event channels of a bundle.

Every time argument is an integer count of microseconds from the reference
time, and a window is half-open.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, SupportsIndex

import numpy as np
import numpy.typing as npt

from timeseries_zarr.attrs import (
    parse_count_level_attrs,
    parse_event_attrs,
    parse_waveform_attrs,
)
from timeseries_zarr.columns import decode_channels_rows, decode_event_text
from timeseries_zarr.constants import (
    CHANNELS_KEY,
    COUNT_KEY,
    DECIMATION_FACTOR,
    DURATIONS_KEY,
    EVENT_DIMENSION,
    EVENTS_KEY,
    EVENTS_SEARCH_LIMIT_BYTES,
    LABEL_DIMENSION,
    LABELS_KEY,
    LEVEL_PERIOD_TOLERANCE,
    MICROSECONDS_PER_SECOND,
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
    # np.array would read equal-length key lists as a second axis.
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
    period_us: float
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
        channel_keys: The channel keys each event applies to. An empty list
            means the whole recording.
        waveform: The snippet of each event.
        label_names: The name of each label value by index, or None when the
            channel has no label_names.
        whole_recording: True when the channel has no channels column. Every
            event of such a channel applies to the whole recording.
    """

    time_us: npt.NDArray[np.integer[Any]]
    duration_us: npt.NDArray[np.integer[Any]] | None
    label: npt.NDArray[np.integer[Any]] | None
    value: npt.NDArray[np.floating[Any]] | None
    text: list[str] | None
    channel_keys: list[list[int]] | None
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
            "channel_keys": self.channel_keys is not None,
        }
        return (
            TIME_COLUMN,
            *(name for name, present in held.items() if present),
        )

    def columns(self) -> Columns:
        """Return time_us and each other per-event column the table holds.

        The order is time_us, duration_us, label, label_name, value, text,
        and channel_keys. label_name, text, and channel_keys are object
        arrays that hold a str or a list of keys per event. waveform and
        whole_recording are not columns.
        """
        present = {
            "duration_us": self.duration_us,
            "label": self.label,
            "label_name": _object_column(self.label_name),
            "value": self.value,
            "text": _object_column(self.text),
            "channel_keys": _object_column(self.channel_keys),
        }
        columns: Columns = {TIME_COLUMN: self.time_us}
        for name, values in present.items():
            if values is not None:
                columns[name] = values
        return columns

    def to_dataframe(self) -> "pd.DataFrame":
        """Return the events as a pandas DataFrame, one row per event."""
        return events_to_dataframe(self)

    def applies_to(self, key: int) -> npt.NDArray[np.bool_]:
        """Return a mask of the events that apply to channel key.

        An event applies to key when its channel_keys names key or is empty.
        Every event applies when whole_recording is true.

        Args:
            key: A channel key.

        Raises:
            ValueError: channel_keys is None and whole_recording is false.
        """
        if self.channel_keys is None:
            if self.whole_recording:
                return np.ones(len(self), dtype=np.bool_)
            raise ValueError(
                "the table has no channel_keys; read with 'channels' among "
                "columns to find the events that apply to a channel"
            )
        return np.array(
            [not keys or key in keys for keys in self.channel_keys],
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
        key: The name of the channel group.
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
        channels: The zarr array of channel keys, or None when absent.
        waveforms: The per-event snippets, or None when absent.
        waveform_mean: The per-label mean snippets, or None when absent.
        count_levels: Each count level by number, in level order.
    """

    key: int
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

    @property
    def kind(self) -> Literal["event"]:
        """Channel kind, which is event."""
        return "event"

    @property
    def num_events(self) -> int:
        """Number of events."""
        return self.events.shape[0]

    @property
    def columns(self) -> tuple[str, ...]:
        """Names of the per-event columns the channel holds, in spec order."""
        return tuple(self._column_arrays())

    def __repr__(self) -> str:
        """Return the key, name, event count, and per-event columns."""
        return (
            f"EventChannel(key={self.key}, name={self.name!r}, "
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
        open. An event without a duration is in the window when its time is
        at or after start_us and before end_us. An event with a duration is
        in the window when it starts before end_us and ends at or after
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
            keep = (ends >= start) | ((durations64 > 0) & (ends < times64))
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
                    f"channel {self.key} has columns {present!r}; columns "
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
                f"channel {self.key}",
                label,
                len(names),
                "the length of label_names",
            )
        text = rows.get(TEXT_KEY)
        channels = rows.get(CHANNELS_KEY)
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
                    text, indices.tolist(), f"channel {self.key}"
                )
            ),
            channel_keys=(
                None if channels is None else decode_channels_rows(channels)
            ),
            waveform=waveform,
            label_names=names,
            whole_recording=self.channels is None,
        )

    def _count_level_choices(self, level: int) -> str:
        """Return the error message for a count level the channel lacks."""
        where = f"channel {self.key}"
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
            rows that carries tsz.period_us.
    """
    array = _checked(arrays.get(name), where, name, _FLOATING_ROWS, rows)
    if array is None:
        return None
    timing = parse_waveform_attrs(array.attrs, f"{where} {name}")
    return Snippets(
        array=array,
        period_us=timing.period_us,
        event_sample=timing.event_sample,
        rows=EVENT_DIMENSION if name == WAVEFORMS_KEY else LABEL_DIMENSION,
    )


def _parse_count_levels(
    where: str,
    groups: Mapping[str, ZarrGroup],
    arrays: Mapping[str, ZarrArray],
    label_names: tuple[str, ...] | None,
) -> Mapping[int, CountLevel]:
    """Return the count levels of a channel keyed by number in level order.

    Raises:
        ValueError: A count level is an array, the level numbers have a gap,
            or a level lacks period_us or count. Also raised when offset_us
            is not a multiple of period_us, count is not a rank-2 integer
            array, the levels disagree on the number of labels, or a
            period_us is not DECIMATION_FACTOR times the one before it.
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
        _check_label_columns(level_where, count, levels.get(1), label_names)
        levels[number] = CountLevel(
            number=number,
            period_us=attrs.period_us,
            offset_us=attrs.offset_us,
            count=count,
        )
    return MappingProxyType(levels)


def _check_period(where: str, period_us: float, previous: CountLevel) -> None:
    """Check that period_us follows the count level previous.

    Raises:
        ValueError: period_us is not DECIMATION_FACTOR times the period_us of
            previous.
    """
    expected = DECIMATION_FACTOR * previous.period_us
    if not math.isclose(period_us, expected, rel_tol=LEVEL_PERIOD_TOLERANCE):
        raise ValueError(
            f"{where}: {TSZ_KEY}.period_us must be {expected!r}, "
            f"{DECIMATION_FACTOR} times that of count level "
            f"{previous.number}, got {period_us!r}"
        )


def _check_label_columns(
    where: str,
    count: ZarrArray,
    first_level: CountLevel | None,
    label_names: tuple[str, ...] | None,
) -> None:
    """Check that count has one column per label.

    The number of labels is the length of label_names, or without
    label_names the number of columns of count level 1.

    Raises:
        ValueError: The number of columns differs from the number of labels.
    """
    columns = count.shape[1]
    if label_names is not None and columns != len(label_names):
        raise ValueError(
            f"{where}: {COUNT_KEY} must have {len(label_names)} columns, one "
            f"per label name, got shape {count.shape}"
        )
    if first_level is not None and columns != first_level.num_labels:
        raise ValueError(
            f"{where}: {COUNT_KEY} must have the {first_level.num_labels} "
            f"columns of count level 1, got shape {count.shape}"
        )


def parse_event_channel(
    key: int,
    attributes: Mapping[str, object],
    groups: Mapping[str, ZarrGroup],
    arrays: Mapping[str, ZarrArray],
) -> EventChannel:
    """Return the event channel that channel group key describes.

    The parser ignores unknown nodes and unknown tsz keys.

    Args:
        key: The channel key.
        attributes: The attributes of the channel group.
        groups: The descendant groups by path below the channel group, such
            as 1.
        arrays: The descendant arrays by path below the channel group, such
            as 1/count.

    Raises:
        ValueError: A required key or array is missing, or an array has the
            wrong rank, dtype kind, or number of rows. Also raised when count
            level numbers have a gap, count levels disagree on the number of
            labels, a count level is stored as an array, or a period_us is
            not DECIMATION_FACTOR times the one before it. The message names
            the channel, and the array or count level at fault.
    """
    where = f"channel {key}"
    fields = parse_event_attrs(attributes, where)
    events = _required(
        arrays.get(EVENTS_KEY), where, EVENTS_KEY, _INTEGER_COLUMN, None
    )
    count = events.shape[0]
    columns = {
        name: _checked(arrays.get(name), where, name, layout, count)
        for name, layout in _COLUMN_LAYOUTS.items()
    }
    return EventChannel(
        key=key,
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
        channels=columns[CHANNELS_KEY],
        waveforms=_parse_snippets(where, arrays, WAVEFORMS_KEY, count),
        waveform_mean=_parse_snippets(where, arrays, WAVEFORM_MEAN_KEY, None),
        count_levels=_parse_count_levels(
            where, groups, arrays, fields.label_names
        ),
    )
