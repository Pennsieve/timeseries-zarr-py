"""Concrete NWB adapters implementing the channel-source protocols."""

import logging
from collections.abc import Iterator, Sequence
from datetime import datetime

import numpy as np
import numpy.typing as npt
from pynwb import NWBFile, TimeSeries
from pynwb.ecephys import ElectricalSeries
from pynwb.misc import Units

from timeseries_zarr.constants import (
    MAX_LABELS,
    MICROSECONDS_PER_SECOND,
    UNIT_TO_UV,
)
from timeseries_zarr.planning import sample_period_us
from timeseries_zarr.types import EventColumn

logger = logging.getLogger(__name__)

MAX_SERIES_RANK = 2
"""Highest data rank a series can have and still be read as channels."""


def _require_rate(series: TimeSeries) -> float:
    """Return the series' sample rate in hertz, raising for a timestamps-only series."""
    if series.rate is None:
        raise ValueError(
            f"irregular sampling is not supported: {series.name} has "
            "timestamps and no rate"
        )
    return float(series.rate)


def _start_us(series: TimeSeries, session_start_time: datetime) -> int:
    """Return wall-clock microseconds of sample 0, rounded to whole microseconds."""
    start_s: float = session_start_time.timestamp() + float(
        series.starting_time
    )
    return round(start_s * MICROSECONDS_PER_SECOND)


def _read_column(
    series: TimeSeries, channel_index: int, start: int, stop: int
) -> npt.NDArray[np.float64]:
    """Return one channel's [start, stop) window as float64.

    A rank-1 series is its own single channel; a rank-2 series is indexed by
    column.
    """
    data = series.data
    window = (
        data[start:stop]
        if len(data.shape) == 1
        else data[start:stop, channel_index]
    )
    return np.asarray(window, dtype=np.float64)


def _channel_count(series: TimeSeries) -> int:
    """Return the number of channels a rank-1 or rank-2 series holds."""
    shape = series.data.shape
    return 1 if len(shape) == 1 else int(shape[1])


class NwbContinuousSource:
    """A continuous channel backed by one column of an NWB ElectricalSeries."""

    def __init__(
        self,
        electrical_series: ElectricalSeries,
        channel_index: int,
        session_start_time: datetime,
    ) -> None:
        """Bind one channel of an ElectricalSeries as a continuous source.

        channel_index selects the column this source exposes; a rank-1 series
        holds one channel at index 0. session_start_time is the recording's
        wall-clock origin. Raises ValueError for a series sampled by timestamps
        rather than a rate.
        """
        self._series = electrical_series
        self._channel_index = channel_index
        self._session_start_time = session_start_time
        self._rate_hz = _require_rate(electrical_series)

    @property
    def id(self) -> str:
        """The selected electrode's table id."""
        electrodes = self._series.electrodes
        row_index = electrodes.data[self._channel_index]
        return str(electrodes.table.id[row_index])

    @property
    def name(self) -> str:
        """The electrode's channel_name column, then its label, then the id."""
        electrodes = self._series.electrodes
        row_index = electrodes.data[self._channel_index]
        table = electrodes.table
        for column in ("channel_name", "label"):
            if column in table.colnames:
                return str(table[column][row_index])
        return self.id

    @property
    def unit(self) -> str:
        """Always "uV"; read_samples normalizes every series to microvolts."""
        return "uV"

    def rate_hz(self) -> float:
        """Return the ElectricalSeries' sample rate in hertz."""
        return self._rate_hz

    def start_us(self) -> int:
        """Return the wall-clock microseconds of sample index 0.

        The session start plus the series' own start offset, rounded to whole
        microseconds.
        """
        return _start_us(self._series, self._session_start_time)

    def num_samples(self) -> int:
        """Return the length of the series' time axis, shared by every channel."""
        return int(self._series.data.shape[0])

    def read_samples(self, start: int, stop: int) -> npt.NDArray[np.float32]:
        """Return the half-open [start, stop) sample window as float32 microvolts.

        Applies the series' affine scaling, then converts from the series' own
        unit to microvolts. Raises ValueError if that unit is not a recognized
        volts family. An empty range (stop <= start) yields a length-0 array.
        """
        column = _read_column(self._series, self._channel_index, start, stop)
        scaled = column * float(self._series.conversion)
        if self._series.channel_conversion is not None:
            scaled = scaled * float(
                self._series.channel_conversion[self._channel_index]
            )
        scaled = scaled + float(self._series.offset)
        unit = str(self._series.unit).lower()
        if unit not in UNIT_TO_UV:
            raise ValueError(f"unsupported ElectricalSeries unit: {unit!r}")
        return (scaled * UNIT_TO_UV[unit]).astype(np.float32)


class NwbTimeSeriesSource:
    """A continuous channel backed by one column of a plain NWB TimeSeries.

    The samples keep the series' own unit unless that unit is in the volts
    family, in which case they are normalized to microvolts the way an
    ElectricalSeries is. A rank-1 series is one channel named after the series;
    a rank-2 series is one channel per column, named series/column.
    """

    def __init__(
        self,
        series: TimeSeries,
        channel_index: int,
        session_start_time: datetime,
    ) -> None:
        """Bind one channel of a TimeSeries as a continuous source.

        channel_index selects the column of a rank-2 series and is 0 for a
        rank-1 series. Raises ValueError for a series sampled by timestamps
        rather than a rate, or whose data has a rank above 2.
        """
        rank = len(series.data.shape)
        if rank > MAX_SERIES_RANK:
            raise ValueError(
                f"unsupported TimeSeries shape: {series.name} has rank {rank}, "
                f"and a channel needs rank 1 or 2"
            )
        self._series = series
        self._channel_index = channel_index
        self._session_start_time = session_start_time
        self._rate_hz = _require_rate(series)
        unit = str(series.unit)
        self._uv_factor = UNIT_TO_UV.get(unit.lower())
        self._unit = unit if self._uv_factor is None else "uV"

    @property
    def id(self) -> str:
        """The series name, with the column index appended for a rank-2 series."""
        if len(self._series.data.shape) == 1:
            return str(self._series.name)
        return f"{self._series.name}/{self._channel_index}"

    @property
    def name(self) -> str:
        """The same value as id."""
        return self.id

    @property
    def unit(self) -> str:
        """The series unit, or "uV" when that unit is in the volts family."""
        return self._unit

    def rate_hz(self) -> float:
        """Return the series' sample rate in hertz."""
        return self._rate_hz

    def start_us(self) -> int:
        """Return the wall-clock microseconds of sample index 0."""
        return _start_us(self._series, self._session_start_time)

    def num_samples(self) -> int:
        """Return the length of the series' time axis, shared by every channel."""
        return int(self._series.data.shape[0])

    def read_samples(self, start: int, stop: int) -> npt.NDArray[np.float32]:
        """Return the [start, stop) sample window as float32 in the channel unit.

        Applies the series' conversion and offset, then the volts-to-microvolts
        factor when the unit is in the volts family. An empty range
        (stop <= start) yields a length-0 array.
        """
        column = _read_column(self._series, self._channel_index, start, stop)
        scaled = column * float(self._series.conversion) + float(
            self._series.offset
        )
        if self._uv_factor is not None:
            scaled = scaled * self._uv_factor
        return scaled.astype(np.float32)


type ContinuousSource = NwbContinuousSource | NwbTimeSeriesSource


class NwbUnitSource:
    """A unit (spike) channel backed by an NWB Units table.

    Flattens the table's per-cluster rows into the per-event streams the
    bundle stores: all spikes merged into one timestamp series sorted
    ascending, each event labeled with its cluster's dense index in table row
    order (not the upstream unit id). The table's waveform_mean column, when it
    has one, becomes the per-label mean snippets.
    """

    def __init__(
        self,
        units: Units,
        waveform_rate_hz: float,
        session_start_time: datetime,
    ) -> None:
        """Bind an NWB Units table as a flattened per-event spike source.

        waveform_rate_hz is the sample rate within a waveform; the table
        carries no rate of its own. session_start_time places event timestamps
        in absolute microseconds. Raises ValueError if the table holds more
        than MAX_LABELS units, past the uint16 label range, or if the units'
        waveform_mean snippets differ in width.
        """
        unit_count = len(units)
        if unit_count > MAX_LABELS:
            raise ValueError(f"a unit channel holds at most {MAX_LABELS} units")

        self._id = str(units.name)
        self._rate_hz = float(waveform_rate_hz)
        self._start_us = round(
            session_start_time.timestamp() * MICROSECONDS_PER_SECOND
        )
        self._label_names = [
            str(units["unit_name"][i])
            if "unit_name" in units.colnames
            else str(units.id[i])
            for i in range(unit_count)
        ]

        session_s = session_start_time.timestamp()
        times: list[npt.NDArray[np.float64]] = []
        labels: list[npt.NDArray[np.uint16]] = []
        for cluster_id in range(unit_count):
            spike_times: npt.NDArray[np.float64] = np.asarray(
                units.get_unit_spike_times(cluster_id), dtype=np.float64
            )
            times.append(spike_times)
            labels.append(
                np.full(spike_times.shape[0], cluster_id, dtype=np.uint16)
            )
        all_times = np.concatenate(times) if times else np.empty(0)
        order = np.argsort(all_times, kind="stable")
        self._events = (
            ((all_times[order] + session_s) * MICROSECONDS_PER_SECOND)
            .round()
            .astype(np.int64)
        )
        self._labels = (
            np.concatenate(labels)[order]
            if labels
            else np.empty(0, dtype=np.uint16)
        )
        self._means: npt.NDArray[np.float32] | None = None
        if unit_count and "waveform_mean" in units.colnames:
            self._means = np.stack(
                [
                    np.asarray(units["waveform_mean"][i], dtype=np.float32)
                    for i in range(unit_count)
                ]
            )

    @property
    def id(self) -> str:
        """The Units table's name."""
        return self._id

    @property
    def name(self) -> str:
        """The Units table's name, the same value as id."""
        return self._id

    @property
    def unit(self) -> str | None:
        """Always None, because spikes have no measurement.

        waveform_mean carries no unit metadata, so its amplitudes are stored
        unscaled.
        """
        return None

    def start_us(self) -> int:
        """Return the earliest wall-clock microseconds the channel holds.

        The recording start, or the first spike when it precedes the start.
        """
        if self._events.shape[0]:
            return min(self._start_us, int(self._events[0]))
        return self._start_us

    def num_events(self) -> int:
        """Return the total number of spike events across all units."""
        return int(self._events.shape[0])

    def columns(self) -> frozenset[EventColumn]:
        """Return the optional columns: labels, and waveform_mean when present."""
        if not self._label_names:
            return frozenset()
        if self._means is None:
            return frozenset({"labels"})
        return frozenset({"labels", "waveform_mean"})

    def label_names(self) -> list[str] | None:
        """Return the unit_name of each unit, or its id, in table row order."""
        return list(self._label_names)

    def num_labels(self) -> int:
        """Return the number of units."""
        return len(self._label_names)

    def max_duration_us(self) -> int | None:
        """Return None, because spikes have no duration."""
        return None

    def text_width(self) -> int:
        """Return 0, because spikes carry no text."""
        return 0

    def read_events(self, start: int, stop: int) -> npt.NDArray[np.int64]:
        """Return the half-open [start, stop) window of event timestamps.

        Absolute-microsecond int64. An empty range (stop <= start) yields a
        length-0 array.
        """
        return self._events[start:stop]

    def read_labels(self, start: int, stop: int) -> npt.NDArray[np.uint16]:
        """Return the [start, stop) window of dense unit indices as uint16."""
        return self._labels[start:stop]

    def read_durations(self, start: int, stop: int) -> npt.NDArray[np.int64]:
        """Raise ValueError, because spikes have no durations column."""
        raise ValueError("a unit channel has no durations column")

    def read_text(self, start: int, stop: int) -> list[str]:
        """Raise ValueError, because spikes have no text column."""
        raise ValueError("a unit channel has no text column")

    def read_channel_ids(self, start: int, stop: int) -> list[list[str]]:
        """Raise ValueError, because spikes have no channels column."""
        raise ValueError("a unit channel has no channels column")

    def waveform_mean(self) -> npt.NDArray[np.float32]:
        """Return the mean snippet of each unit, shaped (units, samples).

        Raises ValueError if the table has no waveform_mean column.
        """
        if self._means is None:
            raise ValueError("the Units table has no waveform_mean column")
        return self._means

    def waveform_period_us(self) -> float:
        """Return the microseconds between adjacent snippet samples."""
        return sample_period_us(self._rate_hz)


def _iter_units_tables(nwbfile: NWBFile) -> Iterator[Units]:
    """Yield every Units table in the file in deterministic discovery order.

    The root Units table first when present, then the Units containers of each
    processing module, modules in name order and their containers in name
    order.
    """
    if nwbfile.units is not None:
        yield nwbfile.units
    for module_name in sorted(nwbfile.processing):
        module = nwbfile.processing[module_name]
        for container_name in sorted(module.data_interfaces):
            container = module.data_interfaces[container_name]
            if isinstance(container, Units):
                yield container


def _holds_channel_data(series: TimeSeries) -> bool:
    """Return whether a series holds numeric samples of rank 1 or 2."""
    data = series.data
    shape = getattr(data, "shape", None)
    dtype = getattr(data, "dtype", None)
    if shape is None or dtype is None:
        array = np.asarray(data)
        shape, dtype = array.shape, array.dtype
    return 1 <= len(shape) <= MAX_SERIES_RANK and bool(
        np.issubdtype(dtype, np.number)
    )


def _iter_plain_series(nwbfile: NWBFile) -> Iterator[TimeSeries]:
    """Yield the acquisition TimeSeries that are channels but not ElectricalSeries.

    A series whose data is not numeric samples of rank 1 or 2 is skipped with a
    warning. Order is the acquisition's insertion order.
    """
    for acq in nwbfile.acquisition.values():
        if isinstance(acq, ElectricalSeries) or not isinstance(acq, TimeSeries):
            continue
        if not _holds_channel_data(acq):
            logger.warning(
                "skipping acquisition %s: its data is not numeric samples of "
                "rank 1 or 2",
                acq.name,
            )
            continue
        yield acq


def _require_unique_ids(
    sources: Sequence[ContinuousSource | NwbUnitSource],
) -> None:
    """Raise ValueError when two sources share a channel id."""
    seen: set[str] = set()
    for source in sources:
        if source.id in seen:
            raise ValueError(
                f"duplicate channel id {source.id!r}: every ElectricalSeries "
                "electrode id, TimeSeries name, and Units table name must be "
                "distinct"
            )
        seen.add(source.id)


def build_sources_from_nwb(
    nwbfile: NWBFile,
) -> tuple[list[ContinuousSource], list[NwbUnitSource]]:
    """Discover the channel sources to write from an open NWB file.

    Every ElectricalSeries in the file's acquisition contributes one continuous
    source per channel column, in series order then channel order. Every other
    acquisition TimeSeries holding numeric samples of rank 1 or 2 follows, one
    source per column in the same order. Each Units table contributes one unit
    source, in _iter_units_tables order, with the waveform rate taken from the
    first ElectricalSeries. Raises ValueError if a Units table is present and
    no ElectricalSeries supplies that rate, if a series is sampled by
    timestamps rather than a rate, or if two sources share an id.
    """
    session_start = nwbfile.session_start_time
    series = [
        acq
        for acq in nwbfile.acquisition.values()
        if isinstance(acq, ElectricalSeries)
    ]
    continuous: list[ContinuousSource] = [
        NwbContinuousSource(es, channel_index, session_start)
        for es in series
        for channel_index in range(_channel_count(es))
    ]
    continuous.extend(
        NwbTimeSeriesSource(plain, channel_index, session_start)
        for plain in _iter_plain_series(nwbfile)
        for channel_index in range(_channel_count(plain))
    )

    units: list[NwbUnitSource] = []
    for table in _iter_units_tables(nwbfile):
        if not series:
            raise ValueError("unit channels need an ElectricalSeries rate")
        waveform_rate_hz = float(series[0].rate)
        units.append(NwbUnitSource(table, waveform_rate_hz, session_start))

    _require_unique_ids([*continuous, *units])
    return continuous, units
