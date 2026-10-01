"""Concrete NWB adapters implementing the channel-source protocols."""

import logging
from collections.abc import Iterator, Sequence
from datetime import date, datetime

import numpy as np
import numpy.typing as npt
from pynwb import NWBFile, TimeSeries
from pynwb.ecephys import ElectricalSeries
from pynwb.misc import Units

from timeseries_zarr.constants import (
    MAX_LABEL_VALUES,
    MICROSECONDS_PER_SECOND,
    UNIT_TO_UV,
)
from timeseries_zarr.grid import build_segments, derive_rate_hz
from timeseries_zarr.nwb_series import (
    channel_count,
    electrode_id,
    electrode_name,
    offset_uv,
    read_column,
    read_uv,
    require_rate,
    start_us,
)
from timeseries_zarr.nwb_timestamped import NwbTimestampedSource
from timeseries_zarr.types import RecordingMeta

logger = logging.getLogger(__name__)

MAX_SERIES_RANK = 2
"""Highest data rank a series can have and still be read as channels."""


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
        self._rate_hz = require_rate(electrical_series)

    @property
    def id(self) -> str:
        """The selected electrode's table id."""
        return electrode_id(self._series, self._channel_index)

    @property
    def name(self) -> str:
        """The electrode's channel_name column, then its label, then the id."""
        return electrode_name(self._series, self._channel_index)

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
        return start_us(self._series, self._session_start_time)

    def num_samples(self) -> int:
        """Return the length of the series' time axis, shared by every channel."""
        return int(self._series.data.shape[0])

    def offset_uv(self) -> float:
        """Return the series' declared DC offset in microvolts."""
        return offset_uv(self._series)

    def read_samples(self, start: int, stop: int) -> npt.NDArray[np.float32]:
        """Return the half-open [start, stop) sample window as float32 microvolts.

        Applies the series' affine scaling, then converts from the series' own
        unit to microvolts. Raises ValueError if that unit is not a recognized
        volts family. An empty range (stop <= start) yields a length-0 array.
        """
        return read_uv(self._series, self._channel_index, start, stop)


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
        self._rate_hz = require_rate(series)
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
        return start_us(self._series, self._session_start_time)

    def num_samples(self) -> int:
        """Return the length of the series' time axis, shared by every channel."""
        return int(self._series.data.shape[0])

    def offset_uv(self) -> float:
        """Return the series' declared DC offset in the channel's own unit."""
        offset = float(self._series.offset)
        return offset if self._uv_factor is None else offset * self._uv_factor

    def read_samples(self, start: int, stop: int) -> npt.NDArray[np.float32]:
        """Return the [start, stop) sample window as float32 in the channel unit.

        Applies the series' conversion and offset, then the volts-to-microvolts
        factor when the unit is in the volts family. An empty range
        (stop <= start) yields a length-0 array.
        """
        column = read_column(self._series, self._channel_index, start, stop)
        scaled = column * float(self._series.conversion) + float(
            self._series.offset
        )
        if self._uv_factor is not None:
            scaled = scaled * self._uv_factor
        return scaled.astype(np.float32)


type ContinuousSource = (
    NwbContinuousSource | NwbTimeSeriesSource | NwbTimestampedSource
)


class NwbUnitSource:
    """A unit (spike) channel backed by an NWB Units table.

    Flattens the table's per-cluster rows into the per-event streams the
    bundle stores: all spikes merged into one timestamp series sorted
    ascending, each event tagged with its cluster's dense u2 label in table
    row order (not the upstream unit id) and carrying that cluster's
    waveform_mean.
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
        labels than the u2 column can address.
        """
        unit_count = len(units)
        if unit_count > MAX_LABEL_VALUES:
            raise ValueError(
                f"an event channel holds at most {MAX_LABEL_VALUES} labels"
            )

        self._id = str(units.name)
        self._unit_count = unit_count
        self._rate_hz = float(waveform_rate_hz)
        self._start_us = round(
            session_start_time.timestamp() * MICROSECONDS_PER_SECOND
        )

        session_s = session_start_time.timestamp()
        times: list[npt.NDArray[np.float64]] = []
        clusters: list[npt.NDArray[np.uint16]] = []
        waveforms: list[npt.NDArray[np.float32]] = []
        for cluster_id in range(unit_count):
            spike_times: npt.NDArray[np.float64] = np.asarray(
                units.get_unit_spike_times(cluster_id), dtype=np.float64
            )
            mean: npt.NDArray[np.float32] = np.asarray(
                units["waveform_mean"][cluster_id], dtype=np.float32
            )
            count = spike_times.shape[0]
            times.append(spike_times)
            clusters.append(np.full(count, cluster_id, dtype=np.uint16))
            waveforms.append(np.broadcast_to(mean, (count, mean.shape[0])))

        all_times = np.concatenate(times)
        order = np.argsort(all_times, kind="stable")
        self._events = (
            ((all_times[order] + session_s) * MICROSECONDS_PER_SECOND)
            .round()
            .astype(np.int64)
        )
        self._labels = np.concatenate(clusters)[order]
        self._waveforms = np.concatenate(waveforms)[order]

    @property
    def id(self) -> str:
        """The Units table's name."""
        return self._id

    @property
    def name(self) -> str:
        """The Units table's name, the same value as id."""
        return self._id

    @property
    def unit(self) -> str:
        """Always "uV".

        waveform_mean carries no unit metadata, so the amplitudes are stored
        unscaled.
        """
        return "uV"

    def rate_hz(self) -> float:
        """Return the waveform sample rate in hertz, as bound at construction."""
        return self._rate_hz

    def start_us(self) -> int:
        """Return the recording start, rounded to whole microseconds."""
        return self._start_us

    def num_events(self) -> int:
        """Return the total number of spike events across all units."""
        return int(self._events.shape[0])

    def num_labels(self) -> int:
        """Return the number of clusters in the table.

        The label space, not the labels seen: a cluster the sorter kept but
        that fired nothing still owns a column of the count pyramid.
        """
        return self._unit_count

    def points_per_event(self) -> int:
        """Return the width of the waveform_mean template, shared by every event."""
        return int(self._waveforms.shape[1])

    def read_events(self, start: int, stop: int) -> npt.NDArray[np.int64]:
        """Return the half-open [start, stop) window of event timestamps.

        Absolute-microsecond int64. An empty range (stop <= start) yields a
        length-0 array.
        """
        return self._events[start:stop]

    def read_labels(self, start: int, stop: int) -> npt.NDArray[np.uint16]:
        """Return the half-open [start, stop) window of per-event labels.

        The cluster that produced each spike, aligned with the events at the
        same indices. An empty range yields a length-0 array.
        """
        return self._labels[start:stop]

    def read_waveforms(self, start: int, stop: int) -> npt.NDArray[np.float32]:
        """Return float32 waveforms for events [start, stop).

        Row k is the waveform_mean of the cluster that produced event k. An
        empty range yields a (0, points_per_event) array.
        """
        return self._waveforms[start:stop]


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


def _electrical_sources(
    series: ElectricalSeries, session_start_time: datetime
) -> list[ContinuousSource]:
    """Return one source per channel of an ElectricalSeries.

    A series carrying a rate is read directly. A series carrying timestamps
    is a recording with breaks in it, and is placed on a uniform grid at a
    rate derived from its own first gap-free run; the grid reports NaN where
    nothing was recorded.
    """
    channels = range(channel_count(series))
    if series.rate is not None:
        return [
            NwbContinuousSource(series, index, session_start_time)
            for index in channels
        ]

    rate_hz = derive_rate_hz(series.timestamps)
    logger.info(
        "%s is sampled by timestamps; gridding at %.6f Hz derived from its "
        "first gap-free run",
        series.name,
        rate_hz,
    )
    # One gap map for the whole series: every channel shares its timestamps.
    segments = build_segments(series.timestamps, rate_hz)
    return [
        NwbTimestampedSource(
            series,
            index,
            session_start_time,
            rate_hz=rate_hz,
            segments=segments,
        )
        for index in channels
    ]


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
        source
        for es in series
        for source in _electrical_sources(es, session_start)
    ]
    continuous.extend(
        NwbTimeSeriesSource(plain, channel_index, session_start)
        for plain in _iter_plain_series(nwbfile)
        for channel_index in range(channel_count(plain))
    )

    units: list[NwbUnitSource] = []
    for table in _iter_units_tables(nwbfile):
        if not series:
            raise ValueError("unit channels need an ElectricalSeries rate")
        waveform_rate_hz = float(series[0].rate)
        units.append(NwbUnitSource(table, waveform_rate_hz, session_start))

    _require_unique_ids([*continuous, *units])
    return continuous, units


def _jsonable(value: object) -> object:
    """Return value as something json.dumps can write, dates as ISO strings."""
    return value.isoformat() if isinstance(value, date | datetime) else value


def _present(fields: dict[str, object]) -> dict[str, object]:
    """Drop the keys NWB left unset, so meta/ shows only what the file knew."""
    return {
        key: _jsonable(value)
        for key, value in fields.items()
        if value is not None
    }


def build_meta_from_nwb(nwbfile: NWBFile) -> RecordingMeta:
    """Return the recording metadata for the bundle's meta/ group.

    Everything identifying that this writer carries across from NWB lands here
    and nowhere else, which is what makes deleting one directory a complete
    de-identification. The content is deliberately unschematized, so the NWB
    subject table crosses over nearly verbatim.

    session.start_us is not set here. The bundle's onset is the earliest start
    across every channel, which only the bundle knows.
    """
    subject = nwbfile.subject
    return RecordingMeta(
        subject=_present(
            {
                "subject_id": subject.subject_id,
                "species": subject.species,
                "sex": subject.sex,
                "age": subject.age,
                "description": subject.description,
                "date_of_birth": subject.date_of_birth,
            }
            if subject is not None
            else {}
        ),
        session=_present(
            {
                "session_id": nwbfile.session_id,
                "description": nwbfile.session_description,
                "identifier": nwbfile.identifier,
            }
        ),
        source=_present(
            {
                "converter": "timeseries-zarr-py",
                "devices": sorted(nwbfile.devices) or None,
            }
        ),
    )
