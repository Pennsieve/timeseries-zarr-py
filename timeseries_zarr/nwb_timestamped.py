"""An NWB adapter presenting a gapped recording as a uniform sample grid."""

from datetime import datetime

import numpy as np
import numpy.typing as npt
from pynwb.ecephys import ElectricalSeries

from timeseries_zarr.constants import MICROSECONDS_PER_SECOND
from timeseries_zarr.grid import (
    Segment,
    build_segments,
    grid_length,
    segments_overlapping,
)
from timeseries_zarr.nwb_series import (
    electrode_id,
    electrode_name,
    offset_uv,
    read_uv,
)


class NwbTimestampedSource:
    """One channel of a timestamped ElectricalSeries, as a regular grid.

    An NWB series sampled by timestamps rather than a rate is how a recording
    with breaks in it has to be written, because a rate asserts unbroken
    regularity. This adapter places each sample on the grid position its
    timestamp names and reports NaN everywhere nothing was recorded, so the
    rest of the writer needs no gap awareness: it sees a channel with a rate
    and a sample count like any other.

    ``num_samples`` is therefore the length of the grid, which exceeds the
    number of recorded samples by the width of the gaps.
    """

    def __init__(
        self,
        electrical_series: ElectricalSeries,
        channel_index: int,
        session_start_time: datetime,
        *,
        rate_hz: float,
        segments: list[Segment] | None = None,
    ) -> None:
        """Bind one channel of a timestamped ElectricalSeries as a source.

        rate_hz is required and never inferred here; the caller decides where
        it comes from, so every construction site states it. Raises ValueError
        when the timestamps do not map onto a grid at that rate.

        segments is the series' gap map, built once and shared by every channel.
        Building it reads every timestamp, 15 GB on a three-week recording, and
        the channels of one series all share the same timestamps. Built here
        when not given.
        """
        self._series = electrical_series
        self._channel_index = channel_index
        self._session_start_time = session_start_time
        self._rate_hz = rate_hz
        self._segments: list[Segment] = (
            segments
            if segments is not None
            else build_segments(electrical_series.timestamps, rate_hz)
        )
        self._grid_length = grid_length(self._segments)

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

    @property
    def segments(self) -> list[Segment]:
        """The contiguous runs of recorded samples, in grid order."""
        return list(self._segments)

    def rate_hz(self) -> float:
        """Return the grid's sample rate in hertz."""
        return self._rate_hz

    def start_us(self) -> int:
        """Return the wall-clock microseconds of grid position 0.

        The session start plus the first timestamp, which is the position the
        grid is anchored on.
        """
        first = (
            float(np.asarray(self._series.timestamps[0:1], dtype=np.float64)[0])
            if self._grid_length
            else 0.0
        )
        start_s = self._session_start_time.timestamp() + first
        return round(start_s * MICROSECONDS_PER_SECOND)

    def num_samples(self) -> int:
        """Return the length of the grid, gaps included."""
        return self._grid_length

    def offset_uv(self) -> float:
        """Return the series' declared DC offset in microvolts.

        The gaps read back NaN, which no offset shifts, so the same number
        serves a gridded channel as a continuous one.
        """
        return offset_uv(self._series)

    def read_samples(self, start: int, stop: int) -> npt.NDArray[np.float32]:
        """Return the half-open [start, stop) grid window as float32 microvolts.

        Positions no run covers read back as NaN, which the format defines as
        no data and the folds propagate into the bins holding them. A window
        lying wholly inside a gap reads nothing from the series at all.
        """
        width = max(stop - start, 0)
        out = np.full(width, np.nan, dtype=np.float32)
        if width == 0:
            return out

        for segment in segments_overlapping(self._segments, start, stop):
            grid_from = max(segment.grid_start, start)
            grid_to = min(segment.grid_stop, stop)
            offset = grid_from - segment.grid_start
            sample_from = segment.sample_start + offset
            sample_to = sample_from + (grid_to - grid_from)
            out[grid_from - start : grid_to - start] = read_uv(
                self._series, self._channel_index, sample_from, sample_to
            )
        return out
