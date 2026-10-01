"""Shared readers for one channel of an NWB series.

Both the rate-sampled and the timestamped adapters read a series the same way;
only their notion of where a sample sits on the time axis differs. Keeping
these here lets each adapter live in its own module without one importing the
other.
"""

from datetime import datetime

import numpy as np
import numpy.typing as npt
from pynwb import TimeSeries
from pynwb.ecephys import ElectricalSeries

from timeseries_zarr.constants import MICROSECONDS_PER_SECOND, UNIT_TO_UV


def require_rate(series: TimeSeries) -> float:
    """Return the series' sample rate in hertz, raising for a timestamps-only series."""
    if series.rate is None:
        raise ValueError(
            f"irregular sampling is not supported: {series.name} has "
            "timestamps and no rate"
        )
    return float(series.rate)


def start_us(series: TimeSeries, session_start_time: datetime) -> int:
    """Return wall-clock microseconds of sample 0, rounded to whole microseconds."""
    start_s: float = session_start_time.timestamp() + float(
        series.starting_time
    )
    return round(start_s * MICROSECONDS_PER_SECOND)


def read_column(
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


def channel_count(series: TimeSeries) -> int:
    """Return the number of channels a rank-1 or rank-2 series holds."""
    shape = series.data.shape
    return 1 if len(shape) == 1 else int(shape[1])


def electrode_id(series: ElectricalSeries, channel_index: int) -> str:
    """Return the selected electrode's table id."""
    electrodes = series.electrodes
    row_index = electrodes.data[channel_index]
    return str(electrodes.table.id[row_index])


def electrode_name(series: ElectricalSeries, channel_index: int) -> str:
    """Return the electrode's channel_name column, then its label, then its id."""
    electrodes = series.electrodes
    row_index = electrodes.data[channel_index]
    table = electrodes.table
    for column in ("channel_name", "label"):
        if column in table.colnames:
            return str(table[column][row_index])
    return electrode_id(series, channel_index)


def read_uv(
    series: ElectricalSeries, channel_index: int, start: int, stop: int
) -> npt.NDArray[np.float32]:
    """Return one channel's [start, stop) window as float32 microvolts.

    Applies the series' affine scaling, then converts from the series' own
    unit. Raises ValueError if that unit is not a recognized volts family.
    """
    column = read_column(series, channel_index, start, stop)
    scaled = column * float(series.conversion)
    if series.channel_conversion is not None:
        scaled = scaled * float(series.channel_conversion[channel_index])
    scaled = scaled + float(series.offset)
    unit = str(series.unit).lower()
    if unit not in UNIT_TO_UV:
        raise ValueError(f"unsupported ElectricalSeries unit: {unit!r}")
    return (scaled * UNIT_TO_UV[unit]).astype(np.float32)


def offset_uv(series: ElectricalSeries) -> float:
    """Return the series' declared DC offset in microvolts.

    NWB applies this after its conversion factors, so it is already in the
    series' own unit and needs only the volts-family conversion read_uv uses.
    An acquisition system that bakes its bias into the samples and declares no
    offset reports 0.0, and this cannot detect that.
    """
    unit = str(series.unit).lower()
    if unit not in UNIT_TO_UV:
        raise ValueError(f"unsupported ElectricalSeries unit: {unit!r}")
    return float(series.offset) * UNIT_TO_UV[unit]
