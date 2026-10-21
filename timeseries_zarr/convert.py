"""Conversions of read results to pandas and xarray objects.

Each conversion imports pandas or xarray when it runs, and raises
ImportError naming the extra to install when that package is missing. A
result indexed by time has an index or a dimension coordinate named
TIME_COLUMN. The xarray attribute that holds a unit is units.
"""

from types import ModuleType
from typing import TYPE_CHECKING, Any, Final

import numpy.typing as npt

from timeseries_zarr.constants import LABEL_DIMENSION
from timeseries_zarr.extras import require

if TYPE_CHECKING:
    import pandas as pd
    import xarray as xr

    from timeseries_zarr.read_continuous import Bins, Samples
    from timeseries_zarr.read_event import Counts, EventTable, Waveform
    from timeseries_zarr.stack import Stack

TIME_COLUMN: Final = "time_us"
"""Name of the time column, index, and dimension of a converted result."""

VALUES_COLUMN: Final = "values"
"""Name of the sample column of converted samples."""

CHANNEL_DIMENSION: Final = "channel"
"""Name of the channel dimension and coordinate of a converted stack."""

_PANDAS: Final = "pandas"
_XARRAY: Final = "xarray"
_UNITS: Final = "units"
_LONG_NAME: Final = "long_name"
_NAME: Final = "name"
_UNIT: Final = "unit"

type Columns = dict[str, npt.NDArray[Any]]
"""Arrays of a result keyed by column name, in column order."""


def _time_frame(pandas: ModuleType, columns: Columns) -> "pd.DataFrame":
    """Return a DataFrame of columns indexed by the time column."""
    frame: pd.DataFrame = pandas.DataFrame(columns).set_index(TIME_COLUMN)
    return frame


def samples_to_series(
    samples: "Samples", *, name: str | None = None
) -> "pd.Series":
    """Return the samples as a Series indexed by time_us.

    Args:
        samples: The samples to convert.
        name: The name of the Series.
    """
    pandas = require(_PANDAS, _PANDAS)
    columns = samples.columns()
    series: pd.Series = pandas.Series(
        columns[VALUES_COLUMN],
        index=pandas.Index(columns[TIME_COLUMN], name=TIME_COLUMN),
        name=name,
    )
    return series


def samples_to_xarray(
    samples: "Samples", *, name: str | None = None, unit: str | None = None
) -> "xr.DataArray":
    """Return the samples as a DataArray along time_us.

    The attributes hold period_us and offset_us, units when unit is given,
    and long_name when name is given.

    Args:
        samples: The samples to convert.
        name: The name of the DataArray.
        unit: The unit of the samples.
    """
    xarray = require(_XARRAY, _XARRAY)
    columns = samples.columns()
    attributes: dict[str, object] = {
        "period_us": samples.period_us,
        "offset_us": samples.offset_us,
    }
    if unit is not None:
        attributes[_UNITS] = unit
    if name is not None:
        attributes[_LONG_NAME] = name
    array: xr.DataArray = xarray.DataArray(
        columns[VALUES_COLUMN],
        coords={TIME_COLUMN: columns[TIME_COLUMN]},
        dims=(TIME_COLUMN,),
        name=name,
        attrs=attributes,
    )
    return array


def bins_to_dataframe(bins: "Bins") -> "pd.DataFrame":
    """Return the bins as a DataFrame indexed by time_us.

    It holds min and max, then mean, var, and count when the bins hold them.
    """
    pandas = require(_PANDAS, _PANDAS)
    return _time_frame(pandas, bins.columns())


def bins_to_xarray(bins: "Bins") -> "xr.Dataset":
    """Return the bins as a Dataset with one variable per statistic.

    The attributes hold level and period_us.
    """
    xarray = require(_XARRAY, _XARRAY)
    statistics = bins.columns()
    time_us = statistics.pop(TIME_COLUMN)
    dataset: xr.Dataset = xarray.Dataset(
        {name: (TIME_COLUMN, values) for name, values in statistics.items()},
        coords={TIME_COLUMN: time_us},
        attrs={"level": bins.level, "period_us": bins.period_us},
    )
    return dataset


def events_to_dataframe(table: "EventTable") -> "pd.DataFrame":
    """Return the events as a DataFrame with one row per event.

    The index numbers the rows from 0, and time_us is a column. Two events
    can share a time.
    """
    pandas = require(_PANDAS, _PANDAS)
    frame: pd.DataFrame = pandas.DataFrame(table.columns())
    return frame


def counts_to_dataframe(counts: "Counts") -> "pd.DataFrame":
    """Return the counts as a DataFrame indexed by time_us.

    It holds one column per label. A label_names entry names each column,
    and without label_names the label index as a decimal string names it.

    Raises:
        ValueError: If two label_names are equal or one is time_us.
    """
    pandas = require(_PANDAS, _PANDAS)
    return _time_frame(pandas, counts.columns())


def counts_to_xarray(counts: "Counts") -> "xr.DataArray":
    """Return the counts as a DataArray of dimensions time_us and label.

    The label coordinate holds label_names, or each label index as a decimal
    string when label_names is None.

    Raises:
        ValueError: If two label_names are equal or one is time_us.
    """
    xarray = require(_XARRAY, _XARRAY)
    columns = counts.columns()
    time_us = columns.pop(TIME_COLUMN)
    array: xr.DataArray = xarray.DataArray(
        counts.count,
        coords={TIME_COLUMN: time_us, LABEL_DIMENSION: list(columns)},
        dims=(TIME_COLUMN, LABEL_DIMENSION),
    )
    return array


def waveform_to_dataframe(waveform: "Waveform") -> "pd.DataFrame":
    """Return the snippets as a DataFrame indexed by time_us.

    It holds one column per snippet row. time_us counts from the event time.
    """
    pandas = require(_PANDAS, _PANDAS)
    return _time_frame(pandas, waveform.columns())


def waveform_to_xarray(waveform: "Waveform") -> "xr.DataArray":
    """Return the snippets as a DataArray with one row per snippet.

    The row dimension takes its name from rows, event or label, and the
    other dimension is time_us. time_us counts from the event time. The
    attributes hold period_us and event_sample.
    """
    xarray = require(_XARRAY, _XARRAY)
    array: xr.DataArray = xarray.DataArray(
        waveform.values,
        coords={TIME_COLUMN: waveform.time_us},
        dims=(waveform.rows, TIME_COLUMN),
        attrs={
            "period_us": waveform.period_us,
            "event_sample": waveform.event_sample,
        },
    )
    return array


def stack_to_dataframe(stack: "Stack") -> "pd.DataFrame":
    """Return the stack as a DataFrame indexed by time_us.

    It holds one column per channel, named by its key as a decimal string.
    The attrs of the DataFrame hold names, units, level, and member.
    """
    pandas = require(_PANDAS, _PANDAS)
    frame = _time_frame(pandas, stack.columns())
    frame.attrs.update(
        {
            "names": stack.names,
            "units": stack.units,
            "level": stack.level,
            "member": stack.member,
        }
    )
    return frame


def stack_to_xarray(stack: "Stack") -> "xr.DataArray":
    """Return the stack as a DataArray of dimensions channel and time_us.

    The channel coordinate holds the keys. The name and unit coordinates
    hold the name and unit of each channel along channel. The attributes
    hold level, period_us, and offset_us, plus member when it is not None
    and units when every channel has the same unit.
    """
    xarray = require(_XARRAY, _XARRAY)
    attributes: dict[str, object] = {"level": stack.level}
    if stack.member is not None:
        attributes["member"] = stack.member
    attributes["period_us"] = stack.period_us
    attributes["offset_us"] = stack.offset_us
    if len(set(stack.units)) == 1:
        attributes[_UNITS] = stack.units[0]
    array: xr.DataArray = xarray.DataArray(
        stack.values,
        coords={
            CHANNEL_DIMENSION: list(stack.keys),
            _NAME: (CHANNEL_DIMENSION, list(stack.names)),
            _UNIT: (CHANNEL_DIMENSION, list(stack.units)),
            TIME_COLUMN: stack.time_us,
        },
        dims=(CHANNEL_DIMENSION, TIME_COLUMN),
        attrs=attributes,
    )
    return array
