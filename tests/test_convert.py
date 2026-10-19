import re
import sys

import numpy as np
import pytest

from tests.conftest import ODD_EVENTS, ODD_LABELS, ODD_WAVEFORM_SAMPLES
from timeseries_zarr.convert import TIME_COLUMN
from timeseries_zarr.extras import missing_extra_message
from timeseries_zarr.read_bundle import open_bundle
from timeseries_zarr.read_continuous import Bins, Samples
from timeseries_zarr.read_event import Counts, EventTable, Waveform

_TEXT = ["onset", "", "caf\N{LATIN SMALL LETTER E WITH ACUTE}"]
_CHANNEL_KEYS = [[0], [], [1, 2]]


@pytest.fixture
def pd():
    return pytest.importorskip("pandas")


@pytest.fixture
def xr():
    return pytest.importorskip("xarray")


def _samples():
    return Samples(
        offset_us=1000,
        first=2,
        period_us=1953.125,
        values=np.array([1.5, np.nan, -2.0], dtype=np.float32),
    )


def _bins(*, full=True):
    def optional(values, dtype):
        return np.array(values, dtype=dtype) if full else None

    return Bins(
        level=2,
        offset_us=500,
        first=4,
        period_us=31250.0,
        min=np.array([-1.0, np.nan], dtype=np.float32),
        max=np.array([3.0, np.nan], dtype=np.float32),
        mean=optional([0.5, np.nan], np.float32),
        var=optional([1.25, np.nan], np.float32),
        count=optional([16, 3], np.uint32),
    )


def _waveform(count=2, rows="label"):
    return Waveform(
        values=np.arange(count * 3, dtype=np.float32).reshape(count, 3),
        period_us=100.0,
        event_sample=1,
        rows=rows,
    )


def _table():
    return EventTable(
        time_us=np.array([10, 20, 20], dtype=np.int64),
        duration_us=np.array([0, 5, 0], dtype=np.int64),
        label=np.array([0, 1, 0], dtype=np.uint16),
        value=np.array([0.25, 0.5, 0.75], dtype=np.float32),
        text=list(_TEXT),
        channel_keys=[list(keys) for keys in _CHANNEL_KEYS],
        waveform=_waveform(3),
        label_names=("seizure", "spike"),
    )


def _times_only_table():
    return EventTable(
        time_us=np.array([10, 20], dtype=np.int64),
        duration_us=None,
        label=None,
        value=None,
        text=None,
        channel_keys=None,
        waveform=_waveform(),
        label_names=None,
        whole_recording=True,
    )


_UNNAMEABLE_LABEL_NAMES = pytest.mark.parametrize(
    "label_names",
    [("noise", "noise"), ("time_us", "b")],
    ids=["repeated", "time"],
)


def _unnameable_message(label_names):
    return re.escape(
        "label_names must be distinct and must not include 'time_us' to "
        f"name count columns, got {label_names!r}"
    )


def _counts(label_names=("ripple", "fast-ripple"), bins=2):
    return Counts(
        level=1,
        first=3,
        period_us=1000.0,
        count=np.arange(bins * 2, dtype=np.uint32).reshape(bins, 2),
        label_names=label_names,
    )


def test_samples_to_series_indexes_the_values_by_time_us(pd):
    samples = _samples()
    series = samples.to_series()
    assert series.name is None
    assert series.dtype == np.float32
    assert series.index.name == TIME_COLUMN
    assert series.index.dtype == np.float64
    np.testing.assert_array_equal(series.index, samples.time_us)
    np.testing.assert_array_equal(series.to_numpy(), samples.values)


def test_samples_to_series_takes_a_name(pd):
    assert _samples().to_series(name="LA1").name == "LA1"


def test_samples_to_xarray_holds_the_values_along_time_us(xr):
    samples = _samples()
    array = samples.to_xarray()
    assert array.dims == (TIME_COLUMN,)
    assert array.name is None
    assert array.dtype == np.float32
    np.testing.assert_array_equal(array[TIME_COLUMN], samples.time_us)
    np.testing.assert_array_equal(array.values, samples.values)
    assert array.attrs == {"period_us": 1953.125, "offset_us": 1000}


def test_samples_to_xarray_adds_units_and_long_name_when_given(xr):
    array = _samples().to_xarray(name="LA1", unit="uV")
    assert array.name == "LA1"
    assert array.attrs == {
        "period_us": 1953.125,
        "offset_us": 1000,
        "units": "uV",
        "long_name": "LA1",
    }


def test_samples_to_xarray_adds_units_without_a_name(xr):
    array = _samples().to_xarray(unit="uV")
    assert array.name is None
    assert array.attrs == {
        "period_us": 1953.125,
        "offset_us": 1000,
        "units": "uV",
    }


def test_samples_to_xarray_adds_long_name_without_a_unit(xr):
    array = _samples().to_xarray(name="LA1")
    assert array.name == "LA1"
    assert array.attrs == {
        "period_us": 1953.125,
        "offset_us": 1000,
        "long_name": "LA1",
    }


def test_bins_to_dataframe_holds_min_and_max_then_each_statistic(pd):
    bins = _bins()
    frame = bins.to_dataframe()
    assert list(frame.columns) == ["min", "max", "mean", "var", "count"]
    assert frame.index.name == TIME_COLUMN
    np.testing.assert_array_equal(frame.index, bins.time_us)
    assert [str(dtype) for dtype in frame.dtypes] == [
        "float32",
        "float32",
        "float32",
        "float32",
        "uint32",
    ]
    np.testing.assert_array_equal(frame["count"], [16, 3])


def test_bins_to_dataframe_omits_absent_statistics(pd):
    assert list(_bins(full=False).to_dataframe().columns) == ["min", "max"]


def test_bins_to_xarray_holds_one_variable_per_statistic(xr):
    bins = _bins()
    dataset = bins.to_xarray()
    assert list(dataset.data_vars) == ["min", "max", "mean", "var", "count"]
    for name in ("min", "max", "mean", "var", "count"):
        assert dataset[name].dims == (TIME_COLUMN,)
        np.testing.assert_array_equal(dataset[name].values, getattr(bins, name))
    assert dataset["count"].dtype == np.uint32
    np.testing.assert_array_equal(dataset[TIME_COLUMN], bins.time_us)
    assert dataset.attrs == {"level": 2, "period_us": 31250.0}


def test_bins_to_xarray_omits_absent_statistics(xr):
    assert list(_bins(full=False).to_xarray().data_vars) == ["min", "max"]


def test_events_to_dataframe_holds_one_row_per_event_in_column_order(pd):
    frame = _table().to_dataframe()
    assert isinstance(frame.index, pd.RangeIndex)
    assert list(frame.columns) == [
        "time_us",
        "duration_us",
        "label",
        "label_name",
        "value",
        "text",
        "channel_keys",
    ]
    assert frame["time_us"].tolist() == [10, 20, 20]
    assert frame["time_us"].dtype == np.int64
    assert frame["duration_us"].dtype == np.int64
    assert frame["label"].dtype == np.uint16
    assert frame["value"].dtype == np.float32
    assert frame["label_name"].tolist() == ["seizure", "spike", "seizure"]
    assert frame["text"].tolist() == _TEXT
    assert frame["channel_keys"].tolist() == _CHANNEL_KEYS


def test_events_to_dataframe_omits_absent_columns_and_the_waveform(pd):
    frame = _times_only_table().to_dataframe()
    assert list(frame.columns) == ["time_us"]
    assert len(frame) == 2


def test_counts_to_dataframe_holds_one_column_per_label_name(pd):
    counts = _counts()
    frame = counts.to_dataframe()
    assert list(frame.columns) == ["ripple", "fast-ripple"]
    assert frame.index.name == TIME_COLUMN
    np.testing.assert_array_equal(frame.index, [3000.0, 4000.0])
    assert all(dtype == np.uint32 for dtype in frame.dtypes)
    np.testing.assert_array_equal(frame.to_numpy(), counts.count)


def test_counts_to_dataframe_names_columns_by_label_index_without_names(pd):
    frame = _counts(label_names=None).to_dataframe()
    assert list(frame.columns) == ["0", "1"]


@_UNNAMEABLE_LABEL_NAMES
def test_counts_to_dataframe_rejects_label_names_that_cannot_name_columns(
    pd, label_names
):
    counts = _counts(label_names=label_names)
    with pytest.raises(ValueError, match=_unnameable_message(label_names)):
        counts.to_dataframe()


def test_counts_to_xarray_has_time_us_and_label_dimensions(xr):
    counts = _counts()
    array = counts.to_xarray()
    assert array.dims == (TIME_COLUMN, "label")
    assert array.dtype == np.uint32
    assert array["label"].values.tolist() == ["ripple", "fast-ripple"]
    np.testing.assert_array_equal(array[TIME_COLUMN], [3000.0, 4000.0])
    np.testing.assert_array_equal(array.values, counts.count)


def test_counts_to_xarray_names_labels_by_label_index_without_names(xr):
    array = _counts(label_names=None).to_xarray()
    assert array["label"].values.tolist() == ["0", "1"]


@_UNNAMEABLE_LABEL_NAMES
def test_counts_to_xarray_rejects_label_names_that_cannot_name_columns(
    xr, label_names
):
    counts = _counts(label_names=label_names)
    with pytest.raises(ValueError, match=_unnameable_message(label_names)):
        counts.to_xarray()


def test_counts_to_xarray_keeps_the_shape_of_empty_counts(xr):
    array = _counts(bins=0).to_xarray()
    assert array.shape == (0, 2)
    assert array.dtype == np.uint32


def test_waveform_to_dataframe_holds_one_column_per_row(pd):
    waveform = _waveform()
    frame = waveform.to_dataframe()
    assert list(frame.columns) == ["0", "1"]
    assert frame.index.name == TIME_COLUMN
    np.testing.assert_array_equal(frame.index, [-100.0, 0.0, 100.0])
    assert all(dtype == np.float32 for dtype in frame.dtypes)
    np.testing.assert_array_equal(frame.to_numpy(), waveform.values.T)


def test_waveform_to_xarray_has_label_and_time_us_dimensions(xr):
    waveform = _waveform()
    array = waveform.to_xarray()
    assert array.dims == ("label", TIME_COLUMN)
    assert array.dtype == np.float32
    np.testing.assert_array_equal(array[TIME_COLUMN], [-100.0, 0.0, 100.0])
    np.testing.assert_array_equal(array.values, waveform.values)
    assert array.attrs == {"period_us": 100.0, "event_sample": 1}


def test_waveform_to_xarray_names_the_row_dimension_by_rows(xr):
    array = _waveform(rows="event").to_xarray()
    assert array.dims == ("event", TIME_COLUMN)


def test_waveform_to_xarray_keeps_the_shape_of_an_empty_waveform(xr):
    array = _waveform(count=0).to_xarray()
    assert array.shape == (0, 3)
    np.testing.assert_array_equal(array[TIME_COLUMN], [-100.0, 0.0, 100.0])


@pytest.mark.parametrize(
    ("read", "dimension", "rows"),
    [
        (lambda channel: channel.read().waveform, "event", ODD_EVENTS),
        (lambda channel: channel.waveforms.read(), "event", ODD_EVENTS),
        (lambda channel: channel.waveform_mean.read(), "label", ODD_LABELS),
    ],
    ids=["table", "waveforms", "waveform_mean"],
)
def test_read_waveforms_convert_with_a_row_per_event_or_label(
    xr, odd_bundle, read, dimension, rows
):
    with open_bundle(odd_bundle) as bundle:
        array = read(bundle.events[9]).to_xarray()
    assert array.dims == (dimension, TIME_COLUMN)
    assert array.shape == (rows, ODD_WAVEFORM_SAMPLES)


@pytest.mark.parametrize(
    "result",
    [_bins(), _bins(full=False), _counts(), _counts(label_names=None)]
    + [_waveform(), _waveform(count=0)],
    ids=["bins", "minmax", "counts", "unnamed", "waveform", "empty"],
)
def test_dataframe_of_columns_matches_the_indexed_conversion(pd, result):
    frame = pd.DataFrame(result.columns()).set_index(TIME_COLUMN)
    pd.testing.assert_frame_equal(frame, result.to_dataframe())


@pytest.mark.parametrize(
    "table", [_table(), _times_only_table()], ids=["full", "times"]
)
def test_dataframe_of_table_columns_matches_the_conversion(pd, table):
    frame = pd.DataFrame(table.columns())
    pd.testing.assert_frame_equal(frame, table.to_dataframe())


def test_dataframe_of_samples_columns_matches_the_series(pd):
    samples = _samples()
    frame = pd.DataFrame(samples.columns()).set_index(TIME_COLUMN)
    pd.testing.assert_series_equal(
        frame["values"], samples.to_series(name="values")
    )


_CONVERSIONS = [
    ("pandas", lambda: _samples().to_series()),
    ("xarray", lambda: _samples().to_xarray()),
    ("pandas", lambda: _bins().to_dataframe()),
    ("xarray", lambda: _bins().to_xarray()),
    ("pandas", lambda: _table().to_dataframe()),
    ("pandas", lambda: _counts().to_dataframe()),
    ("xarray", lambda: _counts().to_xarray()),
    ("pandas", lambda: _waveform().to_dataframe()),
    ("xarray", lambda: _waveform().to_xarray()),
]


@pytest.mark.parametrize(
    ("package", "convert"),
    _CONVERSIONS,
    ids=[
        "samples-series",
        "samples-xarray",
        "bins-dataframe",
        "bins-xarray",
        "events-dataframe",
        "counts-dataframe",
        "counts-xarray",
        "waveform-dataframe",
        "waveform-xarray",
    ],
)
def test_conversion_without_its_package_names_the_extra(
    monkeypatch, package, convert
):
    monkeypatch.setitem(sys.modules, package, None)
    message = missing_extra_message(package, package)
    with pytest.raises(ImportError, match=re.escape(message)) as info:
        convert()
    assert info.value.name == package


def test_mixed_bundle_reads_convert_to_pandas_and_xarray(pd, xr, mixed_bundle):
    with open_bundle(mixed_bundle) as bundle:
        channel = bundle.continuous[0]
        samples = channel.read(0, 1_000_000)
        series = samples.to_series(name=channel.name)
        assert len(series) == 256
        assert series.index[1] == 3906.25
        array = samples.to_xarray(name=channel.name, unit=channel.unit)
        assert array.attrs["units"] == "uV"
        frame = channel.read(level=2).to_dataframe()
        assert list(frame.columns) == ["min", "max", "mean", "count"]
        assert len(frame) == channel.levels[2].bins
        marks = bundle.events[3].read().to_dataframe()
        assert marks["text"].tolist() == [
            "spike",
            "spike",
            "seizure onset",
            "note",
            "seizure",
            "x",
        ]
        assert marks["channel_keys"].tolist() == [[0], [], [0, 1], [], [1], []]
        counts = bundle.events[4].read_counts(level=4).to_xarray()
        assert counts["label"].values.tolist() == ["ripple", "fast-ripple"]
        assert int(counts.sum()) == bundle.events[4].num_events
        mean = bundle.events[2].waveform_mean.read().to_xarray()
        assert mean.shape == (3, 16)
