from datetime import UTC, datetime

import numpy as np
import pytest
from hdmf.common import DynamicTableRegion
from pynwb import TimeSeries
from pynwb.ecephys import ElectricalSeries
from pynwb.file import Subject
from pynwb.misc import Units
from pynwb.testing.mock.base import mock_TimeSeries
from pynwb.testing.mock.device import mock_Device
from pynwb.testing.mock.ecephys import mock_ElectricalSeries
from pynwb.testing.mock.file import mock_NWBFile

import timeseries_zarr.nwb_reader as nwb_reader_module
from timeseries_zarr.nwb_reader import (
    NwbContinuousSource,
    NwbTimeSeriesSource,
    NwbUnitSource,
    build_meta_from_nwb,
    build_sources_from_nwb,
)

STARTED = datetime(2021, 6, 1, 12, 0, tzinfo=UTC)


def _series_with_electrode_columns(extra_columns, n=3):
    """Build an ElectricalSeries whose electrodes table carries extra columns.

    extra_columns maps a column name to its per-electrode values. The returned
    series has n channels wired to those n electrodes in order.
    """
    nwb = mock_NWBFile()
    device = mock_Device(nwbfile=nwb)
    group = nwb.create_electrode_group(
        name="grp", description="d", location="loc", device=device
    )
    for column in extra_columns:
        nwb.add_electrode_column(name=column, description=column)
    for i in range(n):
        nwb.add_electrode(
            group=group,
            location="loc",
            **{col: values[i] for col, values in extra_columns.items()},
        )
    region = DynamicTableRegion(
        name="electrodes",
        data=list(range(n)),
        description="r",
        table=nwb.electrodes,
    )
    return mock_ElectricalSeries(
        nwbfile=nwb, data=np.zeros((4, n)), electrodes=region
    )


def _make_units(unit_specs, *, name="my_units"):
    """Build a pynwb Units table from (spike_times, waveform_mean) pairs."""
    units = Units(name=name)
    for spike_times, waveform_mean in unit_specs:
        units.add_unit(
            spike_times=list(spike_times),
            waveform_mean=np.asarray(waveform_mean, dtype=float),
        )
    return units


# Two units whose spikes interleave in time; dense cluster ids are 0 and 1.
# Sorted events: 0.0(u0) 0.1(u1) 0.2(u0) 0.3(u1) 0.5(u0).
_WM0 = [1.0, 2.0, 3.0, 4.0]
_WM1 = [5.0, 6.0, 7.0, 8.0]
_TWO_UNITS = [([0.0, 0.2, 0.5], _WM0), ([0.1, 0.3], _WM1)]
_SORTED_TIMES = [0.0, 0.1, 0.2, 0.3, 0.5]
_SORTED_CLUSTERS = [0, 1, 0, 1, 0]
_SORTED_WAVEFORMS = [_WM0, _WM1, _WM0, _WM1, _WM0]


def test_continuous_init_stores_references(electrical_series):
    es = electrical_series()
    started = datetime(2021, 6, 1, 12, 0, tzinfo=UTC)
    src = NwbContinuousSource(es, 2, started)
    assert src._series is es
    assert src._channel_index == 2
    assert src._session_start_time == started


def test_rate_hz_from_series_rate(electrical_series):
    es = electrical_series()  # mock rate is 30000.0 Hz
    started = datetime(2021, 6, 1, 12, 0, tzinfo=UTC)
    src = NwbContinuousSource(es, 2, started)
    assert src.rate_hz() == 30000.0


def test_num_samples_is_time_axis_length(electrical_series):
    es = electrical_series()  # mock data shape is (10, 5)
    started = datetime(2021, 6, 1, 12, 0, tzinfo=UTC)
    src = NwbContinuousSource(es, 2, started)
    assert src.num_samples() == 10


def test_start_us_is_session_start_plus_offset_in_microseconds(
    electrical_series,
):
    es = electrical_series()  # mock starting_time is 0.0 s
    started = datetime(2021, 6, 1, 12, 0, tzinfo=UTC)
    src = NwbContinuousSource(es, 2, started)
    expected = round(started.timestamp() * 1_000_000)
    assert src.start_us() == expected


def test_id_is_a_string(electrical_series):
    es = electrical_series()
    started = datetime(2021, 6, 1, 12, 0, tzinfo=UTC)
    src = NwbContinuousSource(es, 2, started)
    assert isinstance(src.id, str)


def test_read_samples_returns_the_channel_column(electrical_series):
    # The mock's affine scaling is identity; 1e6 converts volts to microvolts.
    es = electrical_series()
    src = NwbContinuousSource(es, 2, STARTED)
    expected = (np.asarray(es.data[1:5, 2], dtype=np.float64) * 1e6).astype(
        np.float32
    )
    assert np.array_equal(src.read_samples(1, 5), expected)


def test_read_samples_dtype_is_float32(electrical_series):
    es = electrical_series()
    src = NwbContinuousSource(es, 2, STARTED)
    assert src.read_samples(0, 4).dtype == np.float32


def test_read_samples_shape_matches_window(electrical_series):
    es = electrical_series()
    src = NwbContinuousSource(es, 2, STARTED)
    assert src.read_samples(2, 7).shape == (5,)


def test_read_samples_empty_range_is_length_zero(electrical_series):
    es = electrical_series()
    src = NwbContinuousSource(es, 2, STARTED)
    assert src.read_samples(3, 3).shape == (0,)


def test_read_samples_applies_conversion_and_offset():
    data = np.arange(20, dtype=np.float64).reshape(4, 5)
    es = mock_ElectricalSeries(data=data, conversion=2.0, offset=10.0)
    src = NwbContinuousSource(es, 2, STARTED)
    expected = ((data[0:4, 2] * 2.0 + 10.0) * 1e6).astype(np.float32)
    assert np.array_equal(src.read_samples(0, 4), expected)


def test_read_samples_applies_channel_conversion():
    data = np.arange(20, dtype=np.float64).reshape(4, 5)
    channel_conversion = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    es = mock_ElectricalSeries(
        data=data, conversion=2.0, channel_conversion=channel_conversion
    )
    src = NwbContinuousSource(es, 2, STARTED)
    expected = ((data[0:4, 2] * 2.0 * 3.0) * 1e6).astype(np.float32)
    assert np.array_equal(src.read_samples(0, 4), expected)


def test_read_samples_without_channel_conversion_skips_it():
    data = np.arange(20, dtype=np.float64).reshape(4, 5)
    es = mock_ElectricalSeries(data=data, conversion=2.0)
    assert es.channel_conversion is None
    src = NwbContinuousSource(es, 2, STARTED)
    expected = ((data[0:4, 2] * 2.0) * 1e6).astype(np.float32)
    assert np.array_equal(src.read_samples(0, 4), expected)


def test_read_samples_normalizes_millivolts_to_microvolts():
    data = np.arange(20, dtype=np.float64).reshape(4, 5)
    es = mock_ElectricalSeries(data=data)
    es.fields["unit"] = "millivolts"
    src = NwbContinuousSource(es, 2, STARTED)
    expected = (data[0:4, 2] * 1e3).astype(np.float32)
    assert np.array_equal(src.read_samples(0, 4), expected)


def test_read_samples_raises_on_unknown_unit():
    data = np.arange(20, dtype=np.float64).reshape(4, 5)
    es = mock_ElectricalSeries(data=data)
    es.fields["unit"] = "furlongs"
    src = NwbContinuousSource(es, 2, STARTED)
    with pytest.raises(ValueError, match="furlongs"):
        src.read_samples(0, 4)


def test_continuous_unit_is_microvolts(electrical_series):
    es = electrical_series()
    src = NwbContinuousSource(es, 2, STARTED)
    assert src.unit == "uV"


def test_continuous_name_from_channel_name_column():
    es = _series_with_electrode_columns({"channel_name": ["a", "b", "c"]})
    src = NwbContinuousSource(es, 1, STARTED)
    assert src.name == "b"


def test_continuous_name_falls_back_to_label_column():
    es = _series_with_electrode_columns({"label": ["x", "y", "z"]})
    src = NwbContinuousSource(es, 2, STARTED)
    assert src.name == "z"


def test_continuous_name_prefers_channel_name_over_label():
    es = _series_with_electrode_columns(
        {"channel_name": ["a", "b", "c"], "label": ["x", "y", "z"]}
    )
    src = NwbContinuousSource(es, 0, STARTED)
    assert src.name == "a"


def test_continuous_name_falls_back_to_id_when_no_columns(electrical_series):
    es = electrical_series()
    src = NwbContinuousSource(es, 2, STARTED)
    assert src.name == src.id


def test_unit_num_events_sums_all_units_spikes():
    src = NwbUnitSource(_make_units(_TWO_UNITS), 30000.0, STARTED)
    assert src.num_events() == 5


def test_unit_points_per_event_is_waveform_mean_width():
    src = NwbUnitSource(_make_units(_TWO_UNITS), 30000.0, STARTED)
    assert src.points_per_event() == 4


def test_unit_rate_hz_is_the_passed_rate():
    src = NwbUnitSource(_make_units(_TWO_UNITS), 30000.0, STARTED)
    assert src.rate_hz() == 30000.0


def test_unit_start_us_is_session_start_in_microseconds():
    src = NwbUnitSource(_make_units(_TWO_UNITS), 30000.0, STARTED)
    assert src.start_us() == round(STARTED.timestamp() * 1_000_000)


def test_unit_id_is_the_table_name():
    src = NwbUnitSource(
        _make_units(_TWO_UNITS, name="probeA"), 30000.0, STARTED
    )
    assert src.id == "probeA"


def test_unit_name_is_the_table_name():
    src = NwbUnitSource(
        _make_units(_TWO_UNITS, name="probeA"), 30000.0, STARTED
    )
    assert src.name == "probeA"


def test_unit_unit_is_microvolts():
    src = NwbUnitSource(_make_units(_TWO_UNITS), 30000.0, STARTED)
    assert src.unit == "uV"


def test_unit_read_events_is_ascending_absolute_microseconds():
    src = NwbUnitSource(_make_units(_TWO_UNITS), 30000.0, STARTED)
    base = STARTED.timestamp()
    expected = np.array(
        [round((base + t) * 1_000_000) for t in _SORTED_TIMES],
        dtype=np.int64,
    )
    events = src.read_events(0, 5)
    assert events.dtype == np.int64
    assert np.array_equal(events, expected)
    assert np.all(np.diff(events) >= 0)


def test_unit_read_events_window_is_a_subset():
    src = NwbUnitSource(_make_units(_TWO_UNITS), 30000.0, STARTED)
    assert np.array_equal(src.read_events(1, 3), src.read_events(0, 5)[1:3])


def test_unit_read_labels_are_dense_cluster_ids_aligned_with_events():
    src = NwbUnitSource(_make_units(_TWO_UNITS), 30000.0, STARTED)
    labels = src.read_labels(0, 5)
    assert labels.dtype == np.uint16
    assert np.array_equal(labels, np.array(_SORTED_CLUSTERS, dtype=np.uint16))


def test_unit_read_waveforms_broadcasts_each_clusters_mean():
    src = NwbUnitSource(_make_units(_TWO_UNITS), 30000.0, STARTED)
    waveforms = src.read_waveforms(0, 5)
    assert waveforms.dtype == np.float32
    assert waveforms.shape == (5, 4)
    assert np.array_equal(
        waveforms, np.array(_SORTED_WAVEFORMS, dtype=np.float32)
    )


def test_unit_read_events_empty_range_is_length_zero():
    src = NwbUnitSource(_make_units(_TWO_UNITS), 30000.0, STARTED)
    assert src.read_events(2, 2).shape == (0,)


def test_unit_read_waveforms_empty_range_keeps_point_axis():
    src = NwbUnitSource(_make_units(_TWO_UNITS), 30000.0, STARTED)
    assert src.read_waveforms(2, 2).shape == (0, 4)


def test_unit_init_accepts_more_than_256_labels():
    """The old u1 column capped a sort at 256, which a dense probe passes."""
    specs = [([float(i)], [0.0]) for i in range(300)]
    src = NwbUnitSource(_make_units(specs), 30000.0, STARTED)
    assert src.num_events() == 300
    assert src.read_labels(0, 300).max() == 299


def test_unit_init_rejects_more_labels_than_u2_can_address(monkeypatch):
    # Building 65537 units is not worth the runtime; lower the ceiling.
    monkeypatch.setattr(nwb_reader_module, "MAX_LABEL_VALUES", 2)
    specs = [([float(i)], [0.0]) for i in range(3)]
    with pytest.raises(ValueError, match="at most"):
        NwbUnitSource(_make_units(specs), 30000.0, STARTED)


def test_build_yields_one_continuous_source_per_channel():
    nwb = mock_NWBFile()
    data = np.arange(50, dtype=np.float64).reshape(10, 5)
    mock_ElectricalSeries(nwbfile=nwb, data=data, rate=30000.0)
    continuous, units = build_sources_from_nwb(nwb)
    assert len(continuous) == 5
    assert units == []
    assert np.array_equal(
        continuous[2].read_samples(0, 10),
        (data[:, 2] * 1e6).astype(np.float32),
    )


def test_build_yields_unit_source_with_first_series_rate():
    nwb = mock_NWBFile()
    mock_ElectricalSeries(nwbfile=nwb, rate=30000.0)
    nwb.units = _make_units(_TWO_UNITS, name="units")
    continuous, units = build_sources_from_nwb(nwb)
    assert len(units) == 1
    assert units[0].rate_hz() == 30000.0
    assert units[0].num_events() == 5


def test_build_with_no_electrical_series_yields_no_continuous():
    nwb = mock_NWBFile()
    continuous, units = build_sources_from_nwb(nwb)
    assert continuous == []
    assert units == []


def test_build_with_no_units_yields_no_unit_sources():
    nwb = mock_NWBFile()
    mock_ElectricalSeries(nwbfile=nwb, rate=30000.0)
    _, units = build_sources_from_nwb(nwb)
    assert units == []


def test_build_units_without_series_rate_raises():
    nwb = mock_NWBFile()
    nwb.units = _make_units(_TWO_UNITS, name="units")
    with pytest.raises(ValueError):
        build_sources_from_nwb(nwb)


def _add_units_to_module(nwb, module_name, units_table):
    """Attach a Units table to a processing module, creating the module if absent."""
    if module_name in nwb.processing:
        module = nwb.processing[module_name]
    else:
        module = nwb.create_processing_module(name=module_name, description="d")
    module.add(units_table)


def test_build_discovers_units_in_a_processing_module():
    nwb = mock_NWBFile()
    mock_ElectricalSeries(nwbfile=nwb, rate=30000.0)
    _add_units_to_module(nwb, "ecephys", _make_units(_TWO_UNITS, name="sorted"))
    _, units = build_sources_from_nwb(nwb)
    assert len(units) == 1
    assert units[0].id == "sorted"


def test_build_orders_root_units_before_module_units():
    nwb = mock_NWBFile()
    mock_ElectricalSeries(nwbfile=nwb, rate=30000.0)
    nwb.units = _make_units(_TWO_UNITS, name="units")
    _add_units_to_module(nwb, "ecephys", _make_units(_TWO_UNITS, name="module"))
    _, units = build_sources_from_nwb(nwb)
    assert [u.id for u in units] == ["units", "module"]


def test_build_orders_modules_by_name():
    nwb = mock_NWBFile()
    mock_ElectricalSeries(nwbfile=nwb, rate=30000.0)
    _add_units_to_module(nwb, "zebra", _make_units(_TWO_UNITS, name="z_units"))
    _add_units_to_module(nwb, "alpha", _make_units(_TWO_UNITS, name="a_units"))
    _, units = build_sources_from_nwb(nwb)
    assert [u.id for u in units] == ["a_units", "z_units"]


def test_build_orders_containers_within_a_module_by_name():
    nwb = mock_NWBFile()
    mock_ElectricalSeries(nwbfile=nwb, rate=30000.0)
    _add_units_to_module(nwb, "ecephys", _make_units(_TWO_UNITS, name="second"))
    _add_units_to_module(nwb, "ecephys", _make_units(_TWO_UNITS, name="first"))
    _, units = build_sources_from_nwb(nwb)
    assert [u.id for u in units] == ["first", "second"]


def test_build_module_units_without_series_rate_raises():
    nwb = mock_NWBFile()
    _add_units_to_module(nwb, "ecephys", _make_units(_TWO_UNITS, name="sorted"))
    with pytest.raises(ValueError):
        build_sources_from_nwb(nwb)


def _one_channel_series(data, *, rate=30000.0):
    """Build a rank-1 ElectricalSeries wired to one electrode, in a new file."""
    nwb = mock_NWBFile()
    device = mock_Device(nwbfile=nwb)
    group = nwb.create_electrode_group(
        name="grp", description="d", location="loc", device=device
    )
    nwb.add_electrode(group=group, location="loc")
    region = DynamicTableRegion(
        name="electrodes", data=[0], description="r", table=nwb.electrodes
    )
    series = ElectricalSeries(
        name="single",
        data=np.asarray(data),
        electrodes=region,
        rate=rate,
        starting_time=0.0,
    )
    nwb.add_acquisition(series)
    return nwb, series


def test_continuous_reads_a_rank_one_series_as_one_channel():
    nwb, es = _one_channel_series(np.arange(8, dtype=np.float64))
    src = NwbContinuousSource(es, 0, STARTED)
    assert src.num_samples() == 8
    assert np.array_equal(
        src.read_samples(2, 5), (np.arange(2, 5) * 1e6).astype(np.float32)
    )
    continuous, _ = build_sources_from_nwb(nwb)
    assert len(continuous) == 1


def test_continuous_rejects_a_series_sampled_by_timestamps():
    es = mock_ElectricalSeries(
        data=np.zeros((10, 2)), timestamps=np.arange(10) / 512.0
    )
    with pytest.raises(ValueError, match="irregular sampling"):
        NwbContinuousSource(es, 0, STARTED)


def test_timeseries_source_keeps_a_unit_outside_the_volts_family():
    ts = mock_TimeSeries(
        name="SpO2",
        data=np.arange(10, dtype=np.float64),
        unit="%",
        rate=512.0,
        conversion=2.0,
        offset=1.0,
    )
    src = NwbTimeSeriesSource(ts, 0, STARTED)
    assert src.id == "SpO2"
    assert src.name == "SpO2"
    assert src.unit == "%"
    assert src.rate_hz() == 512.0
    assert src.num_samples() == 10
    assert src.start_us() == round(STARTED.timestamp() * 1_000_000)
    expected = (np.arange(4) * 2.0 + 1.0).astype(np.float32)
    assert np.array_equal(src.read_samples(0, 4), expected)
    assert src.read_samples(0, 4).dtype == np.float32


def test_timeseries_source_normalizes_a_volts_family_unit_to_microvolts():
    ts = mock_TimeSeries(
        name="aux", data=np.arange(10, dtype=np.float64), unit="mV", rate=512.0
    )
    src = NwbTimeSeriesSource(ts, 0, STARTED)
    assert src.unit == "uV"
    assert np.array_equal(
        src.read_samples(0, 3), (np.arange(3) * 1e3).astype(np.float32)
    )


def test_timeseries_source_names_the_columns_of_a_rank_two_series():
    data = np.arange(20, dtype=np.float64).reshape(10, 2)
    ts = mock_TimeSeries(name="resp", data=data, unit="cmH2O", rate=512.0)
    first = NwbTimeSeriesSource(ts, 0, STARTED)
    second = NwbTimeSeriesSource(ts, 1, STARTED)
    assert (first.id, second.id) == ("resp/0", "resp/1")
    assert np.array_equal(
        second.read_samples(0, 3), data[0:3, 1].astype(np.float32)
    )


def test_timeseries_source_rejects_a_series_sampled_by_timestamps():
    ts = mock_TimeSeries(
        name="irregular",
        data=np.zeros(10),
        unit="%",
        timestamps=np.arange(10) / 512.0,
    )
    with pytest.raises(ValueError, match="irregular sampling"):
        NwbTimeSeriesSource(ts, 0, STARTED)


def test_timeseries_source_rejects_data_above_rank_two():
    ts = mock_TimeSeries(
        name="video", data=np.zeros((4, 2, 2)), unit="n/a", rate=1.0
    )
    with pytest.raises(ValueError, match="rank"):
        NwbTimeSeriesSource(ts, 0, STARTED)


def test_build_yields_plain_series_columns_after_electrical_series_columns():
    nwb = mock_NWBFile()
    mock_ElectricalSeries(nwbfile=nwb, data=np.zeros((10, 2)), rate=512.0)
    mock_TimeSeries(
        name="SpO2", nwbfile=nwb, data=np.zeros(10), unit="%", rate=512.0
    )
    mock_TimeSeries(
        name="resp",
        nwbfile=nwb,
        data=np.zeros((10, 2)),
        unit="cmH2O",
        rate=512.0,
    )
    continuous, _ = build_sources_from_nwb(nwb)
    assert [src.id for src in continuous] == [
        "0",
        "1",
        "SpO2",
        "resp/0",
        "resp/1",
    ]
    assert [src.unit for src in continuous] == [
        "uV",
        "uV",
        "%",
        "cmH2O",
        "cmH2O",
    ]


def test_build_skips_a_series_that_is_not_numeric_samples(caplog):
    nwb = mock_NWBFile()
    mock_ElectricalSeries(nwbfile=nwb, data=np.zeros((10, 2)), rate=512.0)
    nwb.add_acquisition(
        TimeSeries(name="notes", data=["a", "b"], unit="n/a", rate=1.0)
    )
    mock_TimeSeries(
        name="video",
        nwbfile=nwb,
        data=np.zeros((4, 2, 2)),
        unit="n/a",
        rate=1.0,
    )
    with caplog.at_level("WARNING"):
        continuous, _ = build_sources_from_nwb(nwb)
    assert [src.id for src in continuous] == ["0", "1"]
    skipped = [record.getMessage() for record in caplog.records]
    assert any("notes" in message for message in skipped)
    assert any("video" in message for message in skipped)


def test_build_rejects_two_sources_with_one_id():
    nwb = mock_NWBFile()
    mock_ElectricalSeries(nwbfile=nwb, data=np.zeros((10, 2)), rate=512.0)
    mock_TimeSeries(
        name="0", nwbfile=nwb, data=np.zeros(10), unit="%", rate=512.0
    )
    with pytest.raises(ValueError, match="duplicate channel id '0'"):
        build_sources_from_nwb(nwb)


def test_build_reports_a_series_sampled_by_timestamps():
    nwb = mock_NWBFile()
    mock_ElectricalSeries(nwbfile=nwb, data=np.zeros((10, 2)), rate=512.0)
    mock_TimeSeries(
        name="irregular",
        nwbfile=nwb,
        data=np.zeros(10),
        unit="%",
        timestamps=np.arange(10) / 512.0,
    )
    with pytest.raises(ValueError, match="irregular sampling.*irregular"):
        build_sources_from_nwb(nwb)


def test_build_reads_a_gapped_series_timestamps_once(monkeypatch):
    """Every channel shares the timestamps, so the gap map is built once."""
    nwb = mock_NWBFile()
    timestamps = np.concatenate([np.arange(64), np.arange(80, 144)]) / 512.0
    mock_ElectricalSeries(
        nwbfile=nwb, data=np.zeros((128, 4)), timestamps=timestamps
    )
    calls = []
    real = nwb_reader_module.build_segments
    monkeypatch.setattr(
        nwb_reader_module,
        "build_segments",
        lambda *args, **kwargs: calls.append(1) or real(*args, **kwargs),
    )
    continuous, _ = build_sources_from_nwb(nwb)
    assert len(continuous) == 4
    assert len(calls) == 1
    # Each channel still sees the gap.
    assert all(source.num_samples() == 144 for source in continuous)


def test_build_meta_from_nwb_carries_the_subject_across():
    nwb = mock_NWBFile(
        subject=Subject(
            subject_id="sub-01",
            species="Homo sapiens",
            sex="F",
            age="P41Y",
            date_of_birth=datetime(1985, 3, 2, tzinfo=UTC),
        )
    )
    meta = build_meta_from_nwb(nwb)
    assert meta.subject["subject_id"] == "sub-01"
    assert meta.subject["species"] == "Homo sapiens"
    assert meta.subject["sex"] == "F"
    # A date has to survive as JSON text, not as a datetime object.
    assert meta.subject["date_of_birth"] == "1985-03-02T00:00:00+00:00"


def test_build_meta_from_nwb_without_a_subject():
    meta = build_meta_from_nwb(mock_NWBFile())
    assert meta.subject == {}
    assert "identifier" in meta.session


def test_build_meta_from_nwb_drops_unset_fields():
    nwb = mock_NWBFile(subject=Subject(subject_id="sub-01"))
    meta = build_meta_from_nwb(nwb)
    # NWB leaves most of the subject table None; meta/ shows only what it knew.
    assert set(meta.subject) == {"subject_id"}


def test_build_meta_from_nwb_records_the_converter_and_devices():
    nwb = mock_NWBFile()
    mock_Device(name="amp-1", nwbfile=nwb)
    meta = build_meta_from_nwb(nwb)
    assert meta.source["converter"] == "timeseries-zarr-py"
    assert meta.source["devices"] == ["amp-1"]


def test_build_meta_from_nwb_carries_no_start_us():
    # The onset is the earliest channel start, which only the bundle knows.
    assert "start_us" not in build_meta_from_nwb(mock_NWBFile()).session


def test_continuous_source_reports_the_declared_offset_in_uv():
    es = mock_ElectricalSeries(rate=100.0, offset=0.5)
    source = NwbContinuousSource(es, 0, datetime(2026, 1, 1, tzinfo=UTC))
    # 0.5 V of bias is 500000 uV, the unit the samples are stored in.
    assert source.offset_uv() == pytest.approx(500_000.0)


def test_continuous_source_reports_no_offset_when_none_is_declared():
    es = mock_ElectricalSeries(rate=100.0)
    source = NwbContinuousSource(es, 0, datetime(2026, 1, 1, tzinfo=UTC))
    assert source.offset_uv() == 0.0


def test_timeseries_source_scales_the_offset_like_its_samples():
    series = mock_TimeSeries(rate=100.0, unit="millivolts", offset=2.0)
    source = NwbTimeSeriesSource(series, 0, datetime(2026, 1, 1, tzinfo=UTC))
    assert source.unit == "uV"
    assert source.offset_uv() == pytest.approx(2000.0)


def test_timeseries_source_keeps_the_offset_in_a_non_volts_unit():
    series = mock_TimeSeries(rate=100.0, unit="degrees", offset=2.0)
    source = NwbTimeSeriesSource(series, 0, datetime(2026, 1, 1, tzinfo=UTC))
    assert source.unit == "degrees"
    assert source.offset_uv() == pytest.approx(2.0)
