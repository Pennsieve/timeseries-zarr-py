from datetime import UTC, datetime

import numpy as np
import pytest
from hdmf.common import DynamicTableRegion
from pynwb.testing.mock.device import mock_Device
from pynwb.testing.mock.ecephys import mock_ElectricalSeries
from pynwb.testing.mock.file import mock_NWBFile

from timeseries_zarr.grid import segments_overlapping
from timeseries_zarr.nwb_reader import build_sources_from_nwb
from timeseries_zarr.nwb_timestamped import NwbTimestampedSource

STARTED = datetime(2021, 6, 1, 12, 0, tzinfo=UTC)
RATE = 512.0
PERIOD = 1.0 / RATE


def _timestamped_series(data, timestamps, *, channels=1):
    """Build an ElectricalSeries carrying timestamps and no rate."""
    nwb = mock_NWBFile(session_start_time=STARTED)
    device = mock_Device(nwbfile=nwb)
    group = nwb.create_electrode_group(
        name="grp", description="d", location="loc", device=device
    )
    for _ in range(channels):
        nwb.add_electrode(group=group, location="loc")
    region = DynamicTableRegion(
        name="electrodes",
        data=list(range(channels)),
        description="r",
        table=nwb.electrodes,
    )
    series = mock_ElectricalSeries(
        nwbfile=nwb,
        data=np.asarray(data, dtype=float),
        electrodes=region,
        timestamps=np.asarray(timestamps, dtype=float),
        rate=None,
        starting_time=None,
    )
    return nwb, series


def _grid_len(before, gap_s, after):
    """Grid positions two runs occupy: the gap is measured from the last
    sample of the first run, not from the end of the run."""
    return (before - 1) + round(gap_s * RATE) + after


def _uv(*values):
    """Expected microvolts for raw values, which mock series carry as volts."""
    return np.asarray(values, dtype=np.float32) * 1e6


def _gapped(before, gap_s, after):
    """Samples and timestamps for two runs separated by gap_s seconds."""
    first = np.arange(before, dtype=np.float64) * PERIOD
    second = first[-1] + gap_s + np.arange(after, dtype=np.float64) * PERIOD
    timestamps = np.concatenate([first, second])
    data = np.arange(1, before + after + 1, dtype=np.float64).reshape(-1, 1)
    return data, timestamps


def test_grid_is_longer_than_the_sample_count_by_the_gap():
    data, timestamps = _gapped(100, 1.0, 100)
    _, series = _timestamped_series(data, timestamps)
    source = NwbTimestampedSource(series, 0, STARTED, rate_hz=RATE)

    assert source.num_samples() == _grid_len(100, 1.0, 100)
    assert source.rate_hz() == RATE
    assert len(source.segments) == 2


def test_gap_reads_back_as_nan_and_samples_survive():
    data, timestamps = _gapped(4, 1.0, 4)
    _, series = _timestamped_series(data, timestamps)
    source = NwbTimestampedSource(series, 0, STARTED, rate_hz=RATE)

    whole = source.read_samples(0, source.num_samples())
    assert np.count_nonzero(~np.isnan(whole)) == 8
    assert np.array_equal(whole[:4], _uv(1, 2, 3, 4))
    assert np.all(np.isnan(whole[4:-4]))
    assert np.array_equal(whole[-4:], _uv(5, 6, 7, 8))


def test_window_inside_a_gap_is_all_nan_and_touches_no_run():
    data, timestamps = _gapped(4, 1.0, 4)
    _, series = _timestamped_series(data, timestamps)
    source = NwbTimestampedSource(series, 0, STARTED, rate_hz=RATE)

    assert segments_overlapping(source.segments, 100, 200) == []
    window = source.read_samples(100, 200)
    assert window.shape == (100,)
    assert np.all(np.isnan(window))


def test_window_straddling_a_boundary_splits_correctly():
    data, timestamps = _gapped(4, 1.0, 4)
    _, series = _timestamped_series(data, timestamps)
    source = NwbTimestampedSource(series, 0, STARTED, rate_hz=RATE)

    window = source.read_samples(2, 6)
    assert np.array_equal(window[:2], _uv(3, 4))
    assert np.all(np.isnan(window[2:]))


def test_empty_window_is_empty():
    data, timestamps = _gapped(4, 1.0, 4)
    _, series = _timestamped_series(data, timestamps)
    source = NwbTimestampedSource(series, 0, STARTED, rate_hz=RATE)
    assert source.read_samples(5, 5).shape == (0,)


def test_start_us_anchors_on_the_first_timestamp():
    data, timestamps = _gapped(4, 1.0, 4)
    _, series = _timestamped_series(data, timestamps + 2.0)
    source = NwbTimestampedSource(series, 0, STARTED, rate_hz=RATE)

    expected = round((STARTED.timestamp() + 2.0) * 1_000_000)
    assert source.start_us() == expected


def test_no_gaps_behaves_like_a_regular_channel():
    data = np.arange(1, 17, dtype=np.float64).reshape(-1, 1)
    timestamps = np.arange(16, dtype=np.float64) * PERIOD
    _, series = _timestamped_series(data, timestamps)
    source = NwbTimestampedSource(series, 0, STARTED, rate_hz=RATE)

    assert source.num_samples() == 16
    assert not np.isnan(source.read_samples(0, 16)).any()


def test_build_sources_grids_a_timestamped_series_instead_of_raising():
    data, timestamps = _gapped(64, 1.0, 64)
    nwb, _ = _timestamped_series(data, timestamps)

    continuous, units = build_sources_from_nwb(nwb)

    assert units == []
    assert len(continuous) == 1
    source = continuous[0]
    assert isinstance(source, NwbTimestampedSource)
    assert source.rate_hz() == pytest.approx(RATE)
    assert source.num_samples() == _grid_len(64, 1.0, 64)
