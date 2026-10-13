import numpy as np
import pytest

from timeseries_zarr.types import ChunkShard
from timeseries_zarr.write_event import EVENTS_KEY, write_events_array
from timeseries_zarr.zarr_io import open_group, write_region


def _sizing():
    return ChunkShard(chunk_shape=(4,), shard_shape=(8,))


def _write(tmp_path, source, origin_us=0, sizing=None):
    group = open_group(tmp_path / "bundle")
    write_events_array(group, source, origin_us, sizing or _sizing(), 5)
    return open_group(tmp_path / "bundle")


def test_write_events_array_stores_times_from_the_origin(
    tmp_path, event_source
):
    source = event_source([1000, 1500, 1500, 9000])
    stored = _write(tmp_path, source, origin_us=1000)[EVENTS_KEY][:]
    assert stored.tolist() == [0, 500, 500, 8000]
    assert stored.dtype == np.int64


def test_write_events_array_with_the_origin_at_zero_stores_the_times(
    tmp_path, event_source
):
    stored = _write(tmp_path, event_source([5, 6, 700]))[EVENTS_KEY][:]
    assert stored.tolist() == [5, 6, 700]


def test_write_events_array_is_exact_for_epoch_scale_microseconds(
    tmp_path, event_source
):
    origin = 1_700_000_000_123_456
    source = event_source([origin, origin + 1, origin + 999_999_999_999])
    stored = _write(tmp_path, source, origin_us=origin)[EVENTS_KEY][:]
    assert stored.tolist() == [0, 1, 999_999_999_999]


def test_write_events_array_declares_the_event_dimension_and_zero_fill(
    tmp_path, event_source
):
    arr = _write(tmp_path, event_source([1, 2, 3]))[EVENTS_KEY]
    assert arr.shape == (3,)
    assert dict(arr.attrs) == {}
    assert arr.metadata.dimension_names == ("event",)
    assert arr.fill_value == 0


def test_write_events_array_leaves_the_sources_times_unchanged(
    tmp_path, event_source
):
    times = np.array([1000, 2000, 3000], dtype=np.int64)
    _write(tmp_path, event_source(times), origin_us=1000)
    assert times.tolist() == [1000, 2000, 3000]


def test_write_events_array_empty_source_creates_a_zero_length_array(
    tmp_path, event_source
):
    arr = _write(tmp_path, event_source([]))[EVENTS_KEY]
    assert arr.shape == (0,)
    assert arr[:].shape == (0,)


def test_write_events_array_allows_equal_times_across_a_block_boundary(
    tmp_path, event_source
):
    times = [0] * 7 + [5, 5, 5] + [9] * 6
    stored = _write(tmp_path, event_source(times))[EVENTS_KEY][:]
    assert stored.tolist() == times


def test_write_events_array_writes_one_whole_shard_per_write(
    tmp_path, event_source, monkeypatch
):
    writes = []

    def spy(array, start, block):
        writes.append((start, block.shape[0]))
        write_region(array, start, block)

    monkeypatch.setattr("timeseries_zarr.write_event.write_region", spy)
    times = np.arange(26, dtype=np.int64) * 10
    stored = _write(tmp_path, event_source(times))[EVENTS_KEY][:]
    assert writes == [(0, 8), (8, 8), (16, 8), (24, 2)]
    assert np.array_equal(stored, times)


def test_write_events_array_rejects_a_time_before_the_origin(
    tmp_path, event_source
):
    with pytest.raises(ValueError, match="precedes the bundle origin by 1 "):
        _write(tmp_path, event_source([999, 1500]), origin_us=1000)


def test_write_events_array_rejects_times_that_decrease_within_a_block(
    tmp_path, event_source
):
    with pytest.raises(ValueError, match="non-decreasing"):
        _write(tmp_path, event_source([10, 30, 20, 40]))


def test_write_events_array_rejects_times_that_decrease_across_a_block_boundary(
    tmp_path, event_source
):
    times = [0, 1, 2, 3, 4, 5, 6, 70, 69, 71]
    with pytest.raises(ValueError, match="non-decreasing"):
        _write(tmp_path, event_source(times))


def test_write_events_array_rejects_a_time_before_the_origin_in_a_later_block(
    tmp_path, event_source
):
    times = [1000] * 8 + [999]
    with pytest.raises(ValueError, match="origin"):
        _write(tmp_path, event_source(times), origin_us=1000)
