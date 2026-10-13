import numpy as np
import pytest

from timeseries_zarr.types import ChunkShard
from timeseries_zarr.write_event import (
    DURATIONS_KEY,
    EVENTS_KEY,
    LABELS_KEY,
    write_durations_array,
    write_events_array,
    write_labels_array,
)
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


def _write_durations(tmp_path, source, sizing=None):
    group = open_group(tmp_path / "bundle")
    write_durations_array(group, source, sizing or _sizing(), 5)
    return open_group(tmp_path / "bundle")


def test_write_durations_array_round_trips_microseconds(tmp_path, event_source):
    durations = [0, 5_000_000, 12, 0, 90_000_000]
    source = event_source([1, 2, 3, 4, 5], durations=durations)
    stored = _write_durations(tmp_path, source)[DURATIONS_KEY][:]
    assert stored.tolist() == durations
    assert stored.dtype == np.int64


def test_write_durations_array_declares_the_event_dimension_and_zero_fill(
    tmp_path, event_source
):
    source = event_source([1, 2], durations=[3, 4])
    arr = _write_durations(tmp_path, source)[DURATIONS_KEY]
    assert arr.shape == (2,)
    assert dict(arr.attrs) == {}
    assert arr.metadata.dimension_names == ("event",)
    assert arr.fill_value == 0


def test_write_durations_array_empty_source_creates_a_zero_length_array(
    tmp_path, event_source
):
    source = event_source([], durations=[], max_duration_us=0)
    arr = _write_durations(tmp_path, source)[DURATIONS_KEY]
    assert arr.shape == (0,)


def test_write_durations_array_accepts_a_duration_equal_to_the_bound(
    tmp_path, event_source
):
    source = event_source([1, 2], durations=[0, 9], max_duration_us=9)
    assert _write_durations(tmp_path, source)[DURATIONS_KEY][:].tolist() == [
        0,
        9,
    ]


def test_write_durations_array_writes_one_whole_shard_per_write(
    tmp_path, event_source, monkeypatch
):
    writes = []

    def spy(array, start, block):
        writes.append((start, block.shape[0]))
        write_region(array, start, block)

    monkeypatch.setattr("timeseries_zarr.write_event.write_region", spy)
    durations = np.arange(26, dtype=np.int64)
    source = event_source(np.arange(26), durations=durations)
    stored = _write_durations(tmp_path, source)[DURATIONS_KEY][:]
    assert writes == [(0, 8), (8, 8), (16, 8), (24, 2)]
    assert np.array_equal(stored, durations)


def test_write_durations_array_rejects_a_negative_duration(
    tmp_path, event_source
):
    source = event_source([1, 2, 3], durations=[4, -1, 2], max_duration_us=10)
    with pytest.raises(ValueError, match="negative: -1"):
        _write_durations(tmp_path, source)


def test_write_durations_array_rejects_a_duration_above_the_bound(
    tmp_path, event_source
):
    source = event_source([1, 2, 3], durations=[4, 11, 2], max_duration_us=10)
    with pytest.raises(ValueError, match="11 exceeds max_duration_us 10"):
        _write_durations(tmp_path, source)


def test_write_durations_array_rejects_a_bad_duration_in_a_later_block(
    tmp_path, event_source
):
    durations = [1] * 8 + [50]
    source = event_source(np.arange(9), durations=durations, max_duration_us=10)
    with pytest.raises(ValueError, match="exceeds"):
        _write_durations(tmp_path, source)


def test_write_durations_array_rejects_a_source_without_a_duration_bound(
    tmp_path, event_source
):
    class Unbounded(type(event_source([0], durations=[0]))):
        def max_duration_us(self):
            return None

    with pytest.raises(ValueError, match="max_duration_us"):
        _write_durations(tmp_path, Unbounded([1], durations=[0]))
    assert list(open_group(tmp_path / "bundle").array_keys()) == []


def _write_labels(tmp_path, source, sizing=None):
    group = open_group(tmp_path / "bundle")
    write_labels_array(group, source, sizing or _sizing(), 5)
    return open_group(tmp_path / "bundle")


def test_write_labels_array_round_trips_uint16_labels(tmp_path, event_source):
    labels = [0, 3, 1, 65535, 2]
    source = event_source([1, 2, 3, 4, 5], labels=labels)
    stored = _write_labels(tmp_path, source)[LABELS_KEY][:]
    assert stored.tolist() == labels
    assert stored.dtype == np.uint16


def test_write_labels_array_declares_the_event_dimension_and_zero_fill(
    tmp_path, event_source
):
    source = event_source([1, 2], labels=[0, 1])
    arr = _write_labels(tmp_path, source)[LABELS_KEY]
    assert arr.shape == (2,)
    assert dict(arr.attrs) == {}
    assert arr.metadata.dimension_names == ("event",)
    assert arr.fill_value == 0


def test_write_labels_array_empty_source_creates_a_zero_length_array(
    tmp_path, event_source
):
    source = event_source([], labels=[], label_names=["a"])
    assert _write_labels(tmp_path, source)[LABELS_KEY].shape == (0,)


def test_write_labels_array_accepts_the_highest_label_the_names_allow(
    tmp_path, event_source
):
    source = event_source([1, 2], labels=[0, 2], label_names=["a", "b", "c"])
    assert _write_labels(tmp_path, source)[LABELS_KEY][:].tolist() == [0, 2]


def test_write_labels_array_writes_one_whole_shard_per_write(
    tmp_path, event_source, monkeypatch
):
    writes = []

    def spy(array, start, block):
        writes.append((start, block.shape[0]))
        write_region(array, start, block)

    monkeypatch.setattr("timeseries_zarr.write_event.write_region", spy)
    labels = np.arange(26) % 5
    source = event_source(np.arange(26), labels=labels)
    stored = _write_labels(tmp_path, source)[LABELS_KEY][:]
    assert writes == [(0, 8), (8, 8), (16, 8), (24, 2)]
    assert np.array_equal(stored, labels)


def test_write_labels_array_rejects_a_label_at_or_above_the_category_count(
    tmp_path, event_source
):
    source = event_source([1, 2, 3], labels=[0, 1, 2], label_names=["a", "b"])
    with pytest.raises(ValueError, match="label 2 is outside the 2 label"):
        _write_labels(tmp_path, source)


def test_write_labels_array_rejects_a_bad_label_in_a_later_block(
    tmp_path, event_source
):
    source = event_source(
        np.arange(9), labels=[0] * 8 + [4], label_names=["a", "b"]
    )
    with pytest.raises(ValueError, match="outside"):
        _write_labels(tmp_path, source)


def test_write_labels_array_rejects_more_categories_than_a_uint16_indexes(
    tmp_path, event_source
):
    source = event_source([1], labels=[0], label_names=["x"] * 65537)
    with pytest.raises(ValueError, match="uint16 limit"):
        _write_labels(tmp_path, source)
    assert list(open_group(tmp_path / "bundle").array_keys()) == []
