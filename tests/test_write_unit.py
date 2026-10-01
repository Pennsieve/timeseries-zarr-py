import numpy as np
import pytest

from timeseries_zarr.attrs import channel_group_attrs
from timeseries_zarr.planning import sample_period_us
from timeseries_zarr.types import ChunkShard, WriteOpts
from timeseries_zarr.write_unit import (
    write_events_array,
    write_labels_array,
    write_unit_channel,
    write_waveforms_array,
)
from timeseries_zarr.zarr_io import open_group, write_region


def _sizing():
    # A chunk of 4 forces multiple blocks.
    return ChunkShard(chunk_shape=(4,), shard_shape=(8,))


def test_write_events_array_round_trips(tmp_path, unit_source):
    events = np.array([0, 10, 25, 25, 40, 100, 1000], dtype=np.int64)
    group = open_group(tmp_path / "bundle")
    write_events_array(group, unit_source(events), 0, _sizing(), 5)
    stored = open_group(tmp_path / "bundle")["events"][:]
    assert np.array_equal(stored, events)
    assert stored.dtype == np.int64


def test_write_events_array_creates_named_array_with_shape_and_dtype(
    tmp_path, unit_source
):
    events = np.array([0, 10, 25, 40], dtype=np.int64)
    group = open_group(tmp_path / "bundle")
    write_events_array(group, unit_source(events), 0, _sizing(), 5)
    arr = open_group(tmp_path / "bundle")["events"]
    assert arr.shape == (4,)
    assert arr.dtype == np.int64


def test_write_events_array_has_no_custom_attrs(tmp_path, unit_source):
    events = np.array([0, 10, 25, 40], dtype=np.int64)
    group = open_group(tmp_path / "bundle")
    write_events_array(group, unit_source(events), 0, _sizing(), 5)
    assert dict(open_group(tmp_path / "bundle")["events"].attrs) == {}


def test_write_events_array_empty_source(tmp_path, unit_source):
    group = open_group(tmp_path / "bundle")
    write_events_array(group, unit_source([]), 0, _sizing(), 5)
    arr = open_group(tmp_path / "bundle")["events"]
    assert arr.shape == (0,)
    assert arr.dtype == np.int64


def test_write_events_array_ties_are_allowed(tmp_path, unit_source):
    events = np.array([5, 5, 5, 5, 5], dtype=np.int64)
    group = open_group(tmp_path / "bundle")
    write_events_array(group, unit_source(events), 0, _sizing(), 5)
    assert np.array_equal(open_group(tmp_path / "bundle")["events"][:], events)


def test_write_events_array_raises_on_descending_within_block(
    tmp_path, unit_source
):
    # Drop at index 3 (5 -> 3), inside the first 4-element block.
    events = np.array([0, 1, 5, 3, 8], dtype=np.int64)
    group = open_group(tmp_path / "bundle")
    with pytest.raises(ValueError):
        write_events_array(group, unit_source(events), 0, _sizing(), 5)


def test_write_events_array_raises_on_descending_across_block_boundary(
    tmp_path, unit_source
):
    # Ascending within each 4-element block, but a drop at index 4 (3 -> 2).
    events = np.array([0, 1, 2, 3, 2, 5, 6, 7], dtype=np.int64)
    group = open_group(tmp_path / "bundle")
    with pytest.raises(ValueError):
        write_events_array(group, unit_source(events), 0, _sizing(), 5)


def test_write_labels_array_round_trips(tmp_path, unit_source):
    events = np.array([0, 10, 25, 40, 55, 70, 85], dtype=np.int64)
    labels = np.array([0, 3, 3, 1, 255, 2, 0], dtype=np.uint16)
    group = open_group(tmp_path / "bundle")
    write_labels_array(group, unit_source(events, labels=labels), _sizing(), 5)
    stored = open_group(tmp_path / "bundle")["labels"][:]
    assert np.array_equal(stored, labels)
    assert stored.dtype == np.uint16


def test_write_labels_array_creates_named_array_with_shape_and_dtype(
    tmp_path, unit_source
):
    events = np.array([0, 10, 25, 40], dtype=np.int64)
    labels = np.array([1, 2, 3, 4], dtype=np.uint16)
    group = open_group(tmp_path / "bundle")
    write_labels_array(group, unit_source(events, labels=labels), _sizing(), 5)
    arr = open_group(tmp_path / "bundle")["labels"]
    assert arr.shape == (4,)
    assert arr.dtype == np.uint16


def test_write_labels_array_has_no_custom_attrs(tmp_path, unit_source):
    events = np.array([0, 10, 25, 40], dtype=np.int64)
    labels = np.array([1, 2, 3, 4], dtype=np.uint16)
    group = open_group(tmp_path / "bundle")
    write_labels_array(group, unit_source(events, labels=labels), _sizing(), 5)
    assert dict(open_group(tmp_path / "bundle")["labels"].attrs) == {}


def test_write_labels_array_empty_source(tmp_path, unit_source):
    group = open_group(tmp_path / "bundle")
    write_labels_array(group, unit_source([]), _sizing(), 5)
    arr = open_group(tmp_path / "bundle")["labels"]
    assert arr.shape == (0,)
    assert arr.dtype == np.uint16


def _sizing_2d(ppe):
    # The points-per-event axis is never chunked.
    return ChunkShard(chunk_shape=(4, ppe), shard_shape=(8, ppe))


def test_write_waveforms_array_round_trips(tmp_path, unit_source):
    events = np.arange(7, dtype=np.int64)
    ppe = 5
    waveforms = np.arange(7 * ppe, dtype=np.float32).reshape(7, ppe)
    group = open_group(tmp_path / "bundle")
    write_waveforms_array(
        group,
        unit_source(events, waveforms=waveforms, points_per_event=ppe),
        62.5,
        _sizing_2d(ppe),
        5,
    )
    stored = open_group(tmp_path / "bundle")["waveforms"][:]
    assert np.array_equal(stored, waveforms)
    assert stored.dtype == np.float32


def test_write_waveforms_array_creates_named_array_with_shape_and_dtype(
    tmp_path, unit_source
):
    events = np.arange(4, dtype=np.int64)
    ppe = 3
    waveforms = np.arange(4 * ppe, dtype=np.float32).reshape(4, ppe)
    group = open_group(tmp_path / "bundle")
    write_waveforms_array(
        group,
        unit_source(events, waveforms=waveforms, points_per_event=ppe),
        62.5,
        _sizing_2d(ppe),
        5,
    )
    arr = open_group(tmp_path / "bundle")["waveforms"]
    assert arr.shape == (4, ppe)
    assert arr.dtype == np.float32


def test_write_waveforms_array_sets_period_us_attr(tmp_path, unit_source):
    events = np.arange(4, dtype=np.int64)
    ppe = 3
    waveforms = np.arange(4 * ppe, dtype=np.float32).reshape(4, ppe)
    group = open_group(tmp_path / "bundle")
    write_waveforms_array(
        group,
        unit_source(events, waveforms=waveforms, points_per_event=ppe),
        62.5,
        _sizing_2d(ppe),
        5,
    )
    assert dict(open_group(tmp_path / "bundle")["waveforms"].attrs) == {
        "period_us": 62.5
    }


def test_write_waveforms_array_empty_source(tmp_path, unit_source):
    ppe = 4
    group = open_group(tmp_path / "bundle")
    write_waveforms_array(
        group,
        unit_source([], points_per_event=ppe),
        62.5,
        _sizing_2d(ppe),
        5,
    )
    arr = open_group(tmp_path / "bundle")["waveforms"]
    assert arr.shape == (0, ppe)
    assert arr.dtype == np.float32


def test_write_unit_channel_creates_subgroup_with_attrs(tmp_path, unit_source):
    events = np.arange(5, dtype=np.int64)
    parent = open_group(tmp_path / "bundle")
    src = unit_source(events, id="N:unit:abc", rate_hz=32000.0, start_us=9)
    write_unit_channel(parent, 2, src, onset_us=0, opts=WriteOpts())
    grp = open_group(tmp_path / "bundle")["2"]
    assert dict(grp.attrs) == channel_group_attrs(
        "N:unit:abc", 32000.0, 9, "event", src.name, src.unit
    )


def test_write_unit_channel_writes_events_and_labels(tmp_path, unit_source):
    events = np.array([0, 10, 25, 40, 55], dtype=np.int64)
    labels = np.array([1, 2, 3, 4, 5], dtype=np.uint16)
    parent = open_group(tmp_path / "bundle")
    write_unit_channel(
        parent,
        0,
        unit_source(events, labels=labels),
        onset_us=0,
        opts=WriteOpts(),
    )
    grp = open_group(tmp_path / "bundle")["0"]
    assert np.array_equal(grp["events"][:], events)
    assert np.array_equal(grp["labels"][:], labels)


def test_write_unit_channel_writes_waveforms_with_period(tmp_path, unit_source):
    events = np.arange(5, dtype=np.int64)
    ppe = 3
    waveforms = np.arange(5 * ppe, dtype=np.float32).reshape(5, ppe)
    parent = open_group(tmp_path / "bundle")
    write_unit_channel(
        parent,
        0,
        unit_source(events, waveforms=waveforms, points_per_event=ppe),
        onset_us=0,
        opts=WriteOpts(),
    )
    wf = open_group(tmp_path / "bundle")["0"]["waveforms"]
    assert np.array_equal(wf[:], waveforms)
    assert dict(wf.attrs) == {"period_us": sample_period_us(32000.0)}


def test_write_unit_channel_returns_none(tmp_path, unit_source):
    parent = open_group(tmp_path / "bundle")
    result = write_unit_channel(
        parent,
        0,
        unit_source(np.arange(3, dtype=np.int64)),
        onset_us=0,
        opts=WriteOpts(),
    )
    assert result is None


def test_write_unit_channel_zero_events_writes_empty_arrays(
    tmp_path, unit_source
):
    parent = open_group(tmp_path / "bundle")
    write_unit_channel(parent, 5, unit_source([]), onset_us=0, opts=WriteOpts())
    grp = open_group(tmp_path / "bundle")["5"]
    assert grp["events"].shape == (0,)
    assert grp["labels"].shape == (0,)
    assert grp["waveforms"].shape == (0, 4)


def test_write_events_array_writes_one_whole_shard_per_write(
    tmp_path, unit_source, monkeypatch
):
    writes = []

    def spy(array, start, block):
        writes.append((start, block.shape[0]))
        write_region(array, start, block)

    monkeypatch.setattr("timeseries_zarr.write_unit.write_region", spy)
    events = np.arange(20, dtype=np.int64) * 10
    group = open_group(tmp_path / "bundle")
    write_events_array(group, unit_source(events), 0, _sizing(), 5)
    assert writes == [(0, 8), (8, 8), (16, 4)]
    assert np.array_equal(open_group(tmp_path / "bundle")["events"][:], events)
