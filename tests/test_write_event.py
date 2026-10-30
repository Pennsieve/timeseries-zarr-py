import dataclasses
import json

import numpy as np
import pytest

from timeseries_zarr.columns import channels_rows
from timeseries_zarr.constants import (
    CHANNELS_KEY,
    COUNT_KEY,
    DURATIONS_KEY,
    EVENTS_KEY,
    LABELS_KEY,
    TEXT_KEY,
    WAVEFORM_MEAN_KEY,
)
from timeseries_zarr.counts import plan_count_levels
from timeseries_zarr.types import ChunkShard, WriteOpts
from timeseries_zarr.write_event import (
    _row_sizing,
    check_event_channel,
    write_channels_array,
    write_count_levels,
    write_durations_array,
    write_event_channel,
    write_events_array,
    write_labels_array,
    write_text_array,
    write_waveform_mean_array,
)
from timeseries_zarr.zarr_io import open_group, write_region


def _sizing():
    return ChunkShard(chunk_shape=(4,), shard_shape=(8,))


def _write(tmp_path, source, reference_us=0, sizing=None):
    group = open_group(tmp_path / "bundle")
    write_events_array(group, source, reference_us, sizing or _sizing(), 5)
    return open_group(tmp_path / "bundle")


def test_write_events_array_stores_times_from_the_reference_time(
    tmp_path, event_source
):
    source = event_source([1000, 1500, 1500, 9000])
    stored = _write(tmp_path, source, reference_us=1000)[EVENTS_KEY][:]
    assert stored.tolist() == [0, 500, 500, 8000]
    assert stored.dtype == np.int64


def test_write_events_array_with_the_reference_time_at_zero_stores_the_times(
    tmp_path, event_source
):
    stored = _write(tmp_path, event_source([5, 6, 700]))[EVENTS_KEY][:]
    assert stored.tolist() == [5, 6, 700]


def test_write_events_array_is_exact_for_epoch_scale_microseconds(
    tmp_path, event_source
):
    start = 1_700_000_000_123_456
    source = event_source([start, start + 1, start + 999_999_999_999])
    stored = _write(tmp_path, source, reference_us=start)[EVENTS_KEY][:]
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
    _write(tmp_path, event_source(times), reference_us=1000)
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


def test_write_events_array_stores_a_time_before_the_reference_as_negative(
    tmp_path, event_source
):
    stored = _write(tmp_path, event_source([999, 1500]), reference_us=1000)
    assert stored[EVENTS_KEY][:].tolist() == [-1, 500]


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


def test_write_events_array_rejects_a_decrease_below_zero_in_a_later_block(
    tmp_path, event_source
):
    times = [1000] * 8 + [999]
    with pytest.raises(ValueError, match="non-decreasing"):
        _write(tmp_path, event_source(times), reference_us=1000)


def test_write_events_array_rejects_a_time_at_the_time_limit(
    tmp_path, event_source
):
    with pytest.raises(ValueError, match=r"an event time.*\(-2\^53, 2\^53\)"):
        _write(tmp_path, event_source([0, 2**53]))


def test_write_events_array_rejects_a_time_the_reference_pushes_to_the_limit(
    tmp_path, event_source
):
    with pytest.raises(ValueError, match="an event time"):
        _write(tmp_path, event_source([0, 10]), reference_us=-(2**53))


def test_write_events_array_accepts_the_times_next_to_the_time_limit(
    tmp_path, event_source
):
    times = [1 - 2**53, 0, 2**53 - 1]
    assert (
        _write(tmp_path, event_source(times))[EVENTS_KEY][:].tolist() == times
    )


def _write_durations(tmp_path, source, sizing=None, reference_us=0):
    group = open_group(tmp_path / "bundle")
    write_durations_array(
        group, source, sizing or _sizing(), 5, reference_us=reference_us
    )
    return open_group(tmp_path / "bundle")


def test_write_durations_array_rejects_an_interval_that_ends_at_the_limit(
    tmp_path, event_source
):
    source = event_source(
        [2**53 - 11, 2**53 - 10], durations=[10, 10], max_duration_us=10
    )
    with pytest.raises(ValueError, match="the end of an interval"):
        _write_durations(tmp_path, source)


def test_write_durations_array_accepts_an_interval_that_ends_below_the_limit(
    tmp_path, event_source
):
    source = event_source(
        [2**53 - 11, 2**53 - 10], durations=[10, 9], max_duration_us=10
    )
    stored = _write_durations(tmp_path, source)[DURATIONS_KEY][:]
    assert stored.tolist() == [10, 9]


def test_write_durations_array_counts_interval_ends_from_the_reference_time(
    tmp_path, event_source
):
    source = event_source(
        [2**53, 2**53 + 4], durations=[5, 5], max_duration_us=5
    )
    stored = _write_durations(tmp_path, source, reference_us=10)[DURATIONS_KEY]
    assert stored[:].tolist() == [5, 5]


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


def test_write_labels_array_accepts_a_channel_without_events_and_names(
    tmp_path, event_source
):
    source = event_source([], labels=[])
    assert source.num_labels() == 1
    assert _write_labels(tmp_path, source)[LABELS_KEY].shape == (0,)


def test_write_labels_array_rejects_a_category_count_above_the_largest_label(
    tmp_path, event_source
):
    class Wide(type(event_source([0]))):
        def num_labels(self):
            return 5

    source = Wide([1, 2, 3], labels=[0, 1, 2])
    with pytest.raises(
        ValueError, match="num_labels.. is 5, but the largest label 2 gives 3"
    ):
        _write_labels(tmp_path, source)


def test_write_labels_array_accepts_names_beyond_the_largest_label(
    tmp_path, event_source
):
    source = event_source([1, 2], labels=[0, 1], label_names=["a", "b", "c"])
    assert _write_labels(tmp_path, source)[LABELS_KEY][:].tolist() == [0, 1]


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


_TEXTS = [
    "seizure onset",
    "",
    "caf\N{LATIN SMALL LETTER E WITH ACUTE}",
    "\N{CJK UNIFIED IDEOGRAPH-65E5}\N{CJK UNIFIED IDEOGRAPH-672C}",
    "x",
    "electrographic seizure",
    "",
    "note",
]


def _text_sizing(width):
    return ChunkShard(chunk_shape=(4, width), shard_shape=(8, width))


def _write_text(tmp_path, source, width):
    group = open_group(tmp_path / "bundle")
    write_text_array(group, source, _text_sizing(width), 5)
    return open_group(tmp_path / "bundle")


def test_write_text_array_stores_zero_padded_utf8_rows(tmp_path, event_source):
    source = event_source(np.arange(8), text=_TEXTS)
    width = source.text_width()
    stored = _write_text(tmp_path, source, width)[TEXT_KEY][:]
    decoded = [bytes(row).rstrip(b"\0").decode("utf-8") for row in stored]
    assert decoded == _TEXTS
    assert stored.dtype == np.uint8
    assert stored.shape == (8, width)


def test_write_text_array_declares_event_and_unnamed_dimensions_and_zero_fill(
    tmp_path, event_source
):
    source = event_source(np.arange(8), text=_TEXTS)
    arr = _write_text(tmp_path, source, source.text_width())[TEXT_KEY]
    assert dict(arr.attrs) == {}
    assert arr.metadata.dimension_names == ("event", None)
    assert arr.fill_value == 0


def test_write_text_array_compresses_with_zstd(tmp_path, event_source):
    source = event_source(np.arange(8), text=_TEXTS)
    _write_text(tmp_path, source, source.text_width())
    meta = json.loads((tmp_path / "bundle" / "text" / "zarr.json").read_text())
    inner = meta["codecs"][0]["configuration"]["codecs"]
    assert meta["codecs"][0]["name"] == "sharding_indexed"
    assert [codec["name"] for codec in inner] == ["bytes", "zstd"]


def test_write_text_array_streams_a_partial_last_block(tmp_path, event_source):
    texts = [f"event {i}" for i in range(10)]
    source = event_source(np.arange(10), text=texts)
    stored = _write_text(tmp_path, source, source.text_width())[TEXT_KEY][:]
    decoded = [bytes(row).rstrip(b"\0").decode("utf-8") for row in stored]
    assert decoded == texts


def test_write_text_array_writes_one_whole_shard_per_write(
    tmp_path, event_source, monkeypatch
):
    writes = []

    def spy(array, start, block):
        writes.append((start, block.shape[0]))
        write_region(array, start, block)

    monkeypatch.setattr("timeseries_zarr.write_event.write_region", spy)
    texts = [f"t{i}" for i in range(26)]
    source = event_source(np.arange(26), text=texts)
    _write_text(tmp_path, source, source.text_width())
    assert writes == [(0, 8), (8, 8), (16, 8), (24, 2)]


def test_write_text_array_empty_source_with_a_width_creates_zero_rows(
    tmp_path, event_source
):
    class Wide(type(event_source([0]))):
        def text_width(self):
            return 5

    source = Wide([], text=[])
    arr = _write_text(tmp_path, source, 5)[TEXT_KEY]
    assert arr.shape == (0, 5)


def test_write_text_array_gives_empty_texts_one_column_of_zero_bytes(
    tmp_path, event_source
):
    source = event_source([1, 2], text=["", ""])
    arr = _write_text(tmp_path, source, 1)[TEXT_KEY]
    assert arr.shape == (2, 1)
    assert not arr[:].any()


def test_write_text_array_rejects_a_negative_width_and_creates_nothing(
    tmp_path, event_source
):
    class Negative(type(event_source([0]))):
        def text_width(self):
            return -1

    source = Negative([1, 2], text=["", ""])
    with pytest.raises(ValueError, match="text_width"):
        _write_text(tmp_path, source, 1)
    assert list(open_group(tmp_path / "bundle").array_keys()) == []


def test_write_text_array_rejects_a_text_longer_than_the_width(
    tmp_path, event_source
):
    class Narrow(type(event_source([0]))):
        def text_width(self):
            return 3

    source = Narrow([1, 2], text=["ok", "too long"])
    with pytest.raises(ValueError, match="width"):
        _write_text(tmp_path, source, 3)


def test_write_text_array_rejects_a_nul_character(tmp_path, event_source):
    source = event_source([1], text=["a\0b"])
    with pytest.raises(ValueError, match="NUL"):
        _write_text(tmp_path, source, source.text_width())


def test_write_text_array_rejects_a_source_that_returns_too_few_texts(
    tmp_path, event_source
):
    class Short(type(event_source([0]))):
        def read_text(self, start, stop):
            return super().read_text(start, stop)[:-1]

    source = Short([1, 2, 3], text=["a", "b", "c"])
    with pytest.raises(ValueError, match="returned 2 texts for 3 events"):
        _write_text(tmp_path, source, 1)


_KEYS = {"a": 0, "b": 1, "c": 9}


def _mask_sizing(width):
    return ChunkShard(chunk_shape=(4, width), shard_shape=(8, width))


def _write_channels(tmp_path, source, max_index=9, keys=_KEYS):
    group = open_group(tmp_path / "bundle")
    sizing = _mask_sizing(channels_rows([[]], max_index).shape[1])
    write_channels_array(group, source, keys, max_index, sizing, 5)
    return open_group(tmp_path / "bundle")


def test_write_channels_array_stores_a_bitmask_row_per_event(
    tmp_path, event_source
):
    ids = [[], ["a"], ["b", "c"], ["a", "c"]]
    source = event_source([1, 2, 3, 4], channel_ids=ids)
    stored = _write_channels(tmp_path, source)[CHANNELS_KEY][:]
    want = channels_rows([[], [0], [1, 9], [0, 9]], 9)
    assert np.array_equal(stored, want)
    assert stored.dtype == np.uint8


def test_write_channels_array_declares_shape_dimensions_and_zero_fill(
    tmp_path, event_source
):
    source = event_source([1, 2], channel_ids=[["a"], []])
    arr = _write_channels(tmp_path, source)[CHANNELS_KEY]
    assert arr.shape == (2, 2)
    assert dict(arr.attrs) == {}
    assert arr.metadata.dimension_names == ("event", None)
    assert arr.fill_value == 0


def test_write_channels_array_compresses_with_zstd(tmp_path, event_source):
    source = event_source([1], channel_ids=[["a"]])
    _write_channels(tmp_path, source)
    meta = json.loads(
        (tmp_path / "bundle" / "channels" / "zarr.json").read_text()
    )
    inner = meta["codecs"][0]["configuration"]["codecs"]
    assert [codec["name"] for codec in inner] == ["bytes", "zstd"]


def test_write_channels_array_width_follows_max_index(tmp_path, event_source):
    source = event_source([1], channel_ids=[["a"]])
    assert _write_channels(tmp_path, source, max_index=0)[
        CHANNELS_KEY
    ].shape == (
        1,
        1,
    )
    assert _write_channels(tmp_path / "w", source, max_index=16)[
        CHANNELS_KEY
    ].shape == (1, 3)


def test_write_channels_array_rows_with_no_channel_are_all_zero(
    tmp_path, event_source
):
    source = event_source(np.arange(10), channel_ids=[[]] * 10)
    stored = _write_channels(tmp_path, source)[CHANNELS_KEY][:]
    assert not stored.any()
    shards = [
        f
        for f in (tmp_path / "bundle" / "channels").rglob("*")
        if f.is_file() and f.name != "zarr.json"
    ]
    assert len(shards) == 2


def test_write_channels_array_empty_source_creates_zero_rows(
    tmp_path, event_source
):
    source = event_source([], channel_ids=[])
    assert _write_channels(tmp_path, source)[CHANNELS_KEY].shape == (0, 2)


def test_write_channels_array_writes_one_whole_shard_per_write(
    tmp_path, event_source, monkeypatch
):
    writes = []

    def spy(array, start, block):
        writes.append((start, block.shape[0]))
        write_region(array, start, block)

    monkeypatch.setattr("timeseries_zarr.write_event.write_region", spy)
    source = event_source(np.arange(26), channel_ids=[["b"]] * 26)
    _write_channels(tmp_path, source)
    assert writes == [(0, 8), (8, 8), (16, 8), (24, 2)]


def test_write_channels_array_rejects_an_unknown_channel_id(
    tmp_path, event_source
):
    source = event_source([1, 2], channel_ids=[["a"], ["zz"]])
    with pytest.raises(ValueError, match="unknown channel id 'zz'"):
        _write_channels(tmp_path, source)


def test_write_channels_array_rejects_an_unknown_id_in_a_later_block(
    tmp_path, event_source
):
    ids = [["a"]] * 8 + [["zz"]]
    source = event_source(np.arange(9), channel_ids=ids)
    with pytest.raises(ValueError, match="unknown"):
        _write_channels(tmp_path, source)


def test_write_channels_array_rejects_an_index_above_max_index(
    tmp_path, event_source
):
    source = event_source([1], channel_ids=[["c"]])
    with pytest.raises(ValueError, match="between 0 and 5"):
        _write_channels(tmp_path, source, max_index=5)


def test_write_channels_array_rejects_a_source_that_returns_too_few_lists(
    tmp_path, event_source
):
    class Short(type(event_source([0]))):
        def read_channel_ids(self, start, stop):
            return super().read_channel_ids(start, stop)[:-1]

    source = Short([1, 2], channel_ids=[["a"], ["b"]])
    with pytest.raises(ValueError, match="returned 1 lists for 2 events"):
        _write_channels(tmp_path, source)


def _means():
    means = np.arange(15, dtype=np.float32).reshape(3, 5) - 7.0
    means[1] = np.nan
    return means


def _write_means(tmp_path, source):
    group = open_group(tmp_path / "bundle")
    write_waveform_mean_array(group, source, 5)
    return open_group(tmp_path / "bundle")


def _means_source(event_source, means=None, period=122.0703125, **kw):
    return event_source(
        [1, 2, 3],
        labels=[0, 1, 2],
        label_names=["a", "b", "c"],
        waveform_mean=_means() if means is None else means,
        waveform_period_us=period,
        **kw,
    )


def test_write_waveform_mean_array_round_trips_float32_means_with_nan(
    tmp_path, event_source
):
    stored = _write_means(tmp_path, _means_source(event_source))[
        WAVEFORM_MEAN_KEY
    ][:]
    assert stored.dtype == np.float32
    assert np.array_equal(stored, _means(), equal_nan=True)


def test_write_waveform_mean_array_carries_the_snippet_period_only(
    tmp_path, event_source
):
    arr = _write_means(tmp_path, _means_source(event_source))[WAVEFORM_MEAN_KEY]
    assert dict(arr.attrs) == {"tsz": {"period_us": 122.0703125}}


def test_write_waveform_mean_array_writes_the_event_sample_of_the_source(
    tmp_path, event_source
):
    source = _means_source(event_source, waveform_event_sample=3)
    arr = _write_means(tmp_path, source)[WAVEFORM_MEAN_KEY]
    assert dict(arr.attrs) == {
        "tsz": {"period_us": 122.0703125, "event_sample": 3}
    }


def test_write_waveform_mean_array_treats_a_source_without_the_method_as_0(
    tmp_path, event_source
):
    class Plain:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            if name == "waveform_event_sample":
                raise AttributeError(name)
            return getattr(self._inner, name)

    source = Plain(_means_source(event_source, waveform_event_sample=3))
    arr = _write_means(tmp_path, source)[WAVEFORM_MEAN_KEY]
    assert "event_sample" not in dict(arr.attrs)["tsz"]


def test_write_waveform_mean_array_rejects_an_event_sample_past_the_samples(
    tmp_path, event_source
):
    source = _means_source(event_source, waveform_event_sample=5)
    with pytest.raises(ValueError, match="event_sample"):
        _write_means(tmp_path, source)
    assert list(open_group(tmp_path / "bundle").array_keys()) == []


def test_write_waveform_mean_array_stores_nan_for_a_label_without_events(
    tmp_path, event_source
):
    source = event_source(
        [1, 2],
        labels=[0, 2],
        label_names=["a", "b", "c"],
        waveform_mean=np.ones((3, 5)),
    )
    stored = _write_means(tmp_path, source)[WAVEFORM_MEAN_KEY][:]
    assert np.isnan(stored[1]).all()
    assert (stored[[0, 2]] == 1).all()


def test_write_waveform_mean_array_accepts_a_nan_row_for_a_label_without_events(
    tmp_path, event_source
):
    means = np.ones((3, 5), np.float32)
    means[1] = np.nan
    source = event_source(
        [1, 2],
        labels=[0, 2],
        label_names=["a", "b", "c"],
        waveform_mean=means,
    )
    stored = _write_means(tmp_path, source)[WAVEFORM_MEAN_KEY][:]
    assert np.isnan(stored[1]).all()


def test_write_waveform_mean_array_has_one_nan_row_without_events_or_names(
    tmp_path, event_source
):
    source = event_source([], labels=[], waveform_mean=np.full((1, 4), np.nan))
    stored = _write_means(tmp_path, source)[WAVEFORM_MEAN_KEY][:]
    assert stored.shape == (1, 4)
    assert np.isnan(stored).all()


def test_write_waveform_mean_array_has_the_name_count_of_rows_without_events(
    tmp_path, event_source
):
    source = event_source(
        [],
        labels=[],
        label_names=["a", "b"],
        waveform_mean=np.full((2, 4), np.nan),
    )
    assert _write_means(tmp_path, source)[WAVEFORM_MEAN_KEY].shape == (2, 4)
    finite = event_source(
        [], labels=[], label_names=["a", "b"], waveform_mean=np.ones((2, 4))
    )
    stored = _write_means(tmp_path / "other", finite)[WAVEFORM_MEAN_KEY][:]
    assert np.isnan(stored).all()


def test_write_waveform_mean_array_names_the_label_axis_and_fills_nan(
    tmp_path, event_source
):
    arr = _write_means(tmp_path, _means_source(event_source))[WAVEFORM_MEAN_KEY]
    assert arr.shape == (3, 5)
    assert arr.metadata.dimension_names == ("label", None)
    assert np.isnan(arr.fill_value)


def test_write_waveform_mean_array_is_written_in_one_whole_write(
    tmp_path, event_source, monkeypatch
):
    writes = []

    def spy(array, start, block):
        writes.append((start, block.shape))
        write_region(array, start, block)

    monkeypatch.setattr("timeseries_zarr.write_event.write_region", spy)
    _write_means(tmp_path, _means_source(event_source))
    assert writes == [(0, (3, 5))]


def test_write_waveform_mean_array_returns_the_array(tmp_path, event_source):
    group = open_group(tmp_path / "bundle")
    arr = write_waveform_mean_array(group, _means_source(event_source), 5)
    assert arr.shape == (3, 5)


def test_write_waveform_mean_array_rejects_a_row_count_other_than_the_labels(
    tmp_path, event_source
):
    source = _means_source(event_source, means=np.zeros((2, 5)))
    with pytest.raises(ValueError, match="2 rows but the channel has 3"):
        _write_means(tmp_path, source)
    assert list(open_group(tmp_path / "bundle").array_keys()) == []


def test_write_waveform_mean_array_rejects_means_that_are_not_two_dimensional(
    tmp_path, event_source
):
    source = _means_source(event_source, means=np.zeros(3))
    with pytest.raises(ValueError, match="labels, samples"):
        _write_means(tmp_path, source)


def test_write_waveform_mean_array_rejects_means_with_no_samples(
    tmp_path, event_source
):
    source = _means_source(event_source, means=np.zeros((3, 0)))
    with pytest.raises(ValueError, match="at least 1 sample"):
        _write_means(tmp_path, source)


@pytest.mark.parametrize("period", [0.0, -1.0, float("nan"), float("inf")])
def test_write_waveform_mean_array_rejects_a_period_that_is_not_finite_and_positive(
    tmp_path, event_source, period
):
    source = _means_source(event_source, period=period)
    with pytest.raises(ValueError, match="period_us"):
        _write_means(tmp_path, source)
    assert list(open_group(tmp_path / "bundle").array_keys()) == []


_COUNT_OPTS = WriteOpts(
    min_bins=256, max_levels=8, inner_len=256, target_shard_bytes=4096
)
_SPAN_US = 700_000_000


def _dense(event_source, n=70_000, labelled=True, start_us=0, seed=0):
    """Return times, labels, and a source with n events over 700 seconds."""
    rng = np.random.default_rng(seed)
    times = np.sort(rng.integers(0, _SPAN_US, n)).astype(np.int64)
    times[-1] = _SPAN_US
    labels = rng.integers(0, 3, n)
    extra = {"labels": labels, "label_names": ["u0", "u1", "u2"]}
    source = event_source(times + start_us, **(extra if labelled else {}))
    return times, (labels if labelled else None), source


def _direct_counts(times, labels, columns, plan):
    period_us = int(plan.period_us)
    counts = np.zeros((plan.bins, columns), dtype=np.uint32)
    column = np.zeros(len(times), dtype=np.int64) if labels is None else labels
    np.add.at(
        counts, (times // period_us - plan.offset_us // period_us, column), 1
    )
    return counts


def _plans(times):
    return plan_count_levels(len(times), int(times[0]), int(times[-1]), 8, 256)


def _write_counts(tmp_path, source, reference_us=0, opts=_COUNT_OPTS):
    group = open_group(tmp_path / "bundle")
    write_count_levels(group, source, reference_us, opts)
    return open_group(tmp_path / "bundle")


def test_write_count_levels_equal_the_direct_count_at_every_level(
    tmp_path, event_source
):
    times, labels, source = _dense(event_source)
    group = _write_counts(tmp_path, source)
    plans = _plans(times)
    assert len(plans) == 4
    for plan in plans:
        want = _direct_counts(times, labels, 3, plan)
        assert np.array_equal(group[str(plan.level)][COUNT_KEY][:], want)


@pytest.mark.parametrize("factor", [2, 3, 10])
def test_write_count_levels_at_another_factor_equal_the_direct_count_at_every_level(
    tmp_path, event_source, factor
):
    times, labels, source = _dense(event_source)
    opts = dataclasses.replace(_COUNT_OPTS, decimation_factor=factor)
    group = _write_counts(tmp_path, source, opts=opts)
    plans = plan_count_levels(
        len(times),
        int(times[0]),
        int(times[-1]),
        opts.level_cap,
        opts.min_bins,
        decimation_factor=factor,
    )
    assert len(plans) >= 2
    assert sorted(group.group_keys(), key=int) == [str(p.level) for p in plans]
    for below, above in zip(plans, plans[1:], strict=False):
        assert above.period_us == factor * below.period_us
    for plan in plans:
        level = group[str(plan.level)]
        assert level.attrs["tsz"]["period_us"] == plan.period_us
        want = _direct_counts(times, labels, 3, plan)
        assert np.array_equal(level[COUNT_KEY][:], want)


def test_write_count_levels_groups_carry_the_period_and_one_count_array(
    tmp_path, event_source
):
    times, _, source = _dense(event_source)
    group = _write_counts(tmp_path, source)
    plans = _plans(times)
    assert sorted(group.group_keys()) == [str(p.level) for p in plans]
    for plan in plans:
        level = group[str(plan.level)]
        assert dict(level.attrs) == {
            "tsz": {"period_us": plan.period_us, "offset_us": plan.offset_us}
        }
        assert list(level.array_keys()) == [COUNT_KEY]


def test_write_count_levels_stores_each_period_as_a_json_integer(
    tmp_path, event_source
):
    times, _, source = _dense(event_source)
    _write_counts(tmp_path, source)
    plans = _plans(times)
    assert plans
    for plan in plans:
        text = (tmp_path / "bundle" / str(plan.level) / "zarr.json").read_text()
        tsz = json.loads(text)["attributes"]["tsz"]
        assert type(tsz["period_us"]) is int
        assert tsz["period_us"] == plan.period_us


def test_write_count_levels_array_shape_dtype_dimensions_and_fill(
    tmp_path, event_source
):
    times, _, source = _dense(event_source)
    group = _write_counts(tmp_path, source)
    first = _plans(times)[0]
    arr = group["1"][COUNT_KEY]
    assert arr.shape == (first.bins, 3)
    assert arr.dtype == np.uint32
    assert arr.metadata.dimension_names == ("time", None)
    assert arr.fill_value == 0
    assert dict(arr.attrs) == {}


def test_write_count_levels_without_labels_has_a_single_column(
    tmp_path, event_source
):
    times, _, source = _dense(event_source, labelled=False)
    group = _write_counts(tmp_path, source)
    for plan in _plans(times):
        want = _direct_counts(times, None, 1, plan)
        assert np.array_equal(group[str(plan.level)][COUNT_KEY][:], want)


def test_write_count_levels_every_level_keeps_the_event_totals(
    tmp_path, event_source
):
    times, labels, source = _dense(event_source)
    group = _write_counts(tmp_path, source)
    totals = np.bincount(labels, minlength=3)
    for level in group.group_keys():
        assert (
            group[level][COUNT_KEY][:].sum(axis=0).tolist() == totals.tolist()
        )


def test_write_count_levels_counts_from_the_reference_time(
    tmp_path, event_source
):
    times, labels, source = _dense(event_source, start_us=5_000_000)
    group = _write_counts(tmp_path, source, reference_us=5_000_000)
    plan = _plans(times)[0]
    want = _direct_counts(times, labels, 3, plan)
    assert np.array_equal(group["1"][COUNT_KEY][:], want)


@pytest.mark.parametrize("after_first", [1, 300_000_000, _SPAN_US + 7])
def test_write_count_levels_counts_events_before_the_reference_time(
    tmp_path, event_source, after_first
):
    times, labels, source = _dense(event_source)
    reference_us = int(times[0]) + after_first
    group = _write_counts(tmp_path, source, reference_us=reference_us)
    shifted = times - reference_us
    plans = _plans(shifted)
    assert plans and plans[0].offset_us <= shifted[0] < 0
    for plan in plans:
        level = group[str(plan.level)]
        assert dict(level.attrs)["tsz"]["offset_us"] == plan.offset_us
        assert plan.offset_us % int(plan.period_us) == 0
        want = _direct_counts(shifted, labels, 3, plan)
        assert np.array_equal(level[COUNT_KEY][:], want)


@pytest.mark.parametrize("n", [0, 1, 1000, 64_000])
def test_write_count_levels_few_events_create_no_levels(
    tmp_path, event_source, n
):
    times = np.arange(n, dtype=np.int64)
    group = _write_counts(tmp_path, event_source(times))
    assert list(group.group_keys()) == []


def test_write_count_levels_level_one_under_min_bins_creates_none(
    tmp_path, event_source
):
    times = np.arange(70_000, dtype=np.int64) * 3
    assert _plans(times) == []
    group = _write_counts(tmp_path, event_source(times))
    assert list(group.group_keys()) == []


def test_write_count_levels_every_write_covers_a_whole_shard(
    tmp_path, event_source, monkeypatch
):
    writes = []

    def spy(array, start, block):
        writes.append(
            (array.name, start, block.shape[0], array.shape[0], array.shards[0])
        )
        write_region(array, start, block)

    monkeypatch.setattr("timeseries_zarr.write_event.write_region", spy)
    _, _, source = _dense(event_source)
    _write_counts(tmp_path, source)
    assert len(writes) > 20
    for name, start, rows, length, shard_rows in writes:
        assert start % shard_rows == 0, name
        assert rows == min(shard_rows, length - start), name


def test_write_count_levels_writes_every_shard_even_where_no_event_falls(
    tmp_path, event_source
):
    times = np.concatenate(
        [np.zeros(64_000, dtype=np.int64), np.full(2_000, _SPAN_US)]
    )
    group = _write_counts(tmp_path, event_source(times))
    arr = group["1"][COUNT_KEY]
    shards = [
        f
        for f in (tmp_path / "bundle" / "1" / COUNT_KEY).rglob("*")
        if f.is_file() and f.name != "zarr.json"
    ]
    assert len(shards) == -(-arr.shape[0] // arr.shards[0])
    assert int(arr[:].sum()) == 66_000


def test_write_count_levels_rejects_times_that_decrease(tmp_path, event_source):
    times, labels, _ = _dense(event_source)
    times[30_000], times[30_001] = times[30_001] + 10_000_000, times[30_000]
    source = event_source(times, labels=labels, label_names=["u0", "u1", "u2"])
    with pytest.raises(ValueError, match="non-decreasing"):
        _write_counts(tmp_path, source)


def test_write_count_levels_rejects_a_label_outside_the_categories(
    tmp_path, event_source
):
    times, labels, _ = _dense(event_source)
    source = event_source(times, labels=labels, label_names=["u0", "u1"])
    with pytest.raises(ValueError, match="outside"):
        _write_counts(tmp_path, source)


def test_row_sizing_keeps_the_column_axis_whole_and_the_chunk_within_a_shard():
    narrow = _row_sizing(1000, 1, 4, _COUNT_OPTS)
    assert narrow == ChunkShard(chunk_shape=(256, 1), shard_shape=(1024, 1))
    wide = _row_sizing(1000, 300, 4, _COUNT_OPTS)
    assert wide == ChunkShard(chunk_shape=(3, 300), shard_shape=(3, 300))


def test_row_sizing_takes_one_row_chunks_when_a_row_exceeds_the_shard_target():
    sizing = _row_sizing(1000, 2000, 4, _COUNT_OPTS)
    assert sizing == ChunkShard(chunk_shape=(1, 2000), shard_shape=(1, 2000))


def test_row_sizing_applies_to_one_byte_text_rows():
    sizing = _row_sizing(1000, 100, 1, _COUNT_OPTS)
    assert sizing.chunk_shape == (40, 100)
    assert sizing.shard_shape == (40, 100)


_CHANNEL_OPTS = WriteOpts(
    min_bins=2, max_levels=8, inner_len=4, target_shard_bytes=256
)
_START_US = 1_000_000


def _full(event_source, n=12, **overrides):
    times = _START_US + np.arange(n) * 1000
    fields = {
        "durations": np.arange(n) * 10,
        "labels": np.arange(n) % 3,
        "label_names": ["onset", "offset", "note"],
        "text": [f"mark {i}" for i in range(n)],
        "channel_ids": [["a"] if i % 2 else [] for i in range(n)],
        "waveform_mean": np.arange(12, dtype=np.float32).reshape(3, 4),
        "unit": "a.u.",
        "id": "evt",
        "name": "marks",
    }
    fields.update(overrides)
    return event_source(times, **fields)


def _write_channel(tmp_path, source, key=7, opts=_CHANNEL_OPTS):
    parent = open_group(tmp_path / "bundle")
    write_event_channel(
        parent,
        key,
        source,
        reference_us=_START_US,
        channel_indices=_KEYS,
        max_index=9,
        opts=opts,
    )
    return open_group(tmp_path / "bundle")


def test_write_event_channel_group_attributes_sit_under_tsz(
    tmp_path, event_source
):
    group = _write_channel(tmp_path, _full(event_source))["7"]
    assert dict(group.attrs) == {
        "tsz": {
            "kind": "event",
            "name": "marks",
            "id": "evt",
            "unit": "a.u.",
            "label_names": ["onset", "offset", "note"],
            "max_duration_us": 110,
        }
    }


_PROVENANCE = {"software": "review-tool 3.2", "settings": {"threshold": 3.5}}


def test_write_event_channel_writes_the_source_provenance_last_under_tsz(
    tmp_path, event_source
):
    source = _full(event_source, provenance=_PROVENANCE)
    group = _write_channel(tmp_path, source)["7"]
    assert dict(group.attrs) == {
        "tsz": {
            "kind": "event",
            "name": "marks",
            "id": "evt",
            "unit": "a.u.",
            "label_names": ["onset", "offset", "note"],
            "max_duration_us": 110,
            "provenance": _PROVENANCE,
        }
    }
    assert list(dict(group.attrs)["tsz"])[-1] == "provenance"


def test_write_event_channel_writes_provenance_on_a_minimal_source(
    tmp_path, event_source
):
    source = event_source(
        [_START_US, _START_US + 5],
        id="e",
        name="plain",
        provenance=_PROVENANCE,
    )
    group = _write_channel(tmp_path, source)["7"]
    assert dict(group.attrs) == {
        "tsz": {
            "kind": "event",
            "name": "plain",
            "id": "e",
            "provenance": _PROVENANCE,
        }
    }
    assert list(group.array_keys()) == ["events"]


def test_write_event_channel_writes_no_provenance_key_without_provenance(
    tmp_path, event_source
):
    group = _write_channel(tmp_path, _full(event_source))["7"]
    assert "provenance" not in dict(group.attrs)["tsz"]


def test_write_event_channel_keeps_provenance_off_the_arrays(
    tmp_path, event_source
):
    source = _full(event_source, provenance=_PROVENANCE)
    group = _write_channel(tmp_path, source)["7"]
    assert all(
        "provenance" not in str(dict(group[key].attrs))
        for key in group.array_keys()
    )


def test_write_event_channel_keeps_provenance_off_the_count_levels(
    tmp_path, event_source
):
    n = 70_000
    times = _START_US + np.sort(
        np.random.default_rng(0).integers(0, _SPAN_US, n)
    ).astype(np.int64)
    times[-1] = _START_US + _SPAN_US
    source = event_source(
        times,
        labels=np.arange(n) % 3,
        label_names=["a", "b", "c"],
        id="d",
        name="d",
        provenance=_PROVENANCE,
    )
    opts = WriteOpts(
        min_bins=256, max_levels=8, inner_len=256, target_shard_bytes=4096
    )
    group = _write_channel(tmp_path, source, opts=opts)["7"]
    levels = [group[key] for key in sorted(group.group_keys())]
    assert len(levels) >= 2
    nodes = [*levels, *(level[COUNT_KEY] for level in levels)]
    assert dict(group.attrs)["tsz"]["provenance"] == _PROVENANCE
    assert all("provenance" not in str(dict(node.attrs)) for node in nodes)


@pytest.mark.parametrize("value", [float("nan"), float("-inf")])
def test_write_event_channel_rejects_a_non_finite_provenance_number(
    tmp_path, event_source, value
):
    source = _full(event_source, provenance={"threshold": value})
    with pytest.raises(ValueError, match="provenance"):
        _write_channel(tmp_path, source)


def test_write_event_channel_writes_every_column_the_source_names(
    tmp_path, event_source
):
    group = _write_channel(tmp_path, _full(event_source))["7"]
    assert sorted(group.array_keys()) == sorted(
        [
            "events",
            "durations",
            "labels",
            "text",
            "channels",
            "waveform_mean",
        ]
    )
    assert list(group.group_keys()) == []


def test_write_event_channel_stores_each_column_as_the_source_gives_it(
    tmp_path, event_source
):
    n = 12
    group = _write_channel(tmp_path, _full(event_source, n=n))["7"]
    assert group["events"][:].tolist() == [i * 1000 for i in range(n)]
    assert group["durations"][:].tolist() == [i * 10 for i in range(n)]
    assert group["labels"][:].tolist() == [i % 3 for i in range(n)]
    decoded = [bytes(r).rstrip(b"\0").decode() for r in group["text"][:]]
    assert decoded == [f"mark {i}" for i in range(n)]
    assert group["channels"][:].tolist() == [
        [(i % 2) and 1, 0] for i in range(n)
    ]
    assert group["waveform_mean"].shape == (3, 4)


def test_write_event_channel_minimal_source_has_only_events(
    tmp_path, event_source
):
    source = event_source([_START_US, _START_US + 5], id="e", name="plain")
    group = _write_channel(tmp_path, source)["7"]
    assert dict(group.attrs) == {
        "tsz": {"kind": "event", "name": "plain", "id": "e"}
    }
    assert list(group.array_keys()) == ["events"]


def test_write_event_channel_omits_label_names_without_a_labels_column(
    tmp_path, event_source
):
    class Named(type(event_source([0]))):
        def label_names(self):
            return ["ignored"]

    source = Named([_START_US], id="e", name="plain")
    group = _write_channel(tmp_path, source)["7"]
    assert "label_names" not in dict(group.attrs)["tsz"]


def test_write_event_channel_requires_labels_for_waveform_mean(
    tmp_path, event_source
):
    source = event_source(
        [_START_US], waveform_mean=np.zeros((1, 4)), id="e", name="x"
    )
    with pytest.raises(ValueError, match="requires a labels column"):
        _write_channel(tmp_path, source)
    assert list(open_group(tmp_path / "bundle").group_keys()) == []


def test_write_event_channel_keeps_the_text_and_channel_width_axes_whole(
    tmp_path, event_source
):
    source = _full(event_source, text=["x" * 40] * 12)
    group = _write_channel(tmp_path, source)["7"]
    assert group["text"].chunks[1] == group["text"].shape[1] == 40
    assert group["channels"].chunks[1] == group["channels"].shape[1] == 2


def test_write_event_channel_writes_count_levels_for_a_dense_channel(
    tmp_path, event_source
):
    n = 70_000
    times = _START_US + np.sort(
        np.random.default_rng(0).integers(0, _SPAN_US, n)
    ).astype(np.int64)
    times[-1] = _START_US + _SPAN_US
    source = event_source(
        times,
        labels=np.arange(n) % 3,
        label_names=["a", "b", "c"],
        id="d",
        name="d",
    )
    opts = WriteOpts(
        min_bins=256, max_levels=8, inner_len=256, target_shard_bytes=4096
    )
    group = _write_channel(tmp_path, source, opts=opts)["7"]
    assert sorted(group.group_keys()) == ["1", "2", "3", "4"]
    assert int(group["1"][COUNT_KEY][:].sum()) == n


def test_write_event_channel_propagates_a_column_the_format_rejects(
    tmp_path, event_source
):
    source = _full(event_source, channel_ids=[["zz"]] * 12)
    with pytest.raises(ValueError, match="unknown channel id"):
        _write_channel(tmp_path, source)


def test_write_event_channel_returns_none(tmp_path, event_source):
    parent = open_group(tmp_path / "bundle")
    result = write_event_channel(
        parent,
        1,
        _full(event_source),
        reference_us=_START_US,
        channel_indices=_KEYS,
        max_index=9,
        opts=_CHANNEL_OPTS,
    )
    assert result is None


def _check_event_channel(source, reference_us=0, opts=_COUNT_OPTS):
    check_event_channel(source, reference_us, opts)


def test_check_event_channel_accepts_a_channel_next_to_the_limits(event_source):
    source = event_source(
        [1 - 2**53, 0, 2**53 - 1],
        durations=[0, 0, 0],
        max_duration_us=2**53 - 1,
    )
    _check_event_channel(source)


def test_check_event_channel_accepts_an_empty_channel(event_source):
    _check_event_channel(event_source([]))


def test_check_event_channel_accepts_labels_without_events_and_names(
    event_source,
):
    _check_event_channel(event_source([], labels=[]))


def test_check_event_channel_accepts_names_without_events(event_source):
    _check_event_channel(event_source([], labels=[], label_names=["a", "b"]))


def test_check_event_channel_rejects_empty_label_names(event_source):
    source = event_source([1], labels=[0], label_names=[])
    with pytest.raises(ValueError, match="label_names must name at least one"):
        _check_event_channel(source)


def test_check_event_channel_rejects_a_category_count_other_than_the_names(
    event_source,
):
    class Wrong(type(event_source([0]))):
        def num_labels(self):
            return 3

    source = Wrong([1], labels=[0], label_names=["a", "b"])
    with pytest.raises(
        ValueError, match="num_labels.. is 3, but label_names gives 2"
    ):
        _check_event_channel(source)


@pytest.mark.parametrize("categories", [0, 2])
def test_check_event_channel_rejects_a_category_count_other_than_1_without_events(
    event_source, categories
):
    class Wrong(type(event_source([0]))):
        def num_labels(self):
            return categories

    with pytest.raises(ValueError, match="gives 1 label categories"):
        _check_event_channel(Wrong([], labels=[]))


def test_check_event_channel_ignores_labels_a_source_does_not_offer(
    event_source,
):
    class Wrong(type(event_source([0]))):
        def num_labels(self):
            return 9

    _check_event_channel(Wrong([1]))


def test_check_event_channel_rejects_2_to_the_32_events(event_source):
    class Huge(type(event_source([0]))):
        def num_events(self):
            return 2**32

    with pytest.raises(
        ValueError, match="fewer than 2.32 events, got 4294967296"
    ):
        _check_event_channel(Huge([0]))


def test_check_event_channel_accepts_the_largest_event_count(event_source):
    class Large(type(event_source([0]))):
        def num_events(self):
            return 2**32 - 1

        def read_events(self, start, stop):
            return np.zeros(stop - start, dtype=np.int64)

    _check_event_channel(Large([0]))


@pytest.mark.parametrize("times", [[0, 2**53], [-(2**53), 0]])
def test_check_event_channel_rejects_a_time_at_either_end_at_the_limit(
    event_source, times
):
    with pytest.raises(ValueError, match=r"an event time.*\(-2\^53, 2\^53\)"):
        _check_event_channel(event_source(times))


def test_check_event_channel_counts_times_from_the_reference(event_source):
    with pytest.raises(ValueError, match="an event time"):
        _check_event_channel(event_source([0, 10]), reference_us=-(2**53))


def test_check_event_channel_rejects_a_duration_bound_at_the_limit(
    event_source,
):
    source = event_source([1], durations=[0], max_duration_us=2**53)
    with pytest.raises(ValueError, match="max_duration_us"):
        _check_event_channel(source)


def test_check_event_channel_rejects_a_count_level_that_ends_past_the_limit(
    event_source,
):
    class Dense(type(event_source([0]))):
        def num_events(self):
            return 100_000

        def read_events(self, start, stop):
            time_us = 0 if start == 0 else 2**53 - 1
            return np.full(stop - start, time_us, dtype=np.int64)

    with pytest.raises(ValueError, match="end of the last bin of count level"):
        _check_event_channel(Dense([0]))


def test_write_event_channel_rejects_a_bad_time_before_it_creates_a_node(
    tmp_path, event_source
):
    group = open_group(tmp_path / "bundle")
    source = event_source([0, 2**53])
    with pytest.raises(ValueError, match="an event time"):
        write_event_channel(
            group,
            0,
            source,
            reference_us=0,
            channel_indices={},
            max_index=0,
            opts=_COUNT_OPTS,
        )
    assert list(group.members()) == []


def test_write_event_channel_without_events_or_names_has_1_label_category(
    tmp_path, event_source
):
    source = event_source(
        [], labels=[], waveform_mean=np.full((1, 4), np.nan), id="e", name="x"
    )
    group = _write_channel(tmp_path, source)["7"]
    assert group[LABELS_KEY].shape == (0,)
    assert group[WAVEFORM_MEAN_KEY].shape == (1, 4)
    assert np.isnan(group[WAVEFORM_MEAN_KEY][:]).all()


def test_write_event_channel_without_events_has_a_nan_row_per_label_name(
    tmp_path, event_source
):
    source = event_source(
        [],
        labels=[],
        label_names=["a", "b"],
        waveform_mean=np.full((2, 4), np.nan),
        id="e",
        name="x",
    )
    group = _write_channel(tmp_path, source)["7"]
    assert group[WAVEFORM_MEAN_KEY].shape == (2, 4)
    assert dict(group.attrs)["tsz"]["label_names"] == ["a", "b"]


def test_write_event_channel_rejects_empty_label_names_before_writing(
    tmp_path, event_source
):
    source = event_source([_START_US], labels=[0], label_names=[], id="e")
    with pytest.raises(ValueError, match="at least one category"):
        _write_channel(tmp_path, source)
    assert list(open_group(tmp_path / "bundle").group_keys()) == []


def test_write_event_channel_keeps_one_count_column_per_label_of_the_data(
    tmp_path, event_source
):
    times, _, source = _dense(event_source, labelled=False)
    group = _write_channel(tmp_path, source, opts=_COUNT_OPTS)["7"]
    assert group["1"]["count"].shape[1] == 1
