import json

import numpy as np
import pytest

from timeseries_zarr.columns import channels_rows
from timeseries_zarr.types import ChunkShard
from timeseries_zarr.write_event import (
    CHANNELS_KEY,
    DURATIONS_KEY,
    EVENTS_KEY,
    LABELS_KEY,
    TEXT_KEY,
    WAVEFORM_MEAN_KEY,
    write_channels_array,
    write_durations_array,
    write_events_array,
    write_labels_array,
    write_text_array,
    write_waveform_mean_array,
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


def test_write_text_array_stores_the_rows_without_a_compression_codec(
    tmp_path, event_source
):
    source = event_source(np.arange(8), text=_TEXTS)
    _write_text(tmp_path, source, source.text_width())
    meta = json.loads((tmp_path / "bundle" / "text" / "zarr.json").read_text())
    inner = meta["codecs"][0]["configuration"]["codecs"]
    assert meta["codecs"][0]["name"] == "sharding_indexed"
    assert [codec["name"] for codec in inner] == ["bytes"]


def test_write_text_array_shard_object_starts_with_the_plain_rows(
    tmp_path, event_source
):
    source = event_source(np.arange(8), text=_TEXTS)
    width = source.text_width()
    stored = _write_text(tmp_path, source, width)[TEXT_KEY][:]
    shard = (tmp_path / "bundle" / "text" / "c" / "0" / "0").read_bytes()
    assert shard.startswith(stored.tobytes())
    assert b"seizure onset" in shard


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


def test_write_text_array_rejects_a_zero_width_and_creates_nothing(
    tmp_path, event_source
):
    source = event_source([1, 2], text=["", ""])
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


def _write_channels(tmp_path, source, max_key=9, keys=_KEYS):
    group = open_group(tmp_path / "bundle")
    sizing = _mask_sizing(channels_rows([[]], max_key).shape[1])
    write_channels_array(group, source, keys, max_key, sizing, 5)
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


def test_write_channels_array_width_follows_max_key(tmp_path, event_source):
    source = event_source([1], channel_ids=[["a"]])
    assert _write_channels(tmp_path, source, max_key=0)[CHANNELS_KEY].shape == (
        1,
        1,
    )
    assert _write_channels(tmp_path / "w", source, max_key=16)[
        CHANNELS_KEY
    ].shape == (1, 3)


def test_write_channels_array_whole_recording_rows_are_all_zero(
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


def test_write_channels_array_rejects_a_key_above_max_key(
    tmp_path, event_source
):
    source = event_source([1], channel_ids=[["c"]])
    with pytest.raises(ValueError, match="between 0 and 5"):
        _write_channels(tmp_path, source, max_key=5)


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


@pytest.mark.parametrize("period", [0.0, -1.0])
def test_write_waveform_mean_array_rejects_a_non_positive_period(
    tmp_path, event_source, period
):
    source = _means_source(event_source, period=period)
    with pytest.raises(ValueError, match="period_us"):
        _write_means(tmp_path, source)
    assert list(open_group(tmp_path / "bundle").array_keys()) == []
