import dataclasses
import json
import re
import sys
from types import MappingProxyType

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from tests.conftest import (
    ODD_CHANNEL_SETS,
    ODD_COUNT_PERIOD_US,
    ODD_EVENT_SAMPLE,
    ODD_EVENTS,
    ODD_FULL_TEXT,
    ODD_FULL_TEXT_ROW,
    ODD_INTERVAL_DURATIONS,
    ODD_INTERVAL_EVENTS,
    ODD_LABELS,
    ODD_TIE,
    ODD_WAVEFORM_PERIOD_US,
    ODD_WAVEFORM_SAMPLES,
)
from timeseries_zarr import read_event
from timeseries_zarr.read_event import (
    CountLevel,
    Counts,
    EventTable,
    Snippets,
    Waveform,
    parse_event_channel,
)
from timeseries_zarr.zarr_io import descendants, open_root

_BUILT_KEY = 4
_BUILT_TSZ = {"kind": "event", "name": "built"}
_BUILT_EVENTS = np.array([10, 20, 30], dtype=np.int64)
_SNIPPET_ATTRS = {"tsz": {"period_us": 100.0, "event_sample": 1}}


def _channel_nodes(root, key):
    """Return the attributes of channel key and its nodes keyed below it."""
    groups, arrays = descendants(root)
    prefix = f"{key}/"

    def below(nodes):
        return {
            path.removeprefix(prefix): node
            for path, node in nodes.items()
            if path.startswith(prefix)
        }

    return groups[str(key)].attrs, below(groups), below(arrays)


def _parse(root, key):
    return parse_event_channel(key, *_channel_nodes(root, key))


def _build(arrays, *, groups=None, tsz=None, chunks=None):
    """Return the root of a MemoryStore bundle that holds event channel 4.

    arrays maps a path below the channel to its data, or to a pair of data
    and attributes. groups maps a path below the channel to its attributes.
    tsz replaces the tsz object of the channel group. chunks maps a path
    below the channel to the chunk shape of its array.
    """
    root = zarr.open_group(MemoryStore(), mode="w", zarr_format=3)
    channel = root.create_group(
        str(_BUILT_KEY),
        attributes={"tsz": dict(_BUILT_TSZ) if tsz is None else tsz},
    )
    for path, attributes in (groups or {}).items():
        channel.create_group(path, attributes=attributes)
    for path, value in arrays.items():
        data, attributes = value if isinstance(value, tuple) else (value, {})
        channel.create_array(
            path,
            data=np.asarray(data),
            attributes=attributes,
            chunks=(chunks or {}).get(path, "auto"),
        )
    return root


def _built(arrays, **options):
    return _parse(_build(arrays, **options), _BUILT_KEY)


def _levels(*periods):
    return {
        str(number): {"tsz": {"period_us": period}}
        for number, period in enumerate(periods, start=1)
    }


def _counts(rows, labels):
    return np.zeros((rows, labels), dtype=np.uint32)


@pytest.fixture(scope="module")
def synthetic(synthetic_bundle):
    root = open_root(synthetic_bundle)
    return {key: _parse(root, key) for key in (3, 4, 5)}


@pytest.fixture(scope="module")
def odd(odd_bundle):
    root = open_root(odd_bundle)
    return {key: _parse(root, key) for key in (9, 12)}


@pytest.fixture(scope="module")
def mixed(mixed_bundle):
    root = open_root(mixed_bundle)
    return {key: _parse(root, key) for key in (2, 3, 4)}


def test_read_returns_the_seven_marks_from_100_to_300_seconds(synthetic):
    table = synthetic[4].read(100_000_000, 300_000_000)
    assert table.time_us.tolist() == [
        111014882,
        115254398,
        120000000,
        123075056,
        142306524,
        148108656,
        222000000,
    ]
    assert table.channel_keys == [[0], [0], [2], [0], [0], [0], []]
    assert table.label_name == ["spike"] * 2 + ["note"] + ["spike"] * 3 + [
        "seizure"
    ]


def test_read_keeps_an_interval_that_ends_at_the_window_start(synthetic):
    table = synthetic[4].read(312_000_000, 320_000_000)
    assert table.time_us.tolist() == [222_000_000]
    assert table.duration_us.tolist() == [90_000_000]
    assert table.label_name == ["seizure"]


def test_read_drops_an_interval_that_ends_before_the_window_start(synthetic):
    assert len(synthetic[4].read(312_000_001, 320_000_000)) == 0


def test_read_decodes_a_text_row_that_fills_its_width(synthetic):
    channel = synthetic[4]
    stored = np.asarray(channel.text[11])
    assert stored.shape == (37,)
    assert np.all(stored != 0)
    assert channel.read().text[11].encode("utf-8") == stored.tobytes()


def test_synthetic_marks_channel_carries_its_attributes(synthetic):
    channel = synthetic[4]
    assert channel.kind == "event"
    assert channel.id == "synth-marks-05"
    assert channel.label_names == ("seizure", "spike", "note")
    assert channel.max_duration_us == 90_000_000
    assert channel.provenance == {"software": "synth-review 0.1"}
    assert channel.count_levels == {}


def test_repr_names_the_key_name_event_count_and_columns(synthetic):
    assert repr(synthetic[4]) == (
        "EventChannel(key=4, name='curated marks', num_events=14, "
        "columns=('durations', 'labels', 'text', 'channels'))"
    )


def test_read_returns_194_events_of_five_units_with_their_waveforms(synthetic):
    table = synthetic[3].read(300_000_000, 330_000_000)
    assert len(table) == 194
    assert np.bincount(table.label, minlength=5).tolist() == [73, 9, 34, 49, 29]
    assert table.waveform.values.shape == (194, 32)
    assert table.waveform.event_sample == 12
    assert table.waveform.period_us == 122.0703125
    assert table.waveform.rows == "event"


def test_read_counts_covers_the_bins_that_overlap_the_window(synthetic):
    counts = synthetic[3].read_counts(300_000_000, 330_000_000, level=1)
    assert (counts.first, counts.first + len(counts)) == (1171, 1290)
    assert int(counts.count.sum()) == 197
    assert counts.time_us[0] == 1171 * 256_000.0
    assert counts.label_names == (
        "unit-0",
        "unit-1",
        "unit-2",
        "unit-3",
        "unit-4",
    )


def test_count_level_2_sums_four_bins_of_level_1(synthetic):
    channel = synthetic[3]
    level_1 = channel.read_counts(level=1).count
    level_2 = channel.read_counts(level=2).count
    np.testing.assert_array_equal(
        level_1.reshape(-1, 4, 5).sum(axis=1), level_2
    )
    assert int(level_1.sum()) == int(level_2.sum()) == 8099
    assert channel.num_events == 8099


def test_waveform_mean_time_axis_is_zero_at_the_event_sample(synthetic):
    snippets = synthetic[3].waveform_mean
    assert snippets.time_us[12] == 0.0
    assert snippets.read().values.shape == (5, 32)


@pytest.mark.parametrize(
    ("bins", "level"),
    [(100, 2), (586, 2), (587, 1), (2344, 1), (2345, 0)],
)
def test_level_for_over_the_whole_channel_picks_the_coarsest_level(
    synthetic, bins, level
):
    assert synthetic[3].level_for(bins=bins) == level


@pytest.mark.parametrize(
    ("bins", "level"), [(31, 2), (32, 1), (119, 1), (120, 0)]
)
def test_level_for_a_window_counts_its_overlapping_bins(synthetic, bins, level):
    assert synthetic[3].level_for(300_000_000, 330_000_000, bins=bins) == level


def test_read_returns_detector_values_and_json_text(synthetic):
    table = synthetic[5].read()
    top = np.argsort(table.value)[::-1][:3]
    np.testing.assert_allclose(
        table.value[top], [0.94737792, 0.88927662, 0.84740794], rtol=1e-7
    )
    assert table.time_us[top].tolist() == [
        489_531_239,
        559_385_070,
        450_611_878,
    ]
    assert json.loads(table.text[0]) == {"score": 0.3294, "band": "ripple"}


def test_read_keeps_both_events_of_a_tie_at_the_window_start(odd):
    tie_us = 1_000_000 + 250_000 * ODD_TIE
    table = odd[9].read(tie_us, tie_us + 250_000)
    assert table.time_us.tolist() == [tie_us, tie_us]


def test_read_drops_both_events_of_a_tie_at_the_window_end(odd):
    tie_us = 1_000_000 + 250_000 * ODD_TIE
    table = odd[9].read(tie_us - 250_000, tie_us)
    assert table.time_us.tolist() == [tie_us - 250_000]


def test_read_without_label_names_gives_labels_and_no_label_name(odd):
    table = odd[9].read()
    assert table.label.tolist() == [row % ODD_LABELS for row in range(64)]
    assert table.label_names is None
    assert table.label_name is None


def test_read_returns_values_text_and_channel_keys(odd):
    table = odd[9].read()
    texts = [f"e{row}" if row % 4 == 1 else "" for row in range(ODD_EVENTS)]
    texts[ODD_FULL_TEXT_ROW] = ODD_FULL_TEXT
    assert table.value.tolist() == [row / 8 for row in range(ODD_EVENTS)]
    assert table.text == texts
    assert table.channel_keys == [
        list(ODD_CHANNEL_SETS[row % len(ODD_CHANNEL_SETS)])
        for row in range(ODD_EVENTS)
    ]


def test_applies_to_marks_events_naming_the_key_or_the_whole_recording(odd):
    table = odd[9].read()
    rows = np.arange(ODD_EVENTS) % len(ODD_CHANNEL_SETS)
    np.testing.assert_array_equal(table.applies_to(0), rows != 1)
    np.testing.assert_array_equal(table.applies_to(5), rows != 0)
    np.testing.assert_array_equal(table.applies_to(7), rows == 2)


def test_applies_to_marks_every_event_of_a_channel_without_channels(odd):
    np.testing.assert_array_equal(odd[12].read().applies_to(0), [True] * 5)


def test_applies_to_rejects_a_read_that_left_out_channels(odd):
    table = odd[9].read(columns=["labels"])
    with pytest.raises(
        ValueError,
        match=(
            "the table has no channel_keys; read with 'channels' among "
            "columns to find the events that apply to a channel"
        ),
    ):
        table.applies_to(5)


def test_waveforms_carry_the_event_sample_and_period(odd):
    snippets = odd[9].waveforms
    assert snippets.rows == "event"
    assert snippets.samples == ODD_WAVEFORM_SAMPLES
    assert snippets.period_us == ODD_WAVEFORM_PERIOD_US
    assert snippets.event_sample == ODD_EVENT_SAMPLE
    assert snippets.time_us[ODD_EVENT_SAMPLE] == 0.0
    np.testing.assert_allclose(
        snippets.time_us,
        (np.arange(ODD_WAVEFORM_SAMPLES) - ODD_EVENT_SAMPLE)
        * ODD_WAVEFORM_PERIOD_US,
    )


def test_read_returns_the_waveform_rows_of_the_kept_events(odd):
    channel = odd[9]
    tie_us = 1_000_000 + 250_000 * ODD_TIE
    waveform = channel.read(tie_us, tie_us + 1).waveform
    np.testing.assert_array_equal(
        waveform.values, channel.waveforms.array[ODD_TIE : ODD_TIE + 2]
    )
    assert waveform.samples == ODD_WAVEFORM_SAMPLES
    assert waveform.period_us == ODD_WAVEFORM_PERIOD_US
    assert waveform.event_sample == ODD_EVENT_SAMPLE
    assert waveform.time_us[ODD_EVENT_SAMPLE] == 0.0
    assert waveform.rows == "event"


def test_waveform_time_s_is_time_us_in_seconds(odd):
    waveform = odd[9].read().waveform
    np.testing.assert_array_equal(waveform.time_s, waveform.time_us / 1e6)
    assert waveform.time_s[ODD_EVENT_SAMPLE] == 0.0


@pytest.mark.parametrize(
    "read",
    [
        lambda channel: channel.read().waveform,
        lambda channel: channel.waveform_mean.read(),
    ],
    ids=["events", "waveform_mean"],
)
def test_waveform_len_counts_the_samples_of_each_snippet(odd, read):
    waveform = read(odd[9])
    assert len(waveform) == len(waveform.time_us) == ODD_WAVEFORM_SAMPLES


def test_waveform_mean_holds_one_mean_row_per_label(odd):
    channel = odd[9]
    waveforms = channel.waveforms.read().values.astype(np.float64)
    labels = np.asarray(channel.labels[:])
    means = channel.waveform_mean.read()
    assert means.values.shape == (ODD_LABELS, ODD_WAVEFORM_SAMPLES)
    assert means.event_sample == ODD_EVENT_SAMPLE
    assert channel.waveform_mean.rows == means.rows == "label"
    for label in range(ODD_LABELS):
        np.testing.assert_allclose(
            means.values[label],
            waveforms[labels == label].mean(axis=0),
            rtol=1e-6,
        )


def test_count_level_1_counts_every_event(odd):
    channel = odd[9]
    level = channel.count_levels[1]
    assert isinstance(level, CountLevel)
    assert level.number == 1
    assert level.period_us == ODD_COUNT_PERIOD_US
    assert (level.bins, level.num_labels) == (17, ODD_LABELS)
    counts = channel.read_counts(level=1)
    assert (counts.level, counts.first, len(counts)) == (1, 0, 17)
    assert int(counts.count.sum()) == ODD_EVENTS
    assert counts.label_names is None
    assert counts.time_us[1] == ODD_COUNT_PERIOD_US
    assert counts.time_s[1] == ODD_COUNT_PERIOD_US / 1e6


def test_read_counts_past_the_last_bin_is_empty(odd):
    counts = odd[9].read_counts(20_000_000, 30_000_000, level=1)
    assert counts.first == 17
    assert counts.count.shape == (0, ODD_LABELS)
    assert counts.time_us.shape == (0,)


def test_count_bins_start_at_the_reference_time_without_an_offset():
    channel = _built(
        {"events": [5000, 6000, 7500], "1/count": _counts(8, 1)},
        groups=_levels(1000.0),
    )
    assert channel.count_levels[1].offset_us == 0
    counts = channel.read_counts(5000, 6000, level=1)
    assert (counts.first, len(counts), counts.offset_us) == (5, 1, 0)
    assert counts.time_us.tolist() == [5000.0]
    assert channel.level_for(8000, 13_000, bins=1) == 0


def test_count_bins_start_at_the_level_offset():
    channel = _built(
        {"events": [-4000, -2500, 1000], "1/count": _counts(6, 1)},
        groups={"1": {"tsz": {"period_us": 1000.0, "offset_us": -4000}}},
    )
    assert channel.count_levels[1].offset_us == -4000
    counts = channel.read_counts(-3000, -1000, level=1)
    assert (counts.first, len(counts), counts.offset_us) == (1, 2, -4000)
    assert counts.time_us.tolist() == [-3000.0, -2000.0]
    assert channel.read_counts(level=1).time_us[0] == -4000.0
    assert len(channel.read_counts(2000, 9000, level=1)) == 0
    assert channel.level_for(-4000, 2000, bins=6) == 1
    assert channel.level_for(bins=6) == 1


def test_count_level_rejects_an_offset_off_the_grid():
    with pytest.raises(
        ValueError, match="count level 1: tsz.offset_us must be a multiple of"
    ):
        _built(
            {"events": [5000], "1/count": _counts(8, 1)},
            groups={"1": {"tsz": {"period_us": 1000.0, "offset_us": 500}}},
        )


def test_count_windows_use_the_exact_period():
    channel = _built(
        {"events": [1_595_200, 1_595_201], "1/count": _counts(3000, 1)},
        groups=_levels(16 * 1e6 / 30_000),
    )
    assert channel.read_counts(1_595_200, None, level=1).first == 2990
    assert channel.level_for(bins=2) == 1


def test_odd_marks_channel_omits_id_and_label_names(odd):
    channel = odd[9]
    assert channel.kind == "event"
    assert channel.id is None
    assert channel.unit == "a.u."
    assert channel.label_names is None
    assert channel.max_duration_us is None
    assert channel.provenance is None
    assert channel.durations is None
    assert isinstance(channel.count_levels, MappingProxyType)
    assert list(channel.count_levels) == [1]
    assert channel.columns == (
        "labels",
        "values",
        "text",
        "channels",
        "waveforms",
    )
    assert repr(channel) == (
        "EventChannel(key=9, name='marks', num_events=64, columns=('labels', "
        "'values', 'text', 'channels', 'waveforms'))"
    )


def test_durations_without_max_duration_us_search_from_the_first_event(odd):
    channel = odd[12]
    assert channel.max_duration_us is None
    assert channel.columns == ("durations",)
    first = channel.read(2_500_000, 3_000_000)
    assert first.time_us.tolist() == [ODD_INTERVAL_EVENTS[0]]
    assert first.duration_us.tolist() == [ODD_INTERVAL_DURATIONS[0]]
    late = channel.read(9_000_000, None)
    assert late.time_us.tolist() == [4_000_000, 9_000_000]


def test_read_with_equal_bounds_keeps_the_intervals_open_at_that_time(odd):
    table = odd[12].read(2_000_000, 2_000_000)
    assert table.time_us.tolist() == [ODD_INTERVAL_EVENTS[0]]


def test_read_with_equal_bounds_on_point_events_is_empty(odd):
    table = odd[9].read(3_500_000, 3_500_000)
    assert len(table) == 0
    assert table.label.shape == (0,)
    assert table.value.shape == (0,)
    assert table.text == []
    assert table.channel_keys == []
    assert table.waveform.values.shape == (0, ODD_WAVEFORM_SAMPLES)
    assert table.label_name is None


@pytest.mark.parametrize(
    ("start_us", "end_us"),
    [(0, 1_000_000), (20_000_000, 30_000_000), (None, 0)],
)
def test_read_outside_the_events_is_empty(odd, start_us, end_us):
    assert len(odd[9].read(start_us, end_us)) == 0


def test_read_with_open_bounds_reads_from_the_first_or_to_the_last_event(odd):
    channel = odd[9]
    tie_us = 1_000_000 + 250_000 * ODD_TIE
    assert len(channel.read()) == ODD_EVENTS
    assert len(channel.read(None, tie_us)) == ODD_TIE
    assert len(channel.read(tie_us, None)) == ODD_EVENTS - ODD_TIE


def test_read_reads_the_events_column_once(odd, monkeypatch):
    channel = odd[9]
    reads = []
    read_rows = read_event.read_rows

    def counting(array, start, stop):
        if array is channel.events:
            reads.append((start, stop))
        return read_rows(array, start, stop)

    monkeypatch.setattr(read_event, "read_rows", counting)
    channel.read(3_000_000, 5_000_000)
    assert reads == [(0, ODD_EVENTS)]


def test_table_times_convert_to_seconds(odd):
    table = odd[12].read()
    assert table.time_s.tolist() == [time / 1e6 for time in ODD_INTERVAL_EVENTS]
    assert table.time_s.dtype == np.float64


def test_read_columns_returns_only_the_named_columns(odd):
    table = odd[9].read(columns=["labels", "waveforms"])
    assert table.label is not None
    assert table.waveform is not None
    assert table.value is None
    assert table.text is None
    assert table.channel_keys is None
    assert table.duration_us is None
    assert len(table) == ODD_EVENTS


def test_read_without_labels_gives_no_label_name(mixed):
    channel = mixed[3]
    table = channel.read(columns=["text"])
    assert channel.label_names is not None
    assert table.label_names == channel.label_names
    assert table.label is None
    assert table.label_name is None


def test_read_with_no_columns_returns_only_the_times(odd):
    table = odd[9].read(columns=[])
    assert len(table) == ODD_EVENTS
    assert table.label is None
    assert table.waveform is None


def test_read_applies_the_interval_rule_without_the_durations_column(odd):
    table = odd[12].read(2_500_000, 3_000_000, columns=[])
    assert table.time_us.tolist() == [ODD_INTERVAL_EVENTS[0]]
    assert table.duration_us is None


def test_read_masks_values_and_waveforms_by_the_interval_rule():
    channel = _built(
        {
            "events": _BUILT_EVENTS,
            "durations": [25, 0, 0],
            "values": [1.0, 2.0, 3.0],
            "waveforms": (
                np.arange(6, dtype=np.float32).reshape(3, 2),
                _SNIPPET_ATTRS,
            ),
        },
        tsz={**_BUILT_TSZ, "max_duration_us": 25},
    )
    table = channel.read(25, 40)
    assert table.time_us.tolist() == [10, 30]
    assert table.value.tolist() == [1.0, 3.0]
    np.testing.assert_array_equal(
        table.waveform.values, [[0.0, 1.0], [4.0, 5.0]]
    )


_BAD_TEXT_ROWS = np.array([[0xFF, 0], [0x62, 0], [0xFF, 0]], dtype=np.uint8)


def test_read_names_the_channel_and_event_of_text_that_is_not_utf8():
    channel = _built({"events": _BUILT_EVENTS, "text": _BAD_TEXT_ROWS})
    with pytest.raises(
        ValueError,
        match=r"^channel 4: event 2 text is not valid UTF-8 at byte 0$",
    ):
        channel.read(20, None)


def test_read_decodes_only_the_text_of_the_events_the_interval_rule_keeps():
    channel = _built(
        {
            "events": _BUILT_EVENTS,
            "durations": [0, 0, 0],
            "text": np.array([[0xFF, 0], [0x62, 0], [0x63, 0]], np.uint8),
        },
        tsz={**_BUILT_TSZ, "max_duration_us": 25},
    )
    table = channel.read(15, None, columns=["text"])
    assert table.time_us.tolist() == [20, 30]
    assert table.text == ["b", "c"]


def test_read_rejects_a_column_the_channel_lacks(odd):
    with pytest.raises(
        ValueError,
        match=(
            r"channel 12 has columns \('durations',\); columns must name only "
            r"these, got 'labels'"
        ),
    ):
        odd[12].read(columns=["labels"])


def test_read_rejects_a_string_of_columns(odd):
    with pytest.raises(
        TypeError,
        match="columns must be a collection of column names, got the string",
    ):
        odd[9].read(columns="labels")


@pytest.mark.parametrize(
    ("start_us", "end_us", "name"),
    [(1.5, None, "start_us"), (None, 2.0, "end_us")],
)
def test_read_rejects_a_float_bound(odd, start_us, end_us, name):
    with pytest.raises(
        TypeError, match=f"{name} must be an integer count of microseconds"
    ):
        odd[9].read(start_us, end_us)


def test_read_counts_rejects_a_float_bound(odd):
    with pytest.raises(TypeError, match="start_us must be an integer"):
        odd[9].read_counts(0.5, None, level=1)


@pytest.mark.parametrize(
    ("level", "shown"),
    [(1.0, "1.0"), (np.float64(1.0), "np.float64(1.0)"), ([1], "[1]")],
)
def test_read_counts_rejects_a_level_that_is_not_an_integer(odd, level, shown):
    with pytest.raises(
        TypeError, match=re.escape(f"level must be an integer, got {shown}")
    ):
        odd[9].read_counts(level=level)


def test_read_counts_accepts_a_numpy_integer_level(odd):
    counts = odd[9].read_counts(level=np.int64(1))
    assert (counts.level, len(counts)) == (1, 17)


def test_read_counts_accepts_numpy_integer_bounds(odd):
    counts = odd[9].read_counts(
        np.int64(1_100_000), np.uint32(3_000_000), level=1
    )
    assert (counts.first, len(counts)) == (1, 2)


def test_level_for_rejects_a_float_bound(odd):
    with pytest.raises(TypeError, match="end_us must be an integer"):
        odd[12].level_for(0, 1.0, bins=1)


def test_read_accepts_numpy_integer_bounds(odd):
    table = odd[9].read(np.int64(3_500_000), np.int32(3_750_000))
    assert len(table) == 2


@pytest.mark.parametrize("method", ["read", "level_for"])
def test_window_rejects_an_end_before_its_start(odd, method):
    options = {"bins": 1} if method == "level_for" else {}
    with pytest.raises(ValueError, match="end_us must be at or after start_us"):
        getattr(odd[9], method)(2_000_000, 1_000_000, **options)


def test_read_counts_rejects_an_end_before_its_start(odd):
    with pytest.raises(ValueError, match="end_us must be at or after start_us"):
        odd[9].read_counts(2_000_000, 1_000_000, level=1)


def test_read_counts_rejects_a_level_the_channel_lacks(odd, mixed):
    with pytest.raises(
        ValueError, match="channel 9 has count level 1; level must be 1, got 2"
    ):
        odd[9].read_counts(level=2)
    with pytest.raises(
        ValueError,
        match="channel 12 has no count levels; read its events with read, "
        "got level 1",
    ):
        odd[12].read_counts(level=1)
    with pytest.raises(
        ValueError,
        match=(
            "channel 4 has count levels 1 through 4; level must be one of "
            "them, got 5"
        ),
    ):
        mixed[4].read_counts(level=5)


def test_read_counts_rejects_level_0_and_names_read(odd, mixed):
    messages = {
        odd[12]: "channel 12: level 0 stands for the events; read them with "
        "read",
        odd[9]: "channel 9: level 0 stands for the events; read them with "
        "read, or pass count level 1",
        mixed[4]: "channel 4: level 0 stands for the events; read them with "
        "read, or pass a count level from 1 through 4",
    }
    for channel, message in messages.items():
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            channel.read_counts(level=0)


@pytest.mark.parametrize(("bins", "level"), [(1, 1), (17, 1), (18, 0)])
def test_level_for_resolves_open_bounds_to_the_events(odd, bins, level):
    channel = odd[9]
    last_us = int(channel.events[-1])
    assert channel.level_for(bins=bins) == level
    assert channel.level_for(1_000_000, last_us + 1, bins=bins) == level
    assert channel.level_for(1_000_000, None, bins=bins) == level
    assert channel.level_for(None, last_us + 1, bins=bins) == level


def test_level_for_resolves_an_open_end_to_the_last_event_plus_one():
    channel = _built(
        {"events": [1000, 2000, 3000], "1/count": _counts(8, 1)},
        groups=_levels(1000.0),
    )
    assert channel.level_for(bins=3) == 1
    assert channel.level_for(bins=4) == 0


@pytest.mark.parametrize(
    ("start_us", "end_us", "bins", "level"),
    [
        (0, 20_000, 3, 1),
        (0, 20_000, 2, 2),
        (4000, 8000, 1, 2),
        (None, None, 3, 1),
    ],
)
def test_level_for_counts_the_bins_each_count_level_returns(
    start_us, end_us, bins, level
):
    channel = _built(
        {
            "events": [1000, 2000, 3000, 4000, 5000, 6000, 7000],
            "1/count": _counts(8, 1),
            "2/count": _counts(2, 1),
        },
        groups=_levels(1000.0, 4000.0),
    )
    assert channel.level_for(start_us, end_us, bins=bins) == level
    low = 1000 if start_us is None else start_us
    high = 7001 if end_us is None else end_us
    assert len(channel.read_counts(low, high, level=level)) >= bins
    if level < len(channel.count_levels):
        coarser = channel.read_counts(low, high, level=level + 1)
        assert len(coarser) < bins


@pytest.mark.parametrize(
    ("start_us", "end_us"), [(30_000_000, None), (None, 500_000)]
)
def test_level_for_one_open_bound_outside_the_events_gives_0(
    odd, start_us, end_us
):
    assert odd[9].level_for(start_us, end_us, bins=1) == 0


def test_level_for_without_count_levels_gives_0(odd):
    assert odd[12].level_for(bins=1) == 0


def test_level_for_rejects_bins_below_1_without_count_levels(odd):
    with pytest.raises(ValueError, match="bins must be at least 1, got 0"):
        odd[12].level_for(bins=0)


def test_level_for_rejects_bins_below_1(odd):
    with pytest.raises(ValueError, match="bins must be at least 1, got 0"):
        odd[9].level_for(bins=0)


@pytest.mark.parametrize("bins", [2.5, float("nan"), "3"])
def test_level_for_rejects_a_bins_value_that_is_not_an_integer(odd, bins):
    without_events = _built(
        {"events": np.zeros(0, np.int64), "1/count": _counts(0, 1)},
        groups=_levels(1000.0),
    )
    for channel in (odd[9], odd[12], without_events):
        with pytest.raises(TypeError, match="bins must be an integer"):
            channel.level_for(bins=bins)


def test_level_for_accepts_numpy_integer_bins(odd):
    assert odd[9].level_for(bins=np.int64(17)) == 1


def test_level_for_accepts_numpy_integer_bounds(odd):
    end = 1_000_000 + 250_000 * ODD_EVENTS
    assert odd[9].level_for(np.int64(0), np.uint32(end), bins=17) == 1


def test_written_marks_read_back_with_their_columns(mixed):
    channel = mixed[3]
    table = channel.read()
    assert channel.max_duration_us == 12_000_000
    assert table.time_us.tolist() == [
        second * 1_000_000 for second in (1, 4, 9, 15, 22, 28)
    ]
    assert table.duration_us.tolist() == [0, 0, 5_000_000, 0, 12_000_000, 0]
    assert table.label_name == [
        "spike",
        "spike",
        "seizure",
        "note",
        "seizure",
        "note",
    ]
    assert table.text == [
        "spike",
        "spike",
        "seizure onset",
        "note",
        "seizure",
        "x",
    ]
    assert table.channel_keys == [[0], [], [0, 1], [], [1], []]


def test_written_interval_stays_in_windows_until_it_ends(mixed):
    channel = mixed[3]
    table = channel.read(30_000_000, 40_000_000)
    assert table.time_us.tolist() == [22_000_000]
    assert table.duration_us.tolist() == [12_000_000]
    assert table.label_name == ["seizure"]
    assert table.text == ["seizure"]
    assert table.channel_keys == [[1]]
    assert channel.read(34_000_000, 40_000_000).time_us.tolist() == [22_000_000]
    assert len(channel.read(34_000_001, 40_000_000)) == 0


def test_written_count_levels_each_count_every_event(mixed):
    channel = mixed[4]
    assert list(channel.count_levels) == [1, 2, 3, 4]
    for number in channel.count_levels:
        counts = channel.read_counts(level=number)
        assert int(counts.count.sum()) == channel.num_events == 70_000
        assert counts.label_names == ("ripple", "fast-ripple")


def test_written_waveform_mean_without_waveforms_reads_back(mixed):
    channel = mixed[2]
    assert channel.waveforms is None
    assert channel.columns == ("labels",)
    assert channel.waveform_mean.read().values.shape == (3, 16)
    assert channel.waveform_mean.event_sample == 0


def _brute_force(times, durations, start_us, end_us):
    """Return the indices of the events in a window by the interval rule."""
    ends = times if durations is None else times + durations
    keep = np.ones(times.shape[0], dtype=bool)
    if start_us is not None:
        keep &= ends >= start_us
    if end_us is not None:
        keep &= times < end_us
    return np.flatnonzero(keep)


def _seeded_windows(rng, low, high, count):
    windows = [(None, None), (low, None), (None, high)]
    for _ in range(count):
        start, end = sorted(rng.integers(low, high, 2).tolist())
        windows.append((start, end))
        windows.append((start, start))
    return windows


@pytest.mark.parametrize("with_durations", [False, True])
def test_bisect_search_agrees_with_the_whole_column_search(
    monkeypatch, with_durations
):
    rng = np.random.default_rng(11)
    times = np.sort(rng.integers(0, 2_000, 400)).astype(np.int64)
    assert np.unique(times).size < times.size
    arrays = {"events": times, "labels": rng.integers(0, 3, 400)}
    tsz = dict(_BUILT_TSZ)
    durations = None
    if with_durations:
        durations = rng.integers(0, 60, 400).astype(np.int64)
        arrays["durations"] = durations
        tsz["max_duration_us"] = 60
    channel = _built(arrays, tsz=tsz, chunks={"events": (4,)})
    windows = _seeded_windows(rng, -100, 2_100, 150)
    whole = [channel.read(start, end) for start, end in windows]
    lookups = []
    read_int = read_event.read_int

    def counting(array, index):
        lookups.append(index)
        return read_int(array, index)

    monkeypatch.setattr(read_event, "read_int", counting)
    monkeypatch.setattr(read_event, "EVENTS_SEARCH_LIMIT_BYTES", 0)
    for (start, end), expected in zip(windows, whole, strict=True):
        bisected = channel.read(start, end)
        indices = _brute_force(times, durations, start, end)
        np.testing.assert_array_equal(bisected.time_us, times[indices])
        np.testing.assert_array_equal(bisected.time_us, expected.time_us)
        np.testing.assert_array_equal(bisected.label, expected.label)
    assert lookups


@pytest.mark.parametrize("limit", [None, 0])
def test_read_keeps_an_open_interval_whose_end_overflows_the_stored_dtype(
    monkeypatch, limit
):
    channel = _built(
        {
            "events": np.array([100, 2_100_000_000], np.int32),
            "durations": np.array([5, 100_000_000], np.int32),
        },
        tsz={**_BUILT_TSZ, "max_duration_us": 100_000_000},
    )
    if limit is not None:
        monkeypatch.setattr(read_event, "EVENTS_SEARCH_LIMIT_BYTES", limit)
    table = channel.read(2_140_000_000, 2_141_000_000)
    assert table.time_us.tolist() == [2_100_000_000]
    assert table.time_us.dtype == table.duration_us.dtype == np.int32


def test_search_reads_events_whole_up_to_the_limit_and_bisects_above(
    monkeypatch,
):
    channel = _built({"events": _BUILT_EVENTS}, chunks={"events": (1,)})
    lookups = []
    read_int = read_event.read_int

    def counting(array, index):
        lookups.append(index)
        return read_int(array, index)

    monkeypatch.setattr(read_event, "read_int", counting)
    monkeypatch.setattr(
        read_event, "EVENTS_SEARCH_LIMIT_BYTES", channel.events.nbytes
    )
    assert channel.read(15, 25).time_us.tolist() == [20]
    assert lookups == []
    monkeypatch.setattr(
        read_event, "EVENTS_SEARCH_LIMIT_BYTES", channel.events.nbytes - 1
    )
    assert channel.read(15, 25).time_us.tolist() == [20]
    assert lookups


def _count_lookups(monkeypatch):
    """Return the list that records the index of each single-event read."""
    lookups = []
    read_int = read_event.read_int

    def counting(array, index):
        lookups.append(index)
        return read_int(array, index)

    monkeypatch.setattr(read_event, "read_int", counting)
    monkeypatch.setattr(read_event, "EVENTS_SEARCH_LIMIT_BYTES", 0)
    return lookups


def test_bisect_probes_until_the_range_fits_one_inner_chunk(monkeypatch):
    times = np.arange(0, 4_000, 10, dtype=np.int64)
    channel = _built({"events": times}, chunks={"events": (4,)})
    lookups = _count_lookups(monkeypatch)
    assert channel.read(1_005, 1_025).time_us.tolist() == [1_010, 1_020]
    # Each of the two searches halves at most 400 events down to 4.
    assert 0 < len(lookups) <= 14


def test_bisect_reads_one_inner_chunk_without_probing_its_rows(monkeypatch):
    times = np.arange(0, 4_000, 10, dtype=np.int64)
    channel = _built({"events": times})
    assert channel.events.chunks[0] >= times.size
    lookups = _count_lookups(monkeypatch)
    assert channel.read(1_005, 1_025).time_us.tolist() == [1_010, 1_020]
    assert lookups == []


@pytest.mark.parametrize("limit", [None, 0])
def test_read_keeps_an_interval_whose_end_passes_the_int64_maximum(
    monkeypatch, limit
):
    last = 2**63 - 10
    channel = _built(
        {
            "events": np.array([5, last], np.int64),
            "durations": np.array([3, 100], np.int64),
        },
        tsz={**_BUILT_TSZ, "max_duration_us": 100},
    )
    if limit is not None:
        monkeypatch.setattr(read_event, "EVENTS_SEARCH_LIMIT_BYTES", limit)
    assert channel.read(2**63 - 5, 2**63 - 4).time_us.tolist() == [last]
    assert channel.read(last, last + 1).time_us.tolist() == [last]


@pytest.mark.parametrize("limit", [None, 0])
def test_read_compares_a_bound_past_the_int64_range_exactly(monkeypatch, limit):
    near = 2**63 - 10
    with_durations = _built(
        {
            "events": np.array([5, near], np.int64),
            "durations": np.array([3, 1], np.int64),
        },
        tsz={**_BUILT_TSZ, "max_duration_us": 3},
    )
    instants = _built({"events": np.array([5, near, 2**63 - 1], np.int64)})
    if limit is not None:
        monkeypatch.setattr(read_event, "EVENTS_SEARCH_LIMIT_BYTES", limit)
    assert with_durations.read(None, 2**63).time_us.tolist() == [5, near]
    assert instants.read(None, 2**63).time_us.tolist() == [5, near, 2**63 - 1]
    assert instants.read(2**63 - 20, 2**63).time_us.tolist() == [
        near,
        2**63 - 1,
    ]
    assert len(instants.read(2**63, None)) == 0
    assert instants.read(-(2**63) - 1, 6).time_us.tolist() == [5]


def test_channel_without_events_reads_empty_and_picks_level_0():
    channel = _built(
        {
            "events": np.zeros(0, dtype=np.int64),
            "1/count": _counts(0, 1),
        },
        groups=_levels(1000.0),
    )
    assert channel.num_events == 0
    assert len(channel.read()) == 0
    assert len(channel.read(0, 100)) == 0
    assert channel.level_for(bins=1) == 0
    assert len(channel.read_counts(level=1)) == 0


def test_read_rejects_a_label_at_or_above_the_label_names():
    channel = _built(
        {"events": _BUILT_EVENTS, "labels": np.array([0, 1, 2], np.uint16)},
        tsz={**_BUILT_TSZ, "label_names": ["a", "b"]},
    )
    assert channel.read(0, 25).label_name == ["a", "b"]
    with pytest.raises(
        ValueError,
        match=(
            "channel 4: labels must be at least 0 and below 2, the length of "
            "label_names, got 2"
        ),
    ):
        channel.read()


def test_read_rejects_a_label_below_0():
    channel = _built(
        {"events": _BUILT_EVENTS, "labels": np.array([0, -1, 1], np.int16)},
        tsz={**_BUILT_TSZ, "label_names": ["a", "b"]},
    )
    assert channel.read(0, 15).label_name == ["a"]
    with pytest.raises(
        ValueError,
        match=(
            "channel 4: labels must be at least 0 and below 2, the length of "
            "label_names, got -1$"
        ),
    ):
        channel.read()


def test_waveform_mean_without_labels_is_tolerated():
    channel = _built(
        {
            "events": _BUILT_EVENTS,
            "waveform_mean": (np.zeros((2, 4), np.float32), _SNIPPET_ATTRS),
        }
    )
    assert channel.labels is None
    assert channel.waveform_mean.samples == 4
    assert channel.waveform_mean.time_us.tolist() == [-100.0, 0.0, 100.0, 200.0]
    assert channel.columns == ()


def test_parse_accepts_dtypes_of_the_right_kind_at_any_width():
    channel = _built(
        {
            "events": np.array([10, 20, 30], np.int32),
            "durations": np.zeros(3, np.uint16),
            "values": np.zeros(3, np.float64),
            "waveforms": (np.zeros((3, 4), np.float64), _SNIPPET_ATTRS),
            "waveform_mean": (np.zeros((2, 4), np.float16), _SNIPPET_ATTRS),
            "1/count": np.zeros((1, 1), np.int64),
        },
        groups=_levels(1000.0),
        tsz={**_BUILT_TSZ, "max_duration_us": 0},
    )
    assert channel.events.dtype == np.int32
    assert channel.durations.dtype == np.uint16
    assert channel.values.dtype == np.float64
    assert channel.waveforms.array.dtype == np.float64
    assert channel.waveform_mean.array.dtype == np.float16
    assert channel.count_levels[1].count.dtype == np.int64


def test_unknown_nodes_and_keys_are_ignored():
    channel = _built(
        {
            "events": _BUILT_EVENTS,
            "scores": np.zeros(3, np.float32),
            "notes/count": _counts(1, 1),
            "0/count": _counts(1, 1),
            "01/count": _counts(1, 1),
            "1/2/count": _counts(1, 1),
            "1/count": _counts(1, 1),
        },
        groups={
            "notes": {},
            "0": {},
            "01": {},
            "1": {"tsz": {"period_us": 1000.0}, "beside": True},
            "1/2": {},
        },
        tsz={**_BUILT_TSZ, "future": "a later minor version"},
    )
    assert list(channel.count_levels) == [1]
    assert channel.columns == ()


def test_count_level_periods_within_the_tolerance_are_accepted():
    channel = _built(
        {
            "events": _BUILT_EVENTS,
            "1/count": _counts(1, 2),
            "2/count": _counts(1, 2),
        },
        groups=_levels(1000.0, 4000.0 * (1 + 1e-12)),
    )
    assert [level.period_us for level in channel.count_levels.values()] == [
        1000.0,
        4000.0 * (1 + 1e-12),
    ]


def test_count_levels_follow_the_label_names():
    channel = _built(
        {
            "events": _BUILT_EVENTS,
            "1/count": _counts(1, 2),
            "2/count": _counts(1, 2),
        },
        groups=_levels(1000.0, 4000.0),
        tsz={**_BUILT_TSZ, "label_names": ["a", "b"]},
    )
    assert [level.num_labels for level in channel.count_levels.values()] == [
        2,
        2,
    ]


_FLOAT_COLUMN = np.zeros(3, np.float32)
_INTEGER_COLUMN = np.zeros(3, np.int64)


@pytest.mark.parametrize(
    ("arrays", "groups", "tsz", "match"),
    [
        pytest.param(
            {},
            None,
            None,
            "channel 4: events is missing; it must be a rank-1 integer array",
            id="events-missing",
        ),
        pytest.param(
            {"events": np.zeros((3, 1), np.int64)},
            None,
            None,
            r"channel 4: events must be a rank-1 integer array, got shape "
            r"\(3, 1\) and dtype int64",
            id="events-rank",
        ),
        pytest.param(
            {"events": _FLOAT_COLUMN},
            None,
            None,
            r"channel 4: events must be a rank-1 integer array, got shape "
            r"\(3,\) and dtype float32",
            id="events-dtype",
        ),
        pytest.param(
            {"events": _BUILT_EVENTS, "durations": np.zeros(2, np.int64)},
            None,
            None,
            r"channel 4: durations must be a rank-1 integer array with 3 "
            r"rows, got shape \(2,\) and dtype int64",
            id="durations-rows",
        ),
        pytest.param(
            {"events": _BUILT_EVENTS, "durations": _FLOAT_COLUMN},
            None,
            None,
            "channel 4: durations must be a rank-1 integer array with 3 rows",
            id="durations-dtype",
        ),
        pytest.param(
            {"events": _BUILT_EVENTS, "durations": np.zeros((3, 1), np.int64)},
            None,
            None,
            "channel 4: durations must be a rank-1 integer array",
            id="durations-rank",
        ),
        pytest.param(
            {"events": _BUILT_EVENTS, "labels": _FLOAT_COLUMN},
            None,
            None,
            "channel 4: labels must be a rank-1 integer array with 3 rows",
            id="labels-dtype",
        ),
        pytest.param(
            {"events": _BUILT_EVENTS, "labels": np.zeros(2, np.uint16)},
            None,
            None,
            r"channel 4: labels must be a rank-1 integer array with 3 rows, "
            r"got shape \(2,\)",
            id="labels-rows",
        ),
        pytest.param(
            {"events": _BUILT_EVENTS, "values": _INTEGER_COLUMN},
            None,
            None,
            "channel 4: values must be a rank-1 floating-point array with 3 "
            "rows",
            id="values-dtype",
        ),
        pytest.param(
            {"events": _BUILT_EVENTS, "values": np.zeros(4, np.float32)},
            None,
            None,
            r"channel 4: values must be a rank-1 floating-point array with 3 "
            r"rows, got shape \(4,\)",
            id="values-rows",
        ),
        pytest.param(
            {"events": _BUILT_EVENTS, "text": np.zeros((3, 4), np.int16)},
            None,
            None,
            "channel 4: text must be a rank-2 uint8 array with 3 rows",
            id="text-dtype",
        ),
        pytest.param(
            {"events": _BUILT_EVENTS, "text": np.zeros(3, np.uint8)},
            None,
            None,
            "channel 4: text must be a rank-2 uint8 array with 3 rows",
            id="text-rank",
        ),
        pytest.param(
            {"events": _BUILT_EVENTS, "text": np.zeros((2, 4), np.uint8)},
            None,
            None,
            r"channel 4: text must be a rank-2 uint8 array with 3 rows, got "
            r"shape \(2, 4\)",
            id="text-rows",
        ),
        pytest.param(
            {"events": _BUILT_EVENTS, "channels": np.zeros((2, 1), np.uint8)},
            None,
            None,
            r"channel 4: channels must be a rank-2 uint8 array with 3 rows, "
            r"got shape \(2, 1\) and dtype uint8",
            id="channels-rows",
        ),
        pytest.param(
            {"events": _BUILT_EVENTS, "channels": np.zeros((3, 1), np.uint16)},
            None,
            None,
            r"channel 4: channels must be a rank-2 uint8 array with 3 rows, "
            r"got shape \(3, 1\) and dtype uint16",
            id="channels-dtype",
        ),
        pytest.param(
            {"events": _BUILT_EVENTS, "channels": np.zeros(3, np.uint8)},
            None,
            None,
            "channel 4: channels must be a rank-2 uint8 array",
            id="channels-rank",
        ),
        pytest.param(
            {
                "events": _BUILT_EVENTS,
                "waveforms": (np.zeros(3, np.float32), _SNIPPET_ATTRS),
            },
            None,
            None,
            "channel 4: waveforms must be a rank-2 floating-point array with "
            "3 rows",
            id="waveforms-rank",
        ),
        pytest.param(
            {
                "events": _BUILT_EVENTS,
                "waveforms": (np.zeros((3, 4), np.int16), _SNIPPET_ATTRS),
            },
            None,
            None,
            "channel 4: waveforms must be a rank-2 floating-point array",
            id="waveforms-dtype",
        ),
        pytest.param(
            {
                "events": _BUILT_EVENTS,
                "waveforms": (np.zeros((2, 4), np.float32), _SNIPPET_ATTRS),
            },
            None,
            None,
            r"channel 4: waveforms must be a rank-2 floating-point array with "
            r"3 rows, got shape \(2, 4\)",
            id="waveforms-rows",
        ),
        pytest.param(
            {
                "events": _BUILT_EVENTS,
                "waveforms": (np.zeros((3, 4), np.float32), {"tsz": {}}),
            },
            None,
            None,
            "channel 4 waveforms: tsz.period_us is missing",
            id="waveforms-period",
        ),
        pytest.param(
            {
                "events": _BUILT_EVENTS,
                "waveform_mean": (np.zeros((2, 4), np.float32), {}),
            },
            None,
            None,
            "channel 4 waveform_mean: tsz is missing",
            id="waveform-mean-attributes",
        ),
        pytest.param(
            {
                "events": _BUILT_EVENTS,
                "waveform_mean": (np.zeros(4, np.float32), _SNIPPET_ATTRS),
            },
            None,
            None,
            r"channel 4: waveform_mean must be a rank-2 floating-point array, "
            r"got shape \(4,\)",
            id="waveform-mean-rank",
        ),
        pytest.param(
            {
                "events": _BUILT_EVENTS,
                "waveform_mean": (np.zeros((2, 4), np.int16), _SNIPPET_ATTRS),
            },
            None,
            None,
            r"channel 4: waveform_mean must be a rank-2 floating-point array, "
            r"got shape \(2, 4\) and dtype int16",
            id="waveform-mean-dtype",
        ),
        pytest.param(
            {
                "events": _BUILT_EVENTS,
                "1/count": _counts(1, 1),
                "3/count": _counts(1, 1),
            },
            {"1": {"tsz": {"period_us": 1000.0}}, "3": {}},
            None,
            "channel 4: count levels must be numbered from 1 with no gap, got "
            "1, 3",
            id="count-level-gap",
        ),
        pytest.param(
            {"events": _BUILT_EVENTS, "1/count": _counts(1, 1)},
            {"1": {"tsz": {}}},
            None,
            "channel 4 count level 1: tsz.period_us is missing",
            id="count-level-period",
        ),
        pytest.param(
            {"events": _BUILT_EVENTS, "1": _counts(1, 1)},
            None,
            None,
            "channel 4: count level 1 is an array; a count level must be a "
            "group that holds count",
            id="count-level-array",
        ),
        pytest.param(
            {"events": _BUILT_EVENTS},
            _levels(1000.0),
            None,
            "channel 4 count level 1: count is missing; it must be a rank-2 "
            "integer array",
            id="count-missing",
        ),
        pytest.param(
            {"events": _BUILT_EVENTS, "1/count": np.zeros(4, np.uint32)},
            _levels(1000.0),
            None,
            r"channel 4 count level 1: count must be a rank-2 integer array, "
            r"got shape \(4,\) and dtype uint32",
            id="count-rank",
        ),
        pytest.param(
            {"events": _BUILT_EVENTS, "1/count": np.zeros((4, 1), np.float32)},
            _levels(1000.0),
            None,
            "channel 4 count level 1: count must be a rank-2 integer array",
            id="count-dtype",
        ),
        pytest.param(
            {
                "events": _BUILT_EVENTS,
                "1/count": _counts(1, 1),
                "2/count": _counts(1, 1),
            },
            _levels(1000.0, 3000.0),
            None,
            "channel 4 count level 2: tsz.period_us must be 4000.0, 4 times "
            "that of count level 1, got 3000.0",
            id="count-level-ratio",
        ),
        pytest.param(
            {
                "events": _BUILT_EVENTS,
                "1/count": _counts(1, 1),
                "2/count": _counts(1, 1),
                "3/count": _counts(1, 1),
            },
            _levels(1000.0, 4000.0, 15000.0),
            None,
            "channel 4 count level 3: tsz.period_us must be 16000.0, 4 times "
            "that of count level 2, got 15000.0",
            id="count-level-ratio-past-level-2",
        ),
        pytest.param(
            {
                "events": _BUILT_EVENTS,
                "1/count": _counts(1, 1),
                "2/count": _counts(1, 1),
            },
            _levels(1000.0, 4000.0 * (1 + 1e-8)),
            None,
            r"channel 4 count level 2: tsz\.period_us must be 4000\.0",
            id="count-level-ratio-past-the-tolerance",
        ),
        pytest.param(
            {
                "events": _BUILT_EVENTS,
                "1/count": _counts(1, 2),
                "2/count": _counts(1, 3),
            },
            _levels(1000.0, 4000.0),
            None,
            r"channel 4 count level 2: count must have the 2 columns of count "
            r"level 1, got shape \(1, 3\)",
            id="count-level-widths",
        ),
        pytest.param(
            {"events": _BUILT_EVENTS, "1/count": _counts(1, 2)},
            _levels(1000.0),
            {**_BUILT_TSZ, "label_names": ["a", "b", "c"]},
            r"channel 4 count level 1: count must have 3 columns, one per "
            r"label name, got shape \(1, 2\)",
            id="count-level-label-names",
        ),
        pytest.param(
            {"events": _BUILT_EVENTS},
            None,
            {"kind": "event"},
            "channel 4: tsz.name is missing",
            id="name-missing",
        ),
        pytest.param(
            {"events": _BUILT_EVENTS},
            None,
            {**_BUILT_TSZ, "max_duration_us": -1},
            "channel 4: tsz.max_duration_us must be an integer from 0",
            id="max-duration-negative",
        ),
    ],
)
def test_parse_rejects_a_malformed_channel(arrays, groups, tsz, match):
    root = _build(arrays, groups=groups, tsz=tsz)
    with pytest.raises(ValueError, match=match):
        _parse(root, _BUILT_KEY)


def test_records_built_by_hand_derive_their_time_axes_and_names():
    values = np.zeros((2, 3), np.float32)
    waveform = Waveform(
        values=values, period_us=10.0, event_sample=2, rows="event"
    )
    assert waveform.samples == len(waveform) == 3
    assert waveform.time_us.tolist() == [-20.0, -10.0, 0.0]
    assert waveform.time_s.tolist() == [-2e-5, -1e-5, 0.0]
    counts = Counts(
        level=2,
        first=3,
        period_us=4000.0,
        offset_us=0,
        count=np.zeros((2, 1), np.uint32),
        label_names=None,
    )
    assert counts.time_us.tolist() == [12_000.0, 16_000.0]
    assert counts.time_s.tolist() == [0.012, 0.016]
    table = EventTable(
        time_us=np.array([5, 6], np.int64),
        duration_us=None,
        label=np.array([1, 0], np.uint16),
        value=None,
        text=None,
        channel_keys=[[], [3]],
        waveform=None,
        label_names=("a", "b"),
    )
    assert table.label_name == ["b", "a"]
    assert table.applies_to(3).tolist() == [True, True]
    assert table.applies_to(2).tolist() == [True, False]
    assert table.whole_recording is False


def test_applies_to_follows_whole_recording_without_channel_keys():
    table = EventTable(
        time_us=np.array([5, 6], np.int64),
        duration_us=None,
        label=None,
        value=None,
        text=None,
        channel_keys=None,
        waveform=None,
        label_names=None,
        whole_recording=True,
    )
    assert table.applies_to(3).tolist() == [True, True]
    with pytest.raises(ValueError, match="the table has no channel_keys"):
        dataclasses.replace(table, whole_recording=False).applies_to(3)


def test_snippets_read_returns_every_row():
    array = _build(
        {
            "events": _BUILT_EVENTS,
            "waveforms": (np.ones((3, 2), np.float32), _SNIPPET_ATTRS),
        }
    )[f"{_BUILT_KEY}/waveforms"]
    snippets = Snippets(
        array=array, period_us=100.0, event_sample=1, rows="label"
    )
    waveform = snippets.read()
    assert waveform.values.tolist() == [[1.0, 1.0]] * 3
    assert (waveform.period_us, waveform.event_sample) == (100.0, 1)
    assert waveform.rows == "label"


def test_count_levels_past_9_parse_in_level_order():
    numbers = range(1, 11)
    arrays = {"events": _BUILT_EVENTS}
    arrays.update({f"{number}/count": _counts(1, 1) for number in numbers})
    periods = [1000.0 * 4 ** (number - 1) for number in numbers]
    root = _build(arrays, groups=_levels(*periods))
    zarr.consolidate_metadata(root.store)
    channel = _parse(open_root(root.store), _BUILT_KEY)
    assert list(channel.count_levels) == list(numbers)


def test_records_are_frozen(odd):
    channel = odd[9]
    table = channel.read()
    counts = channel.read_counts(level=1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        channel.name = "renamed"
    with pytest.raises(dataclasses.FrozenInstanceError):
        table.time_us = table.time_us[:1]
    with pytest.raises(dataclasses.FrozenInstanceError):
        counts.first = 1


def test_records_compare_by_identity(odd_bundle):
    first = _parse(open_root(odd_bundle), 9)
    second = _parse(open_root(odd_bundle), 9)
    assert first != second
    assert {first: "first"}[first] == "first"
    assert len({first, second}) == 2
    assert first.read() != first.read()
    assert first.read_counts(level=1) != first.read_counts(level=1)


def test_records_have_no_instance_dict(odd):
    channel = odd[9]
    records = [
        channel,
        channel.count_levels[1],
        channel.waveforms,
        channel.waveforms.read(),
        channel.read_counts(level=1),
        channel.read(),
    ]
    for record in records:
        assert not hasattr(record, "__dict__"), type(record).__name__


def test_table_columns_hold_the_times_then_each_present_column(mixed):
    table = mixed[3].read()
    columns = table.columns()
    assert list(columns) == [
        "time_us",
        "duration_us",
        "label",
        "label_name",
        "text",
        "channel_keys",
    ]
    assert columns["time_us"] is table.time_us
    assert columns["duration_us"] is table.duration_us
    assert columns["label"] is table.label
    for name in ("label_name", "text", "channel_keys"):
        assert columns[name].dtype == np.object_
        assert columns[name].shape == (len(table),)
    assert columns["label_name"].tolist() == table.label_name
    assert columns["text"].tolist() == table.text
    assert columns["channel_keys"].tolist() == [[0], [], [0, 1], [], [1], []]


def test_table_columns_hold_values_and_leave_out_the_waveform(odd):
    table = odd[9].read()
    assert table.waveform is not None
    columns = table.columns()
    assert list(columns) == [
        "time_us",
        "label",
        "value",
        "text",
        "channel_keys",
    ]
    assert columns["value"] is table.value


def test_table_columns_of_an_empty_window_are_empty_arrays(mixed):
    columns = mixed[3].read(0, 1).columns()
    assert len(columns) == 6
    assert all(values.shape == (0,) for values in columns.values())
    assert columns["text"].dtype == np.object_


def test_counts_columns_name_each_label_column_by_label_name(mixed):
    counts = mixed[4].read_counts(level=4)
    columns = counts.columns()
    assert list(columns) == ["time_us", "ripple", "fast-ripple"]
    np.testing.assert_array_equal(columns["time_us"], counts.time_us)
    np.testing.assert_array_equal(columns["ripple"], counts.count[:, 0])
    np.testing.assert_array_equal(columns["fast-ripple"], counts.count[:, 1])
    assert columns["ripple"].dtype == np.uint32


def test_counts_columns_name_label_columns_by_index_without_label_names(odd):
    columns = odd[9].read_counts(level=1).columns()
    assert list(columns) == ["time_us", "0", "1", "2"]


@pytest.mark.parametrize(
    "label_names", [("a", "a"), ("time_us", "b")], ids=["repeated", "time"]
)
def test_counts_columns_reject_label_names_that_cannot_name_columns(
    label_names,
):
    counts = Counts(
        level=1,
        first=0,
        period_us=1000.0,
        offset_us=0,
        count=np.zeros((2, 2), np.uint32),
        label_names=label_names,
    )
    message = (
        "label_names must be distinct and must not include 'time_us' to "
        f"name count columns, got {label_names!r}"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        counts.columns()


def test_waveform_columns_hold_time_us_and_one_column_per_row(odd):
    waveform = odd[9].waveform_mean.read()
    columns = waveform.columns()
    assert list(columns) == ["time_us", "0", "1", "2"]
    np.testing.assert_array_equal(columns["time_us"], waveform.time_us)
    for row in range(ODD_LABELS):
        np.testing.assert_array_equal(columns[str(row)], waveform.values[row])


def test_waveform_columns_of_no_rows_hold_only_time_us(odd):
    tie_us = 1_000_000 + 250_000 * ODD_TIE
    waveform = odd[9].read(tie_us + 1, tie_us + 2).waveform
    assert waveform.values.shape == (0, ODD_WAVEFORM_SAMPLES)
    assert list(waveform.columns()) == ["time_us"]


def test_table_repr_names_the_length_and_the_columns(odd, mixed):
    for table in [
        odd[9].read(),
        odd[12].read(columns=[]),
        mixed[2].read(),
        mixed[3].read(),
    ]:
        assert repr(table) == (
            f"EventTable(len={len(table)}, columns={tuple(table.columns())!r}, "
            f"whole_recording={table.whole_recording})"
        )


def test_table_repr_has_the_same_length_for_a_short_and_a_long_read(odd):
    channel = odd[9]
    short = channel.read(None, 1_000_000 + 250_000 * 12)
    long = channel.read()
    assert 10 <= len(short) < len(long) < 100
    assert len(repr(short)) == len(repr(long))


def test_table_columns_keep_equal_length_key_lists_as_lists():
    table = EventTable(
        time_us=np.array([5, 6], np.int64),
        duration_us=None,
        label=None,
        value=None,
        text=None,
        channel_keys=[[0, 1], [2, 3]],
        waveform=None,
        label_names=None,
    )
    keys = table.columns()["channel_keys"]
    assert keys.shape == (2,)
    assert keys.tolist() == [[0, 1], [2, 3]]


_RESULT_READS = {
    "waveform": lambda channel: channel.waveform_mean.read(),
    "counts": lambda channel: channel.read_counts(level=1),
    "events": lambda channel: channel.read(),
}


@pytest.mark.parametrize(
    ("record", "method", "package"),
    [
        ("waveform", "to_dataframe", "pandas"),
        ("waveform", "to_xarray", "xarray"),
        ("counts", "to_dataframe", "pandas"),
        ("counts", "to_xarray", "xarray"),
        ("events", "to_dataframe", "pandas"),
    ],
)
def test_conversions_raise_import_error_naming_the_missing_package(
    odd, monkeypatch, record, method, package
):
    result = _RESULT_READS[record](odd[9])
    monkeypatch.setitem(sys.modules, package, None)
    with pytest.raises(ImportError) as info:
        getattr(result, method)()
    assert info.value.name == package
