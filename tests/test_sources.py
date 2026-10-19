import dataclasses
import datetime as dt
import json
import re

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from tests.conftest import (
    ODD_CHANNEL_SETS,
    ODD_EVENT_SAMPLE,
    ODD_EVENTS,
    ODD_FULL_TEXT,
    ODD_FULL_TEXT_ROW,
    ODD_INTERVAL_DURATIONS,
    ODD_LABELS,
    ODD_OFFSET_US,
    ODD_TEXT_WIDTH,
)
from tests.test_bundle import (
    _ACQUIRE,
    _RECORDING,
    _SESSION,
    _SESSION_US,
    _SORTER,
    _SPEC_OPTS,
)
from timeseries_zarr.bundle import write_bundle
from timeseries_zarr.columns import channels_rows
from timeseries_zarr.protocols import (
    ContinuousChannelSource,
    EventChannelSource,
    ProvenanceSource,
)
from timeseries_zarr.read_bundle import open_bundle
from timeseries_zarr.recording import RECORDING_FILE
from timeseries_zarr.sources import (
    BundleContinuousSource,
    BundleEventSource,
    bundle_epoch_us,
    bundle_sources,
    channel_ids,
    recording_meta,
)
from timeseries_zarr.types import RecordingMeta
from timeseries_zarr.zarr_io import read_rows

_EARLY_US = _SESSION_US - 2_000_000
_FAR_START = dt.datetime(
    3000, 1, 1, 0, 0, 0, 123457, tzinfo=dt.timezone(dt.timedelta(hours=-5))
)
_FAR_US = 32_503_698_000_123_457
_MEMBERS = ("minmax", "mean", "var", "count")
_MIXED_IDS = {0: "LA1", 1: "LA2", 2: "units", 3: "marks", 4: "hfo"}
_ODD_IDS = {0: "0", 2: "two", 5: "five", 7: "seven", 9: "9", 12: "12"}
_PERIOD_ATTRS = {"tsz": {"period_us": 100.0}}
_SHIFT_US = 3_000_000


@pytest.fixture(scope="module")
def mixed(mixed_bundle):
    return open_bundle(mixed_bundle)


@pytest.fixture(scope="module")
def odd(odd_bundle):
    return open_bundle(odd_bundle)


@pytest.fixture(scope="module")
def mixed_copy(mixed, tmp_path_factory):
    """Return the path of the mixed bundle rewritten through its sources."""
    output = tmp_path_factory.mktemp("copy") / "bundle"
    write_bundle(
        *bundle_sources(mixed),
        output_path=output,
        opts=_SPEC_OPTS,
        recording=recording_meta(mixed),
    )
    return output


def _rewrite(bundle, output, *, opts=_SPEC_OPTS, recording=None, **options):
    """Write the sources of bundle to output and return the reopened copy."""
    write_bundle(
        *bundle_sources(bundle, **options),
        output_path=output,
        opts=opts,
        recording=recording,
    )
    return open_bundle(output)


def _edit_recording(path, key, time=None):
    """Edit the recording.json of the bundle at path and open the bundle.

    key is a top-level key or session.start_time. A time of None deletes
    key. Any other time replaces the value of key.
    """
    document = json.loads((path / RECORDING_FILE).read_text("utf-8"))
    holder = document
    if key.startswith("session."):
        holder, key = document.setdefault("session", {}), key[len("session.") :]
    if time is None:
        del holder[key]
    else:
        holder[key] = time.isoformat(timespec="microseconds")
    (path / RECORDING_FILE).write_text(json.dumps(document), "utf-8")
    return open_bundle(path)


def _root():
    """Return a writable root group in memory with tsz.version 1.0."""
    return zarr.open_group(
        MemoryStore(),
        mode="w",
        zarr_format=3,
        attributes={"tsz": {"version": "1.0"}},
    )


def _event_bundle(arrays, *, tsz=None, groups=None):
    """Return an open bundle in memory that holds event channel 0.

    arrays maps the path of each array below the channel to its data, or to
    a pair of data and attributes. tsz holds further tsz keys of the channel
    group. groups maps the name of each group below the channel to its
    attributes.
    """
    root = _root()
    channel = root.create_group(
        "0",
        attributes={"tsz": {"kind": "event", "name": "built", **(tsz or {})}},
    )
    for name, attributes in (groups or {}).items():
        channel.create_group(name, attributes=attributes)
    for name, value in arrays.items():
        data, attributes = value if isinstance(value, tuple) else (value, {})
        channel.create_array(name, data=np.asarray(data), attributes=attributes)
    zarr.consolidate_metadata(root.store)
    return open_bundle(root.store)


def _aliased_bundle(first_tsz):
    """Return an open bundle in memory with event channels 0 and 1.

    Channel 1 has the id "0", and its first event names channel 0.
    first_tsz holds further tsz keys of channel 0.
    """
    root = _root()
    first = root.create_group(
        "0",
        attributes={"tsz": {"kind": "event", "name": "first", **first_tsz}},
    )
    first.create_array("events", data=np.array([5], np.int64))
    second = root.create_group(
        "1",
        attributes={"tsz": {"kind": "event", "name": "second", "id": "0"}},
    )
    second.create_array("events", data=np.array([10, 20], np.int64))
    second.create_array("channels", data=channels_rows([[0], []], 1))
    zarr.consolidate_metadata(root.store)
    return open_bundle(root.store)


def _event_source(bundle, key=0, **options):
    return BundleEventSource(
        bundle.channels[key], channel_ids(bundle), epoch_us=0, **options
    )


def _count_reads(monkeypatch):
    """Return a list that collects the path of each array a source reads."""
    paths = []

    def counted(array, start, stop):
        paths.append(array.path)
        return read_rows(array, start, stop)

    monkeypatch.setattr("timeseries_zarr.sources.read_rows", counted)
    return paths


def _member(level, name):
    array = getattr(level, name)
    return None if array is None else array[:]


def _assert_same_continuous(original, copy):
    assert (copy.name, copy.id, copy.unit) == (
        original.name,
        original.id,
        original.unit,
    )
    assert (copy.rate_hz, copy.offset_us) == (
        original.rate_hz,
        original.offset_us,
    )
    assert copy.provenance == original.provenance
    np.testing.assert_array_equal(copy.raw[:], original.raw[:])
    assert list(copy.levels) == list(original.levels)
    for number, level in original.levels.items():
        for name in _MEMBERS:
            np.testing.assert_array_equal(
                _member(copy.levels[number], name), _member(level, name)
            )


def _assert_same_events(original, copy):
    assert (copy.name, copy.id, copy.unit) == (
        original.name,
        original.id,
        original.unit,
    )
    assert copy.label_names == original.label_names
    assert copy.max_duration_us == original.max_duration_us
    assert copy.provenance == original.provenance
    assert copy.columns == original.columns
    table, original_table = copy.read(), original.read()
    np.testing.assert_array_equal(table.time_us, original_table.time_us)
    for name in ("duration_us", "label"):
        np.testing.assert_array_equal(
            getattr(table, name), getattr(original_table, name)
        )
    assert table.text == original_table.text
    assert table.channel_keys == original_table.channel_keys
    if original.waveform_mean is None:
        assert copy.waveform_mean is None
    else:
        np.testing.assert_array_equal(
            copy.waveform_mean.read().values,
            original.waveform_mean.read().values,
        )
        assert copy.waveform_mean.period_us == original.waveform_mean.period_us
    assert list(copy.count_levels) == list(original.count_levels)
    for number, level in original.count_levels.items():
        np.testing.assert_array_equal(
            copy.count_levels[number].count[:], level.count[:]
        )


def _assert_same_channels(original, copy):
    assert list(copy.channels) == list(original.channels)
    for key, channel in original.continuous.items():
        _assert_same_continuous(channel, copy.continuous[key])
    for key, channel in original.events.items():
        _assert_same_events(channel, copy.events[key])


def test_a_continuous_source_follows_the_continuous_and_provenance_protocols(
    mixed,
):
    source = BundleContinuousSource(mixed.channels[0], epoch_us=0)
    assert isinstance(source, ContinuousChannelSource)
    assert isinstance(source, ProvenanceSource)
    assert not isinstance(source, EventChannelSource)


def test_an_event_source_follows_the_event_and_provenance_protocols(mixed):
    source = _event_source(mixed, 3)
    assert isinstance(source, EventChannelSource)
    assert isinstance(source, ProvenanceSource)
    assert not isinstance(source, ContinuousChannelSource)


def test_a_rewrite_of_the_mixed_bundle_reproduces_every_channel(
    mixed, mixed_copy
):
    with open_bundle(mixed_copy) as copy:
        _assert_same_channels(mixed, copy)


def test_a_rewrite_of_the_mixed_bundle_reproduces_both_recording_times(
    mixed, mixed_copy
):
    with open_bundle(mixed_copy) as copy:
        recording = copy.recording
    assert recording.session_start_time.isoformat() == _SESSION.isoformat()
    assert (
        recording.reference_time.isoformat()
        == mixed.recording.reference_time.isoformat()
    )
    assert recording == mixed.recording


def test_a_rewrite_of_the_mixed_bundle_reproduces_every_file_byte_for_byte(
    mixed_bundle, mixed_copy
):
    def files(path):
        return {
            file.relative_to(path).as_posix(): file.read_bytes()
            for file in path.rglob("*")
            if file.is_file()
        }

    assert files(mixed_copy) == files(mixed_bundle)


def test_a_subset_without_the_channels_column_keeps_every_time(mixed, tmp_path):
    copy = _rewrite(
        mixed,
        tmp_path / "subset",
        recording=recording_meta(mixed),
        keys=[1, 3],
        drop_columns={"channels"},
    )
    assert channel_ids(copy) == {0: "LA2", 1: "marks"}
    _assert_same_continuous(mixed.channels[1], copy.channels[0])
    marks = copy.events[1]
    assert marks.channels is None
    assert marks.columns == ("durations", "labels", "text")
    np.testing.assert_array_equal(marks.events[:], mixed.events[3].events[:])
    assert copy.recording == mixed.recording


def test_a_subset_that_leaves_out_a_named_channel_makes_the_writer_raise(
    mixed, tmp_path
):
    with pytest.raises(
        ValueError,
        match=r"channel 3: event \d+ names channel key 0, and the source maps "
        r"the channel keys \[1, 3\]",
    ):
        _rewrite(mixed, tmp_path / "subset", keys=[1, 3])


@pytest.mark.parametrize("first_tsz", [{}, {"id": "0"}])
def test_an_event_that_names_a_dropped_channel_with_a_kept_id_raises(
    tmp_path, first_tsz
):
    bundle = _aliased_bundle(first_tsz)
    message = (
        r"channel 1: event 0 names channel key 0, and the source maps the "
        r"channel keys \[1\]"
    )
    _, events = bundle_sources(bundle, [1])
    with pytest.raises(ValueError, match=message):
        events[0].read_channel_ids(0, 2)
    with pytest.raises(ValueError, match=message):
        _rewrite(bundle, tmp_path / "subset", keys=[1])


@pytest.mark.parametrize("first_tsz", [{}, {"id": "0"}])
def test_bundle_sources_rejects_two_kept_channels_that_share_an_id(first_tsz):
    bundle = _aliased_bundle(first_tsz)
    message = (
        "channels 0 and 1 share the id '0'; keys must name at most one of them"
    )
    with pytest.raises(ValueError, match=message):
        bundle_sources(bundle)
    with pytest.raises(ValueError, match=message):
        bundle_sources(bundle, [1, 0])
    assert bundle_sources(bundle, [0])[1][0].id == "0"


def test_a_rewrite_with_write_var_adds_var_to_every_level(mixed, tmp_path):
    opts = dataclasses.replace(_SPEC_OPTS, write_var=True)
    copy = _rewrite(mixed, tmp_path / "var", opts=opts)
    for key, channel in mixed.continuous.items():
        levels = copy.continuous[key].levels
        assert list(levels) == list(channel.levels)
        for number, level in channel.levels.items():
            assert level.var is None
            assert levels[number].var is not None
            np.testing.assert_array_equal(
                levels[number].minmax[:], level.minmax[:]
            )


def test_channel_ids_gives_a_channel_without_an_id_its_key(odd):
    assert channel_ids(odd) == _ODD_IDS
    assert BundleContinuousSource(odd.channels[0], epoch_us=0).id == "0"
    assert _event_source(odd, 9).id == "9"


def test_channel_ids_maps_each_key_to_the_id_of_its_channel(mixed):
    assert channel_ids(mixed) == _MIXED_IDS


def test_channel_ids_maps_only_the_keys_it_names_in_key_order(mixed):
    assert channel_ids(mixed, [3, 1, 3]) == {1: "LA2", 3: "marks"}
    assert channel_ids(mixed, iter([4, 0])) == {0: "LA1", 4: "hfo"}
    assert channel_ids(mixed, [7]) == {}


@pytest.mark.parametrize("first_tsz", [{}, {"id": "0"}])
def test_channel_ids_rejects_two_named_channels_that_share_an_id(first_tsz):
    bundle = _aliased_bundle(first_tsz)
    with pytest.raises(
        ValueError,
        match="^channels 0 and 1 share the id '0'; keys must name at most "
        "one of them$",
    ):
        channel_ids(bundle)
    assert channel_ids(bundle, [1]) == {1: "0"}


def test_a_rewrite_of_the_odd_bundle_renumbers_the_channels_by_id(
    odd, tmp_path
):
    copy = _rewrite(odd, tmp_path / "odd")
    assert channel_ids(copy) == dict(enumerate(_ODD_IDS.values()))
    renumbered = {0: [0], 2: [1], 5: [2]}
    expected = [
        sorted(key for old in keys for key in renumbered[old])
        for keys in ODD_CHANNEL_SETS
    ]
    marks = copy.events[4].read()
    assert marks.channel_keys[: len(ODD_CHANNEL_SETS)] == expected
    assert marks.text[ODD_FULL_TEXT_ROW] == ODD_FULL_TEXT
    assert copy.continuous[2].offset_us == ODD_OFFSET_US
    assert copy.events[5].max_duration_us == max(ODD_INTERVAL_DURATIONS)


def test_a_rewrite_leaves_out_values_waveforms_and_the_event_sample(
    odd, tmp_path
):
    marks = odd.events[9]
    assert marks.waveform_mean.event_sample == ODD_EVENT_SAMPLE
    assert marks.values is not None
    assert marks.waveforms is not None
    copied = _rewrite(odd, tmp_path / "odd").events[4]
    assert copied.waveform_mean.event_sample == 0
    assert copied.values is None
    assert copied.waveforms is None


def test_num_labels_is_the_length_of_label_names(mixed):
    assert _event_source(mixed, 2).num_labels() == 3


def test_num_labels_is_the_label_columns_of_a_count_level_without_names(odd):
    source = _event_source(odd, 9)
    assert source.label_names() is None
    assert source.num_labels() == ODD_LABELS


def test_num_labels_is_the_rows_of_waveform_mean_without_count_levels(
    tmp_path,
):
    bundle = _event_bundle(
        {
            "events": np.array([10, 20, 30], dtype=np.int64),
            "labels": np.array([0, 1, 1], dtype=np.uint16),
            "waveform_mean": (np.ones((4, 2), np.float32), _PERIOD_ATTRS),
        }
    )
    source = _event_source(bundle)
    assert source.num_labels() == 4
    assert source.waveform_mean().shape == (4, 2)
    assert source.waveform_period_us() == 100.0
    copy = _rewrite(bundle, tmp_path / "copy")
    assert copy.events[0].waveform_mean.read().values.shape == (4, 2)


def test_num_labels_takes_a_count_level_before_waveform_mean_and_labels():
    bundle = _event_bundle(
        {
            "events": np.array([10, 20, 30], dtype=np.int64),
            "labels": np.array([0, 1, 2], dtype=np.uint16),
            "waveform_mean": (np.ones((3, 2), np.float32), _PERIOD_ATTRS),
            "1/count": np.array([[1, 1, 1, 0]], dtype=np.uint32),
        },
        groups={"1": {"tsz": {"period_us": 1000.0}}},
    )
    assert _event_source(bundle).num_labels() == 4


def test_num_labels_is_one_more_than_the_largest_label_without_other_widths(
    tmp_path,
):
    bundle = _event_bundle(
        {
            "events": np.array([10, 20, 30], dtype=np.int64),
            "labels": np.array([0, 4, 1], dtype=np.uint16),
        }
    )
    assert _event_source(bundle).num_labels() == 5
    copy = _rewrite(bundle, tmp_path / "copy")
    assert copy.events[0].labels[:].tolist() == [0, 4, 1]


def test_num_labels_reads_the_labels_column_once(monkeypatch):
    bundle = _event_bundle(
        {
            "events": np.array([10, 20, 30], dtype=np.int64),
            "labels": np.array([0, 4, 1], dtype=np.uint16),
        }
    )
    source = _event_source(bundle)
    reads = _count_reads(monkeypatch)
    assert [source.num_labels(), source.num_labels()] == [5, 5]
    assert reads == ["0/labels"]


def test_num_labels_is_0_for_labels_without_events():
    bundle = _event_bundle(
        {
            "events": np.zeros(0, dtype=np.int64),
            "labels": np.zeros(0, dtype=np.uint16),
        }
    )
    source = _event_source(bundle)
    assert source.columns() == {"labels"}
    assert source.num_labels() == 0


def test_num_labels_is_0_without_a_labels_column(mixed, odd):
    assert _event_source(odd, 12).num_labels() == 0
    dropped = _event_source(mixed, 2, columns=())
    assert dropped.num_labels() == 0


def test_columns_leave_out_values_and_waveforms(odd):
    assert _event_source(odd, 9).columns() == {
        "labels",
        "text",
        "channels",
        "waveform_mean",
    }


def test_columns_leave_out_a_text_column_of_width_0():
    bundle = _event_bundle(
        {
            "events": np.array([10, 20, 30], dtype=np.int64),
            "text": np.zeros((3, 0), dtype=np.uint8),
        }
    )
    source = _event_source(bundle)
    assert source.columns() == frozenset()
    assert source.text_width() == 0
    assert source.read_text(0, 3) == ["", "", ""]


def test_columns_leave_out_waveform_mean_when_labels_are_dropped(mixed):
    continuous, events = bundle_sources(mixed, drop_columns={"labels"})
    assert [source.columns() for source in events] == [
        frozenset(),
        {"durations", "text", "channels"},
        frozenset(),
    ]
    assert len(continuous) == 2


def test_columns_leave_out_waveform_mean_without_a_labels_column(tmp_path):
    bundle = _event_bundle(
        {
            "events": np.array([10, 20, 30], dtype=np.int64),
            "waveform_mean": (np.ones((2, 4), np.float32), _PERIOD_ATTRS),
        }
    )
    assert _event_source(bundle).columns() == frozenset()
    copy = _rewrite(bundle, tmp_path / "copy")
    assert copy.events[0].waveform_mean is None


def test_columns_carry_only_the_named_columns_the_channel_holds(mixed):
    source = _event_source(mixed, 3, columns=["text", "waveform_mean"])
    assert source.columns() == {"text"}


def test_a_source_rejects_columns_given_as_a_string(mixed):
    with pytest.raises(TypeError, match="got the string 'text'"):
        _event_source(mixed, 3, columns="text")


def test_max_duration_us_is_the_bound_of_the_channel(mixed):
    assert _event_source(mixed, 3).max_duration_us() == 12_000_000


def test_max_duration_us_falls_back_to_the_largest_duration(odd):
    source = _event_source(odd, 12)
    assert source.columns() == {"durations"}
    assert source.max_duration_us() == max(ODD_INTERVAL_DURATIONS)
    assert source.read_durations(0, 5).tolist() == list(ODD_INTERVAL_DURATIONS)


def test_max_duration_us_reads_the_durations_column_once(odd, monkeypatch):
    source = _event_source(odd, 12)
    reads = _count_reads(monkeypatch)
    largest = max(ODD_INTERVAL_DURATIONS)
    assert [source.max_duration_us(), source.max_duration_us()] == [
        largest,
        largest,
    ]
    assert reads == ["12/durations"]


def test_max_duration_us_is_0_for_durations_without_events():
    bundle = _event_bundle(
        {
            "events": np.zeros(0, dtype=np.int64),
            "durations": np.zeros(0, dtype=np.int64),
        }
    )
    assert _event_source(bundle).max_duration_us() == 0


def test_max_duration_us_is_none_without_durations(mixed):
    dropped = _event_source(mixed, 3, columns=["labels"])
    assert dropped.max_duration_us() is None
    assert _event_source(mixed, 2).max_duration_us() is None


def test_text_width_is_the_width_of_the_text_column(odd, mixed):
    assert _event_source(odd, 9).text_width() == ODD_TEXT_WIDTH
    assert _event_source(mixed, 2).text_width() == 0


def test_read_text_decodes_each_row(odd):
    texts = _event_source(odd, 9).read_text(0, ODD_EVENTS)
    assert len(texts) == ODD_EVENTS
    assert texts[ODD_FULL_TEXT_ROW] == ODD_FULL_TEXT
    assert texts[:2] == ["", "e1"]


def test_read_text_names_the_channel_and_event_of_text_that_is_not_utf8():
    bundle = _event_bundle(
        {
            "events": np.array([10, 20, 30], dtype=np.int64),
            "text": np.array([[0x61, 0], [0x62, 0], [0xFF, 0]], np.uint8),
        }
    )
    source = _event_source(bundle)
    assert source.read_text(0, 2) == ["a", "b"]
    with pytest.raises(
        ValueError,
        match=r"^channel 0: event 2 text is not valid UTF-8 at byte 0$",
    ):
        source.read_text(1, 3)


def test_read_channel_ids_maps_the_keys_of_the_kept_channels_to_ids(odd):
    expected = [["0"], ["two", "five"], [], ["0"]]
    _, events = bundle_sources(odd)
    assert events[0].read_channel_ids(0, 4) == expected
    _, events = bundle_sources(odd, [0, 2, 5, 9])
    assert events[0].read_channel_ids(0, 4) == expected
    _, events = bundle_sources(odd, [2, 5, 9])
    assert events[0].read_channel_ids(1, 3) == [["two", "five"], []]
    with pytest.raises(
        ValueError,
        match=r"event 0 names channel key 0, and the source maps the channel "
        r"keys \[2, 5, 9\]",
    ):
        events[0].read_channel_ids(0, 1)


def test_read_channel_ids_rejects_a_key_that_names_no_channel(odd):
    source = BundleEventSource(odd.channels[9], {0: "0"}, epoch_us=0)
    with pytest.raises(
        ValueError,
        match=r"channel 9: event 1 names channel key 2, and the source maps "
        r"the channel keys \[0\]",
    ):
        source.read_channel_ids(1, 2)


@pytest.mark.parametrize(
    ("key", "read", "column"),
    [
        (9, lambda source: source.read_durations(0, 1), "durations"),
        (12, lambda source: source.read_labels(0, 1), "labels"),
        (12, lambda source: source.read_text(0, 1), "text"),
        (12, lambda source: source.read_channel_ids(0, 1), "channels"),
        (12, lambda source: source.waveform_mean(), "waveform_mean"),
        (12, lambda source: source.waveform_period_us(), "waveform_mean"),
    ],
)
def test_reading_a_column_the_channel_lacks_raises(odd, key, read, column):
    with pytest.raises(
        ValueError,
        match=f"channel {key} has no {column} column; read only a column "
        "that columns",
    ):
        read(_event_source(odd, key))


def test_reads_return_the_dtypes_the_writer_takes(mixed):
    continuous = BundleContinuousSource(mixed.channels[0], epoch_us=0)
    samples = continuous.read_samples(0, 10)
    assert samples.dtype == np.float32
    np.testing.assert_array_equal(samples, mixed.channels[0].raw[:10])
    events = _event_source(mixed, 3)
    assert events.read_events(0, 6).dtype == np.int64
    assert events.read_durations(0, 6).dtype == np.int64
    assert events.read_labels(0, 6).dtype == np.uint16
    assert _event_source(mixed, 2).waveform_mean().dtype == np.float32


@pytest.mark.parametrize("label", [65536, -1])
def test_read_labels_rejects_a_label_outside_the_uint16_range(tmp_path, label):
    bundle = _event_bundle(
        {
            "events": np.array([10, 20, 30], dtype=np.int64),
            "labels": np.array([0, label, 1], dtype=np.int32),
        },
        tsz={"label_names": ["a", "b"]},
    )
    message = (
        f"channel 0: labels must be at least 0 and below 65536, got {label}"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        _event_source(bundle).read_labels(0, 3)
    output = tmp_path / "copy"
    with pytest.raises(ValueError, match=re.escape(message)):
        _rewrite(bundle, output)
    with pytest.raises(ValueError, match="has no consolidated metadata"):
        open_bundle(output)


def test_sources_copy_the_name_unit_label_names_and_provenance(mixed):
    lead = BundleContinuousSource(mixed.channels[0], epoch_us=0)
    assert (lead.name, lead.unit, lead.rate_hz()) == ("LA1", "uV", 256.0)
    assert lead.num_samples() == mixed.channels[0].num_samples
    assert lead.provenance() == _ACQUIRE
    assert type(lead.provenance()) is dict
    assert lead.provenance() is not mixed.channels[0].provenance
    lead.provenance()["settings"]["gain"] = 8
    assert mixed.channels[0].provenance == _ACQUIRE
    assert (
        BundleContinuousSource(mixed.channels[1], epoch_us=0).provenance()
        is None
    )
    units = _event_source(mixed, 2)
    assert (units.name, units.unit) == ("units", None)
    assert units.label_names() == ["unit-0", "unit-1", "unit-2"]
    assert units.provenance() == _SORTER
    assert units.provenance() is not mixed.channels[2].provenance
    dense = _event_source(mixed, 4)
    assert dense.provenance() is None
    assert dense.num_events() == mixed.channels[4].num_events


def test_the_reference_time_moves_to_the_first_kept_sample_without_a_recording(
    write_mixed, tmp_path
):
    plain = open_bundle(write_mixed("plain", recording=None))
    assert recording_meta(plain) is None
    copy = _rewrite(plain, tmp_path / "copy", keys=[1])
    assert plain.channels[1].offset_us == 500_000
    assert copy.channels[0].offset_us == 0
    np.testing.assert_array_equal(
        copy.channels[0].raw[:], plain.channels[1].raw[:]
    )


def test_a_kept_event_channel_before_the_first_sample_goes_negative(
    write_mixed, tmp_path
):
    plain = open_bundle(write_mixed("plain", recording=None))
    first_spike = int(plain.channels[2].events[0])
    assert 0 < first_spike < 500_000
    copy = _rewrite(plain, tmp_path / "copy", keys=[1, 2])
    assert copy.channels[0].offset_us == 0
    np.testing.assert_array_equal(
        copy.channels[1].events[:], plain.channels[2].events[:] - 500_000
    )
    assert int(copy.channels[1].events[0]) == first_spike - 500_000 < 0


def test_an_epoch_us_override_shifts_every_wall_clock_time(mixed):
    continuous, events = bundle_sources(mixed, epoch_us=1_000)
    assert [source.start_us() for source in continuous] == [1_000, 501_000]
    stored = mixed.channels[3].events[:2]
    assert events[1].read_events(0, 2).tolist() == (stored + 1_000).tolist()


def test_an_epoch_us_override_and_a_session_start_date_the_copy(
    write_mixed, tmp_path
):
    plain = open_bundle(write_mixed("plain", recording=None))
    copy = _rewrite(
        plain,
        tmp_path / "copy",
        recording=RecordingMeta(session_start_time=_SESSION),
        epoch_us=_SESSION_US,
    )
    assert copy.recording.reference_time == _SESSION
    assert copy.recording.session_start_time == _SESSION
    _assert_same_channels(plain, copy)


def test_the_sources_default_to_the_epoch_of_the_reference_time(mixed):
    continuous, events = bundle_sources(mixed)
    assert continuous[1].start_us() == _SESSION_US + 500_000
    first_spike = int(mixed.channels[2].events[0])
    assert events[0].read_events(0, 1).tolist() == [_SESSION_US + first_spike]


def test_bundle_epoch_us_is_the_reference_time_in_microseconds(mixed):
    assert bundle_epoch_us(mixed) == _SESSION_US


def test_bundle_epoch_us_is_the_reference_time_before_the_session_start(
    write_mixed,
):
    bundle = open_bundle(write_mixed("early", base=_EARLY_US))
    assert bundle.recording.session_start_time == _SESSION
    assert bundle_epoch_us(bundle) == _EARLY_US


def test_bundle_epoch_us_is_exact_for_a_reference_time_far_from_the_epoch(
    write_mixed,
):
    path = write_mixed(
        "far",
        recording=RecordingMeta(session_start_time=_FAR_START),
        base=_FAR_US,
    )
    assert bundle_epoch_us(open_bundle(path)) == _FAR_US


def test_bundle_epoch_us_is_0_without_a_reference_time(odd, write_mixed):
    assert bundle_epoch_us(odd) == 0
    bundle = _edit_recording(write_mixed("edited"), "reference_time")
    assert bundle.recording.reference_time is None
    assert bundle_epoch_us(bundle) == 0


def test_bundle_epoch_us_needs_no_session_start(write_mixed):
    subject_only = RecordingMeta(subject={"id": "sub-0931"})
    bundle = open_bundle(write_mixed("subject", recording=subject_only))
    assert bundle.recording.session_start_time is None
    assert bundle_epoch_us(bundle) == _SESSION_US
    assert recording_meta(bundle) == dataclasses.replace(
        subject_only, reference_time=bundle.recording.reference_time
    )


def test_recording_meta_keeps_the_session_start_outside_the_session_object(
    mixed,
):
    meta = recording_meta(mixed)
    assert meta == dataclasses.replace(_RECORDING, reference_time=_SESSION)
    assert "start_time" in mixed.recording.session
    assert meta.session_start_time.utcoffset() == _SESSION.utcoffset()
    assert all(
        type(value) is dict
        for value in (meta.subject, meta.session, meta.source)
    )


def test_recording_meta_shares_no_nested_object_with_the_bundle(write_mixed):
    nested = RecordingMeta(
        subject={"id": "sub-0931", "ages": {"years": [34]}},
        session={"tags": ["a"]},
        source={"tool": {"name": "acquire"}},
    )
    bundle = open_bundle(write_mixed("nested", recording=nested))
    meta = recording_meta(bundle)
    meta.subject["ages"]["years"].append(35)
    meta.session["tags"].append("b")
    meta.source["tool"]["name"] = "changed"
    assert bundle.recording.subject == nested.subject
    assert bundle.recording.session == nested.session
    assert bundle.recording.source == nested.source


def test_a_rewrite_of_a_bundle_without_a_reference_time_stays_undated(
    write_mixed, tmp_path
):
    bundle = _edit_recording(write_mixed("edited"), "reference_time")
    meta = recording_meta(bundle)
    assert meta.session_start_time == _SESSION
    assert meta.reference_time is None
    assert meta.wall_clock is False
    copy = _rewrite(bundle, tmp_path / "copy", recording=meta)
    assert copy.recording == bundle.recording
    _assert_same_channels(bundle, copy)


def test_a_rewrite_of_a_lone_reference_time_keeps_it_and_no_session_start(
    write_mixed, tmp_path
):
    lone = RecordingMeta(session={"id": "ses-01"})
    bundle = open_bundle(write_mixed("lone", recording=lone))
    assert bundle.recording.reference_time == _SESSION
    assert bundle.recording.session_start_time is None
    meta = recording_meta(bundle)
    assert meta == dataclasses.replace(
        lone, reference_time=bundle.recording.reference_time
    )
    copy = _rewrite(bundle, tmp_path / "copy", recording=meta)
    assert copy.recording == bundle.recording
    _assert_same_channels(bundle, copy)


def test_a_rewrite_keeps_a_session_start_later_than_the_reference_time(
    write_mixed, tmp_path
):
    bundle = open_bundle(write_mixed("early", base=_EARLY_US))
    meta = recording_meta(bundle)
    assert meta.session_start_time == _SESSION
    copy = _rewrite(bundle, tmp_path / "copy", recording=meta)
    assert copy.recording == bundle.recording
    assert copy.recording.reference_time != _SESSION
    _assert_same_channels(bundle, copy)


def test_a_session_start_before_the_reference_time_places_no_data(
    write_mixed, tmp_path
):
    earlier = _SESSION - dt.timedelta(microseconds=_SHIFT_US)
    bundle = _edit_recording(write_mixed("late"), "session.start_time", earlier)
    assert bundle.recording.reference_time == _SESSION
    assert bundle.recording.session_start_time == earlier
    copy = _rewrite(bundle, tmp_path / "copy", recording=recording_meta(bundle))
    assert copy.recording == bundle.recording
    _assert_same_channels(bundle, copy)


def test_a_rewrite_with_recording_meta_keeps_the_reference_time_and_every_offset(
    write_mixed, tmp_path
):
    bundle = open_bundle(write_mixed("early", base=_EARLY_US))
    reference = bundle.recording.reference_time
    assert bundle.recording.session_start_time > reference
    assert bundle.channels[1].offset_us == 500_000
    copy = _rewrite(
        bundle, tmp_path / "copy", recording=recording_meta(bundle), keys=[1]
    )
    assert copy.channels[0].offset_us == 500_000
    assert copy.recording == bundle.recording
    np.testing.assert_array_equal(
        copy.channels[0].raw[:], bundle.channels[1].raw[:]
    )


def test_a_rewrite_with_a_later_reference_time_gives_negative_offsets(
    write_mixed, tmp_path
):
    bundle = open_bundle(write_mixed("early", base=_EARLY_US))
    later = bundle.recording.reference_time + dt.timedelta(seconds=3)
    meta = dataclasses.replace(recording_meta(bundle), reference_time=later)
    copy = _rewrite(bundle, tmp_path / "copy", recording=meta)
    assert copy.recording.reference_time == later
    assert [c.offset_us for c in copy.continuous.values()] == [
        -3_000_000,
        -2_500_000,
    ]
    for key, channel in bundle.events.items():
        np.testing.assert_array_equal(
            copy.events[key].events[:], channel.events[:] - 3_000_000
        )
    _assert_same_continuous_data = np.testing.assert_array_equal
    _assert_same_continuous_data(
        copy.channels[0].raw[:], bundle.channels[0].raw[:]
    )


def test_recording_meta_is_none_without_recording_json(odd):
    assert recording_meta(odd) is None


def test_bundle_sources_lists_each_kind_in_key_order(mixed):
    continuous, events = bundle_sources(mixed, [4, 1, 0, 2, 1])
    assert [source.id for source in continuous] == ["LA1", "LA2"]
    assert [source.id for source in events] == ["units", "hfo"]


def test_bundle_sources_rejects_an_unknown_key(mixed):
    with pytest.raises(
        KeyError, match="the bundle has no channel 5; its channel keys are 0, "
    ):
        bundle_sources(mixed, [0, 5])


def test_bundle_sources_lists_no_keys_for_a_bundle_without_channels():
    root = _root()
    zarr.consolidate_metadata(root.store)
    with pytest.raises(KeyError) as raised:
        bundle_sources(open_bundle(root.store), [0])
    assert raised.value.args[0].endswith("its channel keys are none")


def test_bundle_sources_rejects_a_column_outside_event_column(mixed):
    with pytest.raises(
        ValueError, match="drop_columns must name only the columns .* 'values'"
    ):
        bundle_sources(mixed, drop_columns={"values"})


def test_bundle_sources_rejects_drop_columns_given_as_a_string(mixed):
    with pytest.raises(TypeError, match="drop_columns must be a collection"):
        bundle_sources(mixed, drop_columns="channels")


@pytest.mark.parametrize("epoch_us", [1.5, "0", None])
def test_a_source_rejects_an_epoch_us_that_is_not_an_integer(mixed, epoch_us):
    with pytest.raises(TypeError, match="epoch_us must be an integer, got"):
        BundleContinuousSource(mixed.channels[0], epoch_us=epoch_us)
    with pytest.raises(TypeError, match="epoch_us must be an integer, got"):
        BundleEventSource(mixed.channels[2], {}, epoch_us=epoch_us)


def test_bundle_sources_rejects_an_epoch_us_that_is_not_an_integer(mixed):
    with pytest.raises(TypeError, match="epoch_us must be an integer, got 1.5"):
        bundle_sources(mixed, epoch_us=1.5)
    with pytest.raises(TypeError, match="epoch_us must be an integer, got 1.5"):
        bundle_sources(mixed, [], epoch_us=1.5)
