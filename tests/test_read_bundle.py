import importlib.util
import json
import re
import shutil
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
import zarr
from zarr.buffer import default_buffer_prototype
from zarr.storage import MemoryStore, StorePath

from tests.conftest import (
    MEMORY_PREFIX,
    ODD_KEYS,
    ODD_VERSION,
    RUNS_TABLE,
    memory_store,
)
from tests.test_bundle import (
    _ACQUIRE,
    _RECORDING,
    _SESSION,
    _SESSION_US,
    _SORTER,
    _SPEC_OPTS,
)
from tests.test_zarr_io import _RecordingStore
from timeseries_zarr import read_event
from timeseries_zarr.attrs import root_attrs
from timeseries_zarr.bundle import write_bundle
from timeseries_zarr.read_bundle import Bundle, open_bundle, parse_bundle
from timeseries_zarr.read_continuous import ContinuousChannel
from timeseries_zarr.read_event import EventChannel
from timeseries_zarr.recording import RECORDING_FILE
from timeseries_zarr.types import RecordingMeta
from timeseries_zarr.zarr_io import (
    consolidate,
    descendants,
    open_group,
    open_root,
)

_MIXED_KINDS = {
    0: "continuous",
    1: "continuous",
    2: "event",
    3: "event",
    4: "event",
}
_NO_REFERENCE = (
    "recording.json holds no reference_time; datetimes are unavailable"
)
_EACH_LOCATION_KIND = pytest.mark.parametrize(
    "make_location",
    [
        lambda path, store: str(path),
        lambda path, store: path,
        lambda path, store: path.as_uri(),
        lambda path, store: store,
        lambda path, store: StorePath(store, MEMORY_PREFIX),
    ],
    ids=["str", "Path", "file URL", "MemoryStore", "StorePath"],
)
# A fixture that writes a bundle on first use warns from zarr's consolidate.
_WARNINGS_AS_ERRORS = pytest.mark.filterwarnings(
    "error", "ignore:Consolidated metadata is currently not part:UserWarning"
)


def _copy(path, *, edit_root=None, files=None):
    """Return a read-only MemoryStore copy of the bundle at path.

    edit_root changes the parsed root zarr.json in place. files maps a key
    to the bytes that replace its object, or to None to drop the object.
    """
    contents = {
        file.relative_to(path).as_posix(): file.read_bytes()
        for file in path.rglob("*")
        if file.is_file()
    }
    if edit_root is not None:
        document = json.loads(contents["zarr.json"])
        edit_root(document)
        contents["zarr.json"] = json.dumps(document).encode("utf-8")
    for key, data in (files or {}).items():
        if data is None:
            del contents[key]
        else:
            contents[key] = data
    buffer = default_buffer_prototype().buffer
    return MemoryStore(
        {key: buffer.from_bytes(data) for key, data in contents.items()},
        read_only=True,
    )


def _nodes(document):
    """Return the consolidated metadata of every node below the root."""
    return document["consolidated_metadata"]["metadata"]


def _set_channel_field(key, field, value):
    """Return a root edit that sets tsz.field of channel group key."""

    def edit(document):
        _nodes(document)[str(key)]["attributes"]["tsz"][field] = value

    return edit


def _set_version(version):
    def edit(document):
        document["attributes"]["tsz"]["version"] = version

    return edit


@pytest.fixture(scope="module")
def mixed(mixed_bundle):
    return open_bundle(mixed_bundle)


@pytest.fixture(scope="module")
def odd(odd_bundle):
    return open_bundle(odd_bundle)


@_EACH_LOCATION_KIND
def test_open_bundle_opens_each_location_kind(
    mixed_bundle, memory_bundle, make_location
):
    bundle = open_bundle(make_location(mixed_bundle, memory_bundle))
    assert isinstance(bundle, Bundle)
    assert bundle.version == "1.0"
    assert {key: channel.kind for key, channel in bundle.channels.items()} == (
        _MIXED_KINDS
    )
    assert bundle.recording.subject == _RECORDING.subject
    assert dict(bundle.recording.session) == {
        "start_time": _SESSION.isoformat(timespec="microseconds"),
        **_RECORDING.session,
    }
    assert bundle.recording.source == _RECORDING.source


def test_open_bundle_reads_recording_json_below_the_store_path_prefix(
    mixed_bundle,
):
    store = memory_store(mixed_bundle, (MEMORY_PREFIX,))
    bundle = open_bundle(StorePath(store, MEMORY_PREFIX))
    assert bundle.recording.subject == _RECORDING.subject
    assert bundle.location.endswith(f"/{MEMORY_PREFIX}")


def test_parse_bundle_reads_a_root_that_open_root_opened(mixed_bundle):
    bundle = parse_bundle(open_root(mixed_bundle))
    assert list(bundle.channels) == [0, 1, 2, 3, 4]


def test_open_bundle_raises_file_not_found_for_a_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_bundle(tmp_path / "absent")


@pytest.mark.skipif(
    importlib.util.find_spec("fsspec") is not None,
    reason="fsspec is installed",
)
def test_an_empty_memory_url_without_fsspec_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        open_bundle("memory://absent/bundle")


@_EACH_LOCATION_KIND
def test_open_bundle_rejects_storage_options_without_a_remote_url(
    mixed_bundle, memory_bundle, make_location
):
    location = make_location(mixed_bundle, memory_bundle)
    with pytest.raises(TypeError, match="storage_options"):
        open_bundle(location, storage_options={"anon": True})


def test_continuous_and_events_split_the_channels_by_kind(mixed):
    assert list(mixed.continuous) == [0, 1]
    assert list(mixed.events) == [2, 3, 4]
    assert all(
        isinstance(channel, ContinuousChannel)
        for channel in mixed.continuous.values()
    )
    assert all(
        isinstance(channel, EventChannel) for channel in mixed.events.values()
    )


def test_a_bundle_without_runs_reads_with_no_runs_on_any_channel(mixed, odd):
    for bundle in (mixed, odd):
        assert all(
            channel.runs_array is None and channel.runs is None
            for channel in bundle.continuous.values()
        )
        assert all(
            channel.read().runs is None
            for channel in bundle.continuous.values()
        )


def test_open_bundle_reads_the_runs_of_one_channel_and_none_for_the_other(
    runs_bundle,
):
    bundle = open_bundle(runs_bundle)
    assert bundle.continuous[0].runs.tolist() == [list(r) for r in RUNS_TABLE]
    assert bundle.continuous[1].runs is None


def test_open_bundle_rejects_a_runs_array_of_the_wrong_shape(
    runs_bundle,
):
    def narrow(document):
        document["consolidated_metadata"]["metadata"]["0/runs"]["shape"] = [
            2,
            2,
        ]

    with pytest.raises(
        ValueError,
        match=r"channel 0: runs must be an int64 array of shape \(runs, 3\)",
    ):
        open_bundle(_copy(runs_bundle, edit_root=narrow))


def test_channels_is_a_read_only_mapping_in_index_order(mixed):
    assert isinstance(mixed.channels, MappingProxyType)
    assert [channel.index for channel in mixed.channels.values()] == [
        0,
        1,
        2,
        3,
        4,
    ]
    with pytest.raises(TypeError):
        mixed.channels[9] = mixed.channels[0]


def test_bundle_fields_are_frozen(mixed):
    with pytest.raises(AttributeError):
        mixed.version = "1.1"


def test_bundles_compare_by_identity(mixed_bundle):
    first = open_bundle(mixed_bundle)
    second = open_bundle(mixed_bundle)
    assert first != second
    assert len({first, second}) == 2


def test_bundle_has_no_instance_dict(mixed):
    assert not hasattr(mixed, "__dict__")


def test_mixed_recording_holds_the_session_start_as_both_times(mixed):
    assert mixed.recording.reference_time == _SESSION
    assert mixed.recording.session_start_time == _SESSION


def test_the_recording_of_a_bundle_is_hashable(mixed, mixed_bundle):
    again = open_bundle(mixed_bundle).recording
    assert again == mixed.recording
    assert hash(again) == hash(mixed.recording)


def test_odd_bundle_opens_with_a_newer_minor_version_and_an_unknown_index(odd):
    assert odd.version == ODD_VERSION
    assert odd.recording is None


def test_odd_bundle_keeps_the_numbered_groups_and_ignores_other_nodes(
    odd, odd_bundle
):
    groups, arrays = descendants(open_root(odd_bundle))
    assert "index" in arrays
    assert "montages" in groups
    assert tuple(odd.channels) == ODD_KEYS
    assert list(odd.continuous) == [0, 2, 5, 7]
    assert list(odd.events) == [9, 12]


def test_open_bundle_ignores_a_group_whose_name_has_a_leading_zero(
    mixed_bundle,
):
    def add_group(document):
        _nodes(document)["01"] = _nodes(document)["0"]

    bundle = open_bundle(_copy(mixed_bundle, edit_root=add_group))
    assert list(bundle.channels) == [0, 1, 2, 3, 4]


def test_open_bundle_ignores_a_root_array_with_a_numbered_name(mixed_bundle):
    def add_array(document):
        _nodes(document)["8"] = _nodes(document)["0/raw"]

    bundle = open_bundle(_copy(mixed_bundle, edit_root=add_array))
    assert list(bundle.channels) == [0, 1, 2, 3, 4]


def test_channel_finds_a_channel_by_name(mixed):
    assert mixed.channel("curated marks") is mixed.channels[3]
    assert mixed.channel("LA2") is mixed.channels[1]


def test_channel_finds_a_channel_by_id_when_no_name_matches(mixed):
    assert mixed.channel("marks") is mixed.channels[3]
    assert mixed.channel("hfo") is mixed.channels[4]


def test_channel_prefers_a_name_match_to_an_id_match(mixed_bundle):
    store = _copy(
        mixed_bundle, edit_root=_set_channel_field(2, "name", "marks")
    )
    assert open_bundle(store).channel("marks").index == 2


def test_channel_rejects_a_name_that_several_channels_share(mixed_bundle):
    store = _copy(mixed_bundle, edit_root=_set_channel_field(1, "name", "LA1"))
    bundle = open_bundle(store)
    with pytest.raises(
        ValueError, match=r"^channels 0, 1 share the name 'LA1'; select one"
    ):
        bundle.channel("LA1")
    assert bundle.channel("LA2").index == 1


def test_channel_rejects_an_id_that_several_channels_share(mixed_bundle):
    store = _copy(mixed_bundle, edit_root=_set_channel_field(4, "id", "marks"))
    with pytest.raises(
        ValueError, match=r"^channels 3, 4 share the id 'marks'; select one"
    ):
        open_bundle(store).channel("marks")


def test_channel_raises_index_error_for_an_unknown_name_or_id(mixed):
    message = (
        "no channel has the name or id 'LA9'; describe() lists the name and "
        "id of each channel"
    )
    with pytest.raises(KeyError, match=re.escape(message)) as caught:
        mixed.channel("LA9")
    assert caught.value.args == (message,)


def test_describe_lists_the_mixed_bundle_one_channel_a_line(mixed):
    assert mixed.describe().splitlines() == [
        "index  kind        name           id     unit  rate_hz  length"
        "         levels",
        "0      continuous  LA1            LA1    uV    256.0    8192 samples"
        "   1-2",
        "1      continuous  LA2            LA2    uV    512.0    16384 samples"
        "  1-3",
        "2      event       units          units  -     -        400 events"
        "     -",
        "3      event       curated marks  marks  -     -        6 events"
        "       -",
        "4      event       hfo detector   hfo    -     -        70000 events"
        "   1-4",
    ]


def test_describe_shows_a_hyphen_for_each_absent_value(odd):
    assert odd.describe().splitlines() == [
        "index  kind        name       id     unit  rate_hz  length        levels",
        "0      continuous  short      -      uV    512.0    1001 samples  1-2",
        "2      continuous  full       two    uV    512.0    3000 samples  1",
        "5      continuous  late       five   uV    512.0    1024 samples  -",
        "7      continuous  empty      seven  uV    30000.0  0 samples     -",
        "9      event       marks      -      a.u.  -        64 events     1",
        "12     event       intervals  -      -     -        5 events      -",
    ]


def test_describe_names_one_sample_and_one_event_in_the_singular(
    tmp_path, continuous_source, event_source
):
    path = tmp_path / "single"
    write_bundle(
        [continuous_source([1.0], id="one", name="one", rate_hz=512.0)],
        [event_source([0], id="tick", name="tick")],
        output_path=path,
        opts=_SPEC_OPTS,
    )
    lines = open_bundle(path).describe().splitlines()
    assert lines[1:] == [
        "0      continuous  one   one   uV    512.0    1 sample  -",
        "1      event       tick  tick  -     -        1 event   -",
    ]


def test_describe_of_a_bundle_without_channels_is_the_header(tmp_path):
    path = tmp_path / "empty"
    consolidate(open_group(path, root_attrs()))
    bundle = open_bundle(path)
    assert bundle.channels == {}
    assert bundle.describe() == (
        "index  kind  name  id  unit  rate_hz  length  levels"
    )


def test_repr_names_the_location_version_and_channel_counts(
    mixed, mixed_bundle
):
    assert repr(mixed) == (
        f"Bundle({mixed_bundle.as_uri()!r}, version='1.0', continuous=2, "
        "events=3)"
    )


def test_location_of_a_memory_store_is_a_memory_url(memory_bundle):
    bundle = open_bundle(StorePath(memory_bundle, MEMORY_PREFIX))
    assert bundle.location.startswith("memory://")
    assert bundle.location.endswith(f"/{MEMORY_PREFIX}")
    assert repr(bundle).startswith("Bundle('memory://")


def test_location_of_a_local_bundle_is_its_file_url(mixed, mixed_bundle):
    assert mixed.location == mixed_bundle.as_uri()


@pytest.mark.parametrize(
    "make_location",
    [lambda name: name, Path, lambda name: f"file:{name}"],
    ids=["str", "Path", "file URL"],
)
def test_open_bundle_resolves_a_relative_path_at_the_open(
    mixed, mixed_bundle, monkeypatch, make_location
):
    expected = mixed.continuous[0].raw[:]
    monkeypatch.chdir(mixed_bundle.parent)
    bundle = open_bundle(make_location(mixed_bundle.name))
    monkeypatch.chdir(mixed_bundle.parent.parent)
    assert np.isfinite(expected).any()
    np.testing.assert_array_equal(bundle.continuous[0].raw[:], expected)
    assert bundle.location == mixed_bundle.as_uri()


@_WARNINGS_AS_ERRORS
def test_datetimes_counts_from_the_reference_time_in_utc(mixed):
    times = mixed.datetimes([0, 1_000_000])
    assert times.dtype == np.dtype("datetime64[us]")
    assert times.tolist() == [
        np.datetime64("2024-03-14T14:15:00", "us").item(),
        np.datetime64("2024-03-14T14:15:01", "us").item(),
    ]
    assert mixed.datetimes([0])[0] == np.datetime64("2024-03-14T14:15:00")


@_WARNINGS_AS_ERRORS
def test_datetimes_rounds_a_fraction_to_the_nearest_microsecond(mixed):
    times = mixed.datetimes(np.array([1.4, 1.6, -0.6, 0.5, 2.5]))
    assert times.tolist() == [
        np.datetime64("2024-03-14T14:15:00.000001").item(),
        np.datetime64("2024-03-14T14:15:00.000002").item(),
        np.datetime64("2024-03-14T14:14:59.999999").item(),
        np.datetime64("2024-03-14T14:15:00.000000").item(),
        np.datetime64("2024-03-14T14:15:00.000002").item(),
    ]


@_WARNINGS_AS_ERRORS
@pytest.mark.parametrize(
    "time_us",
    [
        np.array([2**53 + 1], dtype=np.int64),
        np.array([2**62 - 1], dtype=np.int64),
        np.array([-(2**62) + 1], dtype=np.int64),
        np.array([2**62 - 1], dtype=np.uint64),
        [2**62 - 1],
    ],
    ids=["2^53 + 1", "2^62 - 1", "-(2^62) + 1", "uint64", "Python int"],
)
def test_datetimes_keeps_every_microsecond_of_an_integer_time(mixed, time_us):
    start = np.datetime64("2024-03-14T14:15:00", "us")
    offset = int(np.asarray(time_us)[0])
    assert mixed.datetimes(time_us)[0] == start + np.timedelta64(offset, "us")


@_WARNINGS_AS_ERRORS
def test_datetimes_keeps_the_shape_and_gives_nat_for_nan(mixed):
    scalar = mixed.datetimes(0)
    assert scalar.shape == ()
    assert scalar == np.datetime64("2024-03-14T14:15:00")
    assert np.isnat(mixed.datetimes([np.nan])).tolist() == [True]
    assert mixed.datetimes(np.zeros((2, 3))).shape == (2, 3)


_OUTSIDE_THE_DATETIME_RANGE = (
    r"^time_us must hold NaN or times below 2\^62 microseconds in magnitude, "
    "got "
)


@_WARNINGS_AS_ERRORS
@pytest.mark.parametrize(
    ("time_us", "shown"),
    [
        (np.inf, "inf"),
        (-np.inf, "-inf"),
        (1e30, "1e+30"),
        (2.0**62, "4.611686018427388e+18"),
        (-(2.0**62), "-4.611686018427388e+18"),
    ],
)
def test_datetimes_rejects_a_time_outside_the_datetime_range(
    mixed, time_us, shown
):
    with pytest.raises(
        ValueError, match=f"{_OUTSIDE_THE_DATETIME_RANGE}{re.escape(shown)}$"
    ):
        mixed.datetimes([0.0, np.nan, time_us])


@_WARNINGS_AS_ERRORS
@pytest.mark.parametrize(
    ("time_us", "shown"),
    [
        (np.array([0, 2**62], dtype=np.int64), "4611686018427387904"),
        (np.array([0, -(2**63)], dtype=np.int64), "-9223372036854775808"),
        (np.array([0, 2**63], dtype=np.uint64), "9223372036854775808"),
        ([0, 10**400], "a number beyond the float64 range"),
    ],
    ids=["2^62", "int64 minimum", "uint64 2^63", "beyond float64"],
)
def test_datetimes_rejects_an_integer_time_outside_the_datetime_range(
    mixed, time_us, shown
):
    with pytest.raises(
        ValueError, match=f"{_OUTSIDE_THE_DATETIME_RANGE}{re.escape(shown)}$"
    ):
        mixed.datetimes(time_us)


@_WARNINGS_AS_ERRORS
def test_datetimes_accepts_the_largest_time_below_2_to_the_62(mixed):
    below = np.nextafter(2.0**62, 0)
    assert mixed.datetimes([below])[0] == np.datetime64(
        "2024-03-14T14:15:00", "us"
    ) + np.timedelta64(int(below), "us")


@pytest.mark.parametrize(
    "start", ["0001-01-01T00:00:00+01:00", "9999-12-31T23:59:59-01:00"]
)
def test_datetimes_rejects_a_reference_time_outside_the_datetime_range(
    mixed_bundle, start
):
    document = {"reference_time": start}
    store = _copy(
        mixed_bundle, files={RECORDING_FILE: json.dumps(document).encode()}
    )
    bundle = open_bundle(store)
    with pytest.raises(
        ValueError,
        match=(
            r"^recording\.json: reference_time must fall within "
            rf"years 1 through 9999 in UTC, got '{re.escape(start)}'$"
        ),
    ):
        bundle.datetimes([0])


def test_datetimes_counts_from_the_reference_time_when_it_precedes_the_session_start(
    write_mixed,
):
    bundle = open_bundle(write_mixed(base=_SESSION_US - 2_000_000))
    assert (
        bundle.recording.session_start_time != bundle.recording.reference_time
    )
    assert bundle.datetimes([0])[0] == np.datetime64("2024-03-14T14:14:58")


def test_datetimes_returns_the_event_times_as_utc_datetimes(mixed):
    table = mixed.channel("curated marks").read()
    assert mixed.datetimes(table.time_us[:2]).tolist() == [
        np.datetime64("2024-03-14T14:15:01", "us").item(),
        np.datetime64("2024-03-14T14:15:04", "us").item(),
    ]


def test_datetimes_rejects_a_bundle_without_a_reference_time(odd_bundle):
    bundle = open_bundle(odd_bundle)
    with pytest.raises(ValueError, match=f"^{_NO_REFERENCE}$"):
        bundle.datetimes([0])


def test_datetimes_rejects_a_recording_without_wall_clock_times(write_mixed):
    undated = RecordingMeta(subject={"id": "sub-0931"}, wall_clock=False)
    bundle = open_bundle(write_mixed(recording=undated))
    assert bundle.recording.subject == {"id": "sub-0931"}
    assert bundle.recording.reference_time is None
    with pytest.raises(ValueError, match=f"^{_NO_REFERENCE}$"):
        bundle.datetimes([0])


def test_datetimes_rejects_a_recording_with_only_a_session_start(
    mixed_bundle,
):
    document = {"session": {"start_time": "2024-03-14T09:15:00-05:00"}}
    store = _copy(
        mixed_bundle, files={RECORDING_FILE: json.dumps(document).encode()}
    )
    bundle = open_bundle(store)
    assert bundle.recording.session_start_time == _SESSION
    with pytest.raises(ValueError, match=f"^{_NO_REFERENCE}$"):
        bundle.datetimes([0])


def test_the_context_manager_closes_the_store(mixed_bundle):
    with open_bundle(mixed_bundle) as bundle:
        assert bundle.root.store._is_open
    assert not bundle.root.store._is_open


def test_the_context_manager_closes_the_store_and_reraises_an_error(
    mixed_bundle,
):
    with (
        pytest.raises(RuntimeError, match="body"),
        open_bundle(mixed_bundle) as bundle,
    ):
        raise RuntimeError("body")
    assert not bundle.root.store._is_open


def test_close_closes_the_store_and_a_second_call_succeeds(mixed_bundle):
    bundle = open_bundle(mixed_bundle)
    bundle.close()
    bundle.close()
    assert not bundle.root.store._is_open


def test_open_bundle_rejects_a_root_without_consolidated_metadata(
    mixed_bundle,
):
    def drop(document):
        del document["consolidated_metadata"]

    with pytest.raises(ValueError, match="has no consolidated metadata"):
        open_bundle(_copy(mixed_bundle, edit_root=drop))


def test_open_bundle_rejects_major_version_2(mixed_bundle):
    store = _copy(mixed_bundle, edit_root=_set_version("2.0"))
    with pytest.raises(
        ValueError,
        match=(
            r"^bundle version '2\.0' has major version 2; this package reads "
            r"major version 1$"
        ),
    ):
        open_bundle(store)


def test_open_bundle_accepts_a_newer_minor_version(mixed_bundle):
    store = _copy(mixed_bundle, edit_root=_set_version("1.7"))
    assert open_bundle(store).version == "1.7"


def test_open_bundle_warns_and_skips_a_channel_of_unknown_kind(mixed_bundle):
    store = _copy(
        mixed_bundle, edit_root=_set_channel_field(2, "kind", "video")
    )
    with pytest.warns(UserWarning) as caught:
        bundle = open_bundle(store)
    assert [str(warning.message) for warning in caught] == [
        "skipped channel 2: tsz.kind must be 'continuous' or 'event', "
        "got 'video'"
    ]
    assert caught[0].filename == __file__
    assert list(bundle.channels) == [0, 1, 3, 4]


def test_open_bundle_rejects_a_channel_group_without_a_kind(mixed_bundle):
    def drop_kind(document):
        del _nodes(document)["2"]["attributes"]["tsz"]["kind"]

    with pytest.raises(ValueError, match=r"^channel 2: tsz\.kind is missing"):
        open_bundle(_copy(mixed_bundle, edit_root=drop_kind))


def test_open_bundle_rejects_a_channel_that_does_not_parse(mixed_bundle):
    store = _copy(mixed_bundle, edit_root=_set_channel_field(0, "rate_hz", 0))
    with pytest.raises(ValueError, match=r"^channel 0: tsz\.rate_hz must be"):
        open_bundle(store)


def test_open_bundle_rejects_a_recording_json_that_does_not_parse(
    mixed_bundle,
):
    store = _copy(mixed_bundle, files={RECORDING_FILE: b"[]"})
    with pytest.raises(
        ValueError, match="recording.json must hold a JSON object"
    ):
        open_bundle(store)


def test_open_bundle_rejects_a_recording_json_that_is_not_utf_8(mixed_bundle):
    store = _copy(mixed_bundle, files={RECORDING_FILE: b"\xff\xfe{}"})
    with pytest.raises(
        ValueError, match=r"^recording\.json is not valid UTF-8 at byte 0$"
    ):
        open_bundle(store)


def test_open_bundle_without_recording_json_has_no_recording(mixed_bundle):
    store = _copy(mixed_bundle, files={RECORDING_FILE: None})
    assert open_bundle(store).recording is None


def test_channel_provenance_is_read_only(mixed_bundle):
    with open_bundle(mixed_bundle) as bundle:
        for key, provenance in ((0, _ACQUIRE), (2, _SORTER)):
            channel = bundle.channels[key]
            assert channel.provenance == provenance
            with pytest.raises(TypeError):
                channel.provenance["x"] = 1


def test_a_change_to_nested_provenance_leaves_the_root_attributes_unchanged(
    mixed_bundle,
):
    with open_bundle(mixed_bundle) as bundle:
        bundle.channels[0].provenance["settings"]["gain"] = 8
        stored = dict(bundle.root["0"].attrs)["tsz"]["provenance"]
        assert stored == _ACQUIRE
        assert parse_bundle(bundle.root).channels[0].provenance == _ACQUIRE


def _touch_every_channel(bundle):
    """Read every field that the metadata gives, on every channel."""
    summary = [bundle.describe(), repr(bundle)]
    for channel in bundle.continuous.values():
        summary.append(
            (channel.num_samples, channel.end_us, channel.provenance)
        )
        for level in channel.levels.values():
            summary.append((level.bins, level.period_us, level.mean))
    for channel in bundle.events.values():
        summary.append((channel.num_events, channel.columns))
        for level in channel.count_levels.values():
            summary.append((level.bins, level.num_labels))
        if channel.waveform_mean is not None:
            summary.append(channel.waveform_mean.time_us)
    return summary


@pytest.mark.parametrize("prefix", ["", MEMORY_PREFIX])
def test_open_bundle_reads_only_the_root_zarr_json_and_recording_json(
    memory_bundle, prefix
):
    store = _RecordingStore(memory_bundle)
    bundle = open_bundle(StorePath(store, prefix) if prefix else store)
    _touch_every_channel(bundle)
    lead = f"{prefix}/" if prefix else ""
    assert store.reads == [f"{lead}zarr.json", f"{lead}{RECORDING_FILE}"]


def test_open_bundle_without_recording_json_reads_only_the_two_root_indices(
    odd_bundle,
):
    store = _RecordingStore(memory_store(odd_bundle))
    bundle = open_bundle(store)
    _touch_every_channel(bundle)
    assert bundle.recording is None
    assert store.reads == ["zarr.json", RECORDING_FILE]


def test_writing_through_a_raw_array_raises(mixed):
    with pytest.raises(ValueError, match="read-only"):
        mixed.continuous[0].raw[0:1] = np.zeros(1, dtype=np.float32)


_SMALL_CHUNK_ROWS = 20
_LARGE_CHUNK_ROWS = 1000
_LARGE_ARRAY_ROWS = 20_000
_STATISTICS = ("min", "max", "mean", "var", "count")


def _relayout(source, destination):
    """Copy the bundle at source to destination without sharding or compression.

    An array of up to _LARGE_ARRAY_ROWS rows takes chunks of _SMALL_CHUNK_ROWS
    rows, and a longer array chunks of _LARGE_CHUNK_ROWS rows. A chunk that
    holds only the fill value is left out.
    """
    original = zarr.open_group(str(source), mode="r")
    copy = zarr.open_group(
        str(destination),
        mode="w",
        zarr_format=3,
        attributes=dict(original.attrs),
    )
    members = sorted(
        original.members(max_depth=None),
        key=lambda member: member[0].count("/"),
    )
    for path, node in members:
        if isinstance(node, zarr.Group):
            copy.create_group(path, attributes=dict(node.attrs))
            continue
        rows = (
            _SMALL_CHUNK_ROWS
            if node.shape[0] <= _LARGE_ARRAY_ROWS
            else _LARGE_CHUNK_ROWS
        )
        array = copy.create_array(
            path,
            shape=node.shape,
            dtype=node.dtype,
            chunks=(rows, *(max(1, size) for size in node.shape[1:])),
            compressors=None,
            fill_value=node.fill_value,
            attributes=dict(node.attrs),
            dimension_names=node.metadata.dimension_names,
            config={"write_empty_chunks": False},
        )
        if node.size:
            array[...] = node[...]
    zarr.consolidate_metadata(str(destination))
    recording = source / RECORDING_FILE
    if recording.exists():
        shutil.copy(recording, destination / RECORDING_FILE)
    return destination


def _middle_third(start_us, end_us):
    third = (end_us - start_us) // 3
    return start_us + third, start_us + 2 * third


def _reads(bundle):
    """Return what reads of every channel, level, and column of bundle give.

    Each channel is read whole and over the middle third of its span.
    """
    values = {"describe": bundle.describe(), "recording": bundle.recording}
    for key, channel in bundle.continuous.items():
        values[key, "spans"] = channel.spans()
        middle = _middle_third(channel.start_us, channel.end_us)
        for window in [(None, None), middle]:
            values[key, window, "raw"] = channel.read(*window).values
            for number in channel.levels:
                bins = channel.read(*window, level=number)
                for member in _STATISTICS:
                    values[key, window, number, member] = getattr(bins, member)
    for key, channel in bundle.events.items():
        times = np.asarray(channel.events[:])
        windows = [(None, None)]
        if times.size:
            windows.append(_middle_third(int(times[0]), int(times[-1]) + 1))
        if channel.waveform_mean is not None:
            values[key, "waveform_mean"] = channel.waveform_mean.read().values
        for window in windows:
            table = channel.read(*window)
            for column, data in table.columns().items():
                values[key, window, column] = data
            if table.waveform is not None:
                values[key, window, "waveform"] = table.waveform.values
            for number in channel.count_levels:
                counts = channel.read_counts(*window, level=number)
                values[key, window, "counts", number] = counts.count
    return values


def _assert_same_reads(expected, found):
    assert found.keys() == expected.keys()
    for name, value in expected.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(found[name], value, err_msg=str(name))
        else:
            assert found[name] == value, name


@pytest.mark.parametrize("fixture", ["mixed_bundle", "odd_bundle"])
@pytest.mark.parametrize(
    "search_limit",
    [None, 0],
    ids=["whole events search", "bisect events search"],
)
def test_reads_match_across_chunk_layouts_and_missing_fill_chunks(
    request, tmp_path, monkeypatch, fixture, search_limit
):
    if search_limit is not None:
        monkeypatch.setattr(
            read_event, "EVENTS_SEARCH_LIMIT_BYTES", search_limit
        )
    source = request.getfixturevalue(fixture)
    copy = _relayout(source, tmp_path / "relaid")
    assert zarr.open_group(str(copy), mode="r")["0/raw"].shards is None
    raw = zarr.open_group(str(source), mode="r")["0/raw"][:]
    fill_chunks = [
        start // _SMALL_CHUNK_ROWS
        for start in range(0, raw.size, _SMALL_CHUNK_ROWS)
        if np.isnan(raw[start : start + _SMALL_CHUNK_ROWS]).all()
    ]
    assert fill_chunks
    for index in fill_chunks:
        assert not (copy / "0" / "raw" / "c" / str(index)).exists()
    _assert_same_reads(_reads(open_bundle(source)), _reads(open_bundle(copy)))


def test_reads_fill_a_missing_shard_with_the_fill_value(mixed_bundle):
    original = open_bundle(mixed_bundle)
    store = memory_store(mixed_bundle, omit=("0/raw/c/1", "4/1/count/c/0/0"))
    bundle = open_bundle(store)
    rows = original.continuous[0].raw.shards[0]
    expected = original.continuous[0].read().values.copy()
    expected[rows : 2 * rows] = np.nan
    np.testing.assert_array_equal(bundle.continuous[0].read().values, expected)
    for member in _STATISTICS:
        np.testing.assert_array_equal(
            getattr(bundle.continuous[0].read(level=1), member),
            getattr(original.continuous[0].read(level=1), member),
        )
    count_rows = original.events[4].count_levels[1].count.shards[0]
    counts = original.events[4].read_counts(level=1).count.copy()
    counts[:count_rows] = 0
    np.testing.assert_array_equal(
        bundle.events[4].read_counts(level=1).count, counts
    )


def _event_bundle_with_channels(channel_count, width):
    """Return an in-memory bundle of channel_count event channels.

    Channel 0 holds a channels column of the given width.
    """
    root = zarr.open_group(
        MemoryStore(),
        mode="w",
        zarr_format=3,
        attributes={"tsz": {"version": "1.0"}},
    )
    for key in range(channel_count):
        channel = root.create_group(
            str(key), attributes={"tsz": {"kind": "event", "name": f"e{key}"}}
        )
        channel.create_array("events", data=np.array([1, 2], dtype=np.int64))
        if key == 0:
            channel.create_array(
                "channels", data=np.zeros((2, width), dtype=np.uint8)
            )
    zarr.consolidate_metadata(root.store)
    return root.store


@pytest.mark.parametrize(("channel_count", "width"), [(8, 1), (9, 2)])
def test_open_bundle_accepts_a_channels_width_of_ceil_channel_count_over_8(
    channel_count, width
):
    store = _event_bundle_with_channels(channel_count, width)
    assert open_bundle(store).events[0].channels.shape == (2, width)


@pytest.mark.parametrize(("channel_count", "width"), [(8, 2), (9, 1)])
def test_open_bundle_rejects_a_channels_width_other_than_ceil_channel_count_over_8(
    channel_count, width
):
    store = _event_bundle_with_channels(channel_count, width)
    with pytest.raises(ValueError, match="channel 0: channels must have"):
        open_bundle(store)
