import dataclasses
import datetime as dt
import json

import numpy as np
import pytest
import zarr
from zarr.buffer import default_buffer_prototype
from zarr.core.sync import sync
from zarr.storage import MemoryStore, StorePath

from tests import conftest
from tests.conftest import (
    MEMORY_PREFIX,
    ODD_COUNT_PERIOD_US,
    ODD_EVENTS,
    ODD_FULL_TEXT,
    ODD_FULL_TEXT_ROW,
    ODD_INTERVAL_DURATIONS,
    ODD_INTERVAL_EVENTS,
    ODD_KEYS,
    ODD_LABELS,
    ODD_NAN_RUN,
    ODD_TEXT_WIDTH,
    ODD_TIE,
    ODD_WAVEFORM_PERIOD_US,
    SYNTHETIC_BUNDLE_ENV,
    find_synthetic_bundle,
    memory_store,
)
from tests.test_bundle import (
    _MIXED_PROVENANCE,
    _RECORDING,
    _SESSION,
    _SESSION_US,
    _SPEC_OPTS,
)
from timeseries_zarr.recording import RECORDING_FILE


def _open(location):
    return zarr.open_group(location, mode="r", use_consolidated=True)


def _tsz(node):
    return dict(node.attrs)["tsz"]


def _arrays(root):
    return {
        path: node
        for path, node in root.members(max_depth=None)
        if isinstance(node, zarr.Array)
    }


def _bins(values, width):
    return [
        values[start : start + width].astype(np.float64)
        for start in range(0, values.shape[0], width)
    ]


async def _store_keys(store):
    return sorted([key async for key in store.list()])


def test_mixed_bundle_holds_two_continuous_and_three_event_channels(
    mixed_bundle,
):
    root = _open(mixed_bundle)
    kinds = {key: _tsz(root[key])["kind"] for key in root.group_keys()}
    assert kinds == {
        "0": "continuous",
        "1": "continuous",
        "2": "event",
        "3": "event",
        "4": "event",
    }
    assert root["0"]["raw"].shape == (8192,)
    assert root["1"]["raw"].shape == (16384,)
    assert [root[key]["events"].shape for key in ("2", "3", "4")] == [
        (400,),
        (6,),
        (70_000,),
    ]
    assert root["0"]["raw"].dtype == np.float32
    assert root["2"]["events"].dtype == np.int64
    assert sorted(root["0"].group_keys()) == ["1", "2"]
    assert sorted(root["1"].group_keys()) == ["1", "2", "3"]
    assert root["0"]["raw"].chunks == (256,)


def test_mixed_bundle_recording_json_holds_both_times_and_the_identity(
    mixed_bundle,
):
    document = json.loads((mixed_bundle / RECORDING_FILE).read_text("utf-8"))
    assert dt.datetime.fromisoformat(document["start_time"]) == _SESSION
    session_start = document["session"]["start_time"]
    assert dt.datetime.fromisoformat(session_start) == _SESSION
    assert "start" not in document
    assert document["subject"] == _RECORDING.subject


def test_mixed_bundle_carries_the_provenance_of_three_channels(mixed_bundle):
    root = _open(mixed_bundle)
    provenance = {
        key: _tsz(root[key]).get("provenance") for key in root.group_keys()
    }
    assert provenance == _MIXED_PROVENANCE


def test_write_mixed_writes_each_variant_under_its_name(write_mixed, tmp_path):
    first = write_mixed()
    second = write_mixed("plain", recording=None, with_provenance=False)
    assert first == tmp_path / "bundle"
    assert second == tmp_path / "plain"
    assert (first / RECORDING_FILE).is_file()
    assert not (second / RECORDING_FILE).exists()
    root = _open(second)
    assert all("provenance" not in _tsz(root[key]) for key in root.group_keys())


def test_write_mixed_passes_the_base_and_the_options_to_the_writer(
    write_mixed,
):
    output = write_mixed(
        base=_SESSION_US - 2_000_000,
        opts=dataclasses.replace(_SPEC_OPTS, max_levels=0),
    )
    document = json.loads((output / RECORDING_FILE).read_text("utf-8"))
    start = dt.datetime.fromisoformat(document["start_time"])
    assert start == _SESSION - dt.timedelta(seconds=2)
    assert list(_open(output)["0"].group_keys()) == []


def test_memory_bundle_is_a_read_only_memory_store(memory_bundle):
    assert isinstance(memory_bundle, MemoryStore)
    assert memory_bundle.read_only


def test_memory_bundle_holds_every_file_at_the_root_and_under_the_prefix(
    memory_bundle, mixed_bundle
):
    names = sorted(
        file.relative_to(mixed_bundle).as_posix()
        for file in mixed_bundle.rglob("*")
        if file.is_file()
    )
    expected = sorted(names + [f"{MEMORY_PREFIX}/{name}" for name in names])
    assert sync(_store_keys(memory_bundle)) == expected
    stored = (mixed_bundle / RECORDING_FILE).read_bytes()
    for key in (RECORDING_FILE, f"{MEMORY_PREFIX}/{RECORDING_FILE}"):
        buffer = sync(
            memory_bundle.get(key, prototype=default_buffer_prototype())
        )
        assert buffer.to_bytes() == stored


@pytest.mark.parametrize(
    "locate",
    [lambda store: store, lambda store: StorePath(store, MEMORY_PREFIX)],
    ids=["root", "prefix"],
)
def test_memory_bundle_opens_with_plain_zarr(
    memory_bundle, mixed_bundle, locate
):
    root = _open(locate(memory_bundle))
    on_disk = _open(mixed_bundle)
    assert root.metadata.consolidated_metadata is not None
    assert sorted(root.group_keys()) == sorted(on_disk.group_keys())
    assert np.array_equal(
        root["1"]["raw"][:], on_disk["1"]["raw"][:], equal_nan=True
    )


def test_memory_store_places_files_at_the_root_by_default(mixed_bundle):
    store = memory_store(mixed_bundle)
    assert sync(_store_keys(store)) == sorted(
        file.relative_to(mixed_bundle).as_posix()
        for file in mixed_bundle.rglob("*")
        if file.is_file()
    )


_ODD_TREE = {
    "0": None,
    "0/raw": ((1001,), np.float32),
    "0/1": None,
    "0/1/minmax": ((251, 2), np.float32),
    "0/1/median": ((251,), np.float32),
    "0/2": None,
    "0/2/minmax": ((63, 2), np.float32),
    "0/2/count": ((63,), np.uint32),
    "2": None,
    "2/raw": ((3000,), np.float32),
    "2/1": None,
    "2/1/minmax": ((750, 2), np.float32),
    "2/1/mean": ((750,), np.float32),
    "2/1/var": ((750,), np.float32),
    "2/1/count": ((750,), np.uint32),
    "5": None,
    "5/raw": ((1024,), np.float32),
    "7": None,
    "7/raw": ((0,), np.float32),
    "9": None,
    "9/events": ((64,), np.int64),
    "9/labels": ((64,), np.uint16),
    "9/values": ((64,), np.float32),
    "9/text": ((64, 5), np.uint8),
    "9/channels": ((64, 2), np.uint8),
    "9/waveforms": ((64, 8), np.float32),
    "9/waveform_mean": ((3, 8), np.float32),
    "9/1": None,
    "9/1/count": ((17, 3), np.uint32),
    "9/notes": None,
    "12": None,
    "12/events": ((5,), np.int64),
    "12/durations": ((5,), np.int64),
    "index": ((6,), np.int64),
    "montages": None,
}


def test_odd_bundle_holds_the_listed_node_tree(odd_bundle):
    root = _open(odd_bundle)
    paths = sorted(path for path, _ in root.members(max_depth=None))
    assert paths == sorted(_ODD_TREE)
    assert sorted(int(key) for key in root.group_keys() if key.isdecimal()) == (
        list(ODD_KEYS)
    )


def test_odd_bundle_arrays_have_the_listed_shapes_and_dtypes(odd_bundle):
    arrays = _arrays(_open(odd_bundle))
    shapes = {
        path: (array.shape, array.dtype) for path, array in arrays.items()
    }
    assert shapes == {
        path: (shape, np.dtype(dtype))
        for path, entry in _ODD_TREE.items()
        if entry is not None
        for shape, dtype in [entry]
    }


def test_odd_bundle_arrays_chunk_and_shard_only_the_first_axis(odd_bundle):
    for path, array in _arrays(_open(odd_bundle)).items():
        trailing = array.shape[1:]
        assert array.chunks == (256, *trailing), path
        assert array.shards == (1024, *trailing), path


def test_odd_bundle_fills_float_arrays_with_nan_and_leaves_text_uncompressed(
    odd_bundle,
):
    for path, array in _arrays(_open(odd_bundle)).items():
        if np.issubdtype(array.dtype, np.floating):
            assert np.isnan(array.fill_value), path
        else:
            assert array.fill_value == 0, path
        assert (array.compressors == ()) == (path == "9/text"), path


def test_odd_bundle_names_the_axes_of_every_channel_array(odd_bundle):
    arrays = _arrays(_open(odd_bundle))
    names = {
        path: array.metadata.dimension_names for path, array in arrays.items()
    }
    assert names.pop("index") is None
    assert names == {
        "0/raw": ("time",),
        "0/1/minmax": ("time", None),
        "0/1/median": ("time",),
        "0/2/minmax": ("time", None),
        "0/2/count": ("time",),
        "2/raw": ("time",),
        "2/1/minmax": ("time", None),
        "2/1/mean": ("time",),
        "2/1/var": ("time",),
        "2/1/count": ("time",),
        "5/raw": ("time",),
        "7/raw": ("time",),
        "9/events": ("event",),
        "9/labels": ("event",),
        "9/values": ("event",),
        "9/text": ("event", None),
        "9/channels": ("event", None),
        "9/waveforms": ("event", None),
        "9/waveform_mean": ("label", None),
        "9/1/count": ("time", None),
        "12/events": ("event",),
        "12/durations": ("event",),
    }


def test_odd_bundle_root_holds_version_1_3_an_unknown_key_and_no_recording(
    odd_bundle,
):
    root = _open(odd_bundle)
    assert _tsz(root) == {"version": "1.3", "future_key": "not in 1.0"}
    assert not (odd_bundle / RECORDING_FILE).exists()


def test_odd_bundle_montages_group_carries_attributes_beside_tsz_only(
    odd_bundle,
):
    assert dict(_open(odd_bundle)["montages"].attrs) == {
        "montage": {"pairs": [[0, 2]]}
    }


@pytest.mark.parametrize(
    ("key", "present", "absent"),
    [
        ("0", {"kind", "name", "rate_hz", "unit"}, {"id", "offset_us"}),
        ("2", {"id", "offset_us"}, set()),
        ("5", {"id", "offset_us"}, set()),
    ],
)
def test_odd_bundle_channel_attributes_hold_and_omit_the_listed_keys(
    odd_bundle, key, present, absent
):
    fields = set(_tsz(_open(odd_bundle)[key]))
    assert present <= fields
    assert not absent & fields


def test_odd_bundle_continuous_channels_hold_the_listed_attributes(
    odd_bundle,
):
    root = _open(odd_bundle)
    assert _tsz(root["2"])["id"] == "two"
    assert _tsz(root["2"])["offset_us"] == 0
    assert _tsz(root["5"])["offset_us"] == 2_000_000
    assert _tsz(root["7"])["rate_hz"] == 30_000.0
    assert [_tsz(root[key])["rate_hz"] for key in ("0", "2", "5")] == [
        512.0
    ] * 3
    assert _tsz(root["0"]["1"]) == _tsz(root["2"]["1"]) == {"period_us": 7812.5}
    assert _tsz(root["0"]["2"]) == {"period_us": 31250.0}


def test_odd_bundle_event_channels_hold_exactly_the_listed_attributes(
    odd_bundle,
):
    root = _open(odd_bundle)
    assert _tsz(root["9"]) == {"kind": "event", "name": "marks", "unit": "a.u."}
    assert _tsz(root["12"]) == {"kind": "event", "name": "intervals"}


@pytest.mark.parametrize(
    ("key", "level", "members"),
    [
        ("0", 1, ("minmax", "median")),
        ("0", 2, ("minmax", "count")),
        ("2", 1, ("minmax", "mean", "var", "count")),
    ],
)
def test_odd_bundle_level_members_are_the_statistics_of_their_bins(
    odd_bundle, key, level, members
):
    channel = _open(odd_bundle)[key]
    bins = _bins(channel["raw"][:], 4**level)
    reference = {
        "minmax": np.array(
            [[np.min(samples), np.max(samples)] for samples in bins]
        ),
        "median": np.array([np.median(samples) for samples in bins]),
        "mean": np.array([np.mean(samples) for samples in bins]),
        "var": np.array([np.var(samples) for samples in bins]),
        "count": np.array([np.isfinite(samples).sum() for samples in bins]),
    }
    for member in members:
        stored = channel[str(level)][member][:]
        np.testing.assert_allclose(
            stored, reference[member].astype(stored.dtype), rtol=1e-6
        )


def test_odd_bundle_last_bin_of_channel_0_is_short(odd_bundle):
    channel = _open(odd_bundle)["0"]
    last = channel["raw"][1000]
    assert channel["1"]["minmax"][250].tolist() == [last, last]
    assert channel["1"]["median"][250] == last
    assert channel["2"]["count"][62] == 1001 - 62 * 16


def test_odd_bundle_nan_run_of_channel_0_reaches_both_levels(odd_bundle):
    channel = _open(odd_bundle)["0"]
    raw = channel["raw"][:]
    start, stop = ODD_NAN_RUN
    assert np.flatnonzero(np.isnan(raw)).tolist() == list(range(start, stop))
    level_1 = np.isnan(channel["1"]["minmax"][:]).any(axis=1)
    assert np.flatnonzero(level_1).tolist() == list(range(25, 35))
    level_2 = np.isnan(channel["2"]["minmax"][:]).any(axis=1)
    assert np.flatnonzero(level_2).tolist() == [6, 7, 8]
    assert channel["2"]["count"][5:10].tolist() == [16, 4, 0, 4, 16]


def test_odd_bundle_channels_without_levels_hold_only_raw(odd_bundle):
    root = _open(odd_bundle)
    for key in ("5", "7"):
        assert list(root[key].group_keys()) == []
        assert list(root[key].array_keys()) == ["raw"]


def test_odd_bundle_channel_9_events_rise_with_one_tie(odd_bundle):
    events = _open(odd_bundle)["9"]["events"][:]
    expected = 1_000_000 + 250_000 * np.arange(ODD_EVENTS, dtype=np.int64)
    expected[ODD_TIE + 1] = expected[ODD_TIE]
    assert events.tolist() == expected.tolist()
    steps = np.diff(events)
    assert (steps >= 0).all()
    assert np.flatnonzero(steps == 0).tolist() == [ODD_TIE]


def test_odd_bundle_channel_9_holds_labels_in_three_categories_and_values(
    odd_bundle,
):
    channel = _open(odd_bundle)["9"]
    labels = channel["labels"][:]
    assert set(labels.tolist()) == set(range(ODD_LABELS))
    assert channel["values"][:].tolist() == [
        row / 8 for row in range(ODD_EVENTS)
    ]


def test_odd_bundle_channel_9_text_has_empty_rows_and_one_full_width_row(
    odd_bundle,
):
    rows = _open(odd_bundle)["9"]["text"][:]
    full = [index for index, row in enumerate(rows) if 0 not in row]
    assert full == [ODD_FULL_TEXT_ROW]
    assert rows[ODD_FULL_TEXT_ROW].tobytes().decode("utf-8") == ODD_FULL_TEXT
    assert len(ODD_FULL_TEXT.encode("utf-8")) == ODD_TEXT_WIDTH
    assert not rows[0].any()
    assert rows[1].tobytes() == b"e1\0\0\0"


def test_odd_bundle_channel_9_channels_name_the_three_key_sets(odd_bundle):
    rows = _open(odd_bundle)["9"]["channels"][:]
    bits = np.unpackbits(rows, axis=1, bitorder="little")
    named = [tuple(np.flatnonzero(row).tolist()) for row in bits]
    key_sets = ((0,), (2, 5), ())
    assert named == [key_sets[row % 3] for row in range(ODD_EVENTS)]


def test_odd_bundle_channel_9_snippet_arrays_carry_period_and_event_sample(
    odd_bundle,
):
    channel = _open(odd_bundle)["9"]
    expected = {
        "period_us": ODD_WAVEFORM_PERIOD_US,
        "event_sample": 3,
    }
    assert _tsz(channel["waveforms"]) == expected
    assert _tsz(channel["waveform_mean"]) == expected


def test_odd_bundle_channel_9_waveform_mean_is_the_mean_of_each_label(
    odd_bundle,
):
    channel = _open(odd_bundle)["9"]
    waveforms = channel["waveforms"][:].astype(np.float64)
    labels = channel["labels"][:]
    expected = [
        waveforms[labels == label].mean(axis=0) for label in range(ODD_LABELS)
    ]
    np.testing.assert_allclose(
        channel["waveform_mean"][:], np.array(expected), rtol=1e-6
    )


def test_odd_bundle_channel_9_count_level_counts_each_label_per_bin(
    odd_bundle,
):
    channel = _open(odd_bundle)["9"]
    events = channel["events"][:]
    labels = channel["labels"][:]
    counts = channel["1"]["count"][:]
    assert _tsz(channel["1"]) == {"period_us": float(ODD_COUNT_PERIOD_US)}
    assert counts.shape[0] == events[-1] // ODD_COUNT_PERIOD_US + 1
    for label in range(ODD_LABELS):
        bins = events[labels == label] // ODD_COUNT_PERIOD_US
        expected = np.bincount(bins, minlength=counts.shape[0])
        assert counts[:, label].tolist() == expected.tolist()
    assert int(counts.sum()) == ODD_EVENTS


def test_odd_bundle_channel_12_holds_durations_without_a_bound(odd_bundle):
    channel = _open(odd_bundle)["12"]
    assert channel["events"][:].tolist() == list(ODD_INTERVAL_EVENTS)
    assert channel["durations"][:].tolist() == list(ODD_INTERVAL_DURATIONS)
    assert "max_duration_us" not in _tsz(channel)


def test_synthetic_bundle_opens_as_a_version_1_bundle(synthetic_bundle):
    root = _open(synthetic_bundle)
    assert _tsz(root)["version"].startswith("1.")
    assert _tsz(root["0"])["kind"] == "continuous"


def test_find_synthetic_bundle_reads_the_path_from_the_environment(
    monkeypatch, mixed_bundle, tmp_path
):
    monkeypatch.setenv(SYNTHETIC_BUNDLE_ENV, str(mixed_bundle))
    monkeypatch.setattr(
        conftest, "DEFAULT_SYNTHETIC_BUNDLE", tmp_path / "absent"
    )
    assert find_synthetic_bundle() == mixed_bundle


def test_find_synthetic_bundle_uses_the_default_when_the_variable_is_empty(
    monkeypatch, mixed_bundle
):
    monkeypatch.setenv(SYNTHETIC_BUNDLE_ENV, "")
    monkeypatch.setattr(conftest, "DEFAULT_SYNTHETIC_BUNDLE", mixed_bundle)
    assert find_synthetic_bundle() == mixed_bundle


def test_find_synthetic_bundle_skips_when_the_path_is_absent(
    monkeypatch, tmp_path
):
    monkeypatch.setenv(SYNTHETIC_BUNDLE_ENV, str(tmp_path / "absent"))
    with pytest.raises(pytest.skip.Exception, match="no synthetic bundle"):
        find_synthetic_bundle()
