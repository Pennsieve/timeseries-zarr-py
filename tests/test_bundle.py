import datetime as dt
import json
import math
import re

import numpy as np
import pytest

from timeseries_zarr.bundle import (
    assign_indices,
    atomic_publish,
    channel_keys,
    max_key,
    origin_us,
    write_all_channels,
    write_bundle,
)
from timeseries_zarr.recording import RECORDING_FILE, session_start_us
from timeseries_zarr.types import RecordingMeta, WriteOpts
from timeseries_zarr.zarr_io import open_group


def _write_all(root, indexed, opts=None):
    sources = [source for _, source in indexed]
    write_all_channels(
        root,
        indexed,
        origin_us(sources),
        channel_keys(indexed),
        opts or _OPTS,
    )


_OPTS = WriteOpts(
    min_bins=2, max_levels=8, inner_len=16, target_shard_bytes=256
)


def test_assign_indices_empty():
    assert assign_indices([], []) == []


def test_assign_indices_continuous_then_event(continuous_source):
    c0 = continuous_source(np.zeros(4, dtype=np.float32), id="c0")
    c1 = continuous_source(np.zeros(4, dtype=np.float32), id="c1")
    # assign_indices does not inspect its sources.
    u0, u1 = object(), object()
    assert assign_indices([c0, c1], [u0, u1]) == [
        (0, c0),
        (1, c1),
        (2, u0),
        (3, u1),
    ]


def test_assign_indices_only_continuous(continuous_source):
    c0 = continuous_source(np.zeros(1, dtype=np.float32))
    assert assign_indices([c0], []) == [(0, c0)]


def test_assign_indices_only_events():
    u0 = object()
    assert assign_indices([], [u0]) == [(0, u0)]


def test_atomic_publish_to_fresh_path(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "zarr.json").write_text("root")
    (staging / "0").mkdir()
    final = tmp_path / "bundle"
    atomic_publish(staging, final)
    assert not staging.exists()
    assert (final / "zarr.json").read_text() == "root"
    assert (final / "0").is_dir()


def test_atomic_publish_replaces_existing_final(tmp_path):
    final = tmp_path / "bundle"
    final.mkdir()
    (final / "stale.txt").write_text("old")
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "zarr.json").write_text("new")
    atomic_publish(staging, final)
    assert not staging.exists()
    assert (final / "zarr.json").read_text() == "new"
    assert not (final / "stale.txt").exists()


def test_atomic_publish_returns_none(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    result = atomic_publish(staging, tmp_path / "bundle")
    assert result is None


def test_atomic_publish_restores_final_on_failure(tmp_path):
    final = tmp_path / "bundle"
    final.mkdir()
    (final / "stale.txt").write_text("old")
    missing_staging = tmp_path / "nope"  # does not exist -> os.replace fails
    with pytest.raises(OSError):
        atomic_publish(missing_staging, final)
    assert (final / "stale.txt").read_text() == "old"
    assert not (tmp_path / "bundle.old").exists()


def test_write_all_channels_dispatches_by_type(
    tmp_path, continuous_source, event_source
):
    root = open_group(tmp_path / "bundle")
    cont = continuous_source(np.arange(64, dtype=np.float32), id="c0")
    event = event_source([5, 6, 7], id="e0")
    _write_all(root, assign_indices([cont], [event]))
    g = open_group(tmp_path / "bundle")
    assert dict(g["0"].attrs)["tsz"]["kind"] == "continuous"
    assert "raw" in list(g["0"].array_keys())
    assert dict(g["1"].attrs)["tsz"]["kind"] == "event"
    assert list(g["1"].array_keys()) == ["events"]


def test_write_all_channels_empty(tmp_path):
    root = open_group(tmp_path / "bundle")
    write_all_channels(root, [], 0, {}, _OPTS)
    assert list(open_group(tmp_path / "bundle").group_keys()) == []


def test_write_all_channels_returns_none(tmp_path, continuous_source):
    root = open_group(tmp_path / "bundle")
    indexed = assign_indices(
        [continuous_source(np.arange(8, dtype=np.float32))], []
    )
    result = write_all_channels(root, indexed, 0, channel_keys(indexed), _OPTS)
    assert result is None


def test_write_all_channels_resolves_event_channel_ids_through_the_keys(
    tmp_path, continuous_source, event_source
):
    root = open_group(tmp_path / "bundle")
    first = continuous_source(np.zeros(8, np.float32), id="c0")
    second = continuous_source(np.zeros(8, np.float32), id="c1")
    event = event_source([5, 6], id="e0", channel_ids=[["c1"], ["c0", "c1"]])
    _write_all(root, assign_indices([first, second], [event]))
    channels = open_group(tmp_path / "bundle")["2"]["channels"][:]
    assert channels.tolist() == [[2], [3]]


def test_write_bundle_publishes_mixed_bundle(
    tmp_path, continuous_source, event_source
):
    staging = tmp_path / "staging"
    final = tmp_path / "bundle"
    samples = np.arange(64, dtype=np.float32)
    cont = continuous_source(samples, id="c0")
    event = event_source([5, 6, 7], id="e0")
    write_bundle(
        [cont], [event], staging_dir=staging, final_dir=final, opts=_OPTS
    )
    assert final.exists()
    assert not staging.exists()
    g = open_group(final)
    assert dict(g["0"].attrs)["tsz"]["kind"] == "continuous"
    assert dict(g["1"].attrs)["tsz"]["kind"] == "event"
    assert np.array_equal(g["0"]["raw"][:], samples)
    assert g["1"]["events"].shape == (3,)


def test_write_bundle_counts_every_time_from_the_earliest_start(
    tmp_path, continuous_source, event_source
):
    cont = continuous_source(np.zeros(8, np.float32), id="c0", start_us=1_000)
    event = event_source([500, 700], id="e0")
    final = tmp_path / "bundle"
    write_bundle(
        [cont], [event], staging_dir=tmp_path / "s", final_dir=final, opts=_OPTS
    )
    g = open_group(final)
    assert dict(g["0"].attrs)["tsz"]["offset_us"] == 500
    assert g["1"]["events"][:].tolist() == [0, 200]


def test_write_bundle_rejects_two_channels_with_one_id_before_writing(
    tmp_path, continuous_source, event_source
):
    staging, final = tmp_path / "staging", tmp_path / "bundle"
    cont = continuous_source(np.zeros(8, np.float32), id="same")
    event = event_source([1], id="same")
    with pytest.raises(ValueError, match="share the id"):
        write_bundle(
            [cont], [event], staging_dir=staging, final_dir=final, opts=_OPTS
        )
    assert not staging.exists()
    assert not final.exists()


def test_write_bundle_consolidates_root_inline(tmp_path, continuous_source):
    staging = tmp_path / "staging"
    final = tmp_path / "bundle"
    cont = continuous_source(np.arange(8, dtype=np.float32))
    write_bundle([cont], [], staging_dir=staging, final_dir=final, opts=_OPTS)
    root_json = json.loads((final / "zarr.json").read_text())
    assert "consolidated_metadata" in root_json
    assert not (final / ".zmetadata").exists()


def test_write_bundle_empty_inputs(tmp_path):
    staging = tmp_path / "staging"
    final = tmp_path / "bundle"
    write_bundle([], [], staging_dir=staging, final_dir=final, opts=_OPTS)
    assert final.exists()
    assert list(open_group(final).group_keys()) == []


def test_write_bundle_returns_none(tmp_path, continuous_source):
    cont = continuous_source(np.arange(8, dtype=np.float32))
    result = write_bundle(
        [cont],
        [],
        staging_dir=tmp_path / "staging",
        final_dir=tmp_path / "bundle",
        opts=_OPTS,
    )
    assert result is None


def test_write_all_channels_places_continuous_channels_relative_to_the_earliest_start(
    tmp_path, continuous_source
):
    root = open_group(tmp_path / "bundle")
    early = continuous_source(
        np.arange(8, dtype=np.float32), id="a", start_us=1_000
    )
    late = continuous_source(
        np.arange(8, dtype=np.float32), id="b", start_us=5_000
    )
    _write_all(root, assign_indices([early, late], []))
    g = open_group(tmp_path / "bundle")
    assert dict(g["0"].attrs)["tsz"]["offset_us"] == 0
    assert dict(g["1"].attrs)["tsz"]["offset_us"] == 4_000


def test_origin_us_is_the_earliest_start_of_any_source(
    continuous_source, event_source
):
    sources = [
        continuous_source(np.zeros(4, np.float32), start_us=5_000),
        event_source([2_000, 9_000]),
        continuous_source(np.zeros(4, np.float32), start_us=7_000),
    ]
    assert origin_us(sources) == 2_000


def test_origin_us_is_the_session_start_when_it_is_earliest(continuous_source):
    sources = [continuous_source(np.zeros(4, np.float32), start_us=5_000)]
    assert origin_us(sources, session_start_us=1_000) == 1_000


def test_origin_us_is_a_source_start_when_the_session_starts_later(
    continuous_source,
):
    sources = [continuous_source(np.zeros(4, np.float32), start_us=5_000)]
    assert origin_us(sources, session_start_us=9_000) == 5_000


def test_origin_us_with_only_a_session_start_is_that_start():
    assert origin_us([], session_start_us=42) == 42


def test_origin_us_of_nothing_is_zero():
    assert origin_us([]) == 0


def test_origin_us_counts_an_empty_event_channel_by_its_start(event_source):
    assert origin_us([event_source([], start_us=77)]) == 77


def test_origin_us_may_be_negative_for_a_start_before_the_epoch(
    continuous_source,
):
    sources = [continuous_source(np.zeros(4, np.float32), start_us=-5)]
    assert origin_us(sources) == -5


def test_channel_keys_maps_every_id_to_its_index(
    continuous_source, event_source
):
    indexed = assign_indices(
        [
            continuous_source(np.zeros(4, np.float32), id="LA1"),
            continuous_source(np.zeros(4, np.float32), id="LA2"),
        ],
        [event_source([1], id="units"), event_source([1], id="marks")],
    )
    assert channel_keys(indexed) == {"LA1": 0, "LA2": 1, "units": 2, "marks": 3}


def test_channel_keys_of_no_channels_is_empty():
    assert channel_keys([]) == {}


def test_channel_keys_rejects_two_channels_with_one_id(continuous_source):
    indexed = assign_indices(
        [
            continuous_source(np.zeros(4, np.float32), id="same"),
            continuous_source(np.zeros(4, np.float32), id="other"),
            continuous_source(np.zeros(4, np.float32), id="same"),
        ],
        [],
    )
    with pytest.raises(
        ValueError, match="channels 0 and 2 share the id 'same'"
    ):
        channel_keys(indexed)


def test_channel_keys_rejects_a_continuous_and_an_event_channel_sharing_an_id(
    continuous_source, event_source
):
    indexed = [
        (0, continuous_source(np.zeros(4, np.float32), id="x")),
        (1, event_source([1], id="x")),
    ]
    with pytest.raises(ValueError, match="share the id 'x'"):
        channel_keys(indexed)


def test_max_key_is_the_largest_index(continuous_source):
    sources = [continuous_source(np.zeros(4, np.float32)) for _ in range(5)]
    assert max_key(assign_indices(sources, [])) == 4


def test_max_key_does_not_assume_contiguous_keys(continuous_source):
    src = continuous_source(np.zeros(4, np.float32))
    assert max_key([(2, src), (9, src), (5, src)]) == 9


def test_max_key_of_no_channels_is_zero():
    assert max_key([]) == 0


_SPEC_OPTS = WriteOpts(
    min_bins=256, max_levels=8, inner_len=256, target_shard_bytes=4096
)
_SESSION = dt.datetime(
    2024, 3, 14, 9, 15, tzinfo=dt.timezone(dt.timedelta(hours=-5))
)
_SESSION_US = session_start_us(
    RecordingMeta(start=_SESSION, subject={}, session={}, source={})
)
_RECORDING = RecordingMeta(
    start=_SESSION,
    subject={"id": "sub-0931", "species": "Homo sapiens"},
    session={"id": "ses-01"},
    source={"converter": "timeseries-zarr-py"},
)


def _signal(n, seed):
    rng = np.random.default_rng(seed)
    samples = (5000.0 + 8.0 * rng.standard_normal(n)).astype(np.float32)
    samples[300:340] = np.nan
    return samples


def _mixed_sources(continuous_source, event_source, base=_SESSION_US):
    rng = np.random.default_rng(3)
    lead = continuous_source(
        _signal(8192, 0), id="LA1", name="LA1", rate_hz=256.0, start_us=base
    )
    lag = continuous_source(
        _signal(16384, 1),
        id="LA2",
        name="LA2",
        rate_hz=512.0,
        start_us=base + 500_000,
    )
    spike_times = np.sort(rng.integers(0, 30_000_000, 400)) + base
    spikes = event_source(
        spike_times,
        labels=rng.integers(0, 3, 400),
        label_names=["unit-0", "unit-1", "unit-2"],
        waveform_mean=rng.standard_normal((3, 16)),
        waveform_period_us=122.0703125,
        id="units",
        name="units",
    )
    marks = event_source(
        base + np.array([1, 4, 9, 15, 22, 28]) * 1_000_000,
        durations=[0, 0, 5_000_000, 0, 12_000_000, 0],
        labels=[1, 1, 0, 2, 0, 2],
        label_names=["seizure", "spike", "note"],
        text=["spike", "spike", "seizure onset", "note", "seizure", "x"],
        channel_ids=[["LA1"], [], ["LA1", "LA2"], [], ["LA2"], []],
        id="marks",
        name="curated marks",
    )
    dense_times = np.sort(rng.integers(0, 700_000_000, 70_000))
    dense_times[-1] = 700_000_000
    dense = event_source(
        dense_times + base,
        labels=rng.integers(0, 2, 70_000),
        label_names=["ripple", "fast-ripple"],
        id="hfo",
        name="hfo detector",
    )
    return [lead, lag], [spikes, marks, dense]


def _read_nodes(path):
    return {
        "/".join(file.parent.relative_to(path).parts): json.loads(
            file.read_text()
        )
        for file in path.rglob("zarr.json")
    }


def _array_ok(node, shape, dtype, names):
    fill = "NaN" if dtype.startswith("float") else 0
    return (
        node["node_type"] == "array"
        and tuple(node["shape"]) == tuple(shape)
        and node["data_type"] == dtype
        and node["fill_value"] == fill
        and node.get("dimension_names") == names
    )


def _children(nodes, key):
    prefix = key + "/"
    return {
        p[len(prefix) :]
        for p in nodes
        if p.startswith(prefix) and "/" not in p[len(prefix) :]
    }


def _check_root(path, nodes):
    root = nodes[""]
    assert root["attributes"] == {"tsz": {"version": "1.0"}}
    assert not any(
        "version" in n["attributes"].get("tsz", {})
        for p, n in nodes.items()
        if p
    )
    consolidated = root["consolidated_metadata"]
    assert consolidated["kind"] == "inline"
    assert consolidated["must_understand"] is False
    assert set(consolidated["metadata"]) == set(nodes) - {""}
    for p, entry in consolidated["metadata"].items():
        stored = {
            k: v for k, v in nodes[p].items() if k != "consolidated_metadata"
        }
        listed = {
            k: v for k, v in entry.items() if k != "consolidated_metadata"
        }
        assert listed == stored, p
    assert all(
        isinstance(n["attributes"].get("tsz"), dict)
        for n in nodes.values()
        if n.get("attributes")
    )
    texts = [(path / p / "zarr.json").read_text() for p in nodes]
    assert not any(re.search(r"\d{4}-\d{2}-\d{2}", text) for text in texts)
    assert not any("sub-0931" in text for text in texts)


def _check_continuous(nodes, key, attrs):
    rate = attrs["rate_hz"]
    assert rate > 0
    assert isinstance(attrs["offset_us"], int) and attrs["offset_us"] >= 0
    samples = nodes[f"{key}/raw"]["shape"][0]
    assert _array_ok(nodes[f"{key}/raw"], (samples,), "float32", ["time"])
    levels = sorted(int(c) for c in _children(nodes, key) if c.isdigit())
    assert levels == list(range(1, len(levels) + 1))
    for k in levels:
        period = nodes[f"{key}/{k}"]["attributes"]["tsz"]["period_us"]
        assert math.isclose(period, 4**k * 1e6 / rate, rel_tol=1e-12)
        members = _children(nodes, f"{key}/{k}")
        assert (
            {"minmax", "mean", "count"}
            <= members
            <= {
                "minmax",
                "mean",
                "var",
                "count",
            }
        )
        bins = math.ceil(samples / 4**k)
        expected = {
            "minmax": ((bins, 2), "float32", ["time", None]),
            "mean": ((bins,), "float32", ["time"]),
            "var": ((bins,), "float32", ["time"]),
            "count": ((bins,), "uint32", ["time"]),
        }
        for member in members:
            assert _array_ok(nodes[f"{key}/{k}/{member}"], *expected[member])


def _check_count_levels(nodes, key, events, labels, categories):
    levels = sorted(int(c) for c in _children(nodes, key) if c.isdigit())
    assert levels == list(range(1, len(levels) + 1))
    data = open_group(PATH_HOLDER["path"])[key]
    previous = None
    for k in levels:
        period = nodes[f"{key}/{k}"]["attributes"]["tsz"]["period_us"]
        assert _children(nodes, f"{key}/{k}") == {"count"}
        node = nodes[f"{key}/{k}/count"]
        bins = node["shape"][0]
        assert _array_ok(node, (bins, categories), "uint32", ["time", None])
        counts = data[str(k)]["count"][:]
        assert bins * period > events[-1]
        assert (
            counts.sum(axis=0).tolist()
            == np.bincount(labels, minlength=categories).tolist()
        )
        if previous is None:
            rung = int(period)
            while rung % 4 == 0 and rung > 1000:
                rung //= 4
            assert float(period).is_integer() and rung == 1000
        else:
            assert period == 4 * previous
        previous = period


PATH_HOLDER = {}


def _check_text_and_channels(nodes, key, data, n, group_keys):
    text = nodes.get(f"{key}/text")
    if text is not None:
        width = text["shape"][1]
        assert _array_ok(text, (n, width), "uint8", ["event", None])
        rows = [bytes(row).rstrip(b"\0") for row in data["text"][:]]
        assert all(b"\0" not in row for row in rows)
        assert max(len(row.decode("utf-8")) for row in rows) <= width
        assert max(len(row) for row in rows) == width
    channels = nodes.get(f"{key}/channels")
    if channels is not None:
        width = (max(group_keys) + 1 + 7) // 8
        assert _array_ok(channels, (n, width), "uint8", ["event", None])
        bits = np.unpackbits(data["channels"][:], axis=1, bitorder="little")
        assert set(np.flatnonzero(bits.any(axis=0)).tolist()) <= set(group_keys)


def _check_event(path, nodes, key, attrs, group_keys):
    n = nodes[f"{key}/events"]["shape"][0]
    assert _array_ok(nodes[f"{key}/events"], (n,), "int64", ["event"])
    data = open_group(path)[key]
    events = data["events"][:]
    assert (events >= 0).all() and (np.diff(events) >= 0).all()
    arrays = {c for c in _children(nodes, key) if not c.isdigit()}
    assert arrays <= {
        "events",
        "durations",
        "labels",
        "text",
        "channels",
        "waveform_mean",
    }
    assert all(
        nodes[f"{key}/{a}"]["shape"][0] == n for a in arrays - {"waveform_mean"}
    )
    labels = data["labels"][:] if "labels" in arrays else None
    names = attrs.get("label_names")
    categories = (
        len(names)
        if names
        else (int(labels.max()) + 1 if labels is not None else 1)
    )
    if labels is not None:
        assert _array_ok(nodes[f"{key}/labels"], (n,), "uint16", ["event"])
        assert int(labels.max()) < categories
    if "durations" in arrays:
        durations = data["durations"][:]
        assert _array_ok(nodes[f"{key}/durations"], (n,), "int64", ["event"])
        assert (
            durations.min() >= 0 and durations.max() <= attrs["max_duration_us"]
        )
    if "waveform_mean" in arrays:
        node = nodes[f"{key}/waveform_mean"]
        assert _array_ok(node, tuple(node["shape"]), "float32", ["label", None])
        assert node["shape"][0] == categories and labels is not None
        assert node["attributes"]["tsz"]["period_us"] > 0
    _check_text_and_channels(nodes, key, data, n, group_keys)
    if any(c.isdigit() for c in _children(nodes, key)):
        PATH_HOLDER["path"] = path
        _check_count_levels(
            nodes,
            key,
            events,
            labels if labels is not None else np.zeros(n, int),
            categories,
        )


def _check_recording(path, nodes):
    document = json.loads((path / RECORDING_FILE).read_text("utf-8"))
    start = dt.datetime.fromisoformat(document["start"])
    assert start.utcoffset() is not None
    assert all(
        isinstance(document.get(k, {}), dict)
        for k in ("subject", "session", "source")
    )
    assert RECORDING_FILE not in nodes[""]["consolidated_metadata"]["metadata"]
    return start


def _validate_bundle(path):
    """Check a bundle against the specification's validation list."""
    nodes = _read_nodes(path)
    _check_root(path, nodes)
    group_keys = sorted(int(p) for p in nodes if p and "/" not in p)
    for key in map(str, group_keys):
        assert key == str(int(key))
        attrs = nodes[key]["attributes"]["tsz"]
        assert isinstance(attrs["name"], str)
        if attrs["kind"] == "continuous":
            _check_continuous(nodes, key, attrs)
        else:
            assert attrs["kind"] == "event"
            _check_event(path, nodes, key, attrs, group_keys)
    return nodes


def _write_mixed(
    tmp_path,
    continuous_source,
    event_source,
    recording=_RECORDING,
    base=_SESSION_US,
):
    continuous, events = _mixed_sources(continuous_source, event_source, base)
    final = tmp_path / "bundle"
    write_bundle(
        continuous,
        events,
        staging_dir=tmp_path / "staging",
        final_dir=final,
        opts=_SPEC_OPTS,
        recording=recording,
    )
    return final


def test_write_bundle_conforms_to_the_specification_validation_list(
    tmp_path, continuous_source, event_source
):
    final = _write_mixed(tmp_path, continuous_source, event_source)
    nodes = _validate_bundle(final)
    assert sorted(p for p in nodes if p and "/" not in p) == [
        "0",
        "1",
        "2",
        "3",
        "4",
    ]
    start = _check_recording(final, nodes)
    assert start == _SESSION


def test_write_bundle_without_a_recording_has_no_recording_json_and_conforms(
    tmp_path, continuous_source, event_source
):
    final = _write_mixed(
        tmp_path, continuous_source, event_source, recording=None
    )
    _validate_bundle(final)
    assert not (final / RECORDING_FILE).exists()


def test_write_bundle_recording_start_is_the_origin_when_a_channel_starts_earlier(
    tmp_path, continuous_source, event_source
):
    final = _write_mixed(
        tmp_path, continuous_source, event_source, base=_SESSION_US - 2_000_000
    )
    nodes = _validate_bundle(final)
    start = _check_recording(final, nodes)
    assert start == _SESSION - dt.timedelta(seconds=2)
    assert nodes["0"]["attributes"]["tsz"]["offset_us"] == 0


def test_write_bundle_keeps_identity_and_the_date_out_of_every_zarr_node(
    tmp_path, continuous_source, event_source
):
    final = _write_mixed(tmp_path, continuous_source, event_source)
    document = json.loads((final / RECORDING_FILE).read_text("utf-8"))
    assert document["subject"] == {"id": "sub-0931", "species": "Homo sapiens"}
    assert "sub-0931" not in (final / "zarr.json").read_text()
    assert document["start"][:10] not in (final / "zarr.json").read_text()


def test_write_bundle_rejects_a_recording_start_without_an_offset_before_writing(
    tmp_path, continuous_source
):
    naive = RecordingMeta(
        start=dt.datetime(2024, 3, 14, 9, 15), subject={}, session={}, source={}
    )
    cont = continuous_source(np.zeros(8, np.float32), id="c0")
    with pytest.raises(ValueError, match="UTC offset"):
        write_bundle(
            [cont],
            [],
            staging_dir=tmp_path / "staging",
            final_dir=tmp_path / "bundle",
            opts=_OPTS,
            recording=naive,
        )
    assert not (tmp_path / "staging").exists()


def test_write_bundle_with_a_channel_too_short_for_a_level_still_conforms(
    tmp_path, continuous_source
):
    short = continuous_source(_signal(100, 5), id="short", name="short")
    final = tmp_path / "bundle"
    write_bundle(
        [short],
        [],
        staging_dir=tmp_path / "staging",
        final_dir=final,
        opts=_SPEC_OPTS,
    )
    nodes = _validate_bundle(final)
    assert sorted(p for p in nodes if p.startswith("0/")) == ["0/raw"]
