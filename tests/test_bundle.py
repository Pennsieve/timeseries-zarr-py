import json

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
from timeseries_zarr.types import WriteOpts
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
