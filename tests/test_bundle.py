import json

import numpy as np
import pytest

from timeseries_zarr.bundle import (
    assign_indices,
    atomic_publish,
    write_all_channels,
    write_bundle,
)
from timeseries_zarr.types import WriteOpts
from timeseries_zarr.zarr_io import open_group

_OPTS = WriteOpts(
    min_bins=2, max_levels=8, inner_len=16, target_shard_bytes=256
)


def test_assign_indices_empty():
    assert assign_indices([], []) == []


def test_assign_indices_continuous_then_unit(continuous_source):
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


def test_assign_indices_only_units():
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
    tmp_path, continuous_source, unit_source
):
    root = open_group(tmp_path / "bundle")
    cont = continuous_source(np.arange(64, dtype=np.float32), id="c0")
    unit = unit_source(np.arange(5, dtype=np.int64), id="u0")
    write_all_channels(root, assign_indices([cont], [unit]), 0, _OPTS)
    g = open_group(tmp_path / "bundle")
    assert dict(g["0"].attrs)["kind"] == "continuous"
    assert "raw" in list(g["0"].array_keys())
    assert dict(g["1"].attrs)["kind"] == "event"
    assert sorted(g["1"].array_keys()) == ["events", "labels", "waveforms"]


def test_write_all_channels_empty(tmp_path):
    root = open_group(tmp_path / "bundle")
    write_all_channels(root, [], 0, _OPTS)
    assert list(open_group(tmp_path / "bundle").group_keys()) == []


def test_write_all_channels_returns_none(tmp_path, continuous_source):
    root = open_group(tmp_path / "bundle")
    indexed = assign_indices(
        [continuous_source(np.arange(8, dtype=np.float32))], []
    )
    assert write_all_channels(root, indexed, 0, _OPTS) is None


def test_write_bundle_publishes_mixed_bundle(
    tmp_path, continuous_source, unit_source
):
    staging = tmp_path / "staging"
    final = tmp_path / "bundle"
    samples = np.arange(64, dtype=np.float32)
    cont = continuous_source(samples, id="c0")
    unit = unit_source(np.arange(5, dtype=np.int64), id="u0")
    write_bundle(
        [cont], [unit], staging_dir=staging, final_dir=final, opts=_OPTS
    )
    assert final.exists()
    assert not staging.exists()
    g = open_group(final)
    assert dict(g["0"].attrs)["kind"] == "continuous"
    assert dict(g["1"].attrs)["kind"] == "event"
    assert np.array_equal(g["0"]["raw"][:], samples)
    assert sorted(g["1"].array_keys()) == ["events", "labels", "waveforms"]


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
