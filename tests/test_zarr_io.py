import json

import numpy as np
from zarr import Group
from zarr.codecs import ShardingCodec, ZstdCodec

from timeseries_zarr.zarr_io import (
    consolidate,
    create_array,
    create_group_with_attrs,
    open_group,
    write_region,
)


def test_open_group_creates_when_absent(tmp_path):
    path = tmp_path / "bundle"
    assert not path.exists()
    group = open_group(path)
    assert isinstance(group, Group)
    assert path.exists()


def test_open_group_is_v3(tmp_path):
    group = open_group(tmp_path / "bundle")
    assert group.metadata.zarr_format == 3


def test_open_group_opens_existing_in_place(tmp_path):
    path = tmp_path / "bundle"
    open_group(path).attrs["marker"] = "kept"
    reopened = open_group(path)
    assert reopened.attrs["marker"] == "kept"


def test_create_group_with_attrs_creates_child(tmp_path):
    path = tmp_path / "bundle"
    root = open_group(path)
    child = create_group_with_attrs(root, "0", {})
    assert isinstance(child, Group)
    assert "0" in open_group(path)


def test_create_group_with_attrs_sets_attrs_verbatim(tmp_path):
    path = tmp_path / "bundle"
    attrs = {
        "id": "N:ch:abc",
        "rate_hz": 32000.0,
        "start_us": 1000,
        "kind": "continuous",
    }
    create_group_with_attrs(open_group(path), "0", attrs)
    assert dict(open_group(path)["0"].attrs) == attrs


def test_create_group_with_attrs_empty_attrs(tmp_path):
    path = tmp_path / "bundle"
    create_group_with_attrs(open_group(path), "0", {})
    assert dict(open_group(path)["0"].attrs) == {}


def test_create_array_shape_and_dtype(tmp_path):
    path = tmp_path / "bundle"
    create_array(
        open_group(path), "lvl", (1000,), np.float32, (256,), (512,), {}, 5
    )
    arr = open_group(path)["lvl"]
    assert arr.shape == (1000,)
    assert arr.dtype == np.float32


def test_create_array_chunk_and_shard_grid(tmp_path):
    path = tmp_path / "bundle"
    create_array(
        open_group(path), "lvl", (1000,), np.float32, (256,), (512,), {}, 5
    )
    arr = open_group(path)["lvl"]
    assert arr.chunks == (256,)
    assert arr.shards == (512,)


def test_create_array_rank2_keeps_envelope_axis(tmp_path):
    path = tmp_path / "bundle"
    create_array(
        open_group(path),
        "lvl",
        (500, 2),
        np.float32,
        (256, 2),
        (512, 2),
        {},
        5,
    )
    arr = open_group(path)["lvl"]
    assert arr.shape == (500, 2)
    assert arr.chunks == (256, 2)
    assert arr.shards == (512, 2)


def test_create_array_uses_zstd_at_level(tmp_path):
    path = tmp_path / "bundle"
    create_array(
        open_group(path), "lvl", (1000,), np.float32, (256,), (512,), {}, 7
    )
    arr = open_group(path)["lvl"]
    sharding = arr.metadata.codecs[0]
    assert isinstance(sharding, ShardingCodec)
    assert any(
        isinstance(c, ZstdCodec) and c.level == 7 for c in sharding.codecs
    )


def test_create_array_sets_attrs_verbatim(tmp_path):
    path = tmp_path / "bundle"
    attrs = {"period_us": 31.25}
    create_array(
        open_group(path), "lvl", (1000,), np.float32, (256,), (512,), attrs, 5
    )
    assert dict(open_group(path)["lvl"].attrs) == attrs


def _make_array(path, shape):
    chunk = (4, *shape[1:])
    shard = (8, *shape[1:])
    return create_array(
        open_group(path), "lvl", shape, np.float32, chunk, shard, {}, 5
    )


def test_write_region_rank1_at_offset(tmp_path):
    path = tmp_path / "bundle"
    arr = _make_array(path, (10,))
    block = np.array([1, 2, 3, 4], dtype=np.float32)
    write_region(arr, 2, block)
    assert np.array_equal(open_group(path)["lvl"][2:6], block)


def test_write_region_rank2_envelope(tmp_path):
    path = tmp_path / "bundle"
    arr = _make_array(path, (10, 2))
    block = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.float32)
    write_region(arr, 1, block)
    assert np.array_equal(open_group(path)["lvl"][1:4], block)


def test_write_region_multiple_regions_tile_full_array(tmp_path):
    path = tmp_path / "bundle"
    arr = _make_array(path, (10,))
    full = np.arange(10, dtype=np.float32)
    write_region(arr, 0, full[:6])
    write_region(arr, 6, full[6:])
    assert np.array_equal(open_group(path)["lvl"][:], full)


def _shard_files(path, name):
    """Return the shard files under the array named name, newest key last."""
    return sorted(
        f.relative_to(path).as_posix()
        for f in (path / name).rglob("*")
        if f.is_file() and f.name != "zarr.json"
    )


def test_create_array_writes_all_fill_shards_rank1(tmp_path):
    path = tmp_path / "bundle"
    arr = create_array(
        open_group(path), "lvl", (12,), np.float32, (4,), (8,), {}, 5
    )
    write_region(arr, 0, np.zeros(8, dtype=np.float32))
    write_region(arr, 8, np.zeros(4, dtype=np.float32))
    assert _shard_files(path, "lvl") == ["lvl/c/0", "lvl/c/1"]


def test_create_array_writes_all_fill_shard_rank2(tmp_path):
    path = tmp_path / "bundle"
    arr = create_array(
        open_group(path), "lvl", (6, 2), np.float32, (4, 2), (8, 2), {}, 5
    )
    write_region(arr, 0, np.zeros((6, 2), dtype=np.float32))
    assert _shard_files(path, "lvl") == ["lvl/c/0/0"]


def _build_tree(path):
    root = open_group(path)
    child = create_group_with_attrs(root, "0", {"kind": "continuous"})
    create_array(child, "lvl", (8,), np.float32, (4,), (8,), {}, 5)
    return root


def test_consolidate_writes_inline_block(tmp_path):
    path = tmp_path / "bundle"
    consolidate(_build_tree(path))
    root_json = json.loads((path / "zarr.json").read_text())
    members = root_json["consolidated_metadata"]["metadata"]
    assert set(members) == {"0", "0/lvl"}


def test_consolidate_no_v2_sidecar(tmp_path):
    path = tmp_path / "bundle"
    consolidate(_build_tree(path))
    assert not (path / ".zmetadata").exists()


def test_consolidate_returns_none(tmp_path):
    path = tmp_path / "bundle"
    assert consolidate(_build_tree(path)) is None


def test_create_array_declares_nan_fill_for_floats(tmp_path):
    group = open_group(tmp_path / "bundle")
    arr = create_array(group, "f", (8,), np.float32, (4,), (8,), {}, 5)
    assert np.isnan(arr.fill_value)


def test_create_array_declares_zero_fill_for_integers(tmp_path):
    group = open_group(tmp_path / "bundle")
    arr = create_array(group, "i", (8,), np.int64, (4,), (8,), {}, 5)
    assert arr.fill_value == 0


def test_unwritten_float_chunk_reads_back_as_no_data(tmp_path):
    """The trap this closes: zarr-python's own default fill is 0.0.

    A chunk nobody wrote would then read back as a zero-volt flatline, which a
    viewer draws as a real recorded signal rather than as a gap.
    """
    group = open_group(tmp_path / "bundle")
    arr = create_array(group, "f", (8,), np.float32, (4,), (8,), {}, 5)
    assert np.all(np.isnan(arr[:]))


def test_fill_value_reaches_the_written_metadata(tmp_path):
    group = open_group(tmp_path / "bundle")
    create_array(group, "f", (8,), np.float32, (4,), (8,), {}, 5)
    node = json.loads((tmp_path / "bundle" / "f" / "zarr.json").read_text())
    # JSON has no NaN literal, so Zarr writes it as the string "NaN".
    assert node["fill_value"] == "NaN"
