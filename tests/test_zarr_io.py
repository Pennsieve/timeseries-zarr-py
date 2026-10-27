import json
import sys

import numpy as np
import pytest
import zarr
from zarr import Array, Group
from zarr.buffer import default_buffer_prototype
from zarr.buffer.cpu import Buffer
from zarr.codecs import ShardingCodec, ZstdCodec
from zarr.core.sync import sync
from zarr.errors import ContainsGroupError, GroupNotFoundError
from zarr.storage import (
    FsspecStore,
    LocalStore,
    LoggingStore,
    MemoryStore,
    StorePath,
    WrapperStore,
)

from timeseries_zarr.attrs import root_attrs
from timeseries_zarr.extras import missing_extra_message
from timeseries_zarr.shards import ArrayWriter, BufferShardSink, FileShardSink
from timeseries_zarr.zarr_io import (
    _local_location,
    array_metadata,
    close_store,
    consolidate,
    create_array,
    create_group_with_attrs,
    create_nodes,
    descendants,
    group_metadata,
    open_group,
    open_root,
    read_int,
    read_rows,
    read_store_text,
    shard_layout,
    store_location,
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


def _node(path, *parts):
    return json.loads((path.joinpath(*parts) / "zarr.json").read_text())


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_create_array_float_arrays_declare_nan_fill(tmp_path, dtype):
    path = tmp_path / "bundle"
    arr = create_array(open_group(path), "a", (16,), dtype, (4,), (8,), {}, 5)
    assert _node(path, "a")["fill_value"] == "NaN"
    assert np.isnan(arr[:]).all()


@pytest.mark.parametrize("dtype", [np.int64, np.uint8, np.uint16, np.uint32])
def test_create_array_integer_arrays_declare_zero_fill(tmp_path, dtype):
    path = tmp_path / "bundle"
    arr = create_array(open_group(path), "a", (16,), dtype, (4,), (8,), {}, 5)
    assert _node(path, "a")["fill_value"] == 0
    assert not arr[:].any()


def test_create_array_stores_dimension_names(tmp_path):
    path = tmp_path / "bundle"
    create_array(
        open_group(path),
        "a",
        (16, 2),
        np.float32,
        (4, 2),
        (8, 2),
        {},
        5,
        dimension_names=("time", None),
    )
    assert _node(path, "a")["dimension_names"] == ["time", None]


def test_create_array_without_dimension_names_stores_none(tmp_path):
    path = tmp_path / "bundle"
    create_array(open_group(path), "a", (16,), np.float32, (4,), (8,), {}, 5)
    assert "dimension_names" not in _node(path, "a")


def test_create_array_rejects_dimension_names_of_the_wrong_length(tmp_path):
    with pytest.raises(ValueError, match="dimension"):
        create_array(
            open_group(tmp_path / "bundle"),
            "a",
            (16, 2),
            np.float32,
            (4, 2),
            (8, 2),
            {},
            5,
            dimension_names=("time",),
        )


def _inner_codecs(path, name):
    sharding = _node(path, name)["codecs"][0]
    return [codec["name"] for codec in sharding["configuration"]["codecs"]]


def test_create_array_compresses_with_zstd_by_default(tmp_path):
    path = tmp_path / "bundle"
    create_array(open_group(path), "a", (16,), np.float32, (4,), (8,), {}, 5)
    assert _inner_codecs(path, "a") == ["bytes", "zstd"]


def test_create_array_uncompressed_keeps_the_sharding_codec(tmp_path):
    path = tmp_path / "bundle"
    create_array(
        open_group(path),
        "a",
        (4, 3),
        np.uint8,
        (4, 3),
        (4, 3),
        {},
        5,
        compress=False,
    )
    assert _node(path, "a")["codecs"][0]["name"] == "sharding_indexed"
    assert _inner_codecs(path, "a") == ["bytes"]


def test_create_array_uncompressed_shard_starts_with_the_plain_rows(tmp_path):
    path = tmp_path / "bundle"
    rows = np.arange(12, dtype=np.uint8).reshape(4, 3) + 65
    arr = create_array(
        open_group(path),
        "a",
        (4, 3),
        np.uint8,
        (4, 3),
        (4, 3),
        {},
        5,
        compress=False,
    )
    arr[:] = rows
    shard = (path / "a" / "c" / "0" / "0").read_bytes()
    assert shard.startswith(rows.tobytes())
    assert np.array_equal(open_group(path)["a"][:], rows)


def test_open_group_writes_attrs_when_it_creates_the_group(tmp_path):
    path = tmp_path / "bundle"
    group = open_group(path, {"tsz": {"version": "1.0"}})
    assert dict(group.attrs) == {"tsz": {"version": "1.0"}}
    assert _node(path)["attributes"] == {"tsz": {"version": "1.0"}}


def test_open_group_without_attrs_creates_a_group_with_none(tmp_path):
    path = tmp_path / "bundle"
    assert dict(open_group(path).attrs) == {}
    assert _node(path)["attributes"] == {}


def test_open_group_ignores_attrs_when_the_group_exists(tmp_path):
    path = tmp_path / "bundle"
    open_group(path, {"tsz": {"version": "1.0"}})
    reopened = open_group(path, {"other": 1})
    assert dict(reopened.attrs) == {"tsz": {"version": "1.0"}}
    assert _node(path)["attributes"] == {"tsz": {"version": "1.0"}}


def test_open_group_reopened_without_attrs_keeps_the_existing_ones(tmp_path):
    path = tmp_path / "bundle"
    open_group(path, root_attrs())
    assert dict(open_group(path).attrs) == root_attrs()


def test_root_attrs_survive_consolidation_beside_the_inline_metadata(tmp_path):
    path = tmp_path / "bundle"
    root = open_group(path, root_attrs())
    create_group_with_attrs(root, "0", {"tsz": {"kind": "continuous"}})
    consolidate(root)
    node = _node(path)
    assert node["attributes"] == root_attrs()
    assert node["consolidated_metadata"]["kind"] == "inline"
    assert node["consolidated_metadata"]["must_understand"] is False
    assert list(node["consolidated_metadata"]["metadata"]) == ["0"]


_ROOT_TSZ = {"version": "1.0"}
_RECORDING_TEXT = json.dumps({"subject": {"id": "sub-01"}})
_PREFIX = "sub/bundle"


def _write_reader_tree(path):
    root = open_group(path, {"tsz": _ROOT_TSZ})
    channel = create_group_with_attrs(
        root, "0", {"tsz": {"kind": "continuous"}}
    )
    raw = create_array(channel, "raw", (10,), np.float32, (4,), (8,), {}, 5)
    write_region(raw, 0, np.arange(10, dtype=np.float32))
    level = create_group_with_attrs(channel, "1", {})
    minmax = create_array(
        level, "minmax", (3, 2), np.float32, (4, 2), (8, 2), {}, 5
    )
    write_region(minmax, 0, np.arange(6, dtype=np.float32).reshape(3, 2))
    event = create_group_with_attrs(root, "4", {"tsz": {"kind": "event"}})
    events = create_array(event, "events", (5,), np.int64, (8,), (8,), {}, 5)
    write_region(events, 0, np.arange(5, dtype=np.int64) * 10)
    consolidate(root)
    (path / "recording.json").write_text(_RECORDING_TEXT)
    return path


@pytest.fixture(scope="module")
def reader_tree(tmp_path_factory):
    return _write_reader_tree(tmp_path_factory.mktemp("reader") / "bundle")


def _memory_store(path, prefix="", *, read_only=True):
    lead = f"{prefix}/" if prefix else ""
    files = {
        lead + file.relative_to(path).as_posix(): Buffer.from_bytes(
            file.read_bytes()
        )
        for file in path.rglob("*")
        if file.is_file()
    }
    return MemoryStore(files, read_only=read_only)


class _RecordingStore(WrapperStore):
    """Wrap a read-only store and record every key read or listed."""

    def __init__(self, store):
        super().__init__(store)
        self.reads = []

    async def get(self, key, prototype, byte_range=None):
        self.reads.append(key)
        return await super().get(key, prototype, byte_range)

    def get_sync(self, key, *, prototype=None, byte_range=None):
        self.reads.append(key)
        return super().get_sync(key, prototype=prototype, byte_range=byte_range)

    async def get_partial_values(self, prototype, key_ranges):
        key_ranges = list(key_ranges)
        self.reads.extend(key for key, _ in key_ranges)
        return await super().get_partial_values(prototype, key_ranges)

    async def exists(self, key):
        self.reads.append(key)
        return await super().exists(key)

    async def list(self):
        self.reads.append("list")
        async for key in super().list():
            yield key

    async def list_prefix(self, prefix):
        self.reads.append(f"list {prefix}")
        async for key in super().list_prefix(prefix):
            yield key

    async def list_dir(self, prefix):
        self.reads.append(f"list {prefix}")
        async for key in super().list_dir(prefix):
            yield key


class _AsyncOnlyStore(WrapperStore):
    """Raise TypeError on a synchronous read, as FsspecStore does."""

    def get_sync(self, key, *, prototype=None, byte_range=None):
        raise TypeError("no synchronous get")


class _NoReadOnlyCopyStore(WrapperStore):
    """Raise NotImplementedError on a read-only copy, as ZipStore does."""

    def with_read_only(self, read_only=False):
        raise NotImplementedError("no read-only copy")


@pytest.mark.parametrize(
    "make_location",
    [
        str,
        lambda path: path,
        LocalStore,
        lambda path: path.as_uri(),
        _memory_store,
        lambda path: StorePath(_memory_store(path, _PREFIX), _PREFIX),
    ],
    ids=["str", "Path", "LocalStore", "file URL", "MemoryStore", "StorePath"],
)
def test_open_root_opens_each_location_kind(reader_tree, make_location):
    root = open_root(make_location(reader_tree))
    assert isinstance(root, Group)
    assert root.attrs["tsz"] == _ROOT_TSZ
    assert root.metadata.consolidated_metadata is not None


@pytest.mark.parametrize(
    "make_location",
    [
        LocalStore,
        lambda path: StorePath(LocalStore(path.parent), path.name),
        lambda path: StorePath(
            _memory_store(path, _PREFIX, read_only=False), _PREFIX
        ),
    ],
    ids=["LocalStore", "LocalStore StorePath", "MemoryStore StorePath"],
)
def test_open_root_opens_a_writable_store_read_only(reader_tree, make_location):
    root = open_root(make_location(reader_tree))
    assert root.store.read_only
    with pytest.raises(ValueError, match="read-only"):
        root.attrs.update({"marker": "written"})


def test_open_root_opens_a_store_path_whose_store_makes_no_read_only_copy(
    reader_tree,
):
    store = _NoReadOnlyCopyStore(
        _memory_store(reader_tree, _PREFIX, read_only=False)
    )
    root = open_root(StorePath(store, _PREFIX))
    assert root.attrs["tsz"] == _ROOT_TSZ
    assert root.store is store


def test_open_root_rejects_storage_options_with_a_local_path(reader_tree):
    with pytest.raises(TypeError, match="storage_options"):
        open_root(str(reader_tree), storage_options={"anon": True})


@pytest.mark.parametrize(
    "make_url",
    [
        lambda path: path.as_uri(),
        lambda path: f"file://localhost{path.as_posix()}",
        lambda path: path.as_uri().replace("file:", "FILE:", 1),
        lambda path: f"file://LOCALHOST{path.as_posix()}",
    ],
    ids=["escaped", "localhost", "uppercase scheme", "uppercase host"],
)
def test_open_root_opens_the_local_path_a_file_url_names(tmp_path, make_url):
    path = _write_reader_tree(tmp_path / "a b" / "bundle")
    assert "%20" in path.as_uri()
    assert open_root(make_url(path)).attrs["tsz"] == _ROOT_TSZ


_URL_DELIMITER_NAMES = ["rec#1.tszarr", "rec?x.tszarr"]


def _write_marked_root(path, marker):
    root = open_group(path, {"tsz": _ROOT_TSZ, "marker": marker})
    consolidate(root)
    return path


@pytest.mark.parametrize("name", _URL_DELIMITER_NAMES)
def test_open_root_opens_a_str_path_that_holds_a_url_delimiter(tmp_path, name):
    path = _write_reader_tree(tmp_path / name)
    root = open_root(str(path))
    assert root.attrs["tsz"] == _ROOT_TSZ
    assert set(descendants(root)[1]) == {"0/raw", "0/1/minmax", "4/events"}


@pytest.mark.parametrize("name", _URL_DELIMITER_NAMES)
def test_open_root_opens_the_whole_str_path_beside_its_shorter_prefix(
    tmp_path, name
):
    _write_marked_root(tmp_path / "rec", "rec")
    path = _write_marked_root(tmp_path / name, name)
    assert open_root(str(path)).attrs["marker"] == name


@pytest.mark.parametrize(
    "url", ["memory://bundles/rec.tszarr", "s3://bucket/rec#1.tszarr"]
)
def test_local_location_returns_a_url_of_another_scheme_unchanged(url):
    assert _local_location(url) == url


@pytest.mark.parametrize("name", [*_URL_DELIMITER_NAMES, "rec#", "rec?"])
def test_open_root_rejects_a_file_url_with_a_query_or_fragment(tmp_path, name):
    _write_marked_root(tmp_path / "rec", "rec")
    with pytest.raises(ValueError, match="query or fragment"):
        open_root("file://" + (tmp_path / name).as_posix())


def test_open_root_rejects_a_file_url_with_a_remote_host():
    with pytest.raises(ValueError, match="localhost or no host, got 'server'"):
        open_root("file://server/share/bundle")


def test_open_root_rejects_a_tree_without_consolidated_metadata(tmp_path):
    path = tmp_path / "bundle"
    create_group_with_attrs(open_group(path, {"tsz": _ROOT_TSZ}), "0", {})
    with pytest.raises(ValueError) as excinfo:
        open_root(str(path))
    assert str(excinfo.value) == (
        f"{path} has no consolidated metadata; a complete bundle carries it "
        "in its root zarr.json"
    )


def test_open_root_raises_file_not_found_for_an_absent_path(tmp_path):
    path = tmp_path / "absent"
    with pytest.raises(FileNotFoundError):
        open_root(path)
    assert not path.exists()


def test_open_root_raises_group_not_found_for_a_store_without_zarr_json():
    with pytest.raises(GroupNotFoundError):
        open_root(MemoryStore({}, read_only=True))


def test_open_root_rejects_a_location_whose_root_is_an_array(tmp_path):
    path = tmp_path / "array"
    zarr.create_array(LocalStore(path), shape=(4,), dtype="float32")
    with pytest.raises(ValueError):
        open_root(path)


def test_open_root_reports_a_corrupt_zarr_json_as_invalid_json(tmp_path):
    path = tmp_path / "bundle"
    path.mkdir()
    (path / "zarr.json").write_text("{not json")
    with pytest.raises(json.JSONDecodeError) as excinfo:
        open_root(path)
    assert "consolidated" not in str(excinfo.value)


def test_open_root_reads_only_the_root_zarr_json(reader_tree):
    store = _RecordingStore(_memory_store(reader_tree))
    open_root(store)
    assert store.reads == ["zarr.json"]


def test_open_root_reads_only_the_root_zarr_json_under_a_prefix(reader_tree):
    store = _RecordingStore(_memory_store(reader_tree, _PREFIX))
    open_root(StorePath(store, _PREFIX))
    assert store.reads == [f"{_PREFIX}/zarr.json"]


def test_descendants_keys_every_node_by_its_path(reader_tree):
    groups, arrays = descendants(open_root(reader_tree))
    assert set(groups) == {"0", "0/1", "4"}
    assert set(arrays) == {"0/raw", "0/1/minmax", "4/events"}
    assert all(isinstance(group, Group) for group in groups.values())
    assert all(isinstance(array, Array) for array in arrays.values())


def test_descendants_keys_paths_relative_to_the_group(reader_tree):
    groups, arrays = descendants(open_root(reader_tree)["0"])
    assert set(groups) == {"1"}
    assert set(arrays) == {"raw", "1/minmax"}


def test_descendants_makes_no_store_reads(reader_tree):
    store = _RecordingStore(_memory_store(reader_tree))
    root = open_root(store)
    store.reads.clear()
    groups, _ = descendants(root)
    descendants(groups["0"])
    assert store.reads == []


def test_read_rows_returns_the_half_open_range(reader_tree):
    raw = open_root(reader_tree)["0/raw"]
    rows = read_rows(raw, 2, 5)
    assert isinstance(rows, np.ndarray)
    assert rows.dtype == np.float32
    assert rows.tolist() == [2.0, 3.0, 4.0]


def test_read_rows_keeps_the_trailing_axis(reader_tree):
    minmax = open_root(reader_tree)["0/1/minmax"]
    assert read_rows(minmax, 1, 3).tolist() == [[2.0, 3.0], [4.0, 5.0]]


def test_read_rows_past_the_end_returns_the_rows_that_exist(reader_tree):
    raw = open_root(reader_tree)["0/raw"]
    assert read_rows(raw, 8, 100).tolist() == [8.0, 9.0]


def test_read_rows_wholly_past_the_end_returns_no_rows(reader_tree):
    minmax = open_root(reader_tree)["0/1/minmax"]
    assert read_rows(minmax, 20, 30).shape == (0, 2)


def test_read_rows_returns_no_rows_for_an_empty_range(reader_tree):
    raw = open_root(reader_tree)["0/raw"]
    assert read_rows(raw, 4, 4).shape == (0,)


def test_read_rows_rejects_a_negative_start(reader_tree):
    raw = open_root(reader_tree)["0/raw"]
    with pytest.raises(ValueError, match=r"0 <= start <= stop, got \[-2, 3\)"):
        read_rows(raw, -2, 3)


def test_read_rows_rejects_a_stop_below_the_start(reader_tree):
    raw = open_root(reader_tree)["0/raw"]
    with pytest.raises(ValueError, match=r"got \[5, 4\)"):
        read_rows(raw, 5, 4)


def test_read_int_returns_a_python_int(reader_tree):
    value = read_int(open_root(reader_tree)["4/events"], 3)
    assert type(value) is int
    assert value == 30


def test_read_int_rejects_a_negative_index(reader_tree):
    events = open_root(reader_tree)["4/events"]
    with pytest.raises(ValueError, match="at least 0, got -1"):
        read_int(events, -1)


def test_read_int_raises_index_error_past_the_last_element(reader_tree):
    with pytest.raises(IndexError):
        read_int(open_root(reader_tree)["4/events"], 5)


def test_read_store_text_returns_the_object_beside_zarr_json(reader_tree):
    root = open_root(reader_tree)
    assert read_store_text(root, "recording.json") == _RECORDING_TEXT


def test_read_store_text_returns_none_for_an_absent_object(reader_tree):
    assert read_store_text(open_root(reader_tree), "absent.json") is None


def test_read_store_text_reads_beside_a_prefixed_root(reader_tree):
    store = _RecordingStore(_memory_store(reader_tree, _PREFIX))
    root = open_root(StorePath(store, _PREFIX))
    assert read_store_text(root, "recording.json") == _RECORDING_TEXT
    assert store.reads[-1] == f"{_PREFIX}/recording.json"


def test_read_store_text_reads_from_a_store_without_synchronous_reads(
    reader_tree,
):
    root = open_root(_AsyncOnlyStore(_memory_store(reader_tree)))
    assert read_store_text(root, "recording.json") == _RECORDING_TEXT


def test_read_store_text_decodes_utf8(tmp_path):
    text = json.dumps({"site": "caf\N{LATIN SMALL LETTER E WITH ACUTE}"})
    path = _write_reader_tree(tmp_path / "bundle")
    (path / "recording.json").write_bytes(text.encode("utf-8"))
    assert read_store_text(open_root(path), "recording.json") == text


def test_read_store_text_rejects_bytes_that_are_not_utf8(tmp_path):
    path = _write_reader_tree(tmp_path / "bundle")
    (path / "recording.json").write_bytes(b"\xff\xfe")
    with pytest.raises(UnicodeDecodeError):
        read_store_text(open_root(path), "recording.json")


def test_store_location_names_a_local_root_by_file_url(reader_tree):
    root = open_root(reader_tree)
    assert store_location(root) == f"file://{reader_tree.as_posix()}"
    assert store_location(root["0"]) == f"file://{reader_tree.as_posix()}/0"


def test_store_location_reopens_a_bundle_whose_name_holds_a_url_delimiter(
    tmp_path,
):
    _write_marked_root(tmp_path / "rec", "rec")
    path = _write_marked_root(tmp_path / "rec#1", "rec#1")
    location = store_location(open_root(path))
    assert open_root(location).attrs["marker"] == "rec#1"


def test_store_location_reopens_a_root_opened_from_a_relative_path(
    tmp_path, monkeypatch
):
    _write_marked_root(tmp_path / "rec", "rec")
    monkeypatch.chdir(tmp_path)
    location = store_location(open_root("rec"))
    monkeypatch.chdir(tmp_path.parent)
    assert open_root(location).attrs["marker"] == "rec"


def test_store_location_ends_with_the_store_path_prefix(reader_tree):
    root = open_root(StorePath(_memory_store(reader_tree, _PREFIX), _PREFIX))
    location = store_location(root)
    assert location.startswith("memory://")
    assert location.endswith(f"/{_PREFIX}")


def test_store_location_names_a_wrapped_local_store_as_the_local_store(
    reader_tree,
):
    store = LoggingStore(LocalStore(reader_tree, read_only=True))
    assert store_location(open_root(store)) == store_location(
        open_root(reader_tree)
    )


def test_close_store_closes_the_store_and_a_second_call_succeeds(
    reader_tree,
):
    root = open_root(reader_tree)
    close_store(root)
    close_store(root)
    assert not root.store._is_open


_REMOTE_EXTRA_MESSAGE = missing_extra_message("fsspec", "remote")


def _block_fsspec(monkeypatch):
    """Make every import of fsspec and of each fsspec submodule fail."""
    monkeypatch.setitem(sys.modules, "fsspec", None)
    # A cached submodule would otherwise import without its parent.
    for name in list(sys.modules):
        if name.startswith("fsspec."):
            monkeypatch.setitem(sys.modules, name, None)


def test_open_root_names_the_remote_extra_when_fsspec_is_missing(monkeypatch):
    _block_fsspec(monkeypatch)
    with pytest.raises(ImportError) as excinfo:
        open_root("s3://bucket/bundle")
    assert str(excinfo.value) == _REMOTE_EXTRA_MESSAGE
    assert excinfo.value.name == "fsspec"
    assert isinstance(excinfo.value.__cause__, ModuleNotFoundError)


def _raise_module_not_found(name):
    def open_group(*args, **kwargs):
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    return open_group


@pytest.mark.parametrize("name", ["fsspec", "fsspec.implementations.http"])
def test_open_root_names_the_remote_extra_for_a_missing_fsspec_module(
    monkeypatch, name
):
    monkeypatch.setattr(zarr, "open_group", _raise_module_not_found(name))
    with pytest.raises(ImportError) as excinfo:
        open_root("s3://bucket/bundle")
    assert str(excinfo.value) == _REMOTE_EXTRA_MESSAGE
    assert excinfo.value.name == "fsspec"
    assert excinfo.value.__cause__.name == name


@pytest.mark.parametrize("name", ["s3fs", "fsspec_xrootd", None])
def test_open_root_lets_another_missing_module_propagate(monkeypatch, name):
    monkeypatch.setattr(zarr, "open_group", _raise_module_not_found(name))
    with pytest.raises(ModuleNotFoundError) as excinfo:
        open_root("s3://bucket/bundle")
    assert excinfo.value.name == name


@pytest.mark.parametrize("name", ["run:1.tszarr", "sub-01:ses-02"])
def test_open_root_opens_a_relative_path_with_a_colon_in_a_local_store(
    tmp_path, monkeypatch, name
):
    _block_fsspec(monkeypatch)
    monkeypatch.chdir(tmp_path)
    _write_marked_root(tmp_path / name, name)
    root = open_root(name)
    assert isinstance(root.store, LocalStore)
    assert root.attrs["marker"] == name


_FSSPEC_DIRECTORY = "/zarr-io"
_FSSPEC_URL = "memory://zarr-io/bundle"
_FSSPEC_LOCATION = "memory:///zarr-io/bundle"


@pytest.fixture
def fsspec_memory_tree(reader_tree):
    """Copy the reader tree to _FSSPEC_URL in the fsspec memory filesystem.

    Returns _FSSPEC_URL and removes the copy after the test.
    """
    fsspec = pytest.importorskip("fsspec")
    filesystem = fsspec.filesystem("memory")
    # The trailing slash copies the contents even when the target exists.
    filesystem.put(
        f"{reader_tree}/", f"{_FSSPEC_DIRECTORY}/bundle", recursive=True
    )
    yield _FSSPEC_URL
    filesystem.rm(_FSSPEC_DIRECTORY, recursive=True)


def test_open_root_opens_a_memory_url_through_an_fsspec_store(
    fsspec_memory_tree,
):
    root = open_root(fsspec_memory_tree)
    assert isinstance(root.store, FsspecStore)
    assert root.attrs["tsz"] == _ROOT_TSZ


def test_store_location_names_an_fsspec_group_by_its_filesystem_url(
    fsspec_memory_tree,
):
    root = open_root(fsspec_memory_tree)
    assert store_location(root) == _FSSPEC_LOCATION
    assert store_location(root["0"]) == f"{_FSSPEC_LOCATION}/0"
    assert open_root(store_location(root)).attrs["tsz"] == _ROOT_TSZ


@pytest.mark.parametrize(
    ("store_url", "prefix", "location"),
    [
        ("memory://zarr-io", "bundle", _FSSPEC_LOCATION),
        ("memory://", "zarr-io/bundle", _FSSPEC_URL),
    ],
    ids=["below a store path", "below the filesystem root"],
)
def test_store_location_joins_the_store_path_to_the_fsspec_path(
    fsspec_memory_tree, store_url, prefix, location
):
    store = FsspecStore.from_url(store_url, read_only=True)
    root = open_root(StorePath(store, prefix))
    assert store_location(root) == location
    assert open_root(location).attrs["tsz"] == _ROOT_TSZ


def test_store_location_names_a_wrapped_fsspec_store_by_its_url(
    fsspec_memory_tree,
):
    store = LoggingStore(FsspecStore.from_url(_FSSPEC_URL, read_only=True))
    assert store_location(open_root(store)) == _FSSPEC_LOCATION


def test_store_location_names_a_directory_filesystem_group_by_its_outer_url(
    fsspec_memory_tree,
):
    fsspec = pytest.importorskip("fsspec")
    dirfs = pytest.importorskip("fsspec.implementations.dirfs")
    wrapper = pytest.importorskip("fsspec.implementations.asyn_wrapper")
    directory = dirfs.DirFileSystem(
        path=_FSSPEC_DIRECTORY, fs=fsspec.filesystem("memory")
    )
    store = FsspecStore(
        wrapper.AsyncFileSystemWrapper(directory, asynchronous=True),
        path="bundle",
        read_only=True,
    )
    assert store_location(open_root(store)) == "dir://bundle"


@pytest.mark.parametrize("name", _URL_DELIMITER_NAMES)
@pytest.mark.parametrize(
    "make_location",
    [
        lambda path: FsspecStore.from_url(f"file://{path}", read_only=True),
        lambda path: FsspecStore.from_url(f"local://{path}", read_only=True),
        lambda path: StorePath(
            FsspecStore.from_url(path.parent.as_uri(), read_only=True),
            path.name,
        ),
    ],
    ids=["file URL", "local URL", "StorePath"],
)
def test_store_location_reopens_an_fsspec_local_store_holding_a_url_delimiter(
    tmp_path, make_location, name
):
    pytest.importorskip("fsspec")
    _write_marked_root(tmp_path / "rec", "rec")
    path = _write_marked_root(tmp_path / name, name)
    root = open_root(make_location(path))
    assert isinstance(root.store, FsspecStore)
    location = store_location(root)
    assert location == path.as_uri()
    assert open_root(location).attrs["marker"] == name


def test_read_store_text_reads_through_an_fsspec_store(fsspec_memory_tree):
    root = open_root(fsspec_memory_tree)
    assert read_store_text(root, "recording.json") == _RECORDING_TEXT
    assert read_store_text(root, "absent.json") is None


def test_close_store_closes_an_fsspec_store_and_a_second_call_succeeds(
    fsspec_memory_tree,
):
    root = open_root(fsspec_memory_tree)
    close_store(root)
    close_store(root)
    assert not root.store._is_open


def test_open_root_names_an_fsspec_store_without_consolidated_metadata_by_url():
    fsspec = pytest.importorskip("fsspec")
    filesystem = fsspec.filesystem("memory")
    node = {"zarr_format": 3, "node_type": "group", "attributes": {}}
    filesystem.pipe("/zarr-io-bare/bundle/zarr.json", json.dumps(node).encode())
    try:
        store = FsspecStore.from_url(
            "memory://zarr-io-bare/bundle", read_only=True
        )
        with pytest.raises(ValueError) as excinfo:
            open_root(store)
        assert str(excinfo.value) == (
            "memory:///zarr-io-bare/bundle has no consolidated metadata; a "
            "complete bundle carries it in its root zarr.json"
        )
    finally:
        filesystem.rm("/zarr-io-bare", recursive=True)


def sharded_array(tmp_path, shape=(1000,), dtype=np.float32, name="raw", **kw):
    chunks = kw.pop("chunks", (64, *shape[1:]))
    shards = kw.pop("shards", (256, *shape[1:]))
    group = open_group(tmp_path / "bundle")
    return create_array(
        group, name, shape, dtype, chunks, shards, {}, kw.pop("level", 3)
    )


def rewrite_metadata(array, edit):
    """Return the array reopened after edit changes its zarr.json text."""
    root = array.store_path.store.root
    path = root / array.store_path.path / "zarr.json"
    path.write_text(edit(path.read_text()))
    return zarr.open_array(LocalStore(root), path=array.store_path.path)


def test_shard_layout_reads_the_geometry_of_a_rank_1_array(tmp_path):
    layout = shard_layout(sharded_array(tmp_path, level=7))
    assert layout.dtype == np.float32
    assert layout.chunk_rows == 64
    assert layout.chunks_per_shard == 4
    assert layout.trailing_shape == ()
    assert layout.length == 1000
    assert layout.zstd_level == 7
    assert np.isnan(layout.fill_value)


def test_shard_layout_reads_the_trailing_shape_of_a_rank_2_array(tmp_path):
    layout = shard_layout(sharded_array(tmp_path, shape=(500, 2)))
    assert layout.trailing_shape == (2,)
    assert layout.length == 500
    assert layout.chunk_rows == 64


def test_shard_layout_takes_zero_as_the_fill_of_an_integer_array(tmp_path):
    layout = shard_layout(sharded_array(tmp_path, dtype=np.uint32))
    assert layout.dtype == np.uint32
    assert layout.fill_value == 0


def test_shard_layout_accepts_a_one_byte_dtype(tmp_path):
    layout = shard_layout(sharded_array(tmp_path, dtype=np.uint8))
    assert layout.dtype == np.uint8


def test_shard_layout_sink_writes_a_shard_file_under_a_local_store(tmp_path):
    array = sharded_array(tmp_path, name="raw")
    layout = shard_layout(array)
    sink = layout.open_sink(2)
    assert isinstance(sink, FileShardSink)
    sink.append(b"x")
    sink.close()
    assert (tmp_path / "bundle" / "raw" / "c" / "2").read_bytes() == b"x"


def test_shard_layout_sink_names_the_chunk_key_of_a_nested_rank_2_array(
    tmp_path,
):
    group = open_group(tmp_path / "bundle")
    child = create_group_with_attrs(group, "0", {})
    array = create_array(
        child, "minmax", (500, 2), np.float32, (64, 2), (256, 2), {}, 3
    )
    sink = shard_layout(array).open_sink(1)
    sink.append(b"y")
    sink.close()
    path = tmp_path / "bundle" / "0" / "minmax" / "c" / "1" / "0"
    assert path.read_bytes() == b"y"


def test_shard_layout_sink_sets_the_key_in_a_store_that_is_not_local():
    store = MemoryStore()
    array = zarr.create_array(
        store,
        name="a/raw",
        shape=(600,),
        chunks=(64,),
        shards=(256,),
        dtype=np.float32,
        compressors=ZstdCodec(level=3),
        zarr_format=3,
    )
    sink = shard_layout(array).open_sink(1)
    assert isinstance(sink, BufferShardSink)
    sink.append(b"zz")
    sink.close()
    stored = sync(store.get("a/raw/c/1", default_buffer_prototype()))
    assert stored.to_bytes() == b"zz"


def test_shard_layout_sink_goes_through_a_wrapped_local_store(tmp_path):
    sharded_array(tmp_path)
    wrapped = LoggingStore(LocalStore(tmp_path / "bundle"))
    array = zarr.open_array(wrapped, path="raw")
    assert isinstance(shard_layout(array).open_sink(0), BufferShardSink)


@pytest.mark.parametrize(
    ("dtype", "shape"),
    [
        (np.float32, (700,)),
        (np.uint32, (700,)),
        (np.float32, (700, 2)),
    ],
)
def test_shard_layout_writes_shards_that_zarr_reads(tmp_path, dtype, shape):
    array = sharded_array(tmp_path, shape=shape, dtype=dtype)
    values = np.arange(np.prod(shape)).reshape(shape).astype(dtype)
    writer = ArrayWriter(shard_layout(array))
    writer.push(values)
    writer.finish()
    reopened = open_group(tmp_path / "bundle")["raw"]
    assert np.array_equal(reopened[:], values)


def test_shard_layout_writes_shards_that_zarr_reads_from_a_memory_store():
    store = MemoryStore()
    array = zarr.create_array(
        store,
        name="raw",
        shape=(700, 2),
        chunks=(64, 2),
        shards=(256, 2),
        dtype=np.float32,
        compressors=ZstdCodec(level=3),
        zarr_format=3,
    )
    values = np.arange(1400, dtype=np.float32).reshape(700, 2)
    writer = ArrayWriter(shard_layout(array))
    writer.push(values)
    writer.finish()
    assert np.array_equal(zarr.open_array(store, path="raw")[:], values)


def test_shard_layout_rejects_an_array_that_is_not_sharded():
    array = zarr.create_array(
        MemoryStore(), shape=(100,), chunks=(10,), dtype=np.float32
    )
    with pytest.raises(ValueError, match="expected one sharding_indexed"):
        shard_layout(array)


def test_shard_layout_rejects_a_v2_array():
    array = zarr.create_array(
        MemoryStore(),
        shape=(100,),
        chunks=(10,),
        dtype=np.float32,
        zarr_format=2,
    )
    with pytest.raises(ValueError, match="must be a Zarr v3 array"):
        shard_layout(array)


def test_shard_layout_rejects_a_shard_without_compression(tmp_path):
    group = open_group(tmp_path / "bundle")
    array = create_array(
        group, "text", (100,), np.uint8, (10,), (20,), {}, 3, compress=False
    )
    with pytest.raises(ValueError, match="inner codecs must be bytes"):
        shard_layout(array)


def test_shard_layout_rejects_a_zstd_frame_checksum(tmp_path):
    array = sharded_array(tmp_path)
    reopened = rewrite_metadata(
        array,
        lambda text: text.replace('"checksum": false', '"checksum": true'),
    )
    with pytest.raises(ValueError, match="without a checksum"):
        shard_layout(reopened)


def test_shard_layout_rejects_a_big_endian_chunk(tmp_path):
    array = sharded_array(tmp_path)

    def edit(text):
        head, tail = text.split('"codecs": [', 1)
        return (
            head
            + '"codecs": ['
            + tail.replace('"endian": "little"', '"endian": "big"', 1)
        )

    with pytest.raises(ValueError, match="inner codecs must be bytes"):
        shard_layout(rewrite_metadata(array, edit))


def test_shard_layout_rejects_the_index_at_the_start(tmp_path):
    array = sharded_array(tmp_path)
    reopened = rewrite_metadata(
        array,
        lambda text: text.replace(
            '"index_location": "end"', '"index_location": "start"'
        ),
    )
    with pytest.raises(ValueError, match="index location must be end"):
        shard_layout(reopened)


def test_shard_layout_rejects_an_index_without_a_checksum(tmp_path):
    array = sharded_array(tmp_path)
    reopened = rewrite_metadata(
        array,
        lambda text: text.replace(
            '{\n            "name": "crc32c"\n          }', '{"name": "bytes"}'
        ),
    )
    with pytest.raises(ValueError, match="index codecs must be bytes"):
        shard_layout(reopened)


def test_shard_layout_rejects_a_chunked_trailing_axis(tmp_path):
    group = open_group(tmp_path / "bundle")
    array = create_array(
        group, "wide", (500, 4), np.float32, (64, 2), (256, 4), {}, 3
    )
    with pytest.raises(ValueError, match="whole rows"):
        shard_layout(array)


def test_shard_layout_rejects_a_read_only_store(tmp_path):
    sharded_array(tmp_path)
    array = zarr.open_array(
        LocalStore(tmp_path / "bundle", read_only=True), path="raw"
    )
    with pytest.raises(ValueError, match="read-only"):
        shard_layout(array)


_ARRAY_CASES = [
    pytest.param(
        (100_000,), np.float32, (8192,), (65_536,), {}, ("time",), id="raw"
    ),
    pytest.param(
        (5000, 2),
        np.float32,
        (1024, 2),
        (4096, 2),
        {},
        ("time", None),
        id="minmax",
    ),
    pytest.param((300,), np.uint32, (64,), (128,), {}, ("time",), id="count"),
    pytest.param(
        (9,), np.float32, (3,), (9,), {"k": [1, "a"]}, None, id="attrs"
    ),
    pytest.param(
        (0,), np.float32, (8192,), (65_536,), {}, ("time",), id="empty"
    ),
]


@pytest.mark.parametrize(
    ("shape", "dtype", "chunk", "shard", "attrs", "names"), _ARRAY_CASES
)
def test_array_metadata_writes_the_zarr_json_that_create_array_writes(
    tmp_path, shape, dtype, chunk, shard, attrs, names
):
    expected = create_array(
        open_group(tmp_path / "expected"),
        "member",
        shape,
        dtype,
        chunk,
        shard,
        attrs,
        4,
        dimension_names=names,
    )
    root = open_group(tmp_path / "actual")
    metadata = array_metadata(
        shape, dtype, chunk, shard, 4, attrs, dimension_names=names
    )
    create_nodes(root, {"member": metadata})
    assert (tmp_path / "actual" / "member" / "zarr.json").read_bytes() == (
        tmp_path / "expected" / "member" / "zarr.json"
    ).read_bytes()
    assert expected.metadata == metadata


def test_group_metadata_writes_the_zarr_json_that_create_group_with_attrs_writes(
    tmp_path,
):
    attrs = {"tsz": {"period_us": 125, "unit": "uV"}}
    create_group_with_attrs(open_group(tmp_path / "expected"), "1", attrs)
    create_nodes(open_group(tmp_path / "actual"), {"1": group_metadata(attrs)})
    assert (tmp_path / "actual" / "1" / "zarr.json").read_bytes() == (
        tmp_path / "expected" / "1" / "zarr.json"
    ).read_bytes()


def test_group_metadata_without_attributes_writes_the_same_zarr_json(tmp_path):
    create_group_with_attrs(open_group(tmp_path / "expected"), "1", {})
    create_nodes(open_group(tmp_path / "actual"), {"1": group_metadata({})})
    assert (tmp_path / "actual" / "1" / "zarr.json").read_bytes() == (
        tmp_path / "expected" / "1" / "zarr.json"
    ).read_bytes()


def test_array_metadata_rejects_dimension_names_of_the_wrong_length():
    with pytest.raises(ValueError, match="dimension_names"):
        array_metadata(
            (8, 2), np.float32, (4, 2), (8, 2), 3, {}, dimension_names=("time",)
        )


def test_create_nodes_returns_a_handle_for_each_path(tmp_path):
    root = open_group(tmp_path / "bundle")
    handles = create_nodes(
        root,
        {
            "0": group_metadata({"a": 1}),
            "0/raw": array_metadata((8,), np.float32, (4,), (8,), 3, {}),
            "0/1": group_metadata({}),
            "0/1/count": array_metadata((2,), np.uint32, (2,), (2,), 3, {}),
        },
    )
    assert sorted(handles) == ["0", "0/1", "0/1/count", "0/raw"]
    assert isinstance(handles["0"], Group)
    assert isinstance(handles["0/raw"], Array)
    assert handles["0/1/count"].path == "0/1/count"
    assert dict(handles["0"].attrs) == {"a": 1}
    assert sorted(open_group(tmp_path / "bundle")["0"].keys()) == ["1", "raw"]


def test_create_nodes_creates_the_nodes_below_a_nested_group(tmp_path):
    root = open_group(tmp_path / "bundle")
    parent = create_group_with_attrs(root, "parent", {"keep": True})
    handles = create_nodes(
        parent,
        {"7/raw": array_metadata((8,), np.float32, (4,), (8,), 3, {})},
    )
    assert handles["7/raw"].path == "parent/7/raw"
    assert (tmp_path / "bundle" / "parent" / "7" / "zarr.json").is_file()
    assert dict(open_group(tmp_path / "bundle")["parent"].attrs) == {
        "keep": True
    }


def test_create_nodes_leaves_an_existing_parent_group_as_it_is(tmp_path):
    root = open_group(tmp_path / "bundle", {"root": 1})
    create_nodes(
        root, {"0": group_metadata({"a": 1}), "0/b": group_metadata({})}
    )
    assert dict(open_group(tmp_path / "bundle").attrs) == {"root": 1}


def test_create_nodes_rejects_a_group_that_exists(tmp_path):
    root = open_group(tmp_path / "bundle")
    create_group_with_attrs(root, "0", {})
    with pytest.raises(ContainsGroupError):
        create_nodes(root, {"0": group_metadata({})})


def test_create_nodes_handle_writes_every_all_fill_shard(tmp_path):
    root = open_group(tmp_path / "bundle")
    handles = create_nodes(
        root,
        {"lvl": array_metadata((12,), np.float32, (4,), (8,), 5, {})},
    )
    array = handles["lvl"]
    write_region(array, 0, np.zeros(8, dtype=np.float32))
    write_region(array, 8, np.zeros(4, dtype=np.float32))
    assert _shard_files(tmp_path / "bundle", "lvl") == ["lvl/c/0", "lvl/c/1"]


def test_create_nodes_works_in_a_memory_store():
    root = zarr.open_group(MemoryStore(), mode="a", zarr_format=3)
    handles = create_nodes(
        root, {"0/raw": array_metadata((8,), np.float32, (4,), (8,), 3, {})}
    )
    assert handles["0/raw"].shape == (8,)
    assert root["0"]["raw"].shape == (8,)
