import json

import numpy as np
import pytest
import zarr
from zarr.core.sync import sync
from zarr.storage import StorePath

from timeseries_zarr.validate_arrays import (
    ArrayLayout,
    check_array,
    find_missing_chunks,
    is_integer,
)
from timeseries_zarr.zarr_io import (
    create_array,
    open_group,
    open_store_path,
    write_region,
)


def _node(path, name):
    return json.loads((path / name / "zarr.json").read_text())


@pytest.fixture
def root(tmp_path):
    return open_group(tmp_path / "bundle")


def _make(root, name, rows, dtype="float32", inner=4, shard=12, trailing=()):
    return create_array(
        root,
        name,
        (rows, *trailing),
        dtype,
        (inner, *trailing),
        (shard, *trailing),
        {},
        3,
    )


def _check(tmp_path, name, shape, dtype="float32", itemsize=4):
    return check_array(
        name,
        _node(tmp_path / "bundle", name),
        dtype,
        itemsize,
        shape,
        "4.1",
    )


def test_an_array_the_writer_creates_has_no_findings(root, tmp_path):
    _make(root, "a", 30)
    layout, findings = _check(tmp_path, "a", (None,))
    assert findings == []
    assert layout == ArrayLayout((30,), (12,), (4,))


def test_a_wide_dtype_without_endian_is_a_finding(root, tmp_path):
    _make(root, "a", 30)
    node = _node(tmp_path / "bundle", "a")
    node["codecs"][0]["configuration"]["codecs"][0] = {"name": "bytes"}
    layout, findings = check_array("a", node, "float32", 4, (None,), "4.1")
    assert layout is None
    assert [f.section for f in findings] == ["7.2"]


def test_a_uint8_array_may_omit_endian(root, tmp_path):
    _make(root, "a", 30, dtype="uint8")
    node = _node(tmp_path / "bundle", "a")
    node["codecs"][0]["configuration"]["codecs"][0] = {"name": "bytes"}
    layout, findings = check_array("a", node, "uint8", 1, (None,), "5.1")
    assert findings == []
    assert layout is not None


def test_a_uint8_array_may_declare_little_endian(root, tmp_path):
    _make(root, "a", 30, dtype="uint8")
    node = _node(tmp_path / "bundle", "a")
    node["codecs"][0]["configuration"]["codecs"][0] = {
        "name": "bytes",
        "configuration": {"endian": "little"},
    }
    assert check_array("a", node, "uint8", 1, (None,), "5.1")[1] == []


def test_an_unsharded_bytes_chain_is_accepted():
    node = {
        "node_type": "array",
        "data_type": "int64",
        "shape": [5],
        "fill_value": 0,
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": [5]},
        },
        "chunk_key_encoding": {"name": "default"},
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
    }
    layout, findings = check_array("a", node, "int64", 8, (5,), "5.1")
    assert findings == []
    assert layout == ArrayLayout((5,), (5,), None)


def test_every_violation_of_one_array_is_reported(root, tmp_path):
    _make(root, "a", 30)
    node = _node(tmp_path / "bundle", "a")
    node["data_type"] = "float64"
    node["fill_value"] = 0
    node["shape"] = [31]
    node["chunk_key_encoding"] = {"name": "v2"}
    layout, findings = check_array("a", node, "float32", 4, (30,), "4.1")
    assert layout is None
    assert sorted(f.section for f in findings) == ["4.1", "4.1", "7.1", "7.3"]


def test_a_group_is_not_an_array(root, tmp_path):
    root.create_group("g")
    layout, findings = _check(tmp_path, "g", (None,))
    assert layout is None
    assert [f.message for f in findings] == ["the node must be an array"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, True),
        (0, True),
        (-3, True),
        (True, False),
        (1.0, False),
        ("1", False),
    ],
)
def test_is_integer_rejects_bools_and_floats(value, expected):
    assert is_integer(value) is expected


def _present(root, name):
    store_path = open_store_path(root.store_path.store)
    node = json.loads(sync((store_path / name / "zarr.json").get()).to_bytes())
    array = zarr.open_array(root.store_path / name, mode="r")
    layout, findings = check_array(
        name, node, str(array.dtype), array.dtype.itemsize, array.shape, "5.1"
    )
    assert findings == []
    assert layout is not None
    return store_path, layout


def test_a_written_sharded_array_has_no_missing_chunks(root):
    array = _make(root, "a", 30)
    array[:] = np.arange(30, dtype=np.float32)
    store_path, layout = _present(root, "a")
    assert find_missing_chunks(store_path, "a", layout) == []


def test_a_shard_that_was_never_written_misses_its_chunks_in_the_extent(root):
    _make(root, "a", 30)
    store_path, layout = _present(root, "a")
    # 30 rows, inner chunks of 4 rows: 8 chunks intersect the extent.
    assert find_missing_chunks(store_path, "a", layout) == [
        f"chunk c/{index} (shard object absent)" for index in range(8)
    ]


def test_an_inner_chunk_with_an_empty_index_entry_is_missing(root):
    array = _make(root, "a", 30)
    array[0:4] = np.ones(4, dtype=np.float32)
    array[8:12] = np.ones(4, dtype=np.float32)
    store_path, layout = _present(root, "a")
    # Shard 0 holds chunks 0 through 2, and shards 1 and 2 were never written.
    assert find_missing_chunks(store_path, "a", layout) == [
        "chunk c/1",
        *(f"chunk c/{index} (shard object absent)" for index in range(3, 8)),
    ]


def test_a_chunk_past_the_extent_is_not_missing(root):
    array = _make(root, "a", 5, inner=4, shard=12)
    array[:] = np.ones(5, dtype=np.float32)
    store_path, layout = _present(root, "a")
    # Inner chunk 2 starts at row 8, past the 5 rows of the array.
    assert find_missing_chunks(store_path, "a", layout) == []


def test_an_array_with_a_zero_length_axis_has_no_chunks(root):
    _make(root, "a", 0)
    store_path, layout = _present(root, "a")
    assert find_missing_chunks(store_path, "a", layout) == []


def test_a_second_shard_is_read_by_its_own_index(root):
    array = _make(root, "a", 30)
    write_region(array, 0, np.ones(12, dtype=np.float32))
    write_region(array, 12, np.ones(12, dtype=np.float32))
    store_path, layout = _present(root, "a")
    assert find_missing_chunks(store_path, "a", layout) == [
        "chunk c/6 (shard object absent)",
        "chunk c/7 (shard object absent)",
    ]


def test_a_two_axis_array_reports_the_coordinates_of_a_missing_chunk(root):
    array = _make(root, "a", 12, trailing=(3,))
    array[0:4] = np.ones((4, 3), dtype=np.float32)
    array[8:12] = np.ones((4, 3), dtype=np.float32)
    store_path, layout = _present(root, "a")
    assert find_missing_chunks(store_path, "a", layout) == ["chunk c/1/0"]


def test_a_truncated_shard_is_reported(root, tmp_path):
    array = _make(root, "a", 30)
    array[:] = np.arange(30, dtype=np.float32)
    shard = tmp_path / "bundle" / "a" / "c" / "0"
    shard.write_bytes(shard.read_bytes()[:10])
    store_path, layout = _present(root, "a")
    (message,) = find_missing_chunks(store_path, "a", layout)[:1]
    assert "too short" in message


def test_an_unsharded_array_checks_each_chunk_key(root, tmp_path):
    store_path = StorePath(root.store_path.store)
    layout = ArrayLayout((10,), (4,), None)
    chunks = tmp_path / "bundle" / "plain" / "c"
    chunks.mkdir(parents=True)
    (chunks / "0").write_bytes(b"x")
    assert find_missing_chunks(store_path, "plain", layout) == [
        "chunk c/1",
        "chunk c/2",
    ]
