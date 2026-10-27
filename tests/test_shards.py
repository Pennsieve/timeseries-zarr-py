import resource
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import google_crc32c
import numpy as np
import pytest
import zarr
from numcodecs.zstd import Zstd

from timeseries_zarr.shards import (
    ArrayWriter,
    BufferShardSink,
    ChunkEncoder,
    FileShardSink,
    ShardLayout,
    ShardWriter,
)
from timeseries_zarr.zarr_io import create_array, open_group

MISSING = 2**64 - 1
CHUNK_ROWS = 64
CHUNKS_PER_SHARD = 4
SHARD_ROWS = CHUNK_ROWS * CHUNKS_PER_SHARD


class ListSink:
    def __init__(self):
        self.parts = []
        self.closed = False

    def append(self, data):
        assert not self.closed
        self.parts.append(data)

    def close(self):
        self.closed = True

    @property
    def data(self):
        return b"".join(self.parts)


def make_layout(
    sinks,
    length,
    dtype=np.float32,
    trailing=(),
    level=3,
    fill=float("nan"),
):
    def open_sink(index):
        sinks[index] = ListSink()
        return sinks[index]

    return ShardLayout(
        dtype=np.dtype(dtype),
        chunk_rows=CHUNK_ROWS,
        chunks_per_shard=CHUNKS_PER_SHARD,
        trailing_shape=trailing,
        length=length,
        zstd_level=level,
        fill_value=fill,
        open_sink=open_sink,
    )


def make_values(length, dtype=np.float32, trailing=(), seed=0):
    rng = np.random.default_rng(seed)
    shape = (length, *trailing)
    if np.issubdtype(np.dtype(dtype), np.floating):
        return rng.integers(-500, 500, shape).astype(dtype)
    return rng.integers(0, 500, shape).astype(dtype)


def split_shard(data, slots=CHUNKS_PER_SHARD):
    """Return the (offset, length) table and the checksum bytes of a shard."""
    size = 16 * slots
    table = np.frombuffer(data[-size - 4 : -4], dtype="<u8").reshape(slots, 2)
    return table, data[-4:]


# ChunkEncoder


def test_encoder_returns_zstd_of_little_endian_bytes():
    layout = make_layout({}, SHARD_ROWS)
    rows = make_values(CHUNK_ROWS)
    expected = Zstd(level=3, checksum=False).encode(rows.tobytes())
    assert ChunkEncoder(layout).encode(rows) == expected


def test_encoder_stores_big_endian_input_little_endian():
    layout = make_layout({}, SHARD_ROWS)
    rows = make_values(CHUNK_ROWS)
    big = rows.astype(">f4")
    assert ChunkEncoder(layout).encode(big) == ChunkEncoder(layout).encode(rows)


def test_encoder_pads_a_short_chunk_with_the_fill_value():
    layout = make_layout({}, SHARD_ROWS)
    rows = make_values(10)
    padded = np.full(CHUNK_ROWS, np.nan, dtype=np.float32)
    padded[:10] = rows
    expected = Zstd(level=3, checksum=False).encode(padded.tobytes())
    assert ChunkEncoder(layout).encode(rows) == expected


def test_encoder_pads_an_integer_chunk_with_zero():
    layout = make_layout({}, SHARD_ROWS, dtype=np.uint32, fill=0)
    rows = make_values(5, np.uint32)
    padded = np.zeros(CHUNK_ROWS, dtype=np.uint32)
    padded[:5] = rows
    expected = Zstd(level=3, checksum=False).encode(padded.tobytes())
    assert ChunkEncoder(layout).encode(rows) == expected


def test_encoder_pads_a_wide_row_chunk():
    layout = make_layout({}, SHARD_ROWS, trailing=(2,))
    rows = make_values(7, trailing=(2,))
    padded = np.full((CHUNK_ROWS, 2), np.nan, dtype=np.float32)
    padded[:7] = rows
    expected = Zstd(level=3, checksum=False).encode(padded.tobytes())
    assert ChunkEncoder(layout).encode(rows) == expected


def test_encoder_encodes_a_strided_view_as_its_rows():
    layout = make_layout({}, SHARD_ROWS)
    wide = make_values(CHUNK_ROWS * 3).reshape(CHUNK_ROWS, 3)
    column = wide[:, 1]
    assert not column.flags.c_contiguous
    encoded = ChunkEncoder(layout).encode(column)
    assert encoded == ChunkEncoder(layout).encode(column.copy())


def test_encoder_rejects_more_rows_than_a_chunk():
    layout = make_layout({}, SHARD_ROWS)
    with pytest.raises(ValueError, match="at most 64 rows"):
        ChunkEncoder(layout).encode(make_values(CHUNK_ROWS + 1))


def test_encoder_rejects_another_row_shape():
    layout = make_layout({}, SHARD_ROWS, trailing=(2,))
    with pytest.raises(ValueError, match=r"row has shape \(2,\)"):
        ChunkEncoder(layout).encode(make_values(CHUNK_ROWS))


# ShardWriter


def test_shard_writer_appends_chunks_then_index_and_checksum():
    sink = ListSink()
    writer = ShardWriter(sink, 4)
    chunks = [b"a" * 5, b"b" * 7, b"c" * 3, b"d" * 11]
    for chunk in chunks:
        writer.append(chunk)
    writer.close()
    data = sink.data
    assert data.startswith(b"".join(chunks))
    table, checksum = split_shard(data)
    assert table.tolist() == [[0, 5], [5, 7], [12, 3], [15, 11]]
    index_bytes = data[-16 * 4 - 4 : -4]
    assert checksum == struct.pack("<I", google_crc32c.value(index_bytes))
    assert len(data) == 26 + 16 * 4 + 4
    assert sink.closed


def test_shard_writer_marks_the_slots_after_the_last_chunk_missing():
    sink = ListSink()
    writer = ShardWriter(sink, 4)
    writer.append(b"xx")
    writer.append(b"yyy")
    writer.close()
    table, _ = split_shard(sink.data)
    assert table.tolist() == [
        [0, 2],
        [2, 3],
        [MISSING, MISSING],
        [MISSING, MISSING],
    ]


def test_shard_writer_rejects_a_chunk_past_the_last_slot():
    writer = ShardWriter(ListSink(), 1)
    writer.append(b"x")
    with pytest.raises(ValueError, match="at most 1 chunks"):
        writer.append(b"y")


# Sinks


def test_file_sink_creates_parent_directories_on_the_first_append(tmp_path):
    path = tmp_path / "a" / "c" / "0"
    sink = FileShardSink(path)
    assert not path.parent.exists()
    sink.append(b"ab")
    sink.append(b"cd")
    sink.close()
    assert path.read_bytes() == b"abcd"


def test_file_sink_replaces_an_existing_file(tmp_path):
    path = tmp_path / "0"
    path.write_bytes(b"old old old")
    sink = FileShardSink(path)
    sink.append(b"new")
    sink.close()
    assert path.read_bytes() == b"new"


def test_file_sink_writes_nothing_before_an_append(tmp_path):
    path = tmp_path / "0"
    sink = FileShardSink(path)
    sink.close()
    assert not path.exists()


def test_file_sink_holds_no_open_file_between_appends(tmp_path):
    limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    in_use = len(list(Path("/dev/fd").iterdir()))
    resource.setrlimit(resource.RLIMIT_NOFILE, (in_use + 32, limit[1]))
    try:
        sinks = [FileShardSink(tmp_path / str(index)) for index in range(128)]
        for sink in sinks:
            sink.append(b"ab")
        for sink in sinks:
            sink.append(b"cd")
        for sink in sinks:
            sink.close()
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, limit)
    assert all(
        (tmp_path / str(index)).read_bytes() == b"abcd" for index in range(128)
    )


def test_buffer_sink_saves_the_joined_bytes_once_on_close():
    saved = []
    sink = BufferShardSink(saved.append)
    sink.append(b"ab")
    sink.append(b"cd")
    assert saved == []
    sink.close()
    assert saved == [b"abcd"]


def test_buffer_sink_shard_equals_file_sink_shard(tmp_path):
    values = make_values(SHARD_ROWS + 10)
    saved = {}

    def buffer_sink(index):
        return BufferShardSink(lambda data: saved.__setitem__(index, data))

    def file_sink(index):
        return FileShardSink(tmp_path / str(index))

    for factory in (buffer_sink, file_sink):
        layout = ShardLayout(
            dtype=np.dtype(np.float32),
            chunk_rows=CHUNK_ROWS,
            chunks_per_shard=CHUNKS_PER_SHARD,
            trailing_shape=(),
            length=values.shape[0],
            zstd_level=3,
            fill_value=float("nan"),
            open_sink=factory,
        )
        writer = ArrayWriter(layout)
        writer.push(values)
        writer.finish()
    assert set(saved) == {0, 1}
    for index, data in saved.items():
        assert (tmp_path / str(index)).read_bytes() == data


# ArrayWriter


def written_shards(values, block_lengths=None, executor=None, **kwargs):
    sinks = {}
    layout = make_layout(
        sinks,
        values.shape[0],
        dtype=values.dtype,
        trailing=values.shape[1:],
        **kwargs,
    )
    writer = ArrayWriter(layout, executor)
    if block_lengths is None:
        writer.push(values)
    else:
        start = 0
        for length in block_lengths:
            writer.push(values[start : start + length])
            start += length
        assert start == values.shape[0]
    writer.finish()
    return {index: sink.data for index, sink in sinks.items()}


def test_array_writer_cuts_blocks_into_whole_chunks():
    values = make_values(SHARD_ROWS * 2 + 100)
    whole = written_shards(values)
    ragged = written_shards(values, [1, 63, 64, 65, 200, 0, 2, 100, 117])
    assert ragged == whole


def test_array_writer_gives_a_row_at_a_time_the_same_shards():
    values = make_values(SHARD_ROWS + 70)
    whole = written_shards(values)
    singles = written_shards(values, [1] * values.shape[0])
    assert singles == whole


def test_array_writer_writes_one_shard_for_each_full_shard_and_the_tail():
    shards = written_shards(make_values(SHARD_ROWS * 2 + 100))
    assert sorted(shards) == [0, 1, 2]


def test_array_writer_marks_chunks_past_the_end_missing():
    shards = written_shards(make_values(SHARD_ROWS + CHUNK_ROWS + 5))
    table, _ = split_shard(shards[1])
    assert (table[:2, 0] != MISSING).all()
    assert (table[2:] == MISSING).all()


def test_array_writer_pads_the_last_chunk_with_the_fill_value():
    values = make_values(CHUNK_ROWS + 3)
    shards = written_shards(values)
    table, _ = split_shard(shards[0])
    offset, length = table[1]
    stored = Zstd(level=3, checksum=False).decode(
        shards[0][int(offset) : int(offset + length)]
    )
    rows = np.frombuffer(stored, dtype="<f4")
    assert rows[:3].tolist() == values[CHUNK_ROWS:].tolist()
    assert np.isnan(rows[3:]).all()


def test_array_writer_closes_a_shard_when_its_last_chunk_arrives():
    sinks = {}
    layout = make_layout(sinks, SHARD_ROWS * 2)
    writer = ArrayWriter(layout)
    writer.push(make_values(SHARD_ROWS - 1))
    assert not sinks[0].closed
    writer.push(make_values(1))
    assert sinks[0].closed
    assert 1 not in sinks


def test_array_writer_opens_a_shard_with_its_first_complete_chunk():
    sinks = {}
    layout = make_layout(sinks, SHARD_ROWS * 2)
    writer = ArrayWriter(layout)
    writer.push(make_values(CHUNK_ROWS - 1))
    assert sinks == {}
    writer.push(make_values(1))
    assert len(sinks[0].parts) == 1


def test_array_writer_with_no_rows_writes_no_shard():
    sinks = {}
    writer = ArrayWriter(make_layout(sinks, 0))
    writer.finish()
    assert sinks == {}


def test_array_writer_with_a_whole_number_of_shards_writes_no_extra_shard():
    shards = written_shards(make_values(SHARD_ROWS * 2))
    assert sorted(shards) == [0, 1]


def test_array_writer_handles_rows_with_a_trailing_axis():
    values = make_values(SHARD_ROWS + 30, trailing=(2,))
    whole = written_shards(values)
    ragged = written_shards(values, [7, 100, 1, 150, 28])
    assert ragged == whole


def test_array_writer_encodes_on_an_executor_to_the_same_bytes():
    values = make_values(SHARD_ROWS * 6 + 9)
    inline = written_shards(values)
    with ThreadPoolExecutor(4) as pool:
        pooled = written_shards(values, executor=pool)
        pooled_ragged = written_shards(
            values, [5, 700, 100, 3, 700, 37], executor=pool
        )
    assert pooled == inline
    assert pooled_ragged == inline


def test_array_writer_runs_a_large_chunk_through_groups(monkeypatch):
    monkeypatch.setattr("timeseries_zarr.shards.ENCODE_GROUP_BYTES", 1)
    values = make_values(SHARD_ROWS * 3 + 1)
    inline = written_shards(values)
    with ThreadPoolExecutor(2) as pool:
        assert written_shards(values, executor=pool) == inline


def test_array_writer_rejects_another_dtype():
    writer = ArrayWriter(make_layout({}, SHARD_ROWS))
    with pytest.raises(ValueError, match="dtype float32, got float64"):
        writer.push(np.zeros(4, dtype=np.float64))


def test_array_writer_rejects_another_row_shape():
    writer = ArrayWriter(make_layout({}, SHARD_ROWS, trailing=(2,)))
    with pytest.raises(ValueError, match=r"row has shape \(2,\), got \(3,\)"):
        writer.push(np.zeros((4, 3), dtype=np.float32))


def test_array_writer_rejects_rows_past_the_length():
    writer = ArrayWriter(make_layout({}, 10))
    with pytest.raises(ValueError, match="has 10 rows, got 11"):
        writer.push(np.zeros(11, dtype=np.float32))


def test_array_writer_rejects_a_finish_short_of_the_length():
    writer = ArrayWriter(make_layout({}, 10))
    writer.push(np.zeros(4, dtype=np.float32))
    with pytest.raises(ValueError, match="has 10 rows, got 4"):
        writer.finish()


# Byte identity with zarr

IDENTITY_LENGTHS = [
    1,
    CHUNK_ROWS - 1,
    CHUNK_ROWS,
    CHUNK_ROWS + 1,
    SHARD_ROWS - 1,
    SHARD_ROWS,
    SHARD_ROWS + 1,
    SHARD_ROWS * 2 + CHUNK_ROWS * 2 + 17,
    SHARD_ROWS * 3,
]
IDENTITY_CASES = [
    pytest.param(np.float32, (), id="float32"),
    pytest.param(np.uint32, (), id="uint32"),
    pytest.param(np.float32, (2,), id="float32-by-2"),
]


def shard_key(index, trailing):
    return "/".join([str(index), *["0"] * len(trailing)])


def file_layout(array_dir: Path, length, dtype, trailing, level):
    fill = float("nan") if np.issubdtype(dtype, np.floating) else 0

    def open_sink(index):
        return FileShardSink(array_dir / "c" / shard_key(index, trailing))

    return ShardLayout(
        dtype=np.dtype(dtype),
        chunk_rows=CHUNK_ROWS,
        chunks_per_shard=CHUNKS_PER_SHARD,
        trailing_shape=trailing,
        length=length,
        zstd_level=level,
        fill_value=fill,
        open_sink=open_sink,
    )


def shard_files(array_dir: Path):
    root = array_dir / "c"
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.mark.parametrize(("dtype", "trailing"), IDENTITY_CASES)
@pytest.mark.parametrize("length", IDENTITY_LENGTHS)
def test_shards_equal_the_bytes_zarr_writes(tmp_path, dtype, trailing, length):
    values = make_values(length, dtype, trailing)
    group = open_group(tmp_path / "zarr")
    chunk = (CHUNK_ROWS, *trailing)
    shard = (SHARD_ROWS, *trailing)
    array = create_array(
        group, "a", (length, *trailing), dtype, chunk, shard, {}, 3
    )
    array[:] = values

    layout = file_layout(tmp_path / "ours" / "a", length, dtype, trailing, 3)
    writer = ArrayWriter(layout)
    start = 0
    for block in (37, 500, 1, 64, 10_000):
        writer.push(values[start : start + block])
        start += block
    writer.finish()

    assert shard_files(tmp_path / "ours" / "a") == shard_files(
        tmp_path / "zarr" / "a"
    )


@pytest.mark.parametrize("level", [1, 5, 9])
def test_shards_equal_the_bytes_zarr_writes_at_each_zstd_level(tmp_path, level):
    length = SHARD_ROWS + 100
    values = make_values(length)
    group = open_group(tmp_path / "zarr")
    array = create_array(
        group,
        "a",
        (length,),
        np.float32,
        (CHUNK_ROWS,),
        (SHARD_ROWS,),
        {},
        level,
    )
    array[:] = values
    writer = ArrayWriter(
        file_layout(tmp_path / "ours" / "a", length, np.float32, (), level)
    )
    writer.push(values)
    writer.finish()
    assert shard_files(tmp_path / "ours" / "a") == shard_files(
        tmp_path / "zarr" / "a"
    )


def test_shards_equal_the_bytes_zarr_writes_for_nan_and_infinity(tmp_path):
    length = SHARD_ROWS + 50
    values = make_values(length)
    values[::7] = np.nan
    values[3::11] = np.inf
    values[5::13] = -np.inf
    group = open_group(tmp_path / "zarr")
    array = create_array(
        group, "a", (length,), np.float32, (CHUNK_ROWS,), (SHARD_ROWS,), {}, 3
    )
    array[:] = values
    writer = ArrayWriter(
        file_layout(tmp_path / "ours" / "a", length, np.float32, (), 3)
    )
    writer.push(values)
    writer.finish()
    assert shard_files(tmp_path / "ours" / "a") == shard_files(
        tmp_path / "zarr" / "a"
    )


@pytest.mark.parametrize(("dtype", "trailing"), IDENTITY_CASES)
def test_zarr_reads_back_the_shards_it_did_not_write(tmp_path, dtype, trailing):
    length = SHARD_ROWS * 2 + CHUNK_ROWS + 5
    values = make_values(length, dtype, trailing)
    group = open_group(tmp_path / "bundle")
    create_array(
        group,
        "a",
        (length, *trailing),
        dtype,
        (CHUNK_ROWS, *trailing),
        (SHARD_ROWS, *trailing),
        {},
        3,
    )
    writer = ArrayWriter(
        file_layout(tmp_path / "bundle" / "a", length, dtype, trailing, 3)
    )
    writer.push(values)
    writer.finish()
    stored = zarr.open_array(tmp_path / "bundle" / "a", mode="r")
    assert np.array_equal(stored[:], values)
