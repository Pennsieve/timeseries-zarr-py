import struct
import threading

import google_crc32c
import numpy as np
import pytest
from numcodecs.zstd import Zstd

from timeseries_zarr.shard_read import (
    _frame_content_size,
    chunk_span,
    decode_chunks,
    parse_index,
)
from timeseries_zarr.shards import (
    BufferShardSink,
    ChunkEncoder,
    ShardLayout,
    ShardWriter,
)

MISSING = 2**64 - 1
CHUNK_ROWS = 64
SLOTS = 4
PATH = "channels/0/raw"
SHARD = 7


def make_encoder(
    dtype=np.float32, trailing=(), fill=float("nan"), rows=CHUNK_ROWS
):
    layout = ShardLayout(
        dtype=np.dtype(dtype),
        chunk_rows=rows,
        chunks_per_shard=SLOTS,
        trailing_shape=trailing,
        length=rows * SLOTS,
        zstd_level=3,
        fill_value=fill,
        open_sink=lambda index: BufferShardSink(lambda data: None),
    )
    return ChunkEncoder(layout)


def write_shard(chunks, slots=SLOTS):
    """Return the bytes of a shard that ShardWriter wrote from encoded chunks."""
    saved = []
    writer = ShardWriter(BufferShardSink(saved.append), slots)
    for chunk in chunks:
        writer.append(chunk)
    writer.close()
    return saved[0]


def make_values(rows, dtype=np.float32, trailing=(), seed=0):
    rng = np.random.default_rng(seed)
    values = rng.integers(0, 500, (rows, *trailing))
    return values.astype(dtype)


def assemble(payload, entries):
    """Return a shard of payload then an index that holds entries."""
    table = np.array(entries, dtype="<u8").reshape(-1, 2).tobytes()
    return payload + table + struct.pack("<I", google_crc32c.value(table))


def index_of(shard, slots=SLOTS):
    size = 16 * slots + 4
    return parse_index(shard[-size:], slots, PATH, SHARD)


def decode(
    shard, first, stop, out, fill=float("nan"), rows=CHUNK_ROWS, slots=SLOTS
):
    index = index_of(shard, slots)
    span = chunk_span(index, first, stop)
    start, end = span if span is not None else (0, 0)
    decode_chunks(
        shard[start:end],
        start,
        index,
        first,
        stop,
        out,
        fill,
        rows,
        PATH,
        SHARD,
    )


# parse_index


def test_parse_index_returns_the_pairs_of_a_full_shard():
    encoder = make_encoder()
    chunks = [
        encoder.encode(make_values(CHUNK_ROWS, seed=k)) for k in range(SLOTS)
    ]
    index = index_of(write_shard(chunks))
    offsets = np.cumsum([0] + [len(chunk) for chunk in chunks[:-1]])
    assert index.dtype == np.uint64
    assert index.shape == (SLOTS, 2)
    assert index[:, 0].tolist() == offsets.tolist()
    assert index[:, 1].tolist() == [len(chunk) for chunk in chunks]


def test_parse_index_marks_the_missing_slots_of_a_last_shard():
    encoder = make_encoder()
    chunks = [encoder.encode(make_values(CHUNK_ROWS, seed=k)) for k in range(2)]
    index = index_of(write_shard(chunks))
    assert index[2:].tolist() == [[MISSING, MISSING]] * 2
    assert index[:2, 1].tolist() == [len(chunk) for chunk in chunks]


def test_parse_index_reads_a_shard_of_one_chunk():
    chunk = make_encoder().encode(make_values(CHUNK_ROWS))
    index = index_of(write_shard([chunk], slots=1), slots=1)
    assert index.tolist() == [[0, len(chunk)]]


def test_parse_index_returns_a_read_only_array_that_owns_its_data():
    shard = write_shard([make_encoder().encode(make_values(CHUNK_ROWS))])
    raw = bytearray(shard[-(16 * SLOTS + 4) :])
    index = parse_index(raw, SLOTS, PATH, SHARD)
    before = index.copy()
    raw[:] = bytes(len(raw))
    assert not index.flags.writeable
    assert np.array_equal(index, before)


def test_parse_index_rejects_a_wrong_checksum():
    shard = write_shard([make_encoder().encode(make_values(CHUNK_ROWS))])
    data = bytearray(shard[-(16 * SLOTS + 4) :])
    data[0] ^= 1
    with pytest.raises(
        ValueError, match=rf"shard {SHARD} of array {PATH}.*CRC-32C"
    ):
        parse_index(bytes(data), SLOTS, PATH, SHARD)


def test_parse_index_rejects_an_index_one_byte_short():
    shard = write_shard([make_encoder().encode(make_values(CHUNK_ROWS))])
    data = shard[-(16 * SLOTS + 4) + 1 :]
    with pytest.raises(
        ValueError,
        match=rf"shard {SHARD} of array {PATH} has 67 bytes, expected 68",
    ):
        parse_index(data, SLOTS, PATH, SHARD)


def test_parse_index_rejects_an_index_one_byte_long():
    shard = write_shard([make_encoder().encode(make_values(CHUNK_ROWS))])
    data = shard[-(16 * SLOTS + 4) - 1 :]
    with pytest.raises(
        ValueError,
        match=rf"shard {SHARD} of array {PATH} has 69 bytes, expected 68",
    ):
        parse_index(data, SLOTS, PATH, SHARD)


def test_parse_index_rejects_an_entry_that_ends_past_the_uint64_range():
    shard = assemble(b"", [(2**63, 2**63), (MISSING, MISSING)])
    with pytest.raises(
        ValueError, match=rf"shard {SHARD} of array {PATH}.*ends past"
    ):
        parse_index(shard, 2, PATH, SHARD)


# chunk_span


def test_chunk_span_returns_none_for_slots_without_a_chunk():
    chunk = make_encoder().encode(make_values(CHUNK_ROWS))
    index = index_of(write_shard([chunk]))
    assert chunk_span(index, 1, SLOTS) is None
    assert chunk_span(index, 2, 2) is None


def test_chunk_span_returns_the_range_of_one_chunk():
    encoder = make_encoder()
    chunks = [encoder.encode(make_values(CHUNK_ROWS, seed=k)) for k in range(3)]
    index = index_of(write_shard(chunks))
    start = len(chunks[0])
    assert chunk_span(index, 1, 2) == (start, start + len(chunks[1]))


def test_chunk_span_returns_the_range_of_a_run_of_chunks():
    encoder = make_encoder()
    chunks = [
        encoder.encode(make_values(CHUNK_ROWS, seed=k)) for k in range(SLOTS)
    ]
    index = index_of(write_shard(chunks))
    assert chunk_span(index, 1, 4) == (len(chunks[0]), sum(map(len, chunks)))


def test_chunk_span_skips_a_missing_slot_in_the_middle_of_a_run():
    payload = b"a" * 10 + b"b" * 20
    shard = assemble(payload, [(0, 10), (MISSING, MISSING), (10, 20)])
    index = parse_index(shard[len(payload) :], 3, PATH, SHARD)
    assert chunk_span(index, 0, 3) == (0, 30)
    assert chunk_span(index, 1, 3) == (10, 30)
    assert chunk_span(index, 1, 2) is None


def test_chunk_span_covers_chunks_out_of_slot_order():
    shard = assemble(b"x" * 60, [(40, 20), (0, 10), (10, 30)])
    index = parse_index(shard[60:], 3, PATH, SHARD)
    assert chunk_span(index, 0, 3) == (0, 60)
    assert chunk_span(index, 0, 1) == (40, 60)


def test_chunk_span_covers_overlapping_chunks():
    shard = assemble(b"x" * 30, [(0, 20), (10, 20)])
    index = parse_index(shard[30:], 2, PATH, SHARD)
    assert chunk_span(index, 0, 2) == (0, 30)


@pytest.mark.parametrize(("first", "stop"), [(-1, 2), (3, 2), (0, SLOTS + 1)])
def test_chunk_span_rejects_slots_outside_the_index(first, stop):
    index = index_of(
        write_shard([make_encoder().encode(make_values(CHUNK_ROWS))])
    )
    with pytest.raises(ValueError, match="must lie within"):
        chunk_span(index, first, stop)


# decode_chunks


@pytest.mark.parametrize(
    ("dtype", "trailing"),
    [(np.float32, ()), (np.float32, (2,)), (np.uint32, ())],
)
def test_decode_matches_numcodecs_decode(dtype, trailing):
    encoder = make_encoder(dtype, trailing, fill=0)
    values = [
        make_values(CHUNK_ROWS, dtype, trailing, seed=k) for k in range(SLOTS)
    ]
    shard = write_shard([encoder.encode(rows) for rows in values])
    out = np.zeros((SLOTS * CHUNK_ROWS, *trailing), dtype=dtype)
    decode(shard, 0, SLOTS, out, fill=0)
    index = index_of(shard)
    for slot, rows in enumerate(values):
        offset, length = (int(v) for v in index[slot])
        expected = np.frombuffer(
            Zstd().decode(shard[offset : offset + length]), dtype=dtype
        ).reshape(rows.shape)
        got = out[slot * CHUNK_ROWS : (slot + 1) * CHUNK_ROWS]
        assert got.tobytes() == expected.tobytes() == rows.tobytes()


def test_decode_fills_a_missing_float_chunk_with_nan():
    shard = write_shard([make_encoder().encode(make_values(CHUNK_ROWS))])
    out = np.zeros(SLOTS * CHUNK_ROWS, dtype=np.float32)
    decode(shard, 0, SLOTS, out)
    assert np.isnan(out[CHUNK_ROWS:]).all()
    assert not np.isnan(out[:CHUNK_ROWS]).any()


def test_decode_fills_a_missing_count_chunk_with_zero():
    encoder = make_encoder(np.uint32, fill=0)
    shard = write_shard([encoder.encode(make_values(CHUNK_ROWS, np.uint32))])
    out = np.ones(SLOTS * CHUNK_ROWS, dtype=np.uint32)
    decode(shard, 0, SLOTS, out, fill=0)
    assert (out[CHUNK_ROWS:] == 0).all()
    assert (out[:CHUNK_ROWS] == make_values(CHUNK_ROWS, np.uint32)).all()


def test_decode_fills_a_run_that_holds_no_chunk():
    shard = write_shard([])
    out = np.zeros(2 * CHUNK_ROWS, dtype=np.float32)
    decode(shard, 1, 3, out)
    assert np.isnan(out).all()


def test_decode_reads_a_run_from_data_that_starts_inside_the_shard():
    encoder = make_encoder()
    values = [make_values(CHUNK_ROWS, seed=k) for k in range(SLOTS)]
    shard = write_shard([encoder.encode(rows) for rows in values])
    out = np.zeros(2 * CHUNK_ROWS, dtype=np.float32)
    decode(shard, 1, 3, out)
    assert out.tobytes() == np.concatenate(values[1:3]).tobytes()


def test_decode_reads_chunks_out_of_slot_order():
    encoder = make_encoder()
    values = [make_values(CHUNK_ROWS, seed=k) for k in range(3)]
    frames = [encoder.encode(rows) for rows in values]
    payload = frames[2] + frames[0] + frames[1]
    entries = [
        (len(frames[2]), len(frames[0])),
        (len(frames[2]) + len(frames[0]), len(frames[1])),
        (0, len(frames[2])),
    ]
    shard = assemble(payload, entries)
    out = np.zeros(3 * CHUNK_ROWS, dtype=np.float32)
    decode(shard, 0, 3, out, slots=3)
    assert out.tobytes() == np.concatenate(values).tobytes()


def test_decode_reads_overlapping_chunks():
    frame = make_encoder().encode(make_values(CHUNK_ROWS))
    shard = assemble(frame, [(0, len(frame)), (0, len(frame))])
    out = np.zeros(2 * CHUNK_ROWS, dtype=np.float32)
    decode(shard, 0, 2, out, slots=2)
    assert out[:CHUNK_ROWS].tobytes() == out[CHUNK_ROWS:].tobytes()
    assert out[:CHUNK_ROWS].tobytes() == make_values(CHUNK_ROWS).tobytes()


def test_decode_writes_only_the_rows_of_the_run():
    encoder = make_encoder()
    shard = write_shard(
        [encoder.encode(make_values(CHUNK_ROWS, seed=k)) for k in range(2)]
    )
    out = np.full(2 * CHUNK_ROWS, -1, dtype=np.float32)
    index = index_of(shard)
    decode_chunks(
        shard[: int(index[0, 1])],
        0,
        index,
        0,
        1,
        out[:CHUNK_ROWS],
        0.0,
        CHUNK_ROWS,
        PATH,
        SHARD,
    )
    assert (out[CHUNK_ROWS:] == -1).all()


@pytest.mark.parametrize("delta", [-1, 1])
def test_decode_rejects_a_chunk_of_one_row_too_few_or_too_many(delta):
    encoder = make_encoder(rows=CHUNK_ROWS + delta)
    frame = encoder.encode(make_values(CHUNK_ROWS + delta))
    shard = assemble(frame, [(0, len(frame))])
    out = np.zeros(CHUNK_ROWS, dtype=np.float32)
    with pytest.raises(
        ValueError,
        match=rf"chunk 0 of shard {SHARD} of array {PATH}.*decodes to",
    ):
        decode(shard, 0, 1, out, slots=1)
    assert (out == 0).all()


def test_decode_reads_a_one_byte_content_size():
    encoder = make_encoder(rows=16)
    values = make_values(16)
    frame = encoder.encode(values)
    assert _frame_content_size(memoryview(frame)) == 64
    shard = assemble(frame, [(0, len(frame))])
    out = np.zeros(16, dtype=np.float32)
    decode(shard, 0, 1, out, rows=16, slots=1)
    assert out.tobytes() == values.tobytes()


def raw_frame(data):
    """Return a Zstd frame of one raw block, with no declared content size."""
    header = (len(data) << 3) | 1
    return bytes.fromhex("28b52ffd0058") + header.to_bytes(3, "little") + data


def test_decode_reads_a_frame_with_no_declared_content_size():
    values = make_values(CHUNK_ROWS)
    frame = raw_frame(values.tobytes())
    assert _frame_content_size(memoryview(frame)) is None
    shard = assemble(frame, [(0, len(frame))])
    out = np.zeros(CHUNK_ROWS, dtype=np.float32)
    decode(shard, 0, 1, out, slots=1)
    assert out.tobytes() == values.tobytes()


def test_decode_rejects_a_frame_with_no_declared_size_and_the_wrong_length():
    frame = raw_frame(make_values(CHUNK_ROWS - 1).tobytes())
    shard = assemble(frame, [(0, len(frame))])
    out = np.zeros(CHUNK_ROWS, dtype=np.float32)
    with pytest.raises(ValueError, match="decodes to 252 bytes, expected 256"):
        decode(shard, 0, 1, out, slots=1)


@pytest.mark.parametrize("cut", [1, 5, 100])
def test_decode_rejects_a_frame_cut_short(cut):
    frame = make_encoder().encode(make_values(CHUNK_ROWS))
    short = frame[:-cut]
    shard = assemble(short, [(0, len(short))])
    out = np.zeros(CHUNK_ROWS, dtype=np.float32)
    with pytest.raises(
        ValueError, match=rf"chunk 0 of shard {SHARD} of array {PATH}.*Zstd"
    ):
        decode(shard, 0, 1, out, slots=1)


def test_decode_rejects_bytes_that_are_not_a_frame():
    shard = assemble(b"z" * 40, [(0, 40)])
    out = np.zeros(CHUNK_ROWS, dtype=np.float32)
    with pytest.raises(ValueError, match="does not decode as Zstd"):
        decode(shard, 0, 1, out, slots=1)


def test_decode_rejects_a_chunk_that_runs_past_the_data():
    frame = make_encoder().encode(make_values(CHUNK_ROWS))
    index = index_of(
        assemble(frame, [(0, len(frame) + 10)] + [(MISSING, MISSING)] * 3)
    )
    out = np.zeros(CHUNK_ROWS, dtype=np.float32)
    with pytest.raises(
        ValueError,
        match=rf"chunk 0 of shard {SHARD} of array {PATH}.*cut short",
    ):
        decode_chunks(frame, 0, index, 0, 1, out, 0.0, CHUNK_ROWS, PATH, SHARD)


def test_decode_rejects_a_chunk_that_starts_before_the_data():
    frame = make_encoder().encode(make_values(CHUNK_ROWS))
    shard = assemble(frame, [(0, len(frame))] + [(MISSING, MISSING)] * 3)
    index = index_of(shard)
    out = np.zeros(CHUNK_ROWS, dtype=np.float32)
    with pytest.raises(ValueError, match="lies at bytes"):
        decode_chunks(frame, 5, index, 0, 1, out, 0.0, CHUNK_ROWS, PATH, SHARD)


@pytest.mark.parametrize(
    "make_out",
    [
        lambda: np.zeros(CHUNK_ROWS + 1, dtype=np.float32),
        lambda: np.zeros(2 * CHUNK_ROWS, dtype=np.float32)[::2],
        lambda: np.frombuffer(bytes(4 * CHUNK_ROWS), dtype=np.float32),
        lambda: np.zeros(CHUNK_ROWS, dtype=">f4"),
    ],
    ids=["rows", "strided", "read-only", "big-endian"],
)
def test_decode_rejects_an_out_array_that_does_not_fit(make_out):
    shard = write_shard([make_encoder().encode(make_values(CHUNK_ROWS))])
    index = index_of(shard)
    with pytest.raises(ValueError, match="out must be"):
        decode_chunks(
            shard, 0, index, 0, 1, make_out(), 0.0, CHUNK_ROWS, PATH, SHARD
        )


def test_decode_rejects_slots_outside_the_index():
    shard = write_shard([make_encoder().encode(make_values(CHUNK_ROWS))])
    index = index_of(shard)
    out = np.zeros(CHUNK_ROWS, dtype=np.float32)
    with pytest.raises(ValueError, match="must lie within"):
        decode_chunks(shard, 0, index, 4, 5, out, 0.0, CHUNK_ROWS, PATH, SHARD)


def test_decode_runs_on_several_threads_into_separate_arrays():
    encoder = make_encoder()
    values = [make_values(CHUNK_ROWS, seed=k) for k in range(SLOTS)]
    shard = write_shard([encoder.encode(rows) for rows in values])
    expected = np.concatenate(values).tobytes()
    results = []

    def work():
        out = np.zeros(SLOTS * CHUNK_ROWS, dtype=np.float32)
        decode(shard, 0, SLOTS, out)
        results.append(out.tobytes())

    threads = [threading.Thread(target=work) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results == [expected] * 8


# frame header


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("28b52ffd2007", 7),
        ("28b52ffd600100", 257),
        ("28b52ffda000000100", 65536),
        ("28b52ffde00000000001000000", 2**32),
        ("28b52ffd0058", None),
        ("28b52ffd4158090500", 5 + 256),
        ("28b52ffd62aabb0100", 257),
        ("28b52ffd63aabbccdd0100", 257),
        ("28b52ffd20", None),
        ("28b52ffde00000", None),
        ("0102030405060708", None),
        ("28b52f", None),
    ],
)
def test_frame_content_size_reads_every_header_form(header, expected):
    frame = bytes.fromhex(header)
    assert _frame_content_size(memoryview(frame)) == expected
