import pytest

from timeseries_zarr.constants import (
    ENVELOPE_PAIR_SIZE,
    FLOAT32_BYTES,
    INNER_CHUNK_SAMPLES,
)
from timeseries_zarr.sizing import (
    channel_writer_bytes,
    chunk_and_shard,
    chunk_shape_for_level,
    level_sizing,
    member_sizing,
    raw_sizing,
    shard_shape_for_level,
)
from timeseries_zarr.types import ChunkShard, WriteOpts


@pytest.mark.parametrize(
    ("level_shape", "inner_len", "expected"),
    [
        ((1000,), 256, (256,)),
        ((100,), 256, (100,)),
        ((256,), 256, (256,)),
        ((1000, 2), 256, (256, 2)),
        ((100, 2), 256, (100, 2)),
        ((0,), 256, (1,)),  # zero-length level -> one-bin floor
        ((0, 2), 256, (1, 2)),
    ],
)
def test_chunk_shape_for_level(level_shape, inner_len, expected):
    assert chunk_shape_for_level(level_shape, inner_len) == expected


def test_chunk_shape_for_level_default_inner_len_rank1():
    assert chunk_shape_for_level((10**9,)) == (INNER_CHUNK_SAMPLES,)


def test_chunk_shape_for_level_default_inner_len_rank2():
    assert chunk_shape_for_level((10**9, 2)) == (INNER_CHUNK_SAMPLES, 2)


@pytest.mark.parametrize(
    ("chunk_shape", "level_shape", "dtype_size", "target", "expected"),
    [
        (
            (262144,),
            (100_000_000,),
            4,
            16 * 2**20,
            (4194304,),
        ),  # target caps the shard
        ((256,), (100_000,), 4, 16 * 2**20, (100096,)),  # target exceeds array
        ((1024,), (100_000,), 4, 1024, (1024,)),  # chunk > target -> one chunk
        (
            (131072, 2),
            (100_000_000, 2),
            4,
            16 * 2**20,
            (2097152, 2),
        ),  # rank-2, 2-axis kept
        ((100,), (100,), 4, 16 * 2**20, (100,)),  # tiny -> one chunk
        ((1,), (0,), 4, 16 * 2**20, (1,)),  # zero-length level -> one chunk
    ],
)
def test_shard_shape_for_level(
    chunk_shape, level_shape, dtype_size, target, expected
):
    assert (
        shard_shape_for_level(chunk_shape, level_shape, dtype_size, target)
        == expected
    )


def test_shard_shape_for_level_default_target():
    assert shard_shape_for_level((262144,), (100_000_000,), 4) == (4194304,)


@pytest.mark.parametrize(
    ("chunk_shape", "level_shape", "dtype_size", "target"),
    [
        ((262144,), (100_000_000,), 4, 16 * 2**20),
        ((256,), (100_000,), 4, 16 * 2**20),
        ((131072, 2), (100_000_000, 2), 4, 16 * 2**20),
        ((1024,), (100_000,), 4, 1024),
    ],
)
def test_shard_shape_is_multiple_of_chunk(
    chunk_shape, level_shape, dtype_size, target
):
    shard = shard_shape_for_level(chunk_shape, level_shape, dtype_size, target)
    assert shard[0] % chunk_shape[0] == 0
    assert shard[1:] == chunk_shape[1:]


@pytest.mark.parametrize(
    ("level_shape", "dtype_size"),
    [
        ((100_000_000,), 4),
        ((100_000_000, 2), 4),
        ((1000,), 8),
        ((500, 2), 1),
    ],
)
def test_chunk_and_shard_composes(level_shape, dtype_size):
    result = chunk_and_shard(level_shape, dtype_size)
    chunk = chunk_shape_for_level(level_shape)
    shard = shard_shape_for_level(chunk, level_shape, dtype_size)
    assert result == ChunkShard(chunk_shape=chunk, shard_shape=shard)


def test_chunk_and_shard_custom_params():
    level_shape = (100_000_000,)
    result = chunk_and_shard(
        level_shape, 4, inner_len=1024, target_shard_bytes=64 * 1024
    )
    chunk = chunk_shape_for_level(level_shape, 1024)
    shard = shard_shape_for_level(chunk, level_shape, 4, 64 * 1024)
    assert result == ChunkShard(chunk_shape=chunk, shard_shape=shard)


def test_chunk_and_shard_concrete():
    result = chunk_and_shard((100_000_000,), 4)
    assert result.chunk_shape == (8192,)
    assert result.shard_shape == (4194304,)


@pytest.mark.parametrize(
    ("level_shape", "expected"),
    [
        ((0,), ChunkShard(chunk_shape=(1,), shard_shape=(1,))),
        ((0, 2), ChunkShard(chunk_shape=(1, 2), shard_shape=(1, 2))),
    ],
)
def test_chunk_and_shard_zero_length_level(level_shape, expected):
    assert chunk_and_shard(level_shape, 4) == expected


@pytest.mark.parametrize(
    ("level_shape", "dtype_size", "expected_shard"),
    [((100_000_000,), 4, (4194304,)), ((100_000_000, 2), 4, (2097152, 2))],
)
def test_chunk_and_shard_fills_the_shard_target(
    level_shape, dtype_size, expected_shard
):
    result = chunk_and_shard(level_shape, dtype_size)
    assert result.shard_shape == expected_shard
    assert result.chunk_shape[0] == INNER_CHUNK_SAMPLES


def test_level_sizing_fills_the_default_shard_target():
    result = level_sizing(100_000_000)
    assert result.chunk_shape == (INNER_CHUNK_SAMPLES, ENVELOPE_PAIR_SIZE)
    assert result.shard_shape == (2097152, ENVELOPE_PAIR_SIZE)


def test_level_sizing_matches_chunk_and_shard_of_the_minmax_shape():
    result = level_sizing(
        100_000_000, inner_len=1024, target_shard_bytes=64 * 1024
    )
    assert result == chunk_and_shard(
        (100_000_000, ENVELOPE_PAIR_SIZE),
        FLOAT32_BYTES,
        inner_len=1024,
        target_shard_bytes=64 * 1024,
    )


def test_level_sizing_caps_the_shard_at_the_chunks_the_level_holds():
    result = level_sizing(10_000)
    assert result.chunk_shape == (INNER_CHUNK_SAMPLES, ENVELOPE_PAIR_SIZE)
    assert result.shard_shape == (2 * INNER_CHUNK_SAMPLES, ENVELOPE_PAIR_SIZE)


@pytest.mark.parametrize("bins", [1, 1000, 8193, 100_000_000])
def test_level_sizing_shard_rows_are_a_multiple_of_chunk_rows(bins):
    result = level_sizing(bins)
    assert result.shard_shape[0] % result.chunk_shape[0] == 0


def test_level_sizing_never_chunks_the_pair_axis():
    result = level_sizing(100_000_000)
    assert result.chunk_shape[1] == result.shard_shape[1] == ENVELOPE_PAIR_SIZE


def test_level_sizing_empty_level_takes_one_row_chunks():
    assert level_sizing(0) == ChunkShard(
        chunk_shape=(1, ENVELOPE_PAIR_SIZE),
        shard_shape=(1, ENVELOPE_PAIR_SIZE),
    )


def test_raw_sizing_matches_chunk_and_shard_of_a_rank_one_float32_array() -> (
    None
):
    default = raw_sizing(100_000_000)
    assert default == chunk_and_shard((100_000_000,), FLOAT32_BYTES)
    assert default == ChunkShard(chunk_shape=(8192,), shard_shape=(4194304,))
    small = raw_sizing(
        100_000_000, inner_len=1024, target_shard_bytes=64 * 1024
    )
    assert small == chunk_and_shard(
        (100_000_000,),
        FLOAT32_BYTES,
        inner_len=1024,
        target_shard_bytes=64 * 1024,
    )
    assert small == ChunkShard(chunk_shape=(1024,), shard_shape=(16384,))


def test_raw_sizing_of_no_samples_takes_one_row_chunks_and_shards() -> None:
    assert raw_sizing(0) == ChunkShard(chunk_shape=(1,), shard_shape=(1,))


def test_member_sizing_drops_the_pair_axis():
    result = member_sizing(level_sizing(100_000_000))
    assert result == ChunkShard(chunk_shape=(8192,), shard_shape=(2097152,))


def test_member_sizing_leaves_a_one_dimensional_sizing_equal():
    sizing = ChunkShard(chunk_shape=(8192,), shard_shape=(65536,))
    assert member_sizing(sizing) == sizing


@pytest.mark.parametrize("bins", [1, 1000, 10_000, 100_000_000])
def test_member_sizing_shares_the_minmax_row_counts(bins):
    minmax = level_sizing(bins)
    member = member_sizing(minmax)
    assert member.chunk_shape == minmax.chunk_shape[:1]
    assert member.shard_shape == minmax.shard_shape[:1]


def test_member_sizing_differs_from_sizing_a_member_by_its_own_width():
    own_width = chunk_and_shard((100_000_000,), FLOAT32_BYTES)
    shared = member_sizing(level_sizing(100_000_000))
    assert own_width.shard_shape[0] == 2 * shared.shard_shape[0]


def test_channel_writer_bytes_at_the_defaults_with_8_levels_is_3_6_mib():
    opts = WriteOpts(max_levels=8)
    chunk = 8192
    raw = chunk * 4 + 512 * 16
    level = chunk * 40 + 256 * 16
    assert channel_writer_bytes(opts) == raw + 8 * level + 2**20
    assert 3.5 < channel_writer_bytes(opts) / 2**20 < 3.7


def test_channel_writer_bytes_without_levels_is_raw_and_the_append_bytes():
    opts = WriteOpts(max_levels=0)
    assert channel_writer_bytes(opts) == 8192 * 4 + 512 * 16 + 2**20


@pytest.mark.parametrize("max_levels", [1, 5, 15])
def test_channel_writer_bytes_grows_by_one_level_for_each_level(max_levels):
    one = channel_writer_bytes(WriteOpts(max_levels=1))
    none = channel_writer_bytes(WriteOpts(max_levels=0))
    assert channel_writer_bytes(WriteOpts(max_levels=max_levels)) == (
        none + max_levels * (one - none)
    )


def test_channel_writer_bytes_scales_with_the_chunk_length():
    def estimate(inner_len):
        opts = WriteOpts(
            inner_len=inner_len, target_shard_bytes=1, max_levels=8
        )
        return channel_writer_bytes(opts)

    assert estimate(200) - estimate(100) == 100 * (4 + 8 * 40)


def test_channel_writer_bytes_counts_one_slot_for_a_shard_of_one_chunk():
    opts = WriteOpts(inner_len=100, target_shard_bytes=1, max_levels=2)
    assert channel_writer_bytes(opts) == (
        100 * 4 + 16 + 2 * (100 * 40 + 16) + 2**20
    )


def test_channel_writer_bytes_does_not_depend_on_write_var_or_workers():
    base = channel_writer_bytes(WriteOpts())
    assert channel_writer_bytes(WriteOpts(write_var=True)) == base
    assert channel_writer_bytes(WriteOpts(max_workers=3)) == base
