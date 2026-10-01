import numpy as np
import pytest

from timeseries_zarr.constants import (
    COUNT_COL,
    MAX_COL,
    MEAN_COL,
    MIN_COL,
    STAT_COLUMNS,
    VALID_COL,
)
from timeseries_zarr.fold import fold_raw_block, fold_stat_block
from timeseries_zarr.planning import bin_counts
from timeseries_zarr.streaming import (
    _rebuffer_and_fold,
    iter_array_blocks,
    iter_level_stat_blocks,
    iter_offset_removed_blocks,
    iter_raw_blocks,
    iter_raw_to_level1,
)


def _split(arr, sizes):
    blocks = []
    start = 0
    for s in sizes:
        blocks.append(arr[start : start + s])
        start += s
    return blocks


def _random_sizes(rng, total):
    sizes = []
    remaining = total
    while remaining > 0:
        s = int(rng.integers(1, min(remaining, 7) + 1))
        sizes.append(s)
        remaining -= s
    return sizes


def _folded(blocks, fold_fn):
    out = list(_rebuffer_and_fold(blocks, fold_fn))
    return (
        np.concatenate(out, axis=0)
        if out
        else np.empty((0, STAT_COLUMNS), dtype=np.float64)
    )


@pytest.mark.parametrize("seed", range(5))
def test_rebuffer_and_fold_raw_is_split_invariant(seed):
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal(37).astype(np.float32)
    blocks = _split(arr, _random_sizes(rng, arr.shape[0]))
    assert np.array_equal(_folded(blocks, fold_raw_block), fold_raw_block(arr))


@pytest.mark.parametrize("seed", range(5))
def test_rebuffer_and_fold_stats_is_split_invariant(seed):
    rng = np.random.default_rng(seed + 100)
    arr = np.empty((29, STAT_COLUMNS), dtype=np.float64)
    arr[:, MIN_COL] = rng.standard_normal(29)
    arr[:, MAX_COL] = arr[:, MIN_COL] + 1.0
    arr[:, MEAN_COL] = rng.standard_normal(29)
    arr[:, COUNT_COL] = 4.0
    arr[:, VALID_COL] = 4.0
    blocks = _split(arr, _random_sizes(rng, arr.shape[0]))
    assert np.allclose(_folded(blocks, fold_stat_block), fold_stat_block(arr))


def test_rebuffer_and_fold_specific_splits_match_whole():
    arr = np.arange(20, dtype=np.float32)
    for sizes in ([20], [1] * 20, [5, 5, 5, 5], [3, 7, 1, 9], [1, 19]):
        blocks = _split(arr, sizes)
        assert np.array_equal(
            _folded(blocks, fold_raw_block), fold_raw_block(arr)
        )


@pytest.mark.parametrize("seed", range(5))
def test_rebuffer_and_fold_carry_stays_below_group(seed):
    rng = np.random.default_rng(seed + 200)
    arr = rng.standard_normal(53).astype(np.float32)
    group = 4
    sizes = _random_sizes(rng, arr.shape[0])
    max_block = max(sizes)
    seen_lengths = []

    def recording_fold(block):
        seen_lengths.append(block.shape[0])
        return fold_raw_block(block)

    out = list(
        _rebuffer_and_fold(_split(arr, sizes), recording_fold, group=group)
    )
    # Every fold but the final flush sees a whole multiple of group.
    for length in seen_lengths[:-1]:
        assert length % group == 0
    assert max(seen_lengths) <= max_block + group - 1
    assert np.array_equal(
        np.concatenate(out, axis=0)
        if out
        else np.empty((0, STAT_COLUMNS), np.float64),
        fold_raw_block(arr),
    )


def test_rebuffer_and_fold_final_carry_is_group_minus_one():
    # Splitting [4, 3] with group=4 carries group - 1 rows into the final flush.
    arr = np.arange(7, dtype=np.float32)
    group = 4
    seen_lengths = []

    def recording_fold(block):
        seen_lengths.append(block.shape[0])
        return fold_raw_block(block)

    list(_rebuffer_and_fold(_split(arr, [4, 3]), recording_fold, group=group))
    assert seen_lengths == [4, group - 1]


def test_rebuffer_and_fold_empty_stream_yields_nothing():
    assert list(_rebuffer_and_fold([], fold_raw_block)) == []


def test_rebuffer_and_fold_single_short_block():
    arr = np.array([3, 1], dtype=np.float32)
    out = _folded([arr], fold_raw_block)
    assert out.shape == (1, STAT_COLUMNS)
    assert np.array_equal(out, fold_raw_block(arr))


def test_iter_raw_blocks_concatenates_to_full_series(continuous_source):
    samples = np.arange(10, dtype=np.float32)
    out = list(iter_raw_blocks(continuous_source(samples), 4))
    assert np.array_equal(np.concatenate(out), samples)


def test_iter_raw_blocks_window_sizes_exact_multiple(continuous_source):
    src = continuous_source(np.arange(12, dtype=np.float32))
    assert [b.shape[0] for b in iter_raw_blocks(src, 4)] == [4, 4, 4]


def test_iter_raw_blocks_window_sizes_with_remainder(continuous_source):
    src = continuous_source(np.arange(10, dtype=np.float32))
    assert [b.shape[0] for b in iter_raw_blocks(src, 4)] == [4, 4, 2]


def test_iter_raw_blocks_block_larger_than_n(continuous_source):
    src = continuous_source(np.arange(3, dtype=np.float32))
    assert [b.shape[0] for b in iter_raw_blocks(src, 10)] == [3]


def test_iter_raw_blocks_empty_source_yields_nothing(continuous_source):
    src = continuous_source(np.empty(0, dtype=np.float32))
    assert list(iter_raw_blocks(src, 4)) == []


@pytest.mark.parametrize("block_samples", [0, -1])
def test_iter_raw_blocks_rejects_nonpositive_block(
    continuous_source, block_samples
):
    src = continuous_source(np.arange(5, dtype=np.float32))
    with pytest.raises(ValueError, match="positive"):
        list(iter_raw_blocks(src, block_samples))


@pytest.mark.parametrize("block_samples", [1, 3, 4, 7, 16, 1000])
def test_iter_raw_to_level1_matches_whole_fold(
    continuous_source, block_samples
):
    samples = np.arange(50, dtype=np.float32)
    out = list(iter_raw_to_level1(continuous_source(samples), block_samples))
    result = (
        np.concatenate(out, axis=0)
        if out
        else np.empty((0, STAT_COLUMNS), dtype=np.float64)
    )
    assert np.array_equal(result, fold_raw_block(samples))


def test_iter_raw_to_level1_empty_source_yields_nothing(continuous_source):
    src = continuous_source(np.empty(0, dtype=np.float32))
    assert list(iter_raw_to_level1(src, 4)) == []


@pytest.mark.parametrize("block_samples", [0, -1])
def test_iter_raw_to_level1_rejects_nonpositive_block(
    continuous_source, block_samples
):
    src = continuous_source(np.arange(8, dtype=np.float32))
    with pytest.raises(ValueError, match="positive"):
        list(iter_raw_to_level1(src, block_samples))


def test_iter_array_blocks_concatenates_to_full_array():
    arr = np.arange(10, dtype=np.float32)
    out = list(iter_array_blocks(arr, 4))
    assert np.array_equal(np.concatenate(out, axis=0), arr)


def test_iter_array_blocks_window_sizes_exact_multiple():
    arr = np.arange(12, dtype=np.float32)
    assert [b.shape[0] for b in iter_array_blocks(arr, 4)] == [4, 4, 4]


def test_iter_array_blocks_window_sizes_with_remainder():
    arr = np.arange(10, dtype=np.float32)
    assert [b.shape[0] for b in iter_array_blocks(arr, 4)] == [4, 4, 2]


def test_iter_array_blocks_block_larger_than_len():
    arr = np.arange(3, dtype=np.float32)
    assert [b.shape[0] for b in iter_array_blocks(arr, 10)] == [3]


def test_iter_array_blocks_rank2_slices_axis0():
    arr = np.arange(14, dtype=np.float32).reshape(7, 2)
    out = list(iter_array_blocks(arr, 4))
    assert [b.shape for b in out] == [(4, 2), (3, 2)]
    assert np.array_equal(np.concatenate(out, axis=0), arr)


def test_iter_array_blocks_empty_array_yields_nothing():
    arr = np.empty(0, dtype=np.float32)
    assert list(iter_array_blocks(arr, 4)) == []


@pytest.mark.parametrize("block_len", [0, -1])
def test_iter_array_blocks_rejects_nonpositive_block(block_len):
    arr = np.arange(5, dtype=np.float32)
    with pytest.raises(ValueError, match="positive"):
        list(iter_array_blocks(arr, block_len))


def test_iter_offset_removed_blocks_subtracts_the_offset():
    arr = np.arange(8, dtype=np.float32)
    out = np.concatenate(list(iter_offset_removed_blocks(arr, 3.0, 4)))
    assert np.array_equal(out, np.arange(8, dtype=np.float64) - 3.0)
    assert out.dtype == np.float64


def test_iter_offset_removed_blocks_keeps_nan():
    arr = np.array([1.0, np.nan, 3.0, 4.0], dtype=np.float32)
    out = np.concatenate(list(iter_offset_removed_blocks(arr, 1.0, 2)))
    assert np.isnan(out[1])
    assert out[0] == 0.0


def test_iter_offset_removed_blocks_recovers_what_float32_would_lose():
    """Why the subtraction is float64: the signal is 1e-2 on a 5e5 offset."""
    offset = 500_000.0
    signal = np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float64)
    raw = (offset + signal).astype(np.float32)
    out = np.concatenate(list(iter_offset_removed_blocks(raw, offset, 4)))
    # float32 near 5e5 resolves about 0.03, so the samples survive only
    # because the subtraction happens before anything is summed.
    assert out.max() - out.min() > 0.0
    assert np.all(np.abs(out) < 1.0)


def test_iter_offset_removed_blocks_empty_yields_nothing():
    assert (
        list(iter_offset_removed_blocks(np.empty(0, np.float32), 1.0, 4)) == []
    )


@pytest.mark.parametrize("block_len", [0, -1])
def test_iter_offset_removed_blocks_rejects_nonpositive_block(block_len):
    with pytest.raises(ValueError, match="positive"):
        list(
            iter_offset_removed_blocks(
                np.arange(4, dtype=np.float32), 0.0, block_len
            )
        )


def test_iter_level_stat_blocks_reassembles_a_written_level():
    raw = np.arange(37, dtype=np.float32)
    level1 = fold_raw_block(raw)
    env = level1[:, MIN_COL : MAX_COL + 1].astype(np.float32)
    mean = level1[:, MEAN_COL].astype(np.float32)
    valid = level1[:, VALID_COL].astype(np.uint16)

    out = np.concatenate(
        list(iter_level_stat_blocks(env, mean, valid, 37, 1, 4)), axis=0
    )
    assert np.array_equal(out[:, MIN_COL : MAX_COL + 1], env)
    assert np.array_equal(out[:, MEAN_COL], mean)
    # valid is read back, not rebuilt: it is data, not arithmetic.
    assert np.array_equal(out[:, VALID_COL], valid)
    # The counts are rebuilt, not read: the trailing bin holds one sample.
    assert np.array_equal(out[:, COUNT_COL], bin_counts(37, 1, 0, env.shape[0]))
    assert out[-1, COUNT_COL] == 1.0


@pytest.mark.parametrize("block_len", [1, 3, 4, 16, 1000])
def test_iter_level_stat_blocks_is_block_invariant(block_len):
    raw = np.arange(37, dtype=np.float32)
    level1 = fold_raw_block(raw)
    env = level1[:, MIN_COL : MAX_COL + 1].astype(np.float32)
    mean = level1[:, MEAN_COL].astype(np.float32)
    valid = level1[:, VALID_COL].astype(np.uint16)
    out = np.concatenate(
        list(iter_level_stat_blocks(env, mean, valid, 37, 1, block_len)), axis=0
    )
    assert out.shape == (env.shape[0], STAT_COLUMNS)
    assert np.array_equal(out[:, COUNT_COL], bin_counts(37, 1, 0, env.shape[0]))


def test_iter_level_stat_blocks_empty_level_yields_nothing():
    env = np.empty((0, 2), dtype=np.float32)
    mean = np.empty(0, dtype=np.float32)
    valid = np.empty(0, dtype=np.uint16)
    assert list(iter_level_stat_blocks(env, mean, valid, 0, 1, 4)) == []


@pytest.mark.parametrize("block_len", [0, -1])
def test_iter_level_stat_blocks_rejects_nonpositive_block(block_len):
    env = np.zeros((4, 2), dtype=np.float32)
    mean = np.zeros(4, dtype=np.float32)
    valid = np.zeros(4, dtype=np.uint16)
    with pytest.raises(ValueError, match="positive"):
        list(iter_level_stat_blocks(env, mean, valid, 16, 1, block_len))
