import numpy as np
import pytest

from timeseries_zarr.constants import STATISTIC_COLUMNS, VAR_COLUMN
from timeseries_zarr.fold import (
    assemble_statistics,
    fold_pair_block,
    fold_raw_block,
    fold_statistics_block,
    split_members,
)
from timeseries_zarr.streaming import (
    _rebuffer_and_fold,
    iter_array_blocks,
    iter_member_blocks,
    iter_raw_blocks,
    regroup,
    tap,
)
from timeseries_zarr.types import LevelMembers


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
        else np.empty((0, 2), dtype=np.float32)
    )


@pytest.mark.parametrize("seed", range(5))
def test_rebuffer_and_fold_raw_is_split_invariant(seed):
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal(37).astype(np.float32)
    blocks = _split(arr, _random_sizes(rng, arr.shape[0]))
    assert np.array_equal(_folded(blocks, fold_raw_block), fold_raw_block(arr))


@pytest.mark.parametrize("seed", range(5))
def test_rebuffer_and_fold_pairs_is_split_invariant(seed):
    rng = np.random.default_rng(seed + 100)
    arr = rng.standard_normal((29, 2)).astype(np.float32)
    blocks = _split(arr, _random_sizes(rng, arr.shape[0]))
    assert np.array_equal(
        _folded(blocks, fold_pair_block), fold_pair_block(arr)
    )


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
        np.concatenate(out, axis=0) if out else np.empty((0, 2), np.float32),
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
    assert out.shape == (1, STATISTIC_COLUMNS)
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


def _gappy(length):
    rng = np.random.default_rng(7)
    raw = (5000.0 + 8.0 * rng.standard_normal(length)).astype(np.float32)
    raw[5] = np.nan
    raw[40] = np.inf
    raw[100:140] = np.nan
    return raw


def _cut(length, size):
    """Return block sizes of `size` rows covering `length` rows."""
    whole, tail = divmod(length, size)
    return [size] * whole + ([tail] if tail else [])


@pytest.mark.parametrize("size", [1, 2, 3, 5, 7, 13])
def test_two_streamed_folds_equal_the_whole_fold_bit_for_bit(size):
    raw = _gappy(997)
    level1 = _rebuffer_and_fold(
        _split(raw, _cut(len(raw), size)), fold_raw_block
    )
    level2 = list(_rebuffer_and_fold(level1, fold_statistics_block))
    want = fold_statistics_block(fold_raw_block(raw))
    assert np.array_equal(np.concatenate(level2, axis=0), want, equal_nan=True)


def _regrouped_sizes(sizes, rows):
    blocks = [np.zeros((size, 2), dtype=np.float32) for size in sizes]
    return [block.shape[0] for block in regroup(blocks, rows)]


@pytest.mark.parametrize(
    ("sizes", "rows", "expected"),
    [
        ([3, 3, 3, 3], 6, [6, 6]),
        ([10], 4, [4, 4, 2]),
        ([1] * 9, 4, [4, 4, 1]),
        ([5, 1, 7], 4, [4, 4, 4, 1]),
        ([4, 4], 4, [4, 4]),
        ([3], 4, [3]),
        ([1, 1], 4, [2]),
        ([2, 2, 2], 1, [1, 1, 1, 1, 1, 1]),
    ],
)
def test_regroup_yields_exact_rows_then_the_remainder(sizes, rows, expected):
    assert _regrouped_sizes(sizes, rows) == expected


@pytest.mark.parametrize("seed", range(5))
def test_regroup_preserves_the_concatenation(seed):
    rng = np.random.default_rng(seed + 300)
    arr = rng.standard_normal((61, 2)).astype(np.float32)
    blocks = _split(arr, _random_sizes(rng, arr.shape[0]))
    out = list(regroup(blocks, 8))
    assert np.array_equal(np.concatenate(out, axis=0), arr)
    assert all(block.shape[0] == 8 for block in out[:-1])
    assert 1 <= out[-1].shape[0] <= 8


def test_regroup_skips_blocks_with_no_rows():
    blocks = [np.arange(3.0), np.empty(0), np.arange(3.0, 7.0), np.empty(0)]
    out = list(regroup(blocks, 4))
    assert [b.shape[0] for b in out] == [4, 3]
    assert np.array_equal(np.concatenate(out), np.arange(7.0))


def test_regroup_empty_stream_yields_nothing():
    assert list(regroup([], 4)) == []


def test_regroup_keeps_the_trailing_axes_of_rank_two_blocks():
    out = list(regroup([np.zeros((6, STATISTIC_COLUMNS))], 4))
    assert [b.shape for b in out] == [
        (4, STATISTIC_COLUMNS),
        (2, STATISTIC_COLUMNS),
    ]


def test_regroup_does_not_alias_the_carry_to_the_input():
    first = np.arange(6.0)
    out = regroup([first, np.arange(6.0, 12.0)], 4)
    next(out)
    first[:] = -1.0
    assert np.array_equal(np.concatenate(list(out)), np.arange(4.0, 12.0))


@pytest.mark.parametrize("rows", [0, -1])
def test_regroup_rejects_non_positive_rows(rows):
    with pytest.raises(ValueError, match="rows"):
        list(regroup([np.arange(4.0)], rows))


def test_tap_passes_each_block_to_the_sink_before_yielding_it():
    events = []
    blocks = [np.arange(2.0), np.arange(2.0, 5.0)]

    def sink(block):
        events.append(("sink", block.shape[0]))

    for block in tap(blocks, sink):
        events.append(("got", block.shape[0]))
    assert events == [("sink", 2), ("got", 2), ("sink", 3), ("got", 3)]


def test_tap_yields_the_blocks_unchanged():
    blocks = [np.arange(2.0), np.arange(2.0, 5.0)]
    out = list(tap(blocks, lambda block: None))
    assert all(got is want for got, want in zip(out, blocks, strict=True))


def test_tap_calls_the_sink_only_as_blocks_are_requested():
    calls = []
    stream = tap([np.arange(2.0), np.arange(2.0)], calls.append)
    assert calls == []
    next(stream)
    assert len(calls) == 1


def test_tap_empty_stream_never_calls_the_sink():
    calls = []
    assert list(tap([], calls.append)) == []
    assert calls == []


def test_tap_propagates_a_sink_exception_to_the_consumer():
    def sink(block):
        raise RuntimeError("sink failed")

    with pytest.raises(RuntimeError, match="sink failed"):
        next(tap([np.arange(2.0)], sink))


def _members(length=997, with_var=True):
    statistics = fold_raw_block(_gappy(length))
    minmax, mean, var, count = split_members(statistics, with_var)
    return LevelMembers(minmax=minmax, mean=mean, var=var, count=count)


@pytest.mark.parametrize("block_len", [1, 3, 4, 7, 16, 1000])
def test_iter_member_blocks_concatenates_to_the_assembled_level(block_len):
    members = _members()
    out = list(iter_member_blocks(members, block_len))
    want = assemble_statistics(
        members.minmax, members.mean, members.var, members.count
    )
    assert np.array_equal(np.concatenate(out, axis=0), want, equal_nan=True)


def test_iter_member_blocks_window_sizes_end_with_the_remainder():
    members = _members(length=150)
    sizes = [b.shape[0] for b in iter_member_blocks(members, 4)]
    assert sizes == [4] * 9 + [2]


def test_iter_member_blocks_without_var_leaves_the_variance_column_nan():
    out = list(iter_member_blocks(_members(with_var=False), 16))
    assert all(np.isnan(block[:, VAR_COLUMN]).all() for block in out)


def test_iter_member_blocks_empty_level_yields_nothing():
    members = LevelMembers(
        minmax=np.empty((0, 2), dtype=np.float32),
        mean=np.empty(0, dtype=np.float32),
        var=None,
        count=np.empty(0, dtype=np.uint32),
    )
    assert list(iter_member_blocks(members, 4)) == []


@pytest.mark.parametrize("block_len", [0, -1])
def test_iter_member_blocks_rejects_non_positive_block_len(block_len):
    with pytest.raises(ValueError, match="positive"):
        list(iter_member_blocks(_members(), block_len))


@pytest.mark.parametrize("name", ["mean", "var", "count"])
@pytest.mark.parametrize("extra", [1, -1])
def test_iter_member_blocks_rejects_members_with_different_row_counts(
    name, extra
):
    members = _members(length=150)
    member = getattr(members, name)
    resized = np.concatenate([member, member[:1]]) if extra > 0 else member[:-1]
    mismatched = LevelMembers(
        **{
            "minmax": members.minmax,
            "mean": members.mean,
            "var": members.var,
            "count": members.count,
            name: resized,
        }
    )
    with pytest.raises(ValueError, match=name):
        list(iter_member_blocks(mismatched, 4))
