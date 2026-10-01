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
from timeseries_zarr.fold import (
    _block_split,
    fold_block,
    fold_raw_block,
    fold_stat_block,
)


def _stats(rows):
    """Build a stat block from (min, max, mean, count) tuples.

    valid defaults to the count, the no-gaps case.
    """
    out = np.empty((len(rows), STAT_COLUMNS), dtype=np.float64)
    out[:, :4] = np.array(rows, dtype=np.float64)
    out[:, VALID_COL] = out[:, COUNT_COL]
    return out


@pytest.mark.parametrize(
    ("length", "block", "expected"),
    [
        (0, 4, (0, 0)),
        (8, 4, (2, 0)),
        (16, 4, (4, 0)),
        (5, 4, (1, 1)),
        (6, 4, (1, 2)),
        (7, 4, (1, 3)),
        (9, 4, (2, 1)),
        (1, 4, (0, 1)),
        (3, 4, (0, 3)),
        (5, 2, (2, 1)),
    ],
)
def test_block_split(length, block, expected):
    assert _block_split(length, block) == expected


def test_block_split_default_block_is_4():
    assert _block_split(7) == (1, 3)


@pytest.mark.parametrize("block", [0, -1])
def test_block_split_rejects_nonpositive_block(block):
    with pytest.raises(ValueError, match="positive"):
        _block_split(8, block)


def test_fold_raw_block_matches_reshape_reduce():
    rng = np.random.default_rng(0)
    raw = rng.standard_normal(64).astype(np.float32)
    out = fold_raw_block(raw)
    blocks = raw.reshape(-1, 4).astype(np.float64)
    assert out.shape == (16, STAT_COLUMNS)
    assert np.array_equal(out[:, MIN_COL], blocks.min(axis=1))
    assert np.array_equal(out[:, MAX_COL], blocks.max(axis=1))
    assert np.allclose(out[:, MEAN_COL], blocks.mean(axis=1))
    assert np.array_equal(out[:, COUNT_COL], np.full(16, 4.0))
    assert np.array_equal(out[:, VALID_COL], np.full(16, 4.0))


@pytest.mark.parametrize(
    ("m", "expected_rows"),
    [(1, 1), (4, 1), (5, 2), (6, 2), (7, 2), (8, 2), (9, 3)],
)
def test_fold_raw_block_shape(m, expected_rows):
    out = fold_raw_block(np.arange(m, dtype=np.float32))
    assert out.shape == (expected_rows, STAT_COLUMNS)


def test_fold_raw_block_tail_folds_partial_block():
    out = fold_raw_block(np.array([0, 1, 2, 3, 4], dtype=np.float32))
    assert out.shape == (2, STAT_COLUMNS)
    assert tuple(out[0]) == (0.0, 3.0, 1.5, 4.0, 4.0)
    assert tuple(out[1]) == (4.0, 4.0, 4.0, 1.0, 1.0)


def test_fold_raw_block_empty():
    out = fold_raw_block(np.array([], dtype=np.float32))
    assert out.shape == (0, STAT_COLUMNS)


def test_fold_raw_block_single_sample():
    out = fold_raw_block(np.array([7.0], dtype=np.float32))
    assert out.shape == (1, STAT_COLUMNS)
    assert tuple(out[0]) == (7.0, 7.0, 7.0, 1.0, 1.0)


def test_fold_raw_block_dtype_is_float64():
    out = fold_raw_block(np.arange(8, dtype=np.float32))
    assert out.dtype == np.float64


def test_fold_raw_block_nan_propagates():
    raw = np.array([1, 2, np.nan, 4, 5, 6, 7, 8], dtype=np.float32)
    out = fold_raw_block(raw)
    assert np.isnan(out[0, MIN_COL])
    assert np.isnan(out[0, MAX_COL])
    assert np.isnan(out[0, MEAN_COL])
    # The count is time support, so a NaN bin still spans its four slots.
    assert out[0, COUNT_COL] == 4.0
    assert tuple(out[1]) == (5.0, 8.0, 6.5, 4.0, 4.0)


def test_fold_stat_block_matches_reshape_reduce():
    rng = np.random.default_rng(1)
    stats = np.empty((16, STAT_COLUMNS), dtype=np.float64)
    stats[:, MIN_COL] = rng.standard_normal(16)
    stats[:, MAX_COL] = stats[:, MIN_COL] + 1.0
    stats[:, MEAN_COL] = rng.standard_normal(16)
    stats[:, COUNT_COL] = 4.0
    stats[:, VALID_COL] = 4.0
    out = fold_stat_block(stats)
    groups = stats.reshape(-1, 4, STAT_COLUMNS)
    assert out.shape == (4, STAT_COLUMNS)
    assert np.array_equal(out[:, MIN_COL], groups[:, :, MIN_COL].min(axis=1))
    assert np.array_equal(out[:, MAX_COL], groups[:, :, MAX_COL].max(axis=1))
    assert np.allclose(out[:, MEAN_COL], groups[:, :, MEAN_COL].mean(axis=1))
    assert np.array_equal(out[:, COUNT_COL], np.full(4, 16.0))


def test_fold_stat_block_weights_the_mean_by_count():
    """A short trailing bin must not pull the mean as hard as a full one."""
    stats = _stats(
        [
            [0.0, 0.0, 0.0, 4.0],
            [0.0, 0.0, 0.0, 4.0],
            [0.0, 0.0, 0.0, 4.0],
            [12.0, 12.0, 12.0, 1.0],
        ]
    )
    out = fold_stat_block(stats)
    # Weighted: 12 over 13 samples. A plain mean of means would say 3.0.
    assert out[0, MEAN_COL] == pytest.approx(12.0 / 13.0)
    assert out[0, COUNT_COL] == 13.0


@pytest.mark.parametrize(
    ("m", "expected_rows"),
    [(1, 1), (4, 1), (5, 2), (7, 2), (8, 2), (9, 3)],
)
def test_fold_stat_block_shape(m, expected_rows):
    stats = np.ones((m, STAT_COLUMNS), dtype=np.float64)
    assert fold_stat_block(stats).shape == (expected_rows, STAT_COLUMNS)


def test_fold_stat_block_tail_folds_partial_block():
    stats = _stats(
        [
            [0.0, 3.0, 1.0, 4.0],
            [1.0, 5.0, 2.0, 4.0],
            [2.0, 4.0, 3.0, 4.0],
            [6.0, 9.0, 4.0, 4.0],
            [7.0, 8.0, 5.0, 4.0],
        ]
    )
    out = fold_stat_block(stats)
    assert out.shape == (2, STAT_COLUMNS)
    assert tuple(out[0]) == (0.0, 9.0, 2.5, 16.0, 16.0)
    assert tuple(out[1]) == (7.0, 8.0, 5.0, 4.0, 4.0)


def test_fold_stat_block_empty():
    out = fold_stat_block(np.empty((0, STAT_COLUMNS), dtype=np.float64))
    assert out.shape == (0, STAT_COLUMNS)


def test_fold_stat_block_single_row_identity():
    out = fold_stat_block(_stats([[2.0, 9.0, 5.0, 4.0]]))
    assert out.shape == (1, STAT_COLUMNS)
    assert tuple(out[0]) == (2.0, 9.0, 5.0, 4.0, 4.0)


def test_fold_stat_block_nan_propagates():
    stats = np.ones((8, STAT_COLUMNS), dtype=np.float64)
    stats[:, COUNT_COL] = 4.0
    stats[:, VALID_COL] = 4.0
    stats[1, MIN_COL] = np.nan
    stats[1, MEAN_COL] = np.nan
    out = fold_stat_block(stats)
    assert np.isnan(out[0, MIN_COL])
    assert np.isnan(out[0, MEAN_COL])
    assert out[0, COUNT_COL] == 16.0
    assert not np.isnan(out[1, MEAN_COL])


def test_fold_stat_block_composes_with_raw_exactly():
    rng = np.random.default_rng(2)
    raw = rng.standard_normal(16).astype(np.float32)
    level2 = fold_stat_block(fold_raw_block(raw))
    assert level2.shape == (1, STAT_COLUMNS)
    assert level2[0, MIN_COL] == raw.min()
    assert level2[0, MAX_COL] == raw.max()
    assert level2[0, MEAN_COL] == pytest.approx(raw.astype(np.float64).mean())
    assert level2[0, COUNT_COL] == 16.0


@pytest.mark.parametrize("m", [16, 17, 18, 19, 32, 33])
def test_fold_stat_block_composes_with_raw_over_tails(m):
    """A partial bin at every ladder rung still yields the exact group mean."""
    rng = np.random.default_rng(3)
    raw = rng.standard_normal(m).astype(np.float32)
    level2 = fold_stat_block(fold_raw_block(raw))
    groups = [raw[i : i + 16].astype(np.float64) for i in range(0, m, 16)]
    assert np.array_equal(level2[:, MIN_COL], [g.min() for g in groups])
    assert np.array_equal(level2[:, MAX_COL], [g.max() for g in groups])
    assert np.allclose(level2[:, MEAN_COL], [g.mean() for g in groups])
    assert np.array_equal(level2[:, COUNT_COL], [g.size for g in groups])


def test_fold_block_rank1_delegates_to_raw():
    raw = np.arange(8, dtype=np.float32)
    assert np.array_equal(fold_block(raw), fold_raw_block(raw))


def test_fold_block_rank2_delegates_to_stats():
    stats = np.ones((8, STAT_COLUMNS), dtype=np.float64)
    assert np.array_equal(fold_block(stats), fold_stat_block(stats))


@pytest.mark.parametrize("ndim", [0, 3])
def test_fold_block_rejects_other_ranks(ndim):
    arr = np.zeros((2,) * ndim, dtype=np.float32)
    with pytest.raises(ValueError, match="expected rank"):
        fold_block(arr)


def test_fold_block_rejects_a_rank2_array_of_the_wrong_width():
    """The old envelope-only shape must not fold silently."""
    with pytest.raises(ValueError, match="expected rank"):
        fold_block(np.zeros((8, 2), dtype=np.float32))
