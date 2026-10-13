import warnings

import numpy as np
import pytest

from timeseries_zarr.constants import (
    COUNT_COL,
    MAX_COL,
    MEAN_COL,
    MIN_COL,
    STAT_COLUMNS,
    VAR_COL,
)
from timeseries_zarr.fold import (
    _block_split,
    fold_block,
    fold_pair_block,
    fold_raw_block,
)


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


def _direct(raw, width=4):
    """Per-bin statistics straight from the definition, in float64."""
    x = np.asarray(raw, dtype=np.float64).copy()
    finite = np.isfinite(x)
    x[~finite] = np.nan
    rows = []
    for i in range(0, len(x), width):
        bin_values, bin_finite = x[i : i + width], finite[i : i + width]
        rows.append(
            [
                bin_values.min(),
                bin_values.max(),
                bin_values.mean(),
                bin_values.var(),
                bin_finite.sum(),
            ]
        )
    return np.array(rows, dtype=np.float64).reshape(-1, STAT_COLUMNS)


def test_fold_raw_block_matches_the_direct_definition():
    rng = np.random.default_rng(0)
    raw = (5000.0 + 8.0 * rng.standard_normal(67)).astype(np.float32)
    raw[5] = np.nan
    raw[20:24] = np.nan
    raw[40] = np.inf
    raw[41] = -np.inf
    out = fold_raw_block(raw)
    want = _direct(raw)
    assert out.shape == (17, STAT_COLUMNS)
    assert np.array_equal(np.isnan(out), np.isnan(want))
    assert np.allclose(out, want, rtol=1e-12, equal_nan=True)


@pytest.mark.parametrize(
    ("m", "expected_rows"),
    [(0, 0), (1, 1), (2, 1), (3, 1), (4, 1), (5, 2), (17, 5)],
)
def test_fold_raw_block_shape(m, expected_rows):
    out = fold_raw_block(np.arange(m, dtype=np.float32))
    assert out.shape == (expected_rows, STAT_COLUMNS)


def test_fold_raw_block_tail_folds_partial_block():
    out = fold_raw_block(np.array([0, 1, 2, 3, 4], dtype=np.float32))
    assert out.shape == (2, STAT_COLUMNS)
    assert tuple(out[0]) == (0.0, 3.0, 1.5, 1.25, 4.0)
    assert tuple(out[1]) == (4.0, 4.0, 4.0, 0.0, 1.0)


def test_fold_raw_block_empty():
    out = fold_raw_block(np.array([], dtype=np.float32))
    assert out.shape == (0, STAT_COLUMNS)
    assert out.dtype == np.float64


def test_fold_raw_block_single_sample():
    out = fold_raw_block(np.array([7.0], dtype=np.float32))
    assert tuple(out[0]) == (7.0, 7.0, 7.0, 0.0, 1.0)


def test_fold_raw_block_dtype_is_float64():
    out = fold_raw_block(np.arange(8, dtype=np.float32))
    assert out.dtype == np.float64


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_fold_raw_block_one_non_finite_sample_makes_stats_nan(bad):
    raw = np.array([1, 2, bad, 4, 5, 6, 7, 8], dtype=np.float32)
    out = fold_raw_block(raw)
    assert np.isnan(out[0, [MIN_COL, MAX_COL, MEAN_COL, VAR_COL]]).all()
    assert out[0, COUNT_COL] == 3.0
    assert tuple(out[1]) == (5.0, 8.0, 6.5, 1.25, 4.0)


def test_fold_raw_block_all_non_finite_bin_has_count_zero():
    raw = np.array([np.nan, np.inf, np.nan, -np.inf], dtype=np.float32)
    out = fold_raw_block(raw)
    assert np.isnan(out[0, :COUNT_COL]).all()
    assert out[0, COUNT_COL] == 0.0


def test_fold_raw_block_raises_no_warning_on_non_finite_input():
    raw = np.array([np.nan, np.nan, np.nan, np.nan, 1.0], dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fold_raw_block(raw)


def test_fold_raw_block_leaves_the_input_unchanged():
    raw = np.array([1.0, np.inf, 3.0, 4.0], dtype=np.float32)
    before = raw.copy()
    fold_raw_block(raw)
    assert np.array_equal(raw, before)


def test_fold_pair_block_matches_reshape_reduce():
    rng = np.random.default_rng(1)
    pairs = rng.standard_normal((16, 2)).astype(np.float32)
    out = fold_pair_block(pairs)
    blocks = pairs.reshape(-1, 4, 2)
    expected = np.stack(
        [blocks[:, :, 0].min(axis=1), blocks[:, :, 1].max(axis=1)], axis=1
    )
    assert out.shape == (4, 2)
    assert np.array_equal(out, expected)


@pytest.mark.parametrize(
    ("m", "expected_rows"),
    [(1, 1), (4, 1), (5, 2), (7, 2), (8, 2), (9, 3)],
)
def test_fold_pair_block_shape(m, expected_rows):
    out = fold_pair_block(np.zeros((m, 2), dtype=np.float32))
    assert out.shape == (expected_rows, 2)


def test_fold_pair_block_tail_folds_partial_block():
    pairs = np.array([[0, 3], [1, 5], [2, 4], [6, 9], [7, 8]], dtype=np.float32)
    out = fold_pair_block(pairs)
    assert out.shape == (2, 2)
    assert tuple(out[0]) == (0.0, 9.0)
    assert tuple(out[1]) == (7.0, 8.0)


def test_fold_pair_block_empty():
    out = fold_pair_block(np.empty((0, 2), dtype=np.float32))
    assert out.shape == (0, 2)


def test_fold_pair_block_single_pair_identity():
    out = fold_pair_block(np.array([[2.0, 9.0]], dtype=np.float32))
    assert out.shape == (1, 2)
    assert tuple(out[0]) == (2.0, 9.0)


def test_fold_pair_block_dtype_is_float32():
    out = fold_pair_block(np.zeros((8, 2), dtype=np.float32))
    assert out.dtype == np.float32


def test_fold_pair_block_nan_propagates():
    pairs = np.array(
        [[1, 3], [np.nan, 5], [2, 4], [0, 6], [7, 8], [7, 8], [7, 9], [7, 8]],
        dtype=np.float32,
    )
    out = fold_pair_block(pairs)
    assert np.isnan(out[0, 0])
    assert out[0, 1] == 6.0
    assert tuple(out[1]) == (7.0, 9.0)


def test_fold_pair_block_composes_with_raw_exactly():
    rng = np.random.default_rng(2)
    raw = rng.standard_normal(16).astype(np.float32)
    level2 = fold_pair_block(fold_block(raw))
    assert level2.shape == (1, 2)
    assert level2[0, 0] == raw.min()
    assert level2[0, 1] == raw.max()


@pytest.mark.parametrize("m", [16, 17, 18, 19, 32, 33])
def test_fold_pair_block_composes_with_raw_over_tails(m):
    rng = np.random.default_rng(3)
    raw = rng.standard_normal(m).astype(np.float32)
    level2 = fold_pair_block(fold_block(raw))
    groups = [raw[i : i + 16] for i in range(0, m, 16)]
    expected = np.stack(
        [[g.min() for g in groups], [g.max() for g in groups]], axis=1
    ).astype(np.float32)
    assert np.array_equal(level2, expected)


def test_fold_pair_block_nan_in_max_column_propagates():
    pairs = np.array([[1, 3], [2, np.nan], [0, 4], [5, 6]], dtype=np.float32)
    out = fold_pair_block(pairs)
    assert out[0, 0] == 0.0
    assert np.isnan(out[0, 1])


def test_fold_block_rank1_reduces_raw_to_float32_pairs():
    raw = np.arange(8, dtype=np.float32)
    out = fold_block(raw)
    assert out.dtype == np.float32
    assert np.array_equal(
        out, fold_raw_block(raw)[:, [MIN_COL, MAX_COL]].astype(np.float32)
    )


def test_fold_block_rank2_delegates_to_pairs():
    pairs = np.arange(16, dtype=np.float32).reshape(8, 2)
    assert np.array_equal(fold_block(pairs), fold_pair_block(pairs))


@pytest.mark.parametrize("ndim", [0, 3])
def test_fold_block_rejects_other_ranks(ndim):
    arr = np.zeros((2,) * ndim, dtype=np.float32)
    with pytest.raises(ValueError, match="rank"):
        fold_block(arr)
