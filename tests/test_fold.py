import warnings

import numpy as np
import pytest

from timeseries_zarr.constants import (
    COUNT_COLUMN,
    MAX_COLUMN,
    MEAN_COLUMN,
    MIN_COLUMN,
    STATISTIC_COLUMNS,
    VAR_COLUMN,
)
from timeseries_zarr.fold import (
    _block_split,
    assemble_statistics,
    fold_raw_block,
    fold_statistics_block,
    split_members,
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
    return np.array(rows, dtype=np.float64).reshape(-1, STATISTIC_COLUMNS)


def test_fold_raw_block_matches_the_direct_definition():
    rng = np.random.default_rng(0)
    raw = (5000.0 + 8.0 * rng.standard_normal(67)).astype(np.float32)
    raw[5] = np.nan
    raw[20:24] = np.nan
    raw[40] = np.inf
    raw[41] = -np.inf
    out = fold_raw_block(raw)
    want = _direct(raw)
    assert out.shape == (17, STATISTIC_COLUMNS)
    assert np.array_equal(np.isnan(out), np.isnan(want))
    assert np.allclose(out, want, rtol=1e-12, equal_nan=True)


@pytest.mark.parametrize(
    ("m", "expected_rows"),
    [(0, 0), (1, 1), (2, 1), (3, 1), (4, 1), (5, 2), (17, 5)],
)
def test_fold_raw_block_shape(m, expected_rows):
    out = fold_raw_block(np.arange(m, dtype=np.float32))
    assert out.shape == (expected_rows, STATISTIC_COLUMNS)


def test_fold_raw_block_tail_folds_partial_block():
    out = fold_raw_block(np.array([0, 1, 2, 3, 4], dtype=np.float32))
    assert out.shape == (2, STATISTIC_COLUMNS)
    assert tuple(out[0]) == (0.0, 3.0, 1.5, 1.25, 4.0)
    assert tuple(out[1]) == (4.0, 4.0, 4.0, 0.0, 1.0)


def test_fold_raw_block_empty():
    out = fold_raw_block(np.array([], dtype=np.float32))
    assert out.shape == (0, STATISTIC_COLUMNS)
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
    assert np.isnan(
        out[0, [MIN_COLUMN, MAX_COLUMN, MEAN_COLUMN, VAR_COLUMN]]
    ).all()
    assert out[0, COUNT_COLUMN] == 3.0
    assert tuple(out[1]) == (5.0, 8.0, 6.5, 1.25, 4.0)


def test_fold_raw_block_all_non_finite_bin_has_count_zero():
    raw = np.array([np.nan, np.inf, np.nan, -np.inf], dtype=np.float32)
    out = fold_raw_block(raw)
    assert np.isnan(out[0, :COUNT_COLUMN]).all()
    assert out[0, COUNT_COLUMN] == 0.0


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


def _reduce(values, width, reduction):
    """The generator's block reduction over consecutive rows, tail kept."""
    whole = (values.shape[0] // width) * width
    grouped = values[:whole].reshape(-1, width, *values.shape[1:])
    out = reduction(grouped, axis=1)
    if values.shape[0] > whole:
        tail = np.asarray(reduction(values[whole:], axis=0))[None]
        out = np.concatenate([out, tail])
    return out


def _generator_fold(statistics):
    """The reference generator's fold, column by column, in float64."""
    minimum, maximum, mean, variance, count = (
        statistics[:, column] for column in range(STATISTIC_COLUMNS)
    )
    total = _reduce(count, 4, np.sum)
    with np.errstate(invalid="ignore", divide="ignore"):
        parent_mean = _reduce(count * mean, 4, np.sum) / total
        repeated = np.repeat(parent_mean, 4)[: len(mean)]
        spread = variance + (mean - repeated) ** 2
        parent_variance = _reduce(count * spread, 4, np.sum) / total
    return np.stack(
        [
            _reduce(minimum, 4, np.min),
            _reduce(maximum, 4, np.max),
            parent_mean,
            parent_variance,
            total,
        ],
        axis=1,
    )


def _gappy_series(length, seed=0):
    rng = np.random.default_rng(seed)
    raw = (5000.0 + 8.0 * rng.standard_normal(length)).astype(np.float32)
    raw[5] = np.nan
    raw[20:24] = np.nan
    raw[40] = np.inf
    raw[41] = -np.inf
    raw[100:260] = np.nan
    return raw


def test_fold_statistics_block_matches_the_generator_fold():
    statistics = fold_raw_block(_gappy_series(4133))
    out = fold_statistics_block(statistics)
    want = _generator_fold(statistics)
    assert np.array_equal(np.isnan(out), np.isnan(want))
    assert np.allclose(out, want, rtol=1e-12, atol=0.0, equal_nan=True)


def test_fold_statistics_block_composes_to_the_direct_definition_over_levels():
    raw = _gappy_series(4133)
    statistics = fold_raw_block(raw)
    for level in range(2, 7):
        statistics = fold_statistics_block(statistics)
        want = _direct(raw, 4**level)
        assert statistics.shape == want.shape
        assert np.array_equal(np.isnan(statistics), np.isnan(want))
        assert np.array_equal(
            statistics[:, [MIN_COLUMN, MAX_COLUMN]], want[:, :2], equal_nan=True
        )
        assert np.array_equal(
            statistics[:, COUNT_COLUMN], want[:, COUNT_COLUMN]
        )
        assert np.allclose(
            statistics[:, [MEAN_COLUMN, VAR_COLUMN]],
            want[:, [MEAN_COLUMN, VAR_COLUMN]],
            rtol=1e-9,
            atol=0.0,
            equal_nan=True,
        )


@pytest.mark.parametrize(
    ("m", "expected_rows"),
    [(0, 0), (1, 1), (2, 1), (3, 1), (4, 1), (5, 2), (17, 5)],
)
def test_fold_statistics_block_shape(m, expected_rows):
    out = fold_statistics_block(
        fold_raw_block(np.arange(4 * m, dtype=np.float32))
    )
    assert out.shape == (expected_rows, STATISTIC_COLUMNS)
    assert out.dtype == np.float64


def test_fold_statistics_block_weights_a_partial_child_by_its_count():
    raw = np.array([0.0] * 12 + [8.0], dtype=np.float32)
    children = fold_raw_block(raw)
    assert children[:, MEAN_COLUMN].mean() == 2.0
    out = fold_statistics_block(children)
    want = _direct(raw, 16)
    assert out.shape == (1, STATISTIC_COLUMNS)
    assert np.allclose(out, want, rtol=1e-12, atol=0.0)
    assert out[0, MEAN_COLUMN] == 8.0 / 13.0


def test_fold_statistics_block_one_nan_child_makes_the_parent_nan():
    raw = np.array([1, 2, np.nan, 4] + list(range(12)), dtype=np.float32)
    out = fold_statistics_block(fold_raw_block(raw))
    assert np.isnan(
        out[0, [MIN_COLUMN, MAX_COLUMN, MEAN_COLUMN, VAR_COLUMN]]
    ).all()
    assert out[0, COUNT_COLUMN] == 15.0


def test_fold_statistics_block_all_nan_children_give_count_zero_without_warning():
    raw = np.full(16, np.nan, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = fold_statistics_block(fold_raw_block(raw))
    assert np.isnan(out[0, :COUNT_COLUMN]).all()
    assert out[0, COUNT_COLUMN] == 0.0


def test_fold_statistics_block_raises_no_warning_on_finite_input():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fold_statistics_block(fold_raw_block(np.arange(64, dtype=np.float32)))


def test_fold_statistics_block_tail_of_one_child_is_that_child():
    raw = np.arange(20, dtype=np.float32)
    children = fold_raw_block(raw)
    out = fold_statistics_block(children)
    assert out.shape == (2, STATISTIC_COLUMNS)
    assert np.array_equal(out[1], children[4])


@pytest.mark.parametrize("shape", [(8,), (8, 2), (2, 4, 5)])
def test_fold_statistics_block_rejects_a_non_statistics_block(shape):
    with pytest.raises(ValueError, match="statistics block"):
        fold_statistics_block(np.zeros(shape, dtype=np.float64))


def _members(length=4133, with_var=True):
    statistics = fold_raw_block(_gappy_series(length))
    return statistics, split_members(statistics, with_var)


def _stored(statistics):
    """The statistics block as it reads back from float32 members."""
    out = statistics.copy()
    out[:, :COUNT_COLUMN] = statistics[:, :COUNT_COLUMN].astype(np.float32)
    return out


def test_split_members_returns_the_on_disk_dtypes_and_shapes():
    statistics, (minmax, mean, var, count) = _members()
    rows = statistics.shape[0]
    assert minmax.shape == (rows, 2)
    assert minmax.dtype == np.float32
    assert mean.shape == var.shape == count.shape == (rows,)
    assert mean.dtype == var.dtype == np.float32
    assert count.dtype == np.uint32


def test_split_members_places_each_statistic_in_its_member():
    statistics, (minmax, mean, var, count) = _members()
    assert np.array_equal(
        minmax[:, 0],
        statistics[:, MIN_COLUMN].astype(np.float32),
        equal_nan=True,
    )
    assert np.array_equal(
        minmax[:, 1],
        statistics[:, MAX_COLUMN].astype(np.float32),
        equal_nan=True,
    )
    assert np.array_equal(
        mean, statistics[:, MEAN_COLUMN].astype(np.float32), equal_nan=True
    )
    assert np.array_equal(
        var, statistics[:, VAR_COLUMN].astype(np.float32), equal_nan=True
    )
    assert np.array_equal(count, statistics[:, COUNT_COLUMN])


def test_split_members_keeps_nan_bins_nan_and_their_counts():
    statistics, (minmax, mean, var, count) = _members()
    nan_bins = np.isnan(statistics[:, MEAN_COLUMN])
    assert nan_bins.any()
    assert np.isnan(minmax[nan_bins]).all()
    assert np.isnan(mean[nan_bins]).all()
    assert np.isnan(var[nan_bins]).all()
    assert (count[nan_bins] <= 3).all()


def test_split_members_without_var_returns_none_and_the_same_others():
    _, with_var = _members(with_var=True)
    _, without = _members(with_var=False)
    assert without[2] is None
    for kept, dropped in zip(with_var[:2], without[:2], strict=True):
        assert np.array_equal(kept, dropped, equal_nan=True)
    assert np.array_equal(with_var[3], without[3])


def test_split_members_stores_a_value_beyond_float32_as_infinity_quietly():
    statistics = np.array([[1.0, 2.0, 1.5, 1e300, 4.0]], dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, _, var, _ = split_members(statistics, True)
    assert np.isposinf(var[0])


def test_split_members_empty_block_yields_zero_rows_in_every_member():
    minmax, mean, var, count = split_members(
        np.empty((0, STATISTIC_COLUMNS), dtype=np.float64), True
    )
    assert minmax.shape == (0, 2)
    assert mean.shape == var.shape == count.shape == (0,)


@pytest.mark.parametrize("shape", [(8,), (8, 2), (2, 4, 5)])
def test_split_members_rejects_a_non_statistics_block(shape):
    with pytest.raises(ValueError, match="statistics block"):
        split_members(np.zeros(shape, dtype=np.float64), True)


def test_assemble_stats_inverts_split_members_up_to_float32_rounding():
    statistics, members = _members()
    out = assemble_statistics(*members)
    assert out.shape == statistics.shape
    assert out.dtype == np.float64
    assert np.array_equal(out, _stored(statistics), equal_nan=True)


def test_assemble_stats_without_var_leaves_the_variance_column_nan():
    statistics, members = _members(with_var=False)
    out = assemble_statistics(*members)
    want = _stored(statistics)
    want[:, VAR_COLUMN] = np.nan
    assert np.array_equal(out, want, equal_nan=True)


def test_assemble_stats_keeps_a_count_beyond_float32_precision_exact():
    count = np.array([2**30, 2**24 + 1], dtype=np.uint32)
    minmax = np.zeros((2, 2), dtype=np.float32)
    mean = np.zeros(2, dtype=np.float32)
    out = assemble_statistics(minmax, mean, None, count)
    assert out[:, COUNT_COLUMN].tolist() == [2.0**30, 2.0**24 + 1]


def test_assemble_stats_empty_level_yields_zero_rows():
    out = assemble_statistics(
        np.empty((0, 2), dtype=np.float32),
        np.empty(0, dtype=np.float32),
        None,
        np.empty(0, dtype=np.uint32),
    )
    assert out.shape == (0, STATISTIC_COLUMNS)


@pytest.mark.parametrize("shape", [(5,), (5, 3), (5, 2, 1)])
def test_assemble_stats_rejects_a_minmax_that_is_not_rows_by_two(shape):
    members = [np.zeros(5, dtype=np.float32) for _ in range(2)]
    with pytest.raises(ValueError, match="minmax"):
        assemble_statistics(
            np.zeros(shape, dtype=np.float32),
            members[0],
            members[1],
            np.zeros(5, dtype=np.uint32),
        )


@pytest.mark.parametrize("member", ["mean", "var", "count"])
def test_assemble_stats_rejects_a_member_with_the_wrong_row_count(member):
    arrays = {
        "mean": np.zeros(5, dtype=np.float32),
        "var": np.zeros(5, dtype=np.float32),
        "count": np.zeros(5, dtype=np.uint32),
    }
    arrays[member] = np.zeros(4, dtype=arrays[member].dtype)
    with pytest.raises(ValueError, match=member):
        assemble_statistics(
            np.zeros((5, 2), dtype=np.float32),
            arrays["mean"],
            arrays["var"],
            arrays["count"],
        )
