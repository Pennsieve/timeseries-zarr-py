import pytest

from timeseries_zarr.planning import (
    level_count,
    level_num_bins,
    level_period_us,
    level_shape,
    plan_levels,
)


@pytest.mark.parametrize(
    ("num_samples", "expected"),
    [
        (0, 0),
        (1, 0),
        (500, 0),
        (4095, 0),
        (4096, 1),
    ],
)
def test_level_count_default_thresholds(num_samples, expected):
    assert level_count(num_samples) == expected


def test_level_count_caps_at_max_levels():
    assert level_count(10**12) == 7


def test_level_count_custom_max_levels():
    assert level_count(10**12, max_levels=3) == 3


def test_level_count_custom_min_bins():
    assert level_count(400, min_bins=100) == 1


def test_level_count_threshold_off_by_one():
    # The gate counts complete bins: 399 samples make 99 and drop level 1, even
    # though level_num_bins keeps the partial bin and reports 100.
    assert level_count(399, min_bins=100) == 0
    assert level_count(400, min_bins=100) == 1


@pytest.mark.parametrize(
    ("num_samples", "level", "expected"),
    [
        (1000, 0, 1000),
        (0, 0, 0),
        (16, 2, 1),
        (5, 1, 2),
        (6, 1, 2),
        (7, 1, 2),
        (8, 1, 2),
        (9, 1, 3),
        (1, 1, 1),
        (3, 1, 1),
        (17, 2, 2),
        (0, 1, 0),
    ],
)
def test_level_num_bins(num_samples, level, expected):
    assert level_num_bins(num_samples, level) == expected


def test_level_num_bins_exact_for_huge_num_samples():
    # 2**53 + 1 has no exact float64 representation. A float-based ceil would
    # return one bin too few.
    assert level_num_bins(2**53 + 1, 1) == 2**51 + 1


@pytest.mark.parametrize(
    ("sample_period_us", "level", "expected"),
    [
        (31.25, 0, 31.25),
        (31.25, 1, 125.0),
        (31.25, 2, 500.0),
        (31.25, 7, 512000.0),
        (1000.0, 1, 4000.0),
    ],
)
def test_level_period_us(sample_period_us, level, expected):
    assert level_period_us(sample_period_us, level) == expected


@pytest.mark.parametrize(
    ("num_samples", "level", "expected"),
    [
        (1000, 1, (250, 2)),
        (1000, 2, (63, 2)),
        (17, 2, (2, 2)),
        (0, 1, (0, 2)),
    ],
)
def test_level_shape(num_samples, level, expected):
    assert level_shape(num_samples, level) == expected


def test_plan_levels_matches_building_blocks():
    n, p0 = 10_000_000, 31.25
    plan = plan_levels(n, p0)
    assert len(plan) == level_count(n)
    for index, lp in enumerate(plan):
        level = index + 1
        assert lp.level == level
        assert lp.shape == level_shape(n, level)
        assert lp.period_us == level_period_us(p0, level)


def test_plan_levels_small_n_has_no_levels():
    # Too short for one level's worth of bins, so it carries raw alone.
    assert plan_levels(500, 31.25) == []


def test_plan_levels_concrete_one_level():
    plan = plan_levels(4096, 31.25)
    assert len(plan) == 1
    assert plan[0].level == 1
    assert plan[0].shape == (1024, 2)
    assert plan[0].period_us == 125.0
