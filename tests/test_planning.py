import pytest

from timeseries_zarr.constants import MAX_LEVELS
from timeseries_zarr.planning import (
    level_bins,
    level_count,
    level_period_us,
    plan_levels,
    raw_shape,
    sample_period_us,
)


@pytest.mark.parametrize(
    ("num_samples", "expected"),
    [
        (0, 0),
        (1, 0),
        (500, 0),
        (4095, 0),
        (4096, 1),
        (16383, 1),
        (16384, 2),
    ],
)
def test_level_count_default_thresholds(num_samples, expected):
    assert level_count(num_samples) == expected


def test_level_count_caps_at_max_levels():
    assert level_count(10**13) == MAX_LEVELS


def test_level_count_custom_max_levels():
    assert level_count(10**12, max_levels=3) == 3


def test_level_count_zero_max_levels_gives_no_levels():
    assert level_count(10**12, max_levels=0) == 0


def test_level_count_custom_min_bins():
    assert level_count(400, min_bins=100) == 1


def test_level_count_threshold_off_by_one():
    # The gate counts complete bins: 399 samples make 99 and drop level 1, even
    # though level_bins keeps the partial bin and reports 100.
    assert level_count(399, min_bins=100) == 0
    assert level_count(400, min_bins=100) == 1


@pytest.mark.parametrize(
    ("num_samples", "level", "expected"),
    [
        (1000, 1, 250),
        (1000, 2, 63),
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
def test_level_bins(num_samples, level, expected):
    assert level_bins(num_samples, level) == expected


def test_level_bins_exact_for_huge_num_samples():
    # 2**53 + 1 has no exact float64 representation. A float-based ceil would
    # return one bin too few.
    assert level_bins(2**53 + 1, 1) == 2**51 + 1


@pytest.mark.parametrize("level", [0, -1])
def test_level_bins_rejects_a_level_below_one(level):
    with pytest.raises(ValueError, match="level"):
        level_bins(1000, level)


@pytest.mark.parametrize(
    ("rate_hz", "expected"),
    [(32000.0, 31.25), (1000.0, 1000.0), (512.0, 1953.125)],
)
def test_sample_period_us(rate_hz, expected):
    assert sample_period_us(rate_hz) == expected


@pytest.mark.parametrize("rate_hz", [0.0, -512.0])
def test_sample_period_us_rejects_a_non_positive_rate(rate_hz):
    with pytest.raises(ValueError, match="rate_hz"):
        sample_period_us(rate_hz)


@pytest.mark.parametrize(
    ("rate_hz", "level", "expected"),
    [
        (32000.0, 1, 125.0),
        (32000.0, 2, 500.0),
        (32000.0, 7, 512000.0),
        (1000.0, 1, 4000.0),
        (2000.0, 3, 32000.0),
    ],
)
def test_level_period_us(rate_hz, level, expected):
    assert level_period_us(rate_hz, level) == expected


@pytest.mark.parametrize("rate_hz", [1000.0, 2048.0, 30000.0, 32000.0])
def test_level_period_us_equals_the_specified_formula_at_every_level(rate_hz):
    for level in range(1, MAX_LEVELS + 1):
        assert level_period_us(rate_hz, level) == 4**level * 1e6 / rate_hz


def test_level_period_us_quadruples_exactly_per_level():
    for level in range(1, MAX_LEVELS):
        assert level_period_us(30000.0, level + 1) == 4 * level_period_us(
            30000.0, level
        )


@pytest.mark.parametrize("rate_hz", [0.0, -1.0])
def test_level_period_us_rejects_a_non_positive_rate(rate_hz):
    with pytest.raises(ValueError, match="rate_hz"):
        level_period_us(rate_hz, 1)


@pytest.mark.parametrize("level", [0, -1])
def test_level_period_us_rejects_a_level_below_one(level):
    with pytest.raises(ValueError, match="level"):
        level_period_us(32000.0, level)


@pytest.mark.parametrize("num_samples", [0, 1, 10**12])
def test_raw_shape_is_one_axis_of_samples(num_samples):
    assert raw_shape(num_samples) == (num_samples,)


def test_plan_levels_matches_building_blocks():
    n, rate_hz = 10_000_000, 32000.0
    plan = plan_levels(n, rate_hz)
    assert len(plan) == level_count(n)
    for k, lp in enumerate(plan, start=1):
        assert lp.level == k
        assert lp.bins == level_bins(n, k)
        assert lp.period_us == level_period_us(rate_hz, k)


def test_plan_levels_starts_at_level_one_with_no_raw_level():
    assert plan_levels(10_000_000, 32000.0)[0].level == 1


@pytest.mark.parametrize("num_samples", [0, 500])
def test_plan_levels_short_channel_has_no_levels(num_samples):
    assert plan_levels(num_samples, 32000.0) == []


def test_plan_levels_concrete_two_levels():
    plan = plan_levels(16384, 32000.0)
    assert [(lp.level, lp.bins, lp.period_us) for lp in plan] == [
        (1, 4096, 125.0),
        (2, 1024, 500.0),
    ]


def test_plan_levels_keeps_the_partial_trailing_bin():
    plan = plan_levels(16387, 32000.0)
    assert [lp.bins for lp in plan] == [4097, 1025]


def test_plan_levels_honors_max_levels_and_min_bins():
    assert len(plan_levels(10**6, 32000.0, max_levels=2)) == 2
    assert len(plan_levels(400, 32000.0, min_bins=100)) == 1


@pytest.mark.parametrize("num_samples", [0, 500, 10**6])
def test_plan_levels_rejects_a_non_positive_rate(num_samples):
    with pytest.raises(ValueError, match="rate_hz"):
        plan_levels(num_samples, 0.0)
