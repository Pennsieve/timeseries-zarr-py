import numpy as np
import pytest

from timeseries_zarr.planning import (
    level_bins,
    level_count,
    level_period_us,
    max_levels_for,
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


@pytest.mark.parametrize(
    ("decimation_factor", "expected"), [(2, 31), (4, 15), (3, 20), (10, 9)]
)
def test_max_levels_for(decimation_factor, expected):
    assert max_levels_for(decimation_factor) == expected


@pytest.mark.parametrize("decimation_factor", [2, 3, 4, 5, 10, 1000])
def test_max_levels_for_is_the_last_level_whose_width_fits_uint32(
    decimation_factor,
):
    limit = np.iinfo(np.uint32).max
    levels = max_levels_for(decimation_factor)
    assert decimation_factor**levels <= limit
    assert decimation_factor ** (levels + 1) > limit


@pytest.mark.parametrize("decimation_factor", [1, 0, -4])
def test_max_levels_for_rejects_a_factor_below_2(decimation_factor):
    with pytest.raises(ValueError, match="decimation_factor"):
        max_levels_for(decimation_factor)


def test_level_count_caps_at_the_most_levels_that_fit():
    assert level_count(10**13) == max_levels_for(4)
    assert level_count(2**40, decimation_factor=2, min_bins=1) == 31
    assert level_count(2**40, decimation_factor=10, min_bins=1) == 9


def test_level_count_at_a_factor_of_10_gates_on_complete_bins():
    assert level_count(10_240, min_bins=1024, decimation_factor=10) == 1
    assert level_count(10_239, min_bins=1024, decimation_factor=10) == 0
    assert level_count(1_024_000, min_bins=1024, decimation_factor=10) == 3


def test_level_count_rejects_a_factor_below_2():
    with pytest.raises(ValueError, match="decimation_factor"):
        level_count(1000, decimation_factor=1)


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
    ("num_samples", "samples_per_bin", "expected"),
    [
        (1000, 4, 250),
        (1000, 16, 63),
        (16, 16, 1),
        (5, 4, 2),
        (6, 4, 2),
        (7, 4, 2),
        (8, 4, 2),
        (9, 4, 3),
        (1, 4, 1),
        (3, 4, 1),
        (17, 16, 2),
        (0, 4, 0),
        (1000, 2, 500),
        (1001, 2, 501),
        (1000, 10, 100),
        (1001, 10, 101),
        (1000, 6, 167),
    ],
)
def test_level_bins(num_samples, samples_per_bin, expected):
    assert level_bins(num_samples, samples_per_bin) == expected


def test_level_bins_exact_for_huge_num_samples():
    # 2**53 + 1 has no exact float64 representation. A float-based ceil would
    # return one bin too few.
    assert level_bins(2**53 + 1, 4) == 2**51 + 1


@pytest.mark.parametrize("rate_hz", [float("nan"), float("inf"), 0.0, -1.0])
def test_plan_levels_rejects_a_rate_that_is_not_finite_and_positive(rate_hz):
    with pytest.raises(ValueError, match="rate_hz must be finite and positive"):
        plan_levels(0, rate_hz)
    with pytest.raises(ValueError, match="rate_hz must be finite and positive"):
        level_period_us(rate_hz, 4)
    with pytest.raises(ValueError, match="rate_hz must be finite and positive"):
        sample_period_us(rate_hz)


@pytest.mark.parametrize("samples_per_bin", [1, 0, -4])
def test_level_bins_rejects_a_width_below_2(samples_per_bin):
    with pytest.raises(ValueError, match="samples_per_bin"):
        level_bins(1000, samples_per_bin)


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
    ("rate_hz", "samples_per_bin", "expected"),
    [
        (32000.0, 4, 125.0),
        (32000.0, 16, 500.0),
        (32000.0, 4**7, 512000.0),
        (1000.0, 4, 4000.0),
        (2000.0, 64, 32000.0),
        (1000.0, 2, 2000.0),
        (1000.0, 10, 10000.0),
    ],
)
def test_level_period_us(rate_hz, samples_per_bin, expected):
    assert level_period_us(rate_hz, samples_per_bin) == expected


@pytest.mark.parametrize("rate_hz", [1000.0, 2048.0, 30000.0, 32000.0])
@pytest.mark.parametrize("decimation_factor", [2, 4, 10])
def test_level_period_us_equals_the_bin_width_over_the_rate(
    rate_hz, decimation_factor
):
    for level in range(1, max_levels_for(decimation_factor) + 1):
        width = decimation_factor**level
        assert level_period_us(rate_hz, width) == width * 1e6 / rate_hz


@pytest.mark.parametrize("rate_hz", [0.0, -1.0])
def test_level_period_us_rejects_a_non_positive_rate(rate_hz):
    with pytest.raises(ValueError, match="rate_hz"):
        level_period_us(rate_hz, 4)


@pytest.mark.parametrize("samples_per_bin", [1, 0, -4])
def test_level_period_us_rejects_a_width_below_2(samples_per_bin):
    with pytest.raises(ValueError, match="samples_per_bin"):
        level_period_us(32000.0, samples_per_bin)


@pytest.mark.parametrize("num_samples", [0, 1, 10**12])
def test_raw_shape_is_one_axis_of_samples(num_samples):
    assert raw_shape(num_samples) == (num_samples,)


def test_plan_levels_matches_building_blocks():
    n, rate_hz = 10_000_000, 32000.0
    plan = plan_levels(n, rate_hz)
    assert len(plan) == level_count(n)
    for k, lp in enumerate(plan, start=1):
        assert lp.level == k
        assert lp.bins == level_bins(n, 4**k)
        assert lp.period_us == level_period_us(rate_hz, 4**k)


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


def test_plan_levels_at_a_factor_of_2_halves_each_width():
    plan = plan_levels(10_000, 1000.0, min_bins=1000, decimation_factor=2)
    assert [(lp.level, lp.bins, lp.period_us) for lp in plan] == [
        (1, 5000, 2000.0),
        (2, 2500, 4000.0),
        (3, 1250, 8000.0),
    ]


def test_plan_levels_at_a_factor_of_10_widens_each_level_tenfold():
    plan = plan_levels(100_001, 1000.0, min_bins=10, decimation_factor=10)
    assert [(lp.level, lp.bins, lp.period_us) for lp in plan] == [
        (1, 10_001, 10_000.0),
        (2, 1001, 100_000.0),
        (3, 101, 1_000_000.0),
        (4, 11, 10_000_000.0),
    ]


def test_plan_levels_at_a_factor_of_2_plans_more_than_15_levels():
    plan = plan_levels(2**22, 512.0, min_bins=1, decimation_factor=2)
    assert len(plan) == 22
    assert len(plan_levels(2**22, 512.0, min_bins=1)) == 11


def test_plan_levels_rejects_a_factor_below_2():
    with pytest.raises(ValueError, match="decimation_factor"):
        plan_levels(10**6, 512.0, decimation_factor=1)


def test_plan_levels_honors_max_levels_and_min_bins():
    assert len(plan_levels(10**6, 32000.0, max_levels=2)) == 2
    assert len(plan_levels(400, 32000.0, min_bins=100)) == 1


@pytest.mark.parametrize("num_samples", [0, 500, 10**6])
def test_plan_levels_rejects_a_non_positive_rate(num_samples):
    with pytest.raises(ValueError, match="rate_hz"):
        plan_levels(num_samples, 0.0)
