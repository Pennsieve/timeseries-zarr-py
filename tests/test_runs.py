from fractions import Fraction

import numpy as np
import pytest

from timeseries_zarr.runs import (
    Placement,
    Run,
    check_run_bound,
    check_source_runs,
    place_runs,
)

RATE_HZ = 1000.0
PERIOD_US = 1000
REFERENCE_US = 5_000_000


def _assert_satisfies_the_spec(
    placement: Placement, rate_hz: float, runs: list[Run], reference_us: int
) -> None:
    """Check every rule that the spec states for a runs table."""
    table = placement.table
    period = Fraction(1_000_000) / Fraction(rate_hz)
    assert table.dtype == np.int64
    assert table.shape == (len(runs), 3)
    assert tuple(table[0]) == (0, runs[0].num_samples, placement.offset_us)
    firsts, counts, times = table[:, 0], table[:, 1], table[:, 2]
    assert np.all(np.diff(firsts) > 0)
    assert np.all(np.diff(times) > 0)
    assert np.all(counts >= 1)
    assert list(counts) == [run.num_samples for run in runs]
    assert np.all(firsts[1:] >= firsts[:-1] + counts[:-1])
    assert firsts[-1] + counts[-1] == placement.num_samples
    for first, time_us in zip(firsts, times, strict=True):
        grid_us = placement.offset_us + int(first) * period
        assert abs(int(time_us) - grid_us) < period
    for run, time_us in zip(runs, times, strict=True):
        assert time_us == run.start_us - reference_us


def _place(runs: list[Run], rate_hz: float = RATE_HZ) -> Placement:
    placement = place_runs(runs, rate_hz, REFERENCE_US)
    _assert_satisfies_the_spec(placement, rate_hz, runs, REFERENCE_US)
    return placement


def test_one_run_places_at_sample_zero():
    placement = _place([Run(REFERENCE_US + 250, 100)])
    assert placement.offset_us == 250
    assert placement.num_samples == 100
    assert placement.table.tolist() == [[0, 100, 250]]


def test_a_resume_on_the_grid_leaves_whole_nan_samples():
    placement = _place([Run(6_000_000, 10), Run(6_015_000, 5)])
    assert placement.num_samples == 15 + 5
    assert placement.table.tolist() == [
        [0, 10, 1_000_000],
        [15, 5, 1_015_000],
    ]


def test_a_resume_with_no_pause_follows_the_previous_run():
    placement = _place([Run(6_000_000, 10), Run(6_010_000, 5)])
    assert placement.table.tolist() == [
        [0, 10, 1_000_000],
        [10, 5, 1_010_000],
    ]
    assert placement.num_samples == 15


def test_a_resume_just_under_half_a_period_late_keeps_the_sample():
    placement = _place([Run(6_000_000, 10), Run(6_015_499, 5)])
    assert placement.table.tolist()[1] == [15, 5, 1_015_499]


def test_a_resume_exactly_half_a_period_late_goes_to_the_next_sample():
    placement = _place([Run(6_000_000, 10), Run(6_015_500, 5)])
    assert placement.table.tolist()[1] == [16, 5, 1_015_500]


def test_a_resume_just_under_half_a_period_early_keeps_the_sample():
    placement = _place([Run(6_000_000, 10), Run(6_014_501, 5)])
    assert placement.table.tolist()[1] == [15, 5, 1_014_501]


def test_a_resume_exactly_half_a_period_early_goes_to_the_later_sample():
    placement = _place([Run(6_000_000, 10), Run(6_014_500, 5)])
    assert placement.table.tolist()[1] == [15, 5, 1_014_500]


def test_a_resume_just_over_half_a_period_early_goes_to_the_earlier_sample():
    placement = _place([Run(6_000_000, 10), Run(6_014_499, 5)])
    assert placement.table.tolist()[1] == [14, 5, 1_014_499]


def test_a_pause_shorter_than_a_period_leaves_no_nan_and_keeps_the_time():
    placement = _place([Run(6_000_000, 10), Run(6_010_400, 5)])
    assert placement.num_samples == 15
    assert placement.table.tolist()[1] == [10, 5, 1_010_400]


def test_a_pause_of_more_than_half_a_period_skips_one_sample():
    placement = _place([Run(6_000_000, 10), Run(6_010_600, 5)])
    assert placement.table.tolist()[1] == [11, 5, 1_010_600]
    assert placement.num_samples == 16


def test_a_resume_one_microsecond_after_the_previous_run_ends_places_with_no_nan():
    placement = _place([Run(6_000_000, 10), Run(6_010_001, 5)])
    assert placement.table.tolist()[1] == [10, 5, 1_010_001]


def test_error_does_not_accumulate_across_many_runs():
    runs = [Run(6_000_000 + 10_437 * i, 10) for i in range(50)]
    placement = _place(runs)
    assert placement.num_samples >= 50 * 10


def test_a_run_before_the_reference_time_has_a_negative_offset():
    placement = _place(
        [Run(REFERENCE_US - 3_000, 4), Run(REFERENCE_US + 2_100, 4)]
    )
    assert placement.offset_us == -3_000
    assert placement.table.tolist() == [[0, 4, -3_000], [5, 4, 2_100]]
    assert placement.num_samples == 9


def test_a_later_run_before_the_reference_time_has_a_negative_time():
    placement = _place(
        [Run(REFERENCE_US - 9_000, 4), Run(REFERENCE_US - 2_000, 4)]
    )
    assert placement.table.tolist() == [[0, 4, -9_000], [7, 4, -2_000]]


def test_a_fractional_period_uses_exact_arithmetic():
    rate_hz = 3.0
    runs = [Run(REFERENCE_US, 3), Run(REFERENCE_US + 2_100_000, 3)]
    placement = _place(runs, rate_hz)
    assert placement.table.tolist() == [[0, 3, 0], [6, 3, 2_100_000]]
    assert placement.num_samples == 9


def test_a_numpy_integer_start_and_count_are_accepted():
    runs = [Run(np.int64(6_000_000), np.int64(4))]
    assert place_runs(runs, RATE_HZ, REFERENCE_US).num_samples == 4


def test_runs_out_of_order_raise():
    with pytest.raises(
        ValueError, match="run 1 starts at 5999000 us, before run 0"
    ):
        place_runs(
            [Run(6_000_000, 2), Run(5_999_000, 2)], RATE_HZ, REFERENCE_US
        )


def test_a_run_that_starts_before_the_previous_run_ends_raises():
    with pytest.raises(ValueError, match="at least one sample period"):
        place_runs(
            [Run(6_000_000, 10), Run(6_009_999, 2)], RATE_HZ, REFERENCE_US
        )


def test_a_run_that_starts_inside_the_previous_run_raises():
    with pytest.raises(ValueError, match="before the samples of run 0 end"):
        place_runs(
            [Run(6_000_000, 10), Run(6_003_000, 2)], RATE_HZ, REFERENCE_US
        )


def test_a_run_of_zero_samples_raises():
    with pytest.raises(ValueError, match="run 1 has 0 samples"):
        place_runs(
            [Run(6_000_000, 10), Run(6_020_000, 0)], RATE_HZ, REFERENCE_US
        )


def test_a_first_run_of_zero_samples_raises():
    with pytest.raises(ValueError, match="run 0 has 0 samples"):
        place_runs([Run(6_000_000, 0)], RATE_HZ, REFERENCE_US)


def test_no_runs_raise():
    with pytest.raises(ValueError, match="at least one run"):
        place_runs([], RATE_HZ, REFERENCE_US)


@pytest.mark.parametrize("rate_hz", [0.0, -1.0, float("nan"), float("inf")])
def test_a_rate_that_is_not_positive_and_finite_raises(rate_hz):
    with pytest.raises(ValueError, match="rate_hz"):
        place_runs([Run(0, 1)], rate_hz, 0)


def test_a_non_integer_start_raises():
    with pytest.raises(TypeError, match="run 0 start_us"):
        place_runs([Run(1.5, 1)], RATE_HZ, 0)  # type: ignore[arg-type]


def test_a_non_integer_count_raises():
    with pytest.raises(TypeError, match="run 0 num_samples"):
        place_runs([Run(0, 1.5)], RATE_HZ, 0)  # type: ignore[arg-type]


def test_a_start_that_breaks_the_period_bound_raises():
    with pytest.raises(
        ValueError, match="under one sample period of 1000.0 us"
    ):
        check_run_bound(first=10, time_us=11_000, offset_us=0, rate_hz=RATE_HZ)


def test_a_start_one_period_early_raises():
    with pytest.raises(ValueError, match="under one sample period"):
        check_run_bound(first=10, time_us=9_000, offset_us=0, rate_hz=RATE_HZ)


def test_a_start_just_under_one_period_off_passes_at_both_signs():
    check_run_bound(first=10, time_us=10_999, offset_us=0, rate_hz=RATE_HZ)
    check_run_bound(first=10, time_us=9_001, offset_us=0, rate_hz=RATE_HZ)


def test_the_bound_uses_the_offset():
    check_run_bound(first=2, time_us=7_500, offset_us=5_000, rate_hz=RATE_HZ)
    with pytest.raises(ValueError, match="under one sample period"):
        check_run_bound(
            first=2, time_us=8_000, offset_us=5_000, rate_hz=RATE_HZ
        )


def test_source_runs_that_agree_with_the_channel_pass():
    check_source_runs([Run(100, 4), Run(9_000, 6)], 100, 10)


def test_a_first_run_that_misses_start_us_raises():
    with pytest.raises(
        ValueError, match="first run starts at 101 us, but start_us is 100"
    ):
        check_source_runs([Run(101, 4)], 100, 4)


def test_run_lengths_that_miss_num_samples_raise():
    with pytest.raises(
        ValueError, match="runs hold 10 samples, but num_samples is 9"
    ):
        check_source_runs([Run(100, 4), Run(9_000, 6)], 100, 9)


def test_empty_source_runs_raise():
    with pytest.raises(ValueError, match="at least one run"):
        check_source_runs([], 0, 0)
