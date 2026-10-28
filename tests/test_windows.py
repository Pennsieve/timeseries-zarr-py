import math
from fractions import Fraction

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from timeseries_zarr.constants import DEFAULT_DECIMATION_FACTOR
from timeseries_zarr.planning import level_bins
from timeseries_zarr.windows import (
    array_error,
    bin_period,
    canonical_number,
    check_label_range,
    check_level_arrays,
    check_level_numbers,
    check_not_string,
    duration_us,
    index_range,
    integer_argument,
    merge_spans,
    select_level,
    time_axis,
    window_bounds,
)

ONE_DAY_US = 86_400 * 10**6


def _entry_masks(times, offset_us, period, length):
    """Return, per time, the entries that end after it and that start before it.

    Every mask holds one exact Fraction comparison per grid entry. The None
    key stands for an open bound and marks every entry.
    """
    starts = [offset_us + i * period for i in range(length)]
    ends = [start + period for start in starts]
    every_entry = np.ones(length, dtype=bool)
    ends_after = {None: every_entry}
    starts_before = {None: every_entry}
    for time_us in times:
        ends_after[time_us] = np.array([end > time_us for end in ends])
        starts_before[time_us] = np.array([start < time_us for start in starts])
    return ends_after, starts_before


def _boundary_times(offset_us, period, length):
    """Return integer times beside every entry boundary and outside the grid."""
    times = {offset_us - 10 * math.ceil(period), offset_us - 1}
    for i in range(length + 1):
        boundary = offset_us + i * period
        low, high = math.floor(boundary), math.ceil(boundary)
        times.update((low - 1, low, high, high + 1))
    times.add(offset_us + math.ceil((length + 10) * period))
    return sorted(times)


def _level_five_populated():
    """Return the populated bins of a 600 s level 5 grid at 512 Hz."""
    populated = np.ones(300, dtype=bool)
    populated[[60, 61]] = False
    return populated


def test_bin_period_is_exact_at_512_hz():
    assert bin_period(512.0, 1) == Fraction(15625, 8)


def test_bin_period_is_exact_at_30_khz():
    assert bin_period(30000.0, 1) == Fraction(100, 3)


def test_bin_period_scales_with_samples_per_bin():
    assert bin_period(256.0, DEFAULT_DECIMATION_FACTOR) == 15625
    assert bin_period(512.0, DEFAULT_DECIMATION_FACTOR**5) == 2_000_000


@pytest.mark.parametrize("rate_hz", [0.0, -512.0, math.nan, math.inf])
def test_bin_period_rejects_a_rate_that_is_not_positive_and_finite(rate_hz):
    with pytest.raises(ValueError, match="rate_hz must be positive and finite"):
        bin_period(rate_hz, 1)


@pytest.mark.parametrize("samples_per_bin", [0, -4])
def test_bin_period_rejects_samples_per_bin_below_one(samples_per_bin):
    with pytest.raises(ValueError, match="samples_per_bin must be at least 1"):
        bin_period(512.0, samples_per_bin)


def test_duration_us_of_3000_samples_at_30_khz_is_100_ms():
    assert duration_us(3000, 30000.0) == 100_000


def test_duration_us_rounds_a_partial_microsecond_up():
    assert duration_us(1, 30000.0) == 34
    assert duration_us(1, 44100.0) == 23


def test_duration_us_is_exact_where_float_multiplication_rounds_up():
    assert duration_us(15, 30000.0) == 500
    assert duration_us(237, 30000.0) == 7900


def test_duration_us_of_no_samples_is_zero():
    assert duration_us(0, 512.0) == 0


def test_duration_us_gives_the_end_of_a_10_minute_channel():
    assert 500_000 + duration_us(153_600, 256.0) == 600_500_000


def test_duration_us_rejects_a_rate_that_is_not_positive():
    with pytest.raises(ValueError, match="rate_hz must be positive"):
        duration_us(10, 0.0)


@pytest.mark.parametrize(
    ("num_samples", "rate_hz", "offset_us"),
    [
        (1001, 30000.0, 0),
        (1001, 44100.0, 123_457),
        (614_400, 1024.0, 0),
        (153_600, 256.0, 500_000),
        (7, 512.0, ONE_DAY_US),
    ],
)
def test_a_window_of_duration_us_covers_every_sample(
    num_samples, rate_hz, offset_us
):
    end_us = offset_us + duration_us(num_samples, rate_hz)
    period = bin_period(rate_hz, 1)
    assert index_range(offset_us, end_us, offset_us, period, num_samples) == (
        0,
        num_samples,
    )


def test_index_range_with_open_bounds_covers_the_grid():
    assert index_range(None, None, 0, Fraction(100), 10) == (0, 10)


def test_index_range_with_an_open_start_begins_at_zero():
    assert index_range(None, 450, 0, Fraction(100), 10) == (0, 5)


def test_index_range_with_an_open_end_stops_at_the_length():
    assert index_range(450, None, 0, Fraction(100), 10) == (4, 10)


def test_index_range_clamps_a_window_wider_than_the_grid():
    assert index_range(-5_000, 5_000, 1_000, Fraction(100), 10) == (0, 10)


def test_index_range_clamps_a_window_that_starts_before_the_grid():
    assert index_range(500, 1_250, 1_000, Fraction(100), 10) == (0, 3)


def test_index_range_clamps_a_window_that_ends_after_the_grid():
    assert index_range(1_750, 9_000, 1_000, Fraction(100), 10) == (7, 10)


def test_index_range_of_a_window_before_the_grid_is_empty_at_zero():
    assert index_range(0, 999, 1_000, Fraction(100), 10) == (0, 0)


def test_index_range_of_a_window_that_ends_at_the_grid_start_is_empty():
    assert index_range(0, 1_000, 1_000, Fraction(100), 10) == (0, 0)


def test_index_range_of_a_window_after_the_grid_is_empty_at_the_length():
    assert index_range(2_000, 3_000, 1_000, Fraction(100), 10) == (10, 10)


def test_index_range_of_a_window_that_starts_at_the_grid_end_is_empty():
    assert index_range(2_000, None, 1_000, Fraction(100), 10) == (10, 10)


@pytest.mark.parametrize(("time_us", "index"), [(1_250, 2), (1_300, 3)])
def test_index_range_of_equal_bounds_is_empty(time_us, index):
    assert index_range(time_us, time_us, 1_000, Fraction(100), 10) == (
        index,
        index,
    )


def test_index_range_of_a_grid_with_no_entries_is_empty():
    assert index_range(0, 10**9, 0, Fraction(100), 0) == (0, 0)


def test_index_range_rejects_an_end_before_the_start():
    with pytest.raises(ValueError, match="end_us must be at or after start_us"):
        index_range(1_000, 999, 0, Fraction(100), 10)


@pytest.mark.parametrize(
    ("start_us", "end_us", "name"),
    [
        (1.5, None, "start_us"),
        (1_000.0, 2_000, "start_us"),
        (None, 2.0, "end_us"),
        (0, np.float64(3.0), "end_us"),
    ],
)
def test_index_range_rejects_a_float_bound(start_us, end_us, name):
    with pytest.raises(
        TypeError, match=f"{name} must be an integer count of microseconds"
    ):
        index_range(start_us, end_us, 0, Fraction(100), 10)


def test_window_bounds_return_ints_and_keep_an_open_side():
    assert window_bounds(np.int64(5), None) == (5, None)
    assert type(window_bounds(np.int64(5), None)[0]) is int
    assert window_bounds(None, np.uint32(9)) == (None, 9)
    assert window_bounds(7, 7) == (7, 7)


def test_window_bounds_reject_an_end_before_the_start():
    message = (
        "end_us must be at or after start_us, got start_us 10 and end_us 9"
    )
    with pytest.raises(ValueError, match=message):
        window_bounds(10, 9)


def test_window_bounds_reject_a_float_bound():
    message = "end_us must be an integer count of microseconds, got 2.5"
    with pytest.raises(TypeError, match=message):
        window_bounds(0, 2.5)


def test_integer_argument_returns_an_int_for_a_numpy_integer():
    value = integer_argument(np.uint16(3), "bins")
    assert value == 3
    assert type(value) is int


@pytest.mark.parametrize(
    ("value", "expectation", "message"),
    [
        (2.0, None, "bins must be an integer, got 2.0"),
        ("3", None, "bins must be an integer, got '3'"),
        (None, "an integer count", "bins must be an integer count, got None"),
    ],
)
def test_integer_argument_rejects_a_value_that_is_not_an_integer(
    value, expectation, message
):
    options = {} if expectation is None else {"expectation": expectation}
    with pytest.raises(TypeError, match=f"^{message}$"):
        integer_argument(value, "bins", **options)


def test_index_range_accepts_numpy_integer_bounds():
    result = index_range(np.int64(150), np.uint32(450), 0, Fraction(100), 10)
    assert result == (1, 5)
    assert all(type(index) is int for index in result)


def test_index_range_excludes_a_bin_that_starts_at_end_us():
    period = bin_period(512.0, DEFAULT_DECIMATION_FACTOR**3)
    assert period == 125_000
    assert index_range(0, 125_000, 0, period, 10) == (0, 1)
    assert index_range(0, 125_001, 0, period, 10) == (0, 2)


def test_index_range_excludes_a_bin_that_ends_at_start_us():
    period = bin_period(512.0, DEFAULT_DECIMATION_FACTOR**3)
    assert index_range(125_000, 250_000, 0, period, 10) == (1, 2)
    assert index_range(124_999, 250_000, 0, period, 10) == (0, 2)


def test_index_range_starts_at_sample_30_for_a_start_of_1000_us_at_30_khz():
    assert math.floor(1000 / (10**6 / 30000.0)) == 29
    assert index_range(1000, None, 0, bin_period(30000.0, 1), 100) == (30, 100)


@pytest.mark.parametrize(
    ("rate_hz", "start_us", "end_us", "length", "stop"),
    [(48000.0, 0, 2_625, 200, 126), (3000.0, None, 21_000, 100, 63)],
    ids=["48 kHz", "3 kHz"],
)
def test_index_range_stops_before_the_sample_that_starts_at_end_us(
    rate_hz, start_us, end_us, length, stop
):
    assert math.ceil(end_us / (10**6 / rate_hz)) == stop + 1
    period = bin_period(rate_hz, 1)
    assert index_range(start_us, end_us, 0, period, length) == (0, stop)


def test_index_range_snaps_raw_samples_outward_at_an_offset():
    period = bin_period(256.0, 1)
    assert index_range(500_000, 503_907, 500_000, period, 153_600) == (0, 2)
    assert index_range(500_000, 503_906, 500_000, period, 153_600) == (0, 1)


def test_index_range_reads_the_last_128_samples_of_an_offset_channel():
    period = bin_period(256.0, 1)
    first, stop = index_range(
        600_000_000, 700_000_000, 500_000, period, 153_600
    )
    assert (first, stop) == (153_472, 153_600)


def test_index_range_reads_level_one_bins_of_an_offset_channel():
    period = bin_period(256.0, DEFAULT_DECIMATION_FACTOR)
    length = level_bins(153_600, DEFAULT_DECIMATION_FACTOR)
    assert index_range(0, 1_000_000, 500_000, period, length) == (0, 32)
    assert index_range(0, 500_000, 500_000, period, length) == (0, 0)


@pytest.mark.parametrize("level", range(6))
@pytest.mark.parametrize(
    ("rate_hz", "offset_us"),
    [(512.0, ONE_DAY_US), (30000.0, 0), (44100.0, 123_457)],
)
def test_index_range_matches_a_brute_force_overlap_reference(
    rate_hz, offset_us, level
):
    period = bin_period(rate_hz, DEFAULT_DECIMATION_FACTOR**level)
    # Float division misplaces 30 kHz boundaries from entry 15 on.
    length = 32
    times = _boundary_times(offset_us, period, length)
    ends_after, starts_before = _entry_masks(times, offset_us, period, length)
    bounds = [None, *times]
    for start_us in bounds:
        for end_us in bounds:
            if (
                start_us is not None
                and end_us is not None
                and end_us < start_us
            ):
                continue
            first, stop = index_range(
                start_us, end_us, offset_us, period, length
            )
            assert 0 <= first <= stop <= length
            assert first == np.count_nonzero(~ends_after[start_us])
            if start_us is not None and start_us == end_us:
                assert first == stop
            else:
                overlap = ends_after[start_us] & starts_before[end_us]
                assert (
                    list(range(first, stop)) == np.flatnonzero(overlap).tolist()
                )


def test_time_axis_at_512_hz_holds_exact_multiples_of_the_period():
    times = time_axis(0, 4096, 0, 1953.125)
    assert times.dtype == np.float64
    np.testing.assert_array_equal(times / 1953.125, np.arange(4096))


def test_time_axis_at_512_hz_is_exact_after_a_1_day_offset():
    first = 10**6
    times = time_axis(first, 100, ONE_DAY_US, 1953.125)
    exact = [
        float(ONE_DAY_US + (first + i) * Fraction(15625, 8)) for i in range(100)
    ]
    assert times.tolist() == exact


def test_time_axis_at_30_khz_stays_near_exact_far_from_the_reference_time():
    first = 10**8
    times = time_axis(first, 1000, 0, 10**6 / 30000.0)
    exact = np.array(
        [float((first + i) * Fraction(100, 3)) for i in range(1000)]
    )
    assert np.max(np.abs(times - exact)) <= 1e-6


def test_time_axis_at_30_khz_stays_near_exact_over_1_million_entries():
    times = time_axis(0, 10**6, 0, 10**6 / 30000.0)
    exact = (np.arange(10**6, dtype=np.int64) * 100) / 3
    assert np.max(np.abs(times - exact)) <= 1e-6


def test_time_axis_adds_the_offset():
    np.testing.assert_array_equal(
        time_axis(2, 3, 500_000, 3906.25), [507_812.5, 511_718.75, 515_625.0]
    )


def test_time_axis_of_no_entries_is_empty():
    times = time_axis(5, 0, 0, 1953.125)
    assert times.shape == (0,)
    assert times.dtype == np.float64


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ({1: 153_600, 2: 38_400, 3: 9_600, 4: 2_400, 5: 600}, 4),
        ({1: 7_680, 2: 1_920, 3: 480, 4: 120, 5: 30}, 2),
        ({1: 512, 2: 128, 3: 32, 4: 8, 5: 2}, 0),
    ],
)
def test_select_level_picks_the_coarsest_level_with_enough_bins(
    counts, expected
):
    assert select_level(counts, 1200) == expected


def test_select_level_accepts_a_level_with_exactly_bins_entries():
    assert select_level({1: 4_800, 2: 1_200, 3: 300}, 1200) == 2


def test_select_level_ignores_the_order_of_the_mapping():
    assert select_level({3: 5_000, 1: 80_000, 2: 20_000}, 4000) == 3


def test_select_level_with_no_levels_is_zero():
    assert select_level({}, 1) == 0


@pytest.mark.parametrize("bins", [0, -1])
def test_select_level_rejects_bins_below_one(bins):
    with pytest.raises(ValueError, match="bins must be at least 1"):
        select_level({1: 100}, bins)


@pytest.mark.parametrize(
    ("start_us", "end_us", "level", "first", "count"),
    [
        (0, 600_000_000, 4, 0, 2_400),
        (300_000_000, 330_000_000, 2, 19_200, 1_920),
        (312_000_000, 314_000_000, 0, 319_488, 2_048),
    ],
)
def test_select_level_picks_levels_4_2_and_0_for_three_windows_of_a_1024_hz_channel(
    start_us, end_us, level, first, count
):
    num_samples, rate_hz = 614_400, 1024.0
    ranges = {
        k: index_range(
            start_us,
            end_us,
            0,
            bin_period(rate_hz, DEFAULT_DECIMATION_FACTOR**k),
            level_bins(num_samples, DEFAULT_DECIMATION_FACTOR**k),
        )
        for k in range(1, 6)
    }
    counts = {k: stop - begin for k, (begin, stop) in ranges.items()}
    assert select_level(counts, 1200) == level
    begin, stop = index_range(
        start_us,
        end_us,
        0,
        bin_period(rate_hz, DEFAULT_DECIMATION_FACTOR**level),
        level_bins(num_samples, DEFAULT_DECIMATION_FACTOR**level)
        if level
        else num_samples,
    )
    assert (begin, stop - begin) == (first, count)


def test_select_level_falls_back_to_raw_when_level_one_is_one_bin_short():
    period_by_level = {
        k: bin_period(256.0, DEFAULT_DECIMATION_FACTOR**k) for k in range(1, 6)
    }
    counts = {}
    for k, period in period_by_level.items():
        begin, stop = index_range(
            0,
            1_000_000,
            500_000,
            period,
            level_bins(153_600, DEFAULT_DECIMATION_FACTOR**k),
        )
        counts[k] = stop - begin
    assert counts[1] == 32
    assert select_level(counts, 33) == 0


def test_merge_spans_of_no_entries_is_empty():
    populated = np.zeros(0, dtype=bool)
    assert merge_spans(populated, 0, 0, Fraction(100), 0, 10**6, 0) == []


def test_merge_spans_of_unpopulated_entries_is_empty():
    populated = np.zeros(8, dtype=bool)
    assert merge_spans(populated, 0, 0, Fraction(100), 0, 10**6, 0) == []


def test_merge_spans_maps_index_i_to_entry_first_plus_i():
    populated = np.array([True, True, False, True])
    assert merge_spans(populated, 10, 500, Fraction(100), 0, 10**6, 0) == [
        (1_500, 1_700),
        (1_800, 1_900),
    ]


def test_merge_spans_clamps_a_span_at_both_limits():
    populated = np.ones(10, dtype=bool)
    assert merge_spans(populated, 0, 0, Fraction(100), 250, 730, 0) == [
        (250, 730)
    ]


def test_merge_spans_drops_spans_that_clamp_to_nothing():
    populated = np.array([True, True, False, True, False, False, True])
    assert merge_spans(populated, 0, 0, Fraction(100), 300, 600, 0) == [
        (300, 400)
    ]


def test_merge_spans_widens_a_span_outward_to_whole_microseconds():
    populated = np.array([False, True, False])
    assert merge_spans(
        populated, 0, 0, bin_period(30000.0, 1), 0, 10**6, 0
    ) == [(33, 67)]


def test_merge_spans_ends_a_run_on_an_exact_boundary_at_30_khz():
    populated = np.ones(15, dtype=bool)
    assert merge_spans(
        populated, 0, 0, bin_period(30000.0, 1), 0, 10**6, 0
    ) == [(0, 500)]


def test_merge_spans_starts_a_run_on_an_exact_boundary_at_48_khz():
    populated = np.ones(10, dtype=bool)
    assert merge_spans(
        populated, 390, 0, bin_period(48000.0, 1), 0, 10**6, 0
    ) == [(8125, 8334)]


def test_merge_spans_joins_spans_whose_rounded_bounds_meet():
    populated = np.array([True, False, True])
    assert merge_spans(populated, 0, 0, Fraction(1, 3), 0, 10**6, 0) == [(0, 1)]


def test_merge_spans_keeps_a_gap_wider_than_gap_us():
    populated = np.array([True, False, False, True])
    assert merge_spans(populated, 0, 0, Fraction(100), 0, 10**6, 199) == [
        (0, 100),
        (300, 400),
    ]


def test_merge_spans_bridges_a_gap_of_gap_us():
    populated = np.array([True, False, False, True])
    assert merge_spans(populated, 0, 0, Fraction(100), 0, 10**6, 200) == [
        (0, 400)
    ]


def test_merge_spans_bridges_into_a_span_clamped_at_the_high_limit():
    populated = np.array([True, False, True])
    assert merge_spans(populated, 0, 0, Fraction(100), 0, 250, 100) == [
        (0, 250)
    ]


def test_merge_spans_reports_the_spans_around_an_unpopulated_run():
    period = bin_period(512.0, DEFAULT_DECIMATION_FACTOR**5)
    spans = merge_spans(
        _level_five_populated(), 0, 0, period, 0, 600_000_000, 0
    )
    assert spans == [(0, 120_000_000), (124_000_000, 600_000_000)]


def test_merge_spans_bridges_a_4_s_gap_with_gap_us_of_4_s():
    period = bin_period(512.0, DEFAULT_DECIMATION_FACTOR**5)
    populated = _level_five_populated()
    assert merge_spans(populated, 0, 0, period, 0, 600_000_000, 4_000_000) == [
        (0, 600_000_000)
    ]
    assert merge_spans(populated, 0, 0, period, 0, 600_000_000, 3_999_999) == [
        (0, 120_000_000),
        (124_000_000, 600_000_000),
    ]


def test_merge_spans_reports_level_one_spans_at_512_hz():
    period = bin_period(512.0, DEFAULT_DECIMATION_FACTOR)
    assert period == Fraction(15625, 2)
    populated = np.ones(76_800, dtype=bool)
    populated[15_360:15_936] = False
    populated[51_226:51_328] = False
    assert merge_spans(populated, 0, 0, period, 0, 600_000_000, 0) == [
        (0, 120_000_000),
        (124_500_000, 400_203_125),
        (401_000_000, 600_000_000),
    ]


@pytest.mark.parametrize(
    ("name", "number"),
    [("0", 0), ("7", 7), ("12", 12), ("1024", 1024)],
)
def test_canonical_number_reads_a_decimal_name(name, number):
    assert canonical_number(name) == number


@pytest.mark.parametrize(
    "name",
    [
        "01",
        "00",
        "",
        "-1",
        "+1",
        " 1",
        "1.0",
        "1/minmax",
        "raw",
        "\N{ARABIC-INDIC DIGIT THREE}",
        "\N{SUPERSCRIPT TWO}",
    ],
)
def test_canonical_number_rejects_a_name_that_is_not_a_canonical_decimal(name):
    assert canonical_number(name) is None


def test_check_level_numbers_sorts_the_numbers_past_9():
    names = [str(number) for number in range(10, 0, -1)]
    assert check_level_numbers("channel 0", "levels", names) == list(
        range(1, 11)
    )


def test_check_level_numbers_ignores_names_that_are_not_level_numbers():
    names = ["2", "0", "01", "notes", "1/minmax", "1"]
    assert check_level_numbers("channel 0", "levels", names) == [1, 2]


def test_check_level_numbers_of_no_levels_is_empty():
    assert check_level_numbers("channel 0", "levels", ["0", "raw"]) == []


def test_check_level_arrays_accepts_names_that_are_not_level_numbers():
    names = ["raw", "0", "01", "1/minmax", "notes"]
    check_level_arrays("channel 0", "level", "minmax", names)


def test_check_level_arrays_rejects_an_array_with_a_level_name():
    message = (
        "channel 3: count level 2 is an array; a count level must be a group "
        "that holds count"
    )
    with pytest.raises(ValueError, match=message):
        check_level_arrays("channel 3", "count level", "count", ["events", "2"])


@pytest.mark.parametrize(
    ("names", "listed"), [(["2", "3"], "2, 3"), (["1", "3", "4"], "1, 3, 4")]
)
def test_check_level_numbers_rejects_a_gap(names, listed):
    message = (
        f"channel 3: count levels must be numbered from 1 with no gap, got "
        f"{listed}"
    )
    with pytest.raises(ValueError, match=message):
        check_level_numbers("channel 3", "count levels", names)


def test_array_error_for_a_missing_array_names_the_expectation():
    error = array_error(
        "channel 0", "raw", "a rank-1 floating-point array", None
    )
    assert str(error) == (
        "channel 0: raw is missing; it must be a rank-1 floating-point array"
    )


def test_array_error_for_a_wrong_layout_gives_the_shape_and_dtype():
    array = zarr.create_array(MemoryStore(), shape=(2, 3), dtype="int32")
    error = array_error("channel 0", "raw", "a rank-1 array", array)
    assert str(error) == (
        "channel 0: raw must be a rank-1 array, got shape (2, 3) and dtype "
        "int32"
    )


@pytest.mark.parametrize(
    "labels", [[0, 3, 1], [], [3]], ids=["several", "none", "last"]
)
def test_check_label_range_accepts_labels_from_0_to_below_the_limit(labels):
    check_label_range("channel 0", np.array(labels, dtype=np.int64), 4)


@pytest.mark.parametrize(("labels", "value"), [([0, -1, 2], -1), ([0, 4], 4)])
def test_check_label_range_names_a_label_out_of_range(labels, value):
    message = f"^channel 0: labels must be at least 0 and below 4, got {value}$"
    with pytest.raises(ValueError, match=message):
        check_label_range("channel 0", np.array(labels, dtype=np.int64), 4)


def test_check_label_range_puts_the_limit_phrase_after_the_limit():
    message = (
        "^channel 3: labels must be at least 0 and below 2, the length of "
        "label_names, got 2$"
    )
    with pytest.raises(ValueError, match=message):
        check_label_range(
            "channel 3", np.array([2]), 2, "the length of label_names"
        )


def test_check_not_string_accepts_a_collection_of_names():
    check_not_string(["text"], "columns")


def test_check_not_string_rejects_a_string_and_names_the_argument():
    message = (
        "^drop_columns must be a collection of column names, got the string "
        "'text'$"
    )
    with pytest.raises(TypeError, match=message):
        check_not_string("text", "drop_columns")
