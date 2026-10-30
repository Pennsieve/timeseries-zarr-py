from functools import partial

import numpy as np
import pytest

from timeseries_zarr.constants import (
    COUNT_LADDER_BASE_US,
    EVENT_LEVEL_THRESHOLD,
    max_levels_for,
)
from timeseries_zarr.counts import (
    bin_offset_us,
    count_bins,
    fold_count_block,
    iter_count_blocks,
    level1_period_us,
    plan_count_levels,
)
from timeseries_zarr.streaming import _rebuffer_and_fold

_DAY_US = 86_400 * 1_000_000

_fold_by_4 = partial(fold_count_block, block=4)

_SPAN_PER_WIDTH = 200
"""Microseconds of span per microsecond of width when n_events is 1000.

With 1000 events, 10,000 * last / (1000 * 2000) is last / 200 microseconds.
"""


def test_level1_period_us_million_spikes_over_a_day_is_256_ms():
    day_us = 86_400 * 1_000_000
    assert level1_period_us(10**6, day_us) == 256_000


def test_level1_period_us_threshold_channel_over_ten_minutes_is_16_ms():
    assert level1_period_us(64_000, 600 * 1_000_000) == 16_000


def test_level1_period_us_is_never_finer_than_the_ladder_base():
    assert level1_period_us(100_000, 500_000) == COUNT_LADDER_BASE_US


@pytest.mark.parametrize("n_events", [1, 1000, 10**9])
def test_level1_period_us_events_all_at_one_time_give_the_base(
    n_events,
):
    assert level1_period_us(n_events, 0) == COUNT_LADDER_BASE_US


@pytest.mark.parametrize("rung", range(7))
def test_level1_period_us_width_exactly_on_a_rung_keeps_that_rung(rung):
    width_us = COUNT_LADDER_BASE_US * 4**rung
    assert level1_period_us(1000, _SPAN_PER_WIDTH * width_us) == width_us


@pytest.mark.parametrize("rung", range(1, 7))
def test_level1_period_us_width_one_below_a_rung_takes_the_rung_beneath(rung):
    width_us = COUNT_LADDER_BASE_US * 4**rung
    below = level1_period_us(1000, _SPAN_PER_WIDTH * width_us - 1)
    assert below == COUNT_LADDER_BASE_US * 4 ** (rung - 1)


def test_level1_period_us_width_one_below_the_base_stays_on_the_base():
    last_us = _SPAN_PER_WIDTH * COUNT_LADDER_BASE_US - 1
    assert level1_period_us(1000, last_us) == COUNT_LADDER_BASE_US


def test_level1_period_us_is_exact_for_spans_beyond_float_precision():
    last_us = 2**62
    width_us = 10_000 * last_us // 2000
    period = level1_period_us(1, last_us)
    assert period <= width_us < period * 4
    assert period % COUNT_LADDER_BASE_US == 0


@pytest.mark.parametrize("n_events", [1, 7, 1000, 64_000, 10**7])
@pytest.mark.parametrize("last_us", [0, 1, 999_999, 10**9, 86_400 * 10**6])
def test_level1_period_us_lies_on_the_ladder(n_events, last_us):
    rung = level1_period_us(n_events, last_us) // COUNT_LADDER_BASE_US
    assert rung >= 1
    assert rung & (rung - 1) == 0
    assert (rung.bit_length() - 1) % 2 == 0


def test_level1_period_us_does_not_widen_as_events_are_added():
    periods = [level1_period_us(n, 10**10) for n in (10, 10**3, 10**5, 10**7)]
    assert periods == sorted(periods, reverse=True)


def test_level1_period_us_does_not_narrow_as_the_span_grows():
    periods = [level1_period_us(10**5, last) for last in (10**6, 10**8, 10**10)]
    assert periods == sorted(periods)


def test_level1_period_us_climbs_a_ladder_of_the_decimation_factor():
    assert level1_period_us(10**6, _DAY_US, 10) == 100_000
    assert level1_period_us(10**6, _DAY_US, 2) == 256_000
    assert level1_period_us(10**6, _DAY_US, 3) == 243_000


@pytest.mark.parametrize("factor", [2, 3, 10])
def test_level1_period_us_lies_on_the_ladder_of_the_decimation_factor(factor):
    for last_us in (10**6, 10**8, 10**10):
        period = level1_period_us(10**5, last_us, factor)
        rung = period // COUNT_LADDER_BASE_US
        assert rung * COUNT_LADDER_BASE_US == period
        while rung > 1:
            assert rung % factor == 0
            rung //= factor


@pytest.mark.parametrize("factor", [1, 0, -4])
def test_level1_period_us_rejects_a_decimation_factor_below_2(factor):
    with pytest.raises(ValueError, match="decimation_factor"):
        level1_period_us(10, 1000, factor)


@pytest.mark.parametrize("n_events", [0, -1])
def test_level1_period_us_rejects_fewer_than_one_event(n_events):
    with pytest.raises(ValueError, match="n_events"):
        level1_period_us(n_events, 1000)


def test_level1_period_us_rejects_a_negative_span():
    with pytest.raises(ValueError, match="span_us"):
        level1_period_us(10, -1)


@pytest.mark.parametrize(
    ("first_us", "period_us", "expected"),
    [
        (0, 1000, 0),
        (999, 1000, 0),
        (1000, 1000, 1000),
        (-1, 1000, -1000),
        (-32_000, 16_000, -32_000),
        (-32_001, 16_000, -48_000),
    ],
)
def test_bin_offset_us_rounds_the_first_event_down_to_the_grid(
    first_us, period_us, expected
):
    assert bin_offset_us(first_us, period_us) == expected


def test_bin_offset_us_rejects_a_period_below_one():
    with pytest.raises(ValueError, match="period_us"):
        bin_offset_us(0, 0)


@pytest.mark.parametrize(
    ("last_us", "period_us", "expected"),
    [
        (2500, 1000, 3),
        (2999, 1000, 3),
        (3000, 1000, 4),
        (3001, 1000, 4),
        (1000, 1000, 2),
        (999, 1000, 1),
        (0, 1000, 1),
        (0, 1, 1),
        (7, 1, 8),
    ],
)
def test_count_bins_holds_the_bin_of_the_last_event_plus_one(
    last_us, period_us, expected
):
    assert count_bins(0, last_us, period_us) == expected


def test_count_bins_keeps_an_event_on_a_period_boundary():
    # A ceiling of span over period would give 3 and drop the event at 3000.
    assert count_bins(0, 3000, 1000) == 4


def test_count_bins_lone_event_has_one_bin():
    assert count_bins(0, 0, 256_000) == 1
    assert count_bins(-7, -7, 256_000) == 1


@pytest.mark.parametrize(
    ("first_us", "last_us", "period_us", "expected"),
    [
        (-32_000, 0, 16_000, 3),
        (-1, 0, 1000, 2),
        (-2500, -1500, 1000, 2),
        (2500, 2999, 1000, 1),
        (-1000, 999, 1000, 2),
    ],
)
def test_count_bins_counts_from_the_first_events_bin_through_the_last(
    first_us, last_us, period_us, expected
):
    assert count_bins(first_us, last_us, period_us) == expected


def test_count_bins_is_exact_for_spans_beyond_float_precision():
    assert count_bins(0, 2**62, 1000) == 2**62 // 1000 + 1


@pytest.mark.parametrize("period_us", [1, 1000, 4000, 256_000, 1_024_000])
@pytest.mark.parametrize(
    "last_us", [0, 1, 999, 1000, 3_999_999, 86_400 * 10**6]
)
def test_count_bins_last_event_falls_inside_the_last_bin(last_us, period_us):
    bins = count_bins(0, last_us, period_us)
    assert (bins - 1) * period_us <= last_us < bins * period_us


@pytest.mark.parametrize("period_us", [0, -1])
def test_count_bins_rejects_a_period_below_one(period_us):
    with pytest.raises(ValueError, match="period_us"):
        count_bins(0, 1000, period_us)


def test_count_bins_rejects_a_last_event_before_the_first():
    with pytest.raises(ValueError, match="precedes"):
        count_bins(5, 4, 1000)


def _summary(plans):
    return [(plan.level, plan.bins, plan.period_us) for plan in plans]


def test_plan_count_levels_million_spikes_over_a_day_has_five_levels():
    plans = plan_count_levels(10**6, 0, _DAY_US)
    assert _summary(plans) == [
        (1, 337_501, 256_000.0),
        (2, 84_376, 1_024_000.0),
        (3, 21_094, 4_096_000.0),
        (4, 5_274, 16_384_000.0),
        (5, 1_319, 65_536_000.0),
    ]


def test_plan_count_levels_threshold_channel_gets_none_and_one_more_gets_three():
    assert plan_count_levels(EVENT_LEVEL_THRESHOLD, 0, 600 * 10**6) == []
    plans = plan_count_levels(EVENT_LEVEL_THRESHOLD + 1, 0, 600 * 10**6)
    assert _summary(plans) == [
        (1, 37_501, 16_000.0),
        (2, 9_376, 64_000.0),
        (3, 2_344, 256_000.0),
    ]


@pytest.mark.parametrize("n_events", [0, 1, 1000, EVENT_LEVEL_THRESHOLD])
def test_plan_count_levels_small_channel_gets_no_levels(n_events):
    assert plan_count_levels(n_events, 0, _DAY_US) == []


def test_plan_count_levels_level_one_under_min_bins_gets_none():
    assert plan_count_levels(100_000, 0, 500_000) == []


def test_plan_count_levels_honors_min_bins():
    plans = plan_count_levels(100_000, 0, 500_000, min_bins=256)
    assert _summary(plans) == [(1, 501, 1000.0)]


@pytest.mark.parametrize(
    ("max_levels", "expected"), [(0, 0), (1, 1), (2, 2), (max_levels_for(4), 5)]
)
def test_plan_count_levels_honors_max_levels(max_levels, expected):
    plans = plan_count_levels(10**6, 0, _DAY_US, max_levels=max_levels)
    assert len(plans) == expected


@pytest.mark.parametrize("n_events", [64_001, 10**6, 10**8])
@pytest.mark.parametrize("last_us", [10**9, 600 * 10**6, _DAY_US, 7 * _DAY_US])
@pytest.mark.parametrize("min_bins", [256, 1024])
def test_plan_count_levels_follows_the_ladder_rules(
    n_events, last_us, min_bins
):
    plans = plan_count_levels(n_events, 0, last_us, min_bins=min_bins)
    for plan in plans:
        assert plan.bins >= min_bins
    for below, above in zip(plans, plans[1:], strict=False):
        assert above.level == below.level + 1
        assert above.period_us == 4 * below.period_us
        assert above.bins == -(-below.bins // 4)
    if plans:
        assert plans[0].bins == count_bins(
            0, last_us, level1_period_us(n_events, last_us)
        )
        assert -(-plans[-1].bins // 4) < min_bins or len(
            plans
        ) == max_levels_for(4)


def test_plan_count_levels_at_2x_doubles_the_period_of_each_level():
    plans = plan_count_levels(10**6, 0, _DAY_US, decimation_factor=2)
    assert plans[0].period_us == 256_000.0
    assert len(plans) == 9
    for below, above in zip(plans, plans[1:], strict=False):
        assert above.period_us == 2 * below.period_us
        assert above.bins == -(-below.bins // 2)
    for plan in plans:
        assert plan.bins == count_bins(0, _DAY_US, int(plan.period_us))


def test_plan_count_levels_at_10x_multiplies_the_period_of_each_level_by_10():
    plans = plan_count_levels(10**6, 0, _DAY_US, decimation_factor=10)
    assert _summary(plans) == [
        (1, 864_001, 100_000.0),
        (2, 86_401, 1_000_000.0),
        (3, 8_641, 10_000_000.0),
    ]


def test_plan_count_levels_caps_the_levels_at_what_the_factor_allows():
    plans = plan_count_levels(
        10**6, 0, _DAY_US, min_bins=1, decimation_factor=10
    )
    assert len(plans) == 9
    plans = plan_count_levels(
        10**6, 0, _DAY_US, max_levels=3, min_bins=1, decimation_factor=10
    )
    assert len(plans) == 3


@pytest.mark.parametrize("factor", [1, 0, -4])
def test_plan_count_levels_rejects_a_decimation_factor_below_2(factor):
    with pytest.raises(ValueError, match="decimation_factor"):
        plan_count_levels(10**6, 0, _DAY_US, decimation_factor=factor)


def test_plan_count_levels_accepts_the_largest_uint32_event_count():
    plans = plan_count_levels(2**32 - 1, 0, _DAY_US)
    assert plans
    assert plans[0].period_us == COUNT_LADDER_BASE_US


def test_plan_count_levels_rejects_more_events_than_a_uint32_count_holds():
    with pytest.raises(ValueError, match="uint32"):
        plan_count_levels(2**32, 0, _DAY_US)


def test_plan_count_levels_carries_every_period_as_an_int():
    plans = plan_count_levels(10**6, 0, _DAY_US)
    assert len(plans) >= 2
    assert all(type(plan.period_us) is int for plan in plans)


def test_plan_count_levels_rejects_a_last_bin_that_ends_past_the_time_limit():
    with pytest.raises(
        ValueError, match="end of the last bin of count level 1"
    ):
        plan_count_levels(100_000, 0, 2**53 - 1)


def test_plan_count_levels_rejects_an_offset_at_the_time_limit():
    with pytest.raises(ValueError, match="offset_us of count level 1"):
        plan_count_levels(100_000, -(2**53), 0)


def test_plan_count_levels_rejects_a_negative_event_count():
    with pytest.raises(ValueError, match="n_events"):
        plan_count_levels(-1, 0, _DAY_US)


@pytest.mark.parametrize("n_events", [0, 10, 10**6])
def test_plan_count_levels_rejects_a_last_event_before_the_first(n_events):
    with pytest.raises(ValueError, match="precedes"):
        plan_count_levels(n_events, 0, -1)


def test_plan_count_levels_anchors_every_level_at_or_before_a_negative_first():
    plans = plan_count_levels(10**6, -32_000, _DAY_US - 32_000)
    assert [plan.level for plan in plans] == [1, 2, 3, 4, 5]
    for plan in plans:
        period = int(plan.period_us)
        assert plan.offset_us % period == 0
        assert plan.offset_us <= -32_000 < plan.offset_us + period
        assert plan.bins == count_bins(-32_000, _DAY_US - 32_000, period)
        assert plan.offset_us + plan.bins * period > _DAY_US - 32_000


def test_plan_count_levels_offsets_are_zero_for_events_from_the_reference():
    assert all(
        plan.offset_us == 0 for plan in plan_count_levels(10**6, 0, _DAY_US)
    )


def test_plan_count_levels_bins_above_level_one_cover_the_level_below():
    plans = plan_count_levels(10**6, 12_345, _DAY_US)
    for below, above in zip(plans, plans[1:], strict=False):
        shift = (below.offset_us - above.offset_us) // int(below.period_us)
        assert 0 <= shift < 4
        assert above.bins == -(-(below.bins + shift) // 4)


def _direct(times, labels, n_labels, period_us, bins, offset_us=0):
    """Counts straight from the definition with an unbuffered scatter add."""
    counts = np.zeros((bins, max(1, n_labels)), dtype=np.uint32)
    column = np.zeros(len(times), dtype=np.int64) if labels is None else labels
    np.add.at(counts, (times // period_us - offset_us // period_us, column), 1)
    return counts


def _blocks(times, labels, sizes):
    """Split events into blocks of the given sizes, which may be zero."""
    out, start = [], 0
    for size in sizes:
        stop = start + size
        out.append(
            (times[start:stop], None if labels is None else labels[start:stop])
        )
        start = stop
    return out


def _collect(event_blocks, n_labels, period_us, bins, block_bins, offset_us=0):
    return list(
        iter_count_blocks(
            event_blocks, n_labels, period_us, offset_us, bins, block_bins
        )
    )


def _events(seed, count=300, period_us=1000, bins=137, n_labels=5):
    rng = np.random.default_rng(seed)
    times = np.sort(rng.integers(0, bins * period_us, count)).astype(np.int64)
    labels = rng.integers(0, max(1, n_labels), count).astype(np.uint16)
    return times, labels


@pytest.mark.parametrize("block_bins", [1, 3, 4, 7, 64, 1000])
@pytest.mark.parametrize("seed", range(3))
def test_iter_count_blocks_equals_the_direct_count(block_bins, seed):
    times, labels = _events(seed)
    rng = np.random.default_rng(seed + 50)
    sizes = []
    left = len(times)
    while left:
        size = int(rng.integers(0, min(left, 40) + 1))
        sizes.append(size)
        left -= size
    got = _collect(_blocks(times, labels, sizes), 5, 1000, 137, block_bins)
    assert np.array_equal(
        np.concatenate(got), _direct(times, labels, 5, 1000, 137)
    )


def test_iter_count_blocks_every_block_but_the_last_has_block_bins_rows():
    times, labels = _events(1)
    got = _collect(_blocks(times, labels, [len(times)]), 5, 1000, 137, 20)
    assert [block.shape[0] for block in got] == [20] * 6 + [17]
    assert all(block.shape == (20, 5) for block in got[:-1])
    assert got[-1].dtype == np.uint32


def test_iter_count_blocks_bins_not_divisible_by_block_bins_end_with_a_remainder():
    got = _collect([], 0, 1000, 10, 4)
    assert [block.shape for block in got] == [(4, 1), (4, 1), (2, 1)]


def test_iter_count_blocks_with_no_events_covers_the_axis_with_zeros():
    got = _collect([], 3, 1000, 9, 4)
    assert all(not block.any() for block in got)
    assert np.concatenate(got).shape == (9, 3)


def test_iter_count_blocks_an_axis_of_no_bins_yields_nothing():
    assert _collect([], 0, 1000, 0, 4) == []


def test_iter_count_blocks_counts_events_after_the_last_block_boundary():
    times = np.array([0, 9_500], dtype=np.int64)
    got = _collect([(times, None)], 0, 1000, 10, 4)
    counts = np.concatenate(got)
    assert counts[:, 0].tolist() == [1, 0, 0, 0, 0, 0, 0, 0, 0, 1]


def test_iter_count_blocks_event_on_a_period_boundary_opens_its_own_bin():
    times = np.array([999, 1000, 2000], dtype=np.int64)
    counts = np.concatenate(_collect([(times, None)], 0, 1000, 3, 2))
    assert counts[:, 0].tolist() == [1, 1, 1]


def test_iter_count_blocks_equal_times_accumulate_in_one_bin():
    times = np.full(5, 2500, dtype=np.int64)
    labels = np.array([0, 0, 1, 1, 1], dtype=np.uint16)
    counts = np.concatenate(_collect([(times, labels)], 2, 1000, 4, 4))
    assert counts[2].tolist() == [2, 3]
    assert int(counts.sum()) == 5


def test_iter_count_blocks_a_gap_wider_than_a_block_emits_empty_blocks():
    times = np.array([0, 90_500], dtype=np.int64)
    got = _collect([(times, None)], 0, 1000, 100, 8)
    assert [block.shape[0] for block in got] == [8] * 12 + [4]
    assert int(np.concatenate(got).sum()) == 2
    assert all(not block.any() for block in got[1:11])


def test_iter_count_blocks_without_labels_counts_every_event_in_column_zero():
    times = np.array([0, 1000, 1000], dtype=np.int64)
    counts = np.concatenate(_collect([(times, None)], 3, 1000, 2, 2))
    assert counts.tolist() == [[1, 0, 0], [2, 0, 0]]


def test_iter_count_blocks_zero_labels_means_a_single_column():
    times = np.array([0, 1000], dtype=np.int64)
    counts = np.concatenate(_collect([(times, None)], 0, 1000, 2, 2))
    assert counts.shape == (2, 1)


def test_iter_count_blocks_skips_empty_event_blocks():
    times = np.array([0, 1500], dtype=np.int64)
    empty = np.empty(0, dtype=np.int64)
    blocks = [
        (empty, None),
        (times[:1], None),
        (empty, None),
        (times[1:], None),
    ]
    counts = np.concatenate(_collect(blocks, 0, 1000, 2, 2))
    assert counts[:, 0].tolist() == [1, 1]


def test_iter_count_blocks_yielded_blocks_do_not_share_memory():
    times, labels = _events(2)
    got = _collect(_blocks(times, labels, [len(times)]), 5, 1000, 137, 4)
    assert not np.shares_memory(got[0], got[1])


@pytest.mark.parametrize(
    ("times", "match"),
    [
        ([3_000, 2_000], "non-decreasing"),
        ([-1, 0], "precedes bin 0"),
        ([0, 4_000], "past the 4 bins"),
    ],
)
def test_iter_count_blocks_rejects_bad_times_within_a_block(times, match):
    block = (np.array(times, dtype=np.int64), None)
    with pytest.raises(ValueError, match=match):
        _collect([block], 0, 1000, 4, 2)


def test_iter_count_blocks_rejects_times_that_decrease_across_blocks():
    first = (np.array([3_000], dtype=np.int64), None)
    second = (np.array([2_000], dtype=np.int64), None)
    with pytest.raises(ValueError, match="non-decreasing"):
        _collect([first, second], 0, 1000, 8, 4)


def test_iter_count_blocks_rejects_a_label_outside_the_columns():
    times = np.array([0, 1000], dtype=np.int64)
    labels = np.array([0, 3], dtype=np.uint16)
    with pytest.raises(ValueError, match="label 3"):
        _collect([(times, labels)], 3, 1000, 2, 2)


def test_iter_count_blocks_rejects_labels_that_differ_in_shape_from_times():
    times = np.array([0, 1000], dtype=np.int64)
    labels = np.array([0], dtype=np.uint16)
    with pytest.raises(ValueError, match="shape"):
        _collect([(times, labels)], 3, 1000, 2, 2)


@pytest.mark.parametrize(
    ("n_labels", "period_us", "bins", "block_bins", "name"),
    [
        (1, 1000, 4, 0, "block_bins"),
        (1, 0, 4, 2, "period_us"),
        (1, 1000, -1, 2, "bins"),
        (-1, 1000, 4, 2, "n_labels"),
    ],
)
def test_iter_count_blocks_rejects_bad_geometry(
    n_labels, period_us, bins, block_bins, name
):
    with pytest.raises(ValueError, match=name):
        _collect([], n_labels, period_us, bins, block_bins)


def test_iter_count_blocks_rejects_an_offset_off_the_grid():
    with pytest.raises(ValueError, match="offset_us"):
        _collect([], 1, 1000, 4, 2, offset_us=500)


@pytest.mark.parametrize("block_bins", [1, 3, 7, 1000])
def test_iter_count_blocks_counts_events_before_the_reference_time(block_bins):
    times, labels = _events(9)
    times = times - 40_000
    offset = -40_000
    got = _collect(
        _blocks(times, labels, [100, 0, 200]), 5, 1000, 137, block_bins, offset
    )
    assert np.array_equal(
        np.concatenate(got), _direct(times, labels, 5, 1000, 137, offset)
    )


def test_iter_count_blocks_rejects_an_event_before_bin_zero():
    block = (np.array([-1001, 0], dtype=np.int64), None)
    with pytest.raises(ValueError, match="precedes bin 0 at -1000"):
        _collect([block], 0, 1000, 4, 2, offset_us=-1000)


@pytest.mark.parametrize("offset_below", [-3000, -2000, -1000])
def test_a_zero_row_prefix_folds_a_level_onto_an_earlier_grid(offset_below):
    times, labels = _events(5, bins=1000, n_labels=4)
    times[0] = 0
    times = times + offset_below
    below = _direct(times, labels, 4, 1000, 1000, offset_below)
    offset_above = bin_offset_us(int(times[0]), 4000)
    assert offset_above == -4000
    shift = (offset_below - offset_above) // 1000
    assert shift == (offset_below + 4000) // 1000
    prefix = np.zeros((shift, 4), dtype=np.uint32)
    folded = np.concatenate(
        list(_rebuffer_and_fold([prefix, below], _fold_by_4, group=4))
    )
    bins_above = count_bins(int(times[0]), int(times[-1]), 4000)
    assert np.array_equal(
        folded, _direct(times, labels, 4, 4000, bins_above, offset_above)
    )


def _level1(seed=0, bins=1000, n_labels=4):
    times, labels = _events(seed, count=900, bins=bins, n_labels=n_labels)
    return times, labels, _direct(times, labels, n_labels, 1000, bins)


@pytest.mark.parametrize("level", [2, 3, 4, 5])
def test_fold_count_block_equals_the_direct_count_at_every_level(level):
    times, labels, counts = _level1()
    for _ in range(level - 1):
        counts = fold_count_block(counts, 4)
    period_us = 1000 * 4 ** (level - 1)
    bins = -(-1000 // 4 ** (level - 1))
    assert counts.shape == (bins, 4)
    assert np.array_equal(counts, _direct(times, labels, 4, period_us, bins))


@pytest.mark.parametrize(
    ("rows", "expected"),
    [(0, 0), (1, 1), (3, 1), (4, 1), (5, 2), (8, 2), (9, 3), (17, 5)],
)
def test_fold_count_block_shape(rows, expected):
    out = fold_count_block(np.ones((rows, 3), dtype=np.uint32), 4)
    assert out.shape == (expected, 3)
    assert out.dtype == np.uint32


def test_fold_count_block_sums_a_partial_tail_into_one_row():
    counts = np.arange(10, dtype=np.uint32).reshape(5, 2)
    assert fold_count_block(counts, 4).tolist() == [[12, 16], [8, 9]]


@pytest.mark.parametrize(
    ("rows", "block", "expected"),
    [(0, 2, 0), (1, 2, 1), (2, 2, 1), (5, 2, 3), (10, 10, 1), (11, 10, 2)],
)
def test_fold_count_block_shape_at_another_block_size(rows, block, expected):
    out = fold_count_block(np.ones((rows, 3), dtype=np.uint32), block)
    assert out.shape == (expected, 3)


def test_fold_count_block_at_block_2_sums_pairs_and_a_single_tail_row():
    counts = np.arange(10, dtype=np.uint32).reshape(5, 2)
    assert fold_count_block(counts, 2).tolist() == [[2, 4], [10, 12], [8, 9]]


def test_fold_count_block_at_block_10_keeps_the_total_of_every_column():
    counts = np.random.default_rng(5).integers(0, 50, (37, 6)).astype(np.uint32)
    out = fold_count_block(counts, 10)
    assert np.array_equal(out.sum(axis=0), counts.sum(axis=0))
    assert np.array_equal(out[0], counts[:10].sum(axis=0))
    assert np.array_equal(out[-1], counts[30:].sum(axis=0))


@pytest.mark.parametrize("block", [0, -1])
def test_fold_count_block_rejects_a_block_below_1(block):
    with pytest.raises(ValueError, match="block"):
        fold_count_block(np.ones((4, 2), dtype=np.uint32), block)


def test_fold_count_block_keeps_the_total_of_every_column():
    counts = np.random.default_rng(3).integers(0, 50, (37, 6)).astype(np.uint32)
    assert np.array_equal(
        fold_count_block(counts, 4).sum(axis=0), counts.sum(axis=0)
    )


def test_fold_count_block_is_exact_at_the_top_of_the_uint32_range():
    counts = np.array([[2**30], [2**30], [2**30], [2**30 - 1]], dtype=np.uint32)
    assert fold_count_block(counts, 4)[0, 0] == 2**32 - 1


def test_fold_count_block_leaves_the_input_unchanged():
    counts = np.arange(12, dtype=np.uint32).reshape(6, 2)
    before = counts.copy()
    fold_count_block(counts, 4)
    assert np.array_equal(counts, before)


@pytest.mark.parametrize("size", [1, 2, 3, 5, 7, 13])
def test_streamed_count_folds_equal_the_whole_fold(size):
    _, _, counts = _level1(seed=4)
    pieces = [counts[i : i + size] for i in range(0, len(counts), size)]
    level2 = _rebuffer_and_fold(pieces, _fold_by_4, group=4)
    level3 = list(_rebuffer_and_fold(level2, _fold_by_4, group=4))
    want = fold_count_block(fold_count_block(counts, 4), 4)
    assert np.array_equal(np.concatenate(level3), want)


@pytest.mark.parametrize("shape", [(8,), (2, 4, 3)])
def test_fold_count_block_rejects_counts_that_are_not_rows_by_columns(shape):
    with pytest.raises(ValueError, match="rows, columns"):
        fold_count_block(np.zeros(shape, dtype=np.uint32), 4)
