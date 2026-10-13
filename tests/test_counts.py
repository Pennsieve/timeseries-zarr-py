import pytest

from timeseries_zarr.constants import COUNT_LADDER_BASE_US
from timeseries_zarr.counts import count_bins, level1_period_us

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
def test_level1_period_us_events_all_at_the_origin_give_the_base(n_events):
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


@pytest.mark.parametrize("n_events", [0, -1])
def test_level1_period_us_rejects_fewer_than_one_event(n_events):
    with pytest.raises(ValueError, match="n_events"):
        level1_period_us(n_events, 1000)


def test_level1_period_us_rejects_a_negative_last_event_time():
    with pytest.raises(ValueError, match="last_event_us"):
        level1_period_us(10, -1)


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
    assert count_bins(last_us, period_us) == expected


def test_count_bins_keeps_an_event_on_a_period_boundary():
    # A ceiling of span over period would give 3 and drop the event at 3000.
    assert count_bins(3000, 1000) == 4


def test_count_bins_lone_event_at_the_origin_has_one_bin():
    assert count_bins(0, 256_000) == 1


def test_count_bins_is_exact_for_spans_beyond_float_precision():
    assert count_bins(2**62, 1000) == 2**62 // 1000 + 1


@pytest.mark.parametrize("period_us", [1, 1000, 4000, 256_000, 1_024_000])
@pytest.mark.parametrize(
    "last_us", [0, 1, 999, 1000, 3_999_999, 86_400 * 10**6]
)
def test_count_bins_last_event_falls_inside_the_last_bin(last_us, period_us):
    bins = count_bins(last_us, period_us)
    assert (bins - 1) * period_us <= last_us < bins * period_us


@pytest.mark.parametrize("period_us", [0, -1])
def test_count_bins_rejects_a_period_below_one(period_us):
    with pytest.raises(ValueError, match="period_us"):
        count_bins(1000, period_us)


def test_count_bins_rejects_a_negative_last_event_time():
    with pytest.raises(ValueError, match="last_event_us"):
        count_bins(-1, 1000)
