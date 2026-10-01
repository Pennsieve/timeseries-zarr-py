import numpy as np
import pytest

from timeseries_zarr.grid import (
    Segment,
    build_segments,
    derive_rate_hz,
    grid_length,
    segments_overlapping,
)

RATE = 512.0
PERIOD = 1.0 / RATE


def _regular(n, start=0.0, rate=RATE):
    return start + np.arange(n, dtype=np.float64) / rate


def _dithered(n):
    """Timestamps alternating between two values averaging exactly 1/512.

    The shape a real MEF-derived recording has: neither value is the true
    period, but their mean is.
    """
    steps = np.empty(n - 1, dtype=np.float64)
    steps[0::2] = 0.001953
    steps[1::2] = 0.00195325
    return np.concatenate([[0.0], np.cumsum(steps)])


def test_no_gaps_is_one_segment_spanning_every_sample():
    segments = build_segments(_regular(1000), RATE)
    assert segments == [Segment(0, 1000, 0, 1000)]
    assert grid_length(segments) == 1000


def test_gap_lengthens_the_grid_beyond_the_sample_count():
    # 100 samples, a one-second break, then 100 more.
    ts = np.concatenate(
        [_regular(100), _regular(100, start=100 * PERIOD + 1.0)]
    )
    segments = build_segments(ts, RATE)

    assert len(segments) == 2
    assert segments[0] == Segment(0, 100, 0, 100)
    assert segments[1].grid_start == 100 + int(RATE)
    assert grid_length(segments) > len(ts)
    assert grid_length(segments) == segments[1].grid_stop


def test_several_gaps_keep_sample_spans_contiguous():
    ts = np.concatenate(
        [
            _regular(50),
            _regular(30, start=50 * PERIOD + 2.0),
            _regular(20, start=50 * PERIOD + 2.0 + 30 * PERIOD + 5.0),
        ]
    )
    segments = build_segments(ts, RATE)

    assert [s.length for s in segments] == [50, 30, 20]
    assert [s.sample_start for s in segments] == [0, 50, 80]
    assert segments[-1].sample_stop == 100


def test_empty_timestamps_produce_no_segments():
    assert build_segments(np.empty(0, dtype=np.float64), RATE) == []
    assert grid_length([]) == 0


def test_block_boundary_does_not_split_a_run():
    ts = _regular(1000)
    assert build_segments(ts, RATE, block=64) == build_segments(ts, RATE)


def test_gap_falling_on_a_block_boundary_is_still_one_gap():
    ts = np.concatenate([_regular(64), _regular(64, start=64 * PERIOD + 1.0)])
    segments = build_segments(ts, RATE, block=64)
    assert len(segments) == 2
    assert segments[0].sample_stop == 64


def test_rate_too_low_to_separate_samples_raises():
    with pytest.raises(ValueError, match="rate is too low"):
        build_segments(_regular(100), RATE / 4)


def test_non_positive_rate_raises():
    with pytest.raises(ValueError, match="rate must be positive"):
        build_segments(_regular(10), 0.0)


def test_irregular_spacing_raises_rather_than_resampling():
    # Jittered intervals put two samples on one grid position.
    rng = np.random.default_rng(0)
    ts = np.cumsum(rng.uniform(PERIOD * 0.5, PERIOD * 1.4, size=500))
    with pytest.raises(ValueError, match="rate is too low"):
        build_segments(ts, RATE)


def test_rate_too_high_is_caught_rather_than_read_as_all_gaps():
    # Every sample lands two positions on, which segmentation alone cannot
    # tell from a recording that is nothing but gaps.
    with pytest.raises(ValueError, match="irregular rather than gapped"):
        build_segments(_regular(500), RATE * 2)


def test_single_sample_needs_no_rate_agreement():
    assert build_segments(np.zeros(1, dtype=np.float64), RATE) == [
        Segment(0, 1, 0, 1)
    ]


def test_derive_rate_reads_through_the_dither():
    # Unequal counts of the two intervals, as a real recording has: the
    # median lands on one of them, the span averages them out.
    ts = _dithered(2000)
    median_rate = 1.0 / float(np.median(np.diff(ts)))

    assert derive_rate_hz(ts) == pytest.approx(512.0, abs=1e-3)
    assert abs(median_rate - 512.0) > 0.03


def test_derive_rate_stops_at_the_first_gap():
    ts = np.concatenate(
        [_regular(500), _regular(500, start=500 * PERIOD + 30.0)]
    )
    assert derive_rate_hz(ts) == pytest.approx(RATE)


def test_derive_rate_skips_a_one_sample_first_run():
    # A MEF-derived recording opened with one sample, then a 3.3 s gap.
    ts = np.concatenate([[0.0], _regular(2000, start=3.3555)])
    assert derive_rate_hz(ts) == pytest.approx(RATE)


def test_derive_rate_uses_the_longest_run():
    # The short first run measures the dither badly; the long one averages it.
    short = _dithered(4)
    long = _dithered(4000) + short[-1] + 30.0
    ts = np.concatenate([short, long])
    assert derive_rate_hz(ts) == pytest.approx(512.0, abs=1e-3)


def test_derive_rate_needs_two_timestamps():
    with pytest.raises(ValueError, match="at least two timestamps"):
        derive_rate_hz(np.zeros(1, dtype=np.float64))


def test_dithered_timestamps_grid_without_drift():
    segments = build_segments(_dithered(20_000), 512.0)
    assert segments == [Segment(0, 20_000, 0, 20_000)]


def test_segments_overlapping_selects_only_intersecting_runs():
    segments = [Segment(0, 10, 0, 10), Segment(50, 60, 10, 20)]

    assert segments_overlapping(segments, 0, 5) == [segments[0]]
    assert segments_overlapping(segments, 55, 60) == [segments[1]]
    assert segments_overlapping(segments, 5, 55) == segments
    assert segments_overlapping(segments, 20, 40) == []
    assert segments_overlapping(segments, 10, 10) == []
