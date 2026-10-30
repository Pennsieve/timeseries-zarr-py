import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from timeseries_zarr.validate_full import (
    LevelArrays,
    _bin_statistics,
    _exact_mismatch,
    _within_tolerance,
    check_continuous_levels,
    check_count_level,
    check_raw_outside_runs,
    check_waveform_mean,
)

FLOAT32_MAXIMUM = float(np.finfo(np.float32).max)
NAN = float("nan")


def _array(data):
    data = np.asarray(data)
    array = zarr.create_array(
        MemoryStore(),
        shape=data.shape,
        dtype=data.dtype,
        chunks=data.shape,
        fill_value=0,
    )
    array[...] = data
    return array


def _level(samples, width, **changes):
    """Return the arrays of a level that holds the exact statistics."""
    statistics = _bin_statistics(np.asarray(samples, np.float32), width)
    with np.errstate(over="ignore"):
        variance = statistics.var.astype(np.float32)
    members = {
        "count": statistics.count.astype(np.uint32),
        "minmax": statistics.minmax,
        "mean": statistics.mean.astype(np.float32),
        "var": variance,
    }
    members.update(changes)
    return LevelArrays(
        path="0/1",
        width=width,
        count=_array(members["count"]),
        minmax=_array(members["minmax"]),
        mean=_array(members["mean"]),
        var=_array(members["var"]),
    )


# Statistics of a block.


def test_bin_statistics_cover_the_finite_samples_and_a_short_last_bin():
    block = np.array(
        [1, 2, np.nan, 4, 5, np.inf, -np.inf, np.nan, 7, 9, 11], np.float32
    )
    statistics = _bin_statistics(block, 4)
    assert statistics.count.tolist() == [3, 1, 3]
    assert statistics.minmax[0].tolist() == [1, 4]
    assert statistics.minmax[1].tolist() == [5, 5]
    assert statistics.minmax[2].tolist() == [7, 11]
    assert statistics.mean.tolist() == [7 / 3, 5, 9]
    assert statistics.var[1] == 0
    assert statistics.var[2] == pytest.approx(8 / 3)


def test_a_bin_with_no_finite_sample_has_count_0_and_nan_elsewhere():
    statistics = _bin_statistics(
        np.array([np.nan, np.inf, 1, 2], np.float32), 2
    )
    assert statistics.count.tolist() == [0, 2]
    assert np.isnan(statistics.minmax[0]).all()
    assert np.isnan(statistics.mean[0])
    assert np.isnan(statistics.var[0])
    assert np.isnan(statistics.max_abs[0])


# Comparison rules.


def test_exact_comparison_treats_nan_as_equal_and_zero_signs_as_equal():
    stored = np.array([[np.nan, -0.0], [1.0, 2.0]], np.float32)
    expected = np.array([[np.nan, 0.0], [1.0, 3.0]], np.float32)
    assert _exact_mismatch(stored, expected).tolist() == [False, True]


def test_exact_comparison_separates_nan_from_a_number():
    stored = np.array([np.nan, 1.0], np.float32)
    expected = np.array([1.0, np.nan], np.float32)
    assert _exact_mismatch(stored, expected).tolist() == [True, True]


def _passes(stored, reference, max_abs, *, variance=False):
    return bool(
        _within_tolerance(
            np.array([stored], np.float32),
            np.array([reference], np.float64),
            np.array([max_abs], np.float64),
            variance=variance,
        )[0]
    )


@pytest.mark.parametrize(
    ("stored", "reference", "expected"),
    [(NAN, NAN, True), (1.0, NAN, False), (NAN, 1.0, False)],
)
def test_a_nan_reference_passes_only_a_nan_value(stored, reference, expected):
    assert _passes(stored, reference, 1.0) is expected


def test_a_reference_above_the_float32_maximum_passes_only_positive_infinity():
    reference = 2 * FLOAT32_MAXIMUM
    assert _passes(np.inf, reference, FLOAT32_MAXIMUM, variance=True)
    assert not _passes(
        FLOAT32_MAXIMUM, reference, FLOAT32_MAXIMUM, variance=True
    )
    assert not _passes(-np.inf, reference, FLOAT32_MAXIMUM, variance=True)


def test_infinity_fails_against_a_finite_reference():
    assert not _passes(np.inf, 1.0, 1.0)
    assert not _passes(np.inf, 1.0, 1.0, variance=True)


def test_the_mean_tolerance_scales_with_the_larger_of_the_mean_and_max_abs():
    limit = 2.0**-20
    assert _passes(1000.0 + 0.9 * limit * 5000, 1000.0, 5000.0)
    assert not _passes(1000.0 + 1.5 * limit * 5000, 1000.0, 5000.0)
    # A mean near 0 of samples of large magnitude gets the max_abs scale.
    assert _passes(0.9 * limit * 5000, 0.0, 5000.0)
    assert not _passes(1.5 * limit * 5000, 0.0, 5000.0)


def test_the_variance_tolerance_has_a_floor_from_max_abs():
    limit = 2.0**-20
    # A variance of 0 from samples of magnitude 1000 allows 2^-40 * 1000^2.
    floor = limit * limit * 1000.0**2
    assert _passes(0.9 * floor, 0.0, 1000.0, variance=True)
    assert not _passes(1.5 * floor, 0.0, 1000.0, variance=True)
    assert _passes(100.0 * (1 + 0.9 * limit), 100.0, 1000.0, variance=True)
    assert not _passes(100.0 * (1 + 1.5 * limit), 100.0, 1000.0, variance=True)


# Continuous levels.


def _noisy(count=100, seed=0, offset=5000.0):
    rng = np.random.default_rng(seed)
    samples = (offset + 8.0 * rng.standard_normal(count)).astype(np.float32)
    samples[10:14] = np.nan
    samples[30] = np.inf
    return samples


def test_a_level_that_holds_the_exact_statistics_has_no_findings():
    samples = _noisy()
    assert check_continuous_levels(_array(samples), [_level(samples, 4)]) == []


def test_several_levels_of_one_channel_are_checked_in_one_read():
    samples = _noisy()
    levels = [_level(samples, 4), _level(samples, 16)]
    assert check_continuous_levels(_array(samples), levels) == []


def test_a_level_with_a_short_last_bin_has_no_findings():
    samples = _noisy(103)
    assert check_continuous_levels(_array(samples), [_level(samples, 16)]) == []


def test_a_wrong_count_is_reported_with_its_first_bin():
    samples = _noisy()
    count = _level(samples, 4).count[:]
    count[3] += 1
    count[9] += 1
    findings = check_continuous_levels(
        _array(samples), [_level(samples, 4, count=count)]
    )
    assert [f.path for f in findings] == ["0/1/count"]
    assert findings[0].section == "8.4"
    assert "bin 3" in findings[0].message
    assert "2 bins differ" in findings[0].message


def test_minmax_must_match_exactly():
    samples = _noisy()
    minmax = _level(samples, 4).minmax[:]
    minmax[0, 0] = np.nextafter(minmax[0, 0], np.float32(0))
    findings = check_continuous_levels(
        _array(samples), [_level(samples, 4, minmax=minmax)]
    )
    assert [f.path for f in findings] == ["0/1/minmax"]


def test_minus_zero_equals_plus_zero_in_minmax():
    samples = np.array([0.0, 1.0, 2.0, 3.0], np.float32)
    minmax = np.array([[-0.0, 1.0], [2.0, 3.0]], np.float32)
    assert (
        check_continuous_levels(
            _array(samples), [_level(samples, 2, minmax=minmax)]
        )
        == []
    )


def test_an_empty_bin_must_store_nan_in_every_member():
    samples = np.array([np.nan, np.nan, 1.0, 2.0], np.float32)
    mean = np.array([0.0, 1.5], np.float32)
    findings = check_continuous_levels(
        _array(samples), [_level(samples, 2, mean=mean)]
    )
    assert [f.path for f in findings] == ["0/1/mean"]


def test_a_mean_within_the_tolerance_passes_and_one_outside_fails():
    samples = _noisy()
    mean = _level(samples, 4).mean[:]
    near = mean.copy()
    near[0] = np.nextafter(near[0], np.float32(1e9))
    assert (
        check_continuous_levels(
            _array(samples), [_level(samples, 4, mean=near)]
        )
        == []
    )
    far = mean.copy()
    far[0] += 0.01
    findings = check_continuous_levels(
        _array(samples), [_level(samples, 4, mean=far)]
    )
    assert [f.path for f in findings] == ["0/1/mean"]


def test_the_variance_of_a_channel_with_a_large_offset_is_checked_in_binary64():
    samples = _noisy(offset=1e6)
    assert check_continuous_levels(_array(samples), [_level(samples, 8)]) == []


def test_an_overflowing_variance_must_be_positive_infinity():
    samples = np.array([3e38, -3e38, 3e38, -3e38], np.float32)
    exact = _level(samples, 4)
    assert exact.var[0] == np.inf
    assert check_continuous_levels(_array(samples), [exact]) == []
    finite = _level(samples, 4, var=np.array([3e38], np.float32))
    findings = check_continuous_levels(_array(samples), [finite])
    assert [f.path for f in findings] == ["0/1/var"]


def test_a_level_without_mean_and_var_checks_the_other_members():
    samples = _noisy()
    level = _level(samples, 4)
    level = LevelArrays(level.path, 4, level.count, level.minmax, None, None)
    assert check_continuous_levels(_array(samples), [level]) == []


def test_no_levels_read_nothing():
    assert check_continuous_levels(_array(np.zeros(4, np.float32)), []) == []


# Samples outside the runs.


def test_raw_samples_outside_every_run_must_be_nan():
    table = np.array([[0, 4, 0], [8, 4, 4000]], np.int64)
    samples = np.full(12, 1.0, np.float32)
    samples[4:8] = np.nan
    assert check_raw_outside_runs("0/raw", _array(samples), table) == []
    samples[6] = 0.0
    findings = check_raw_outside_runs("0/raw", _array(samples), table)
    assert [(f.path, f.section) for f in findings] == [("0/raw", "4.2")]
    assert findings[0].message.startswith("raw sample 6 lies outside")


# Count levels.


def _counts(events, labels, label_count, offset, period):
    bins = (events[-1] - offset) // period + 1
    counts = np.zeros((bins, label_count), np.uint32)
    for time, label in zip(events, labels, strict=True):
        counts[(time - offset) // period, label] += 1
    return counts


def test_a_count_level_that_matches_the_events_has_no_findings():
    events = np.array([5, 5, 120, 999, 1000, 2500], np.int64)
    labels = np.array([0, 1, 1, 0, 2, 2], np.uint16)
    counts = _counts(events, labels, 3, -1000, 1000)
    assert (
        check_count_level(
            "1/1/count", events, labels, 3, -1000, 1000, _array(counts)
        )
        == []
    )


def test_a_count_level_without_labels_counts_every_event_as_label_0():
    events = np.array([0, 10, 25], np.int64)
    counts = _counts(events, [0, 0, 0], 1, 0, 10)
    assert check_count_level("c", events, None, 1, 0, 10, _array(counts)) == []


def test_a_wrong_count_is_reported_with_its_bin_and_label():
    events = np.array([0, 10, 25], np.int64)
    labels = np.array([0, 1, 1], np.uint16)
    counts = _counts(events, labels, 2, 0, 10)
    counts[2, 1] = 0
    counts[0, 0] = 5
    findings = check_count_level("c", events, labels, 2, 0, 10, _array(counts))
    assert len(findings) == 1
    assert findings[0].section == "8.4"
    assert "bin 0 holds 5 events of label 0" in findings[0].message
    assert "2 counts differ" in findings[0].message


def test_a_count_array_of_the_wrong_shape_is_left_to_structural_validation():
    events = np.array([0, 10], np.int64)
    assert (
        check_count_level(
            "c", events, None, 1, 0, 10, _array(np.zeros((5, 1), np.uint32))
        )
        == []
    )


# Waveform means.


def _waveforms():
    waveforms = np.array(
        [
            [1.0, 2.0, np.nan],
            [3.0, np.inf, np.nan],
            [5.0, 6.0, 7.0],
            [9.0, 9.0, 9.0],
        ],
        np.float32,
    )
    labels = np.array([0, 0, 2, 0], np.uint16)
    return waveforms, labels


def test_waveform_mean_over_finite_values_has_no_findings():
    waveforms, labels = _waveforms()
    mean = np.array(
        [[13 / 3, 11 / 2, 9.0], [np.nan] * 3, [5.0, 6.0, 7.0]], np.float32
    )
    assert (
        check_waveform_mean("w", _array(waveforms), _array(mean), labels, 3)
        == []
    )


def test_a_label_without_events_must_hold_nan():
    waveforms, labels = _waveforms()
    mean = np.array(
        [[13 / 3, 11 / 2, 9.0], [0.0] * 3, [5.0, 6.0, 7.0]], np.float32
    )
    findings = check_waveform_mean(
        "w", _array(waveforms), _array(mean), labels, 3
    )
    assert len(findings) == 1
    assert "row 1 sample 0" in findings[0].message
    assert "3 values differ" in findings[0].message


def test_a_waveform_mean_off_by_more_than_the_tolerance_is_reported():
    waveforms, labels = _waveforms()
    mean = np.array(
        [[13 / 3, 11 / 2, 9.0], [np.nan] * 3, [5.0, 6.5, 7.0]], np.float32
    )
    findings = check_waveform_mean(
        "w", _array(waveforms), _array(mean), labels, 3
    )
    assert "row 2 sample 1" in findings[0].message


def test_a_label_at_or_above_the_label_count_is_ignored_by_waveform_mean():
    waveforms, labels = _waveforms()
    labels[3] = 7
    mean = np.array(
        [[2.0, 2.0, np.nan], [np.nan] * 3, [5.0, 6.0, 7.0]], np.float32
    )
    assert (
        check_waveform_mean("w", _array(waveforms), _array(mean), labels, 3)
        == []
    )
